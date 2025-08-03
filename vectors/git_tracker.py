#!/usr/bin/env python3
"""
Git Change Tracker - Tracks file modifications for incremental indexing
Integrates with git to identify changed files since last index
"""

import os
import subprocess
import json
from pathlib import Path
from typing import List, Dict, Set, Optional, Tuple
from datetime import datetime
import hashlib


class GitChangeTracker:
    """Track git changes for incremental indexing"""
    
    def __init__(self, repo_path: Path, index_state_file: Path = None):
        """
        Initialize git change tracker
        
        Args:
            repo_path: Path to git repository
            index_state_file: Path to store index state (default: .index_state.json)
        """
        self.repo_path = Path(repo_path).resolve()
        self.index_state_file = index_state_file or (self.repo_path / '.vectors' / '.index_state.json')
        self.index_state_file.parent.mkdir(parents=True, exist_ok=True)
        self.current_state = self._load_state()
        
    def _load_state(self) -> Dict:
        """Load the last index state"""
        if self.index_state_file.exists():
            try:
                with open(self.index_state_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {
            'last_commit': None,
            'last_index_time': None,
            'indexed_files': {},
            'version': '1.0'
        }
        
    def _save_state(self):
        """Save current index state"""
        with open(self.index_state_file, 'w') as f:
            json.dump(self.current_state, f, indent=2)
            
    def _run_git_command(self, cmd: List[str]) -> Optional[str]:
        """Run a git command and return output"""
        try:
            result = subprocess.run(
                ['git'] + cmd,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            return None
            
    def get_current_commit(self) -> Optional[str]:
        """Get current git commit hash"""
        return self._run_git_command(['rev-parse', 'HEAD'])
        
    def get_changed_files(self) -> Dict[str, str]:
        """
        Get all changed files since last index
        
        Returns:
            Dict mapping file paths to change types ('A'dded, 'M'odified, 'D'eleted)
        """
        changes = {}
        
        # Get current commit
        current_commit = self.get_current_commit()
        if not current_commit:
            # Not a git repo, return all files as added
            return self._get_all_files_as_added()
            
        last_commit = self.current_state.get('last_commit')
        
        if not last_commit:
            # First index, get all files
            return self._get_all_files_as_added()
            
        # Get changes between commits
        diff_output = self._run_git_command([
            'diff', '--name-status', last_commit, current_commit
        ])
        
        if diff_output:
            for line in diff_output.split('\n'):
                if line:
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        status, filepath = parts[0], parts[1]
                        changes[filepath] = status[0]  # First char of status
                        
        # Also check for untracked files
        untracked = self._run_git_command(['ls-files', '--others', '--exclude-standard'])
        if untracked:
            for filepath in untracked.split('\n'):
                if filepath:
                    changes[filepath] = 'A'
                    
        # Check for uncommitted changes
        uncommitted = self._run_git_command(['diff', '--name-status'])
        if uncommitted:
            for line in uncommitted.split('\n'):
                if line:
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        status, filepath = parts[0], parts[1]
                        changes[filepath] = 'M'  # Modified
                        
        return changes
        
    def _get_all_files_as_added(self) -> Dict[str, str]:
        """Get all files in repo as added (for initial index)"""
        changes = {}
        
        # Try to use git ls-files
        files_output = self._run_git_command(['ls-files'])
        if files_output:
            for filepath in files_output.split('\n'):
                if filepath:
                    changes[filepath] = 'A'
        else:
            # Fallback to filesystem scan
            for file_path in self.repo_path.rglob('*'):
                if file_path.is_file():
                    try:
                        rel_path = file_path.relative_to(self.repo_path)
                        changes[str(rel_path).replace('\\', '/')] = 'A'
                    except ValueError:
                        pass
                        
        return changes
        
    def get_files_to_reindex(self, supported_extensions: Set[str]) -> Tuple[List[Path], List[Path]]:
        """
        Get files that need to be reindexed
        
        Args:
            supported_extensions: Set of supported file extensions
            
        Returns:
            Tuple of (files_to_index, files_to_delete)
        """
        changes = self.get_changed_files()
        files_to_index = []
        files_to_delete = []
        
        for filepath, status in changes.items():
            file_path = self.repo_path / filepath
            
            # Check if file has supported extension
            if file_path.suffix.lower() in supported_extensions:
                if status == 'D':
                    files_to_delete.append(file_path)
                elif status in ['A', 'M']:
                    if file_path.exists():
                        files_to_index.append(file_path)
                        
        return files_to_index, files_to_delete
        
    def calculate_file_hash(self, file_path: Path) -> str:
        """Calculate hash of file contents"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except:
            return ""
            
    def has_file_changed(self, file_path: Path) -> bool:
        """Check if a specific file has changed since last index"""
        try:
            rel_path = str(file_path.relative_to(self.repo_path)).replace('\\', '/')
        except ValueError:
            return False
        current_hash = self.calculate_file_hash(file_path)
        
        if not current_hash:
            return False
            
        last_hash = self.current_state.get('indexed_files', {}).get(rel_path)
        return current_hash != last_hash
        
    def mark_indexed(self, files: List[Path]):
        """Mark files as indexed with their current hash"""
        for file_path in files:
            try:
                rel_path = str(file_path.relative_to(self.repo_path)).replace('\\', '/')
                file_hash = self.calculate_file_hash(file_path)
                if file_hash:
                    self.current_state['indexed_files'][rel_path] = file_hash
            except:
                pass
                
        self.current_state['last_commit'] = self.get_current_commit()
        self.current_state['last_index_time'] = datetime.now().isoformat()
        self._save_state()
        
    def mark_deleted(self, files: List[Path]):
        """Remove deleted files from index state"""
        for file_path in files:
            try:
                rel_path = str(file_path.relative_to(self.repo_path)).replace('\\', '/')
                self.current_state['indexed_files'].pop(rel_path, None)
            except:
                pass
                
        self._save_state()
        
    def get_index_stats(self) -> Dict:
        """Get statistics about the index state"""
        return {
            'last_commit': self.current_state.get('last_commit'),
            'last_index_time': self.current_state.get('last_index_time'),
            'total_indexed_files': len(self.current_state.get('indexed_files', {})),
            'current_commit': self.get_current_commit()
        }
        
    def reset_state(self):
        """Reset the index state (force full reindex)"""
        self.current_state = {
            'last_commit': None,
            'last_index_time': None,
            'indexed_files': {},
            'version': '1.0'
        }
        self._save_state()


class IncrementalIndexer:
    """Helper class to integrate git tracking with indexing"""
    
    def __init__(self, repo_path: Path, db_path: Path):
        self.repo_path = Path(repo_path)
        self.db_path = Path(db_path)
        self.tracker = GitChangeTracker(repo_path)
        
    def get_incremental_changes(self, supported_extensions: Set[str]) -> Dict:
        """
        Get incremental changes that need processing
        
        Returns:
            Dict with files to add, update, and delete
        """
        files_to_index, files_to_delete = self.tracker.get_files_to_reindex(supported_extensions)
        
        # Separate adds and updates based on whether they existed before
        adds = []
        updates = []
        
        for file_path in files_to_index:
            try:
                rel_path = str(file_path.relative_to(self.repo_path)).replace('\\', '/')
            except ValueError:
                continue
            if rel_path in self.tracker.current_state.get('indexed_files', {}):
                updates.append(file_path)
            else:
                adds.append(file_path)
                
        return {
            'add': adds,
            'update': updates,
            'delete': files_to_delete,
            'stats': self.tracker.get_index_stats()
        }
        
    def should_full_reindex(self) -> bool:
        """Determine if a full reindex is needed"""
        # Full reindex if no previous state
        if not self.tracker.current_state.get('last_commit'):
            return True
            
        # Full reindex if database doesn't exist
        if not self.db_path.exists():
            return True
            
        # Check if too many changes (>50% of files)
        changes = self.tracker.get_changed_files()
        if len(changes) > len(self.tracker.current_state.get('indexed_files', {})) * 0.5:
            return True
            
        return False
        
    def mark_completed(self, indexed_files: List[Path], deleted_files: List[Path]):
        """Mark indexing as completed"""
        self.tracker.mark_indexed(indexed_files)
        self.tracker.mark_deleted(deleted_files)