#!/usr/bin/env python3
"""Direct test of indexing without gitignore"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from indexer_universal import UniversalIndexer

# Create indexer
indexer = UniversalIndexer(
    root_dir="1_multi_language",
    db_dir="test_db_direct",
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Bypass gitignore
indexer.gitignore_parser.patterns = []

# Collect files manually
root = Path("1_multi_language").resolve()
print(f"Scanning {root}")

doc_files = []
code_files = []
config_files = []

for file_path in root.rglob('*'):
    if file_path.is_file():
        ext = file_path.suffix.lower()
        print(f"Found: {file_path.name} ({ext})")
        
        if ext in ['.md', '.txt', '.rst']:
            doc_files.append(file_path)
        elif ext in ['.json', '.yaml', '.yml', '.toml', '.ini', '.conf', '.xml']:
            config_files.append(file_path)
        elif ext in ['.py', '.rs', '.js', '.jsx', '.ts', '.tsx', '.go', '.java', '.c', '.cpp']:
            code_files.append(file_path)

print(f"\nFiles to index:")
print(f"  Docs: {len(doc_files)}")
print(f"  Code: {len(code_files)}")
print(f"  Config: {len(config_files)}")

# Try indexing
if doc_files or code_files or config_files:
    print("\nRunning indexer...")
    success = indexer.run()
    print(f"Success: {success}")
    
    if success:
        print(f"\nStats:")
        print(f"  Total files: {indexer.stats['total_files']}")
        print(f"  Total chunks: {indexer.stats['total_chunks']}")
        print(f"  Languages: {dict(indexer.stats['languages'])}")
else:
    print("No files to index!")