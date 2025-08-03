#!/usr/bin/env python3
"""
Installation script for MCP RAG Indexer Server
Handles Windows, macOS, and Linux installation
"""

import os
import sys
import json
import shutil
import subprocess
from pathlib import Path
import argparse


def run_command(cmd, check=True):
    """Run a command and return result"""
    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=check, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        if check:
            raise
        return e


def check_prerequisites():
    """Check that required tools are installed"""
    print("Checking prerequisites...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("ERROR: Python 3.8 or newer is required")
        return False
        
    print(f"[OK] Python {sys.version_info.major}.{sys.version_info.minor}")
    
    # Check pip
    try:
        import pip
        print(f"[OK] pip installed")
    except ImportError:
        print("ERROR: pip is not installed")
        return False
        
    # Check git
    try:
        result = run_command(["git", "--version"], check=False)
        if result.returncode == 0:
            print("[OK] Git installed")
        else:
            print("WARNING: Git not found (optional for basic functionality)")
    except FileNotFoundError:
        print("WARNING: Git not found (optional for basic functionality)")
        
    return True


def install_dependencies():
    """Install Python dependencies"""
    print("\nInstalling dependencies...")
    
    # Install from requirements.txt
    requirements_file = Path(__file__).parent / "requirements.txt"
    if requirements_file.exists():
        run_command([sys.executable, "-m", "pip", "install", "-r", str(requirements_file)])
    else:
        # Fallback to manual installation
        packages = [
            "mcp>=1.0.0",
            "langchain>=0.3.0",
            "langchain-community>=0.3.0", 
            "langchain-huggingface>=0.1.0",
            "chromadb>=0.5.0",
            "sentence-transformers>=3.0.0",
            "gitpython>=3.1.0",
            "psutil>=5.9.0",
            "pyyaml>=6.0",
            "tomli>=2.0.0",
            "torch>=2.0.0",
            "numpy>=1.24.0"
        ]
        
        for package in packages:
            run_command([sys.executable, "-m", "pip", "install", package])


def create_executable():
    """Create executable script"""
    print("\nCreating executable...")
    
    script_dir = Path(__file__).parent
    server_script = script_dir / "mcp_rag_server.py"
    
    if not server_script.exists():
        print("ERROR: mcp_rag_server.py not found")
        return False
        
    if sys.platform == "win32":
        # Windows: Create .bat wrapper
        scripts_dir = Path(sys.executable).parent / "Scripts"
        if not scripts_dir.exists():
            # Try user Scripts directory  
            import site
            scripts_dir = Path(site.USER_BASE) / "Scripts"
            scripts_dir.mkdir(exist_ok=True)
            
        bat_file = scripts_dir / "mcp-rag-indexer.bat"
        bat_content = f'''@echo off
python "{server_script}" %*
'''
        bat_file.write_text(bat_content)
        print(f"[OK] Created wrapper: {bat_file}")
        
        # Also try to create .exe if PyInstaller is available
        try:
            import PyInstaller
            exe_file = scripts_dir / "mcp-rag-indexer.exe"
            run_command([
                "pyinstaller", 
                "--onefile",
                "--name", "mcp-rag-indexer",
                "--distpath", str(scripts_dir),
                str(server_script)
            ], check=False)
            if exe_file.exists():
                print(f"[OK] Created executable: {exe_file}")
        except ImportError:
            print("PyInstaller not available - using .bat wrapper")
            
    else:
        # Unix: Create shell script
        bin_dir = Path(sys.executable).parent
        if "site-packages" in str(bin_dir):
            # Virtual environment
            bin_dir = bin_dir.parent / "bin"
        
        script_file = bin_dir / "mcp-rag-indexer"
        script_content = f'''#!/usr/bin/env python3
import sys
sys.path.insert(0, "{script_dir}")
from mcp_rag_server import main
import asyncio
asyncio.run(main())
'''
        script_file.write_text(script_content)
        script_file.chmod(0o755)
        print(f"[OK] Created script: {script_file}")
        
    return True


def configure_claude():
    """Configure Claude Code"""
    print("\nConfiguring Claude Code...")
    
    # Find Claude config file
    if sys.platform == "win32":
        config_file = Path.home() / ".claude.json"
    else:
        config_file = Path.home() / ".claude.json"
        
    # Determine command path
    if sys.platform == "win32":
        import site
        scripts_dir = Path(site.USER_BASE) / "Scripts"
        command_path = str(scripts_dir / "mcp-rag-indexer.exe")
        if not Path(command_path).exists():
            command_path = str(scripts_dir / "mcp-rag-indexer.bat")
        # Escape backslashes for JSON
        command_path = command_path.replace("\\", "\\\\")
    else:
        command_path = "mcp-rag-indexer"
        
    # MCP server configuration
    mcp_config = {
        "rag-indexer": {
            "type": "stdio",
            "command": command_path,
            "args": ["--log-level", "info"]
        }
    }
    
    # Load existing config or create new
    if config_file.exists():
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
        except:
            config = {}
    else:
        config = {}
        
    # Add MCP servers section
    if "mcpServers" not in config:
        config["mcpServers"] = {}
        
    config["mcpServers"].update(mcp_config)
    
    # Save config
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
        
    print(f"[OK] Updated Claude configuration: {config_file}")
    print(f"[OK] Added MCP server: rag-indexer")
    print(f"  Command: {command_path}")
    
    return True


def test_installation():
    """Test the installation"""
    print("\nTesting installation...")
    
    # Test import
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from mcp_rag_server import RAGIndexerServer
        server = RAGIndexerServer()
        print("[OK] MCP server can be imported and initialized")
        server.shutdown()
    except Exception as e:
        print(f"[FAIL] Import test failed: {e}")
        return False
        
    # Test version command
    try:
        script_path = Path(__file__).parent / "mcp_rag_server.py"
        result = run_command([sys.executable, str(script_path), "--version"], check=False)
        if result.returncode == 0:
            print("[OK] Version command works")
        else:
            print("[FAIL] Version command failed")
            return False
    except Exception as e:
        print(f"[FAIL] Version test failed: {e}")
        return False
        
    return True


def main():
    """Main installation function"""
    parser = argparse.ArgumentParser(description="Install MCP RAG Indexer")
    parser.add_argument("--skip-deps", action="store_true", help="Skip dependency installation")
    parser.add_argument("--skip-config", action="store_true", help="Skip Claude configuration")
    parser.add_argument("--test-only", action="store_true", help="Only run tests")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("MCP RAG Indexer - Installation Script")
    print("=" * 60)
    
    if args.test_only:
        success = test_installation()
        sys.exit(0 if success else 1)
        
    if not check_prerequisites():
        print("\nInstallation failed - prerequisites not met")
        sys.exit(1)
        
    try:
        if not args.skip_deps:
            install_dependencies()
            
        if not create_executable():
            print("\nWARNING: Executable creation failed")
            
        if not args.skip_config:
            if not configure_claude():
                print("\nWARNING: Claude configuration failed")
                
        if not test_installation():
            print("\nWARNING: Installation tests failed")
            
        print("\n" + "=" * 60)
        print("INSTALLATION COMPLETE!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Restart Claude Code completely")
        print("2. Check MCP connection: Type '/mcp' in Claude")
        print("3. Index your first project: 'Index C:\\your\\project\\path'")
        print("4. Query the code: 'Find authentication functions'")
        print("\nTroubleshooting:")
        print("- Check logs at: ~/.mcp-rag-indexer/server.log") 
        print("- Run tests: python install_mcp_server.py --test-only")
        print("- Manual config: Edit ~/.claude.json")
        
    except KeyboardInterrupt:
        print("\nInstallation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nInstallation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()