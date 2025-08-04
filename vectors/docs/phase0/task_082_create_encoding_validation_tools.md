# Micro-Task 082: Create Encoding Validation Tools

## Objective
Create tools to validate that all test data files use proper UTF-8 encoding and handle Windows-specific encoding requirements.

## Context
Windows file systems and text processing can have encoding issues that affect vector search functionality. These tools will ensure all test data is properly encoded and can be processed correctly on Windows systems.

## Prerequisites
- Task 081 completed (Pattern reference guide created)

## Time Estimate
8 minutes

## Instructions
1. Navigate to test data directory: `cd data\test_files`
2. Create `validate_encoding.py`:
   ```python
   #!/usr/bin/env python3
   """
   Encoding validation tool for test data files.
   Ensures all files are properly UTF-8 encoded for Windows compatibility.
   """
   
   import os
   import sys
   import chardet
   from pathlib import Path
   
   def validate_utf8_encoding(file_path):
       """Validate that a file is UTF-8 encoded."""
       try:
           with open(file_path, 'rb') as f:
               raw_data = f.read()
           
           # Check if it's valid UTF-8
           try:
               raw_data.decode('utf-8')
               return True, "Valid UTF-8"
           except UnicodeDecodeError as e:
               return False, f"UTF-8 decode error: {e}"
       
       except Exception as e:
           return False, f"File read error: {e}"
   
   def detect_encoding(file_path):
       """Detect the encoding of a file."""
       try:
           with open(file_path, 'rb') as f:
               raw_data = f.read()
           
           result = chardet.detect(raw_data)
           return result['encoding'], result['confidence']
       
       except Exception as e:
           return None, 0.0
   
   def check_bom(file_path):
       """Check if file has UTF-8 BOM (which can cause issues)."""
       try:
           with open(file_path, 'rb') as f:
               first_bytes = f.read(3)
           
           # UTF-8 BOM is EF BB BF
           if first_bytes == b'\xef\xbb\xbf':
               return True
           return False
       
       except Exception as e:
           return False
   
   def validate_directory(directory_path):
       """Validate all files in a directory."""
       directory = Path(directory_path)
       results = []
       
       for file_path in directory.rglob('*'):
           if file_path.is_file() and not file_path.name.startswith('.'):
               # Skip binary files and executables
               if file_path.suffix.lower() in ['.exe', '.dll', '.so', '.dylib', '.bin']:
                   continue
               
               is_valid, message = validate_utf8_encoding(file_path)
               encoding, confidence = detect_encoding(file_path)
               has_bom = check_bom(file_path)
               
               results.append({
                   'file': str(file_path),
                   'valid_utf8': is_valid,
                   'message': message,
                   'detected_encoding': encoding,
                   'confidence': confidence,
                   'has_bom': has_bom
               })
       
       return results
   
   def main():
       """Main validation function."""
       if len(sys.argv) > 1:
           target_dir = sys.argv[1]
       else:
           target_dir = '.'
       
       print(f"Validating encoding for files in: {target_dir}")
       print("=" * 60)
       
       results = validate_directory(target_dir)
       
       valid_count = 0
       invalid_count = 0
       bom_count = 0
       
       for result in results:
           status = "✓" if result['valid_utf8'] else "✗"
           bom_status = " (BOM)" if result['has_bom'] else ""
           
           print(f"{status} {result['file']}")
           print(f"   Encoding: {result['detected_encoding']} (confidence: {result['confidence']:.2f}){bom_status}")
           
           if not result['valid_utf8']:
               print(f"   Error: {result['message']}")
               invalid_count += 1
           else:
               valid_count += 1
           
           if result['has_bom']:
               bom_count += 1
           
           print()
       
       print("=" * 60)
       print(f"Summary:")
       print(f"  Valid UTF-8 files: {valid_count}")
       print(f"  Invalid files: {invalid_count}")
       print(f"  Files with BOM: {bom_count}")
       
       if invalid_count > 0:
           print(f"\n⚠️  {invalid_count} files have encoding issues!")
           return 1
       
       if bom_count > 0:
           print(f"\n⚠️  {bom_count} files have UTF-8 BOM (may cause parsing issues)")
       
       print("\n✅ All files are properly encoded!")
       return 0
   
   if __name__ == "__main__":
       sys.exit(main())
   ```
3. Create `fix_encoding.py`:
   ```python
   #!/usr/bin/env python3
   """
   Encoding fix tool for test data files.
   Converts files to proper UTF-8 encoding without BOM.
   """
   
   import os
   import sys
   import shutil
   from pathlib import Path
   
   def remove_bom(file_path):
       """Remove UTF-8 BOM from a file."""
       try:
           with open(file_path, 'rb') as f:
               content = f.read()
           
           # Check for UTF-8 BOM
           if content.startswith(b'\xef\xbb\xbf'):
               # Remove BOM and write back
               with open(file_path, 'wb') as f:
                   f.write(content[3:])
               return True, "BOM removed"
           else:
               return False, "No BOM found"
       
       except Exception as e:
           return False, f"Error: {e}"
   
   def convert_to_utf8(file_path, source_encoding):
       """Convert a file from source encoding to UTF-8."""
       try:
           # Backup original file
           backup_path = f"{file_path}.backup"
           shutil.copy2(file_path, backup_path)
           
           # Read with source encoding
           with open(file_path, 'r', encoding=source_encoding) as f:
               content = f.read()
           
           # Write as UTF-8 without BOM
           with open(file_path, 'w', encoding='utf-8') as f:
               f.write(content)
           
           return True, f"Converted from {source_encoding} to UTF-8"
       
       except Exception as e:
           # Restore backup if conversion failed
           if os.path.exists(f"{file_path}.backup"):
               shutil.move(f"{file_path}.backup", file_path)
           return False, f"Conversion failed: {e}"
   
   def main():
       """Main fix function."""
       if len(sys.argv) > 1:
           target_dir = sys.argv[1]
       else:
           target_dir = '.'
       
       print(f"Fixing encoding for files in: {target_dir}")
       print("=" * 60)
       
       # First run validation to identify issues
       from validate_encoding import validate_directory
       results = validate_directory(target_dir)
       
       fixed_count = 0
       error_count = 0
       
       for result in results:
           file_path = result['file']
           
           # Remove BOM if present
           if result['has_bom']:
               success, message = remove_bom(file_path)
               if success:
                   print(f"✓ {file_path}: {message}")
                   fixed_count += 1
               else:
                   print(f"✗ {file_path}: {message}")
                   error_count += 1
           
           # Convert encoding if not UTF-8
           if not result['valid_utf8'] and result['detected_encoding']:
               if result['detected_encoding'].lower() not in ['utf-8', 'ascii']:
                   success, message = convert_to_utf8(file_path, result['detected_encoding'])
                   if success:
                       print(f"✓ {file_path}: {message}")
                       fixed_count += 1
                   else:
                       print(f"✗ {file_path}: {message}")
                       error_count += 1
       
       print("=" * 60)
       print(f"Summary:")
       print(f"  Files fixed: {fixed_count}")
       print(f"  Errors: {error_count}")
       
       if error_count > 0:
           print(f"\n⚠️  {error_count} files could not be fixed!")
           return 1
       
       print("\n✅ All encoding issues fixed!")
       return 0
   
   if __name__ == "__main__":
       sys.exit(main())
   ```
4. Create `encoding_check.bat`:
   ```batch
   @echo off
   echo Testing encoding validation tools...
   
   REM Check if Python is available
   python --version >nul 2>&1
   if errorlevel 1 (
       echo ERROR: Python is not installed or not in PATH
       exit /b 1
   )
   
   REM Install required packages if needed
   echo Installing required Python packages...
   pip install chardet >nul 2>&1
   
   REM Run encoding validation
   echo Running encoding validation...
   python validate_encoding.py .
   
   if errorlevel 1 (
       echo.
       echo Encoding issues found. Running fix tool...
       python fix_encoding.py .
   ) else (
       echo.
       echo All files are properly encoded!
   )
   
   echo.
   echo Encoding validation complete.
   ```
5. Return to root: `cd ..\..`
6. Commit: `git add data\test_files\validate_encoding.py data\test_files\fix_encoding.py data\test_files\encoding_check.bat && git commit -m "task_082: Create encoding validation and fix tools"`

## Expected Output
- Python script to validate UTF-8 encoding
- Python script to fix encoding issues
- Batch script for Windows automation
- BOM detection and removal capability

## Success Criteria
- [ ] Encoding validation script created
- [ ] Encoding fix script created
- [ ] Windows batch automation script created
- [ ] BOM detection implemented
- [ ] All tools committed to Git

## Validation Commands
```cmd
cd data\test_files
python validate_encoding.py .
encoding_check.bat
```

## Next Task
task_083_setup_file_generation_templates.md

## Notes
- Tools handle Windows-specific encoding requirements
- BOM detection prevents parsing issues in text processing
- Backup functionality protects against conversion errors
- Compatible with Windows Python installations