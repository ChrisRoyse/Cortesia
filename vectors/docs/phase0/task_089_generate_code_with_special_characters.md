# Micro-Task 089: Generate Code with Special Characters

## Objective
Generate code files containing various programming languages with special characters to test vector search handling of code patterns.

## Context
Code files contain special characters that have semantic meaning different from natural language. The vector search system must distinguish between special characters in code contexts versus regular text, maintaining accurate indexing and retrieval.

## Prerequisites
- Task 088 completed (Basic text generation validated)

## Time Estimate
10 minutes

## Instructions
1. Navigate to test data directory: `cd data\test_files`
2. Create code generation script `generate_code_special_chars.py`:
   ```python
   #!/usr/bin/env python3
   """
   Generate code samples with special characters for vector search testing.
   """
   
   import os
   import sys
   from pathlib import Path
   sys.path.append('templates')
   from template_generator import TestFileGenerator
   
   def generate_code_samples():
       """Generate code files with various special character patterns."""
       generator = TestFileGenerator()
       
       # Sample 1: Python code with special characters
       python_code = '''"""
   Python module demonstrating special character usage in code.
   Tests vector search handling of Python-specific syntax patterns.
   """
   
   import sys
   import json
   from typing import Dict, List, Optional, Union, Callable
   from collections import defaultdict
   
   # Dictionary comprehension with special characters
   config = {
       "api_endpoint": "https://api.example.com/v1",
       "timeout": 30,
       "retries": 3,
       "headers": {
           "Content-Type": "application/json",
           "User-Agent": "MyApp/1.0"
       }
   }
   
   # List comprehension with filtering
   numbers = [x**2 for x in range(10) if x % 2 == 0]
   
   # Lambda functions with special operators
   add_one = lambda x: x + 1
   multiply = lambda x, y: x * y
   filter_even = lambda items: [x for x in items if x % 2 == 0]
   
   # Class definition with special methods
   class DataProcessor:
       """Processes data with various operations."""
       
       def __init__(self, name: str, config: Dict[str, Any]):
           self.name = name
           self.config = config
           self._cache: Dict[str, Any] = defaultdict(list)
       
       def __str__(self) -> str:
           return f"DataProcessor(name='{self.name}')"
       
       def __repr__(self) -> str:
           return f"DataProcessor(name='{self.name}', config={self.config})"
       
       def __getitem__(self, key: str) -> Any:
           return self.config.get(key)
       
       def __setitem__(self, key: str, value: Any) -> None:
           self.config[key] = value
       
       def process_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
           """Process data with transformations."""
           result = []
           
           for item in data:
               # Dictionary manipulation with special characters
               processed = {
                   **item,
                   "processed_at": datetime.now().isoformat(),
                   "checksum": hashlib.md5(str(item).encode()).hexdigest()[:8],
                   "metadata": {
                       "processor": self.name,
                       "version": "1.0.0"
                   }
               }
               
               # Complex conditional with multiple operators
               if (item.get("status") == "active" 
                   and item.get("priority", 0) > 5 
                   and item.get("category") in ["urgent", "high"]):
                   processed["requires_attention"] = True
               
               result.append(processed)
           
           return result
   
   # Function with complex type hints and default parameters
   def analyze_metrics(
       data: List[Dict[str, Union[int, float]]], 
       threshold: float = 0.8,
       groupby: Optional[str] = None,
       transform: Callable[[float], float] = lambda x: x
   ) -> Dict[str, Any]:
       """Analyze metrics with special character operations."""
       
       # Error handling with specific exceptions
       try:
           if not data:
               raise ValueError("Data cannot be empty")
           
           # Complex list operations
           values = [transform(item.get("value", 0)) for item in data]
           filtered = [v for v in values if v >= threshold]
           
           # Statistical calculations with special operators
           result = {
               "count": len(filtered),
               "sum": sum(filtered),
               "avg": sum(filtered) / len(filtered) if filtered else 0,
               "min": min(filtered) if filtered else None,
               "max": max(filtered) if filtered else None,
               "above_threshold": len(filtered) / len(values) * 100
           }
           
           # Nested dictionary operations
           if groupby:
               groups = defaultdict(list)
               for item in data:
                   groups[item.get(groupby, "unknown")].append(item.get("value", 0))
               
               result["groups"] = {
                   k: {
                       "count": len(v),
                       "sum": sum(v),
                       "avg": sum(v) / len(v) if v else 0
                   }
                   for k, v in groups.items()
               }
           
           return result
       
       except (ZeroDivisionError, ValueError, KeyError) as e:
           return {"error": str(e), "type": type(e).__name__}
   
   # Main execution with special character patterns
   if __name__ == "__main__":
       # Command line argument parsing
       import argparse
       
       parser = argparse.ArgumentParser(description="Process data files")
       parser.add_argument("--input", "-i", required=True, help="Input file path")
       parser.add_argument("--output", "-o", default="output.json", help="Output file path")
       parser.add_argument("--threshold", "-t", type=float, default=0.5, help="Threshold value")
       
       args = parser.parse_args()
       
       # File operations with error handling
       try:
           with open(args.input, 'r', encoding='utf-8') as f:
               input_data = json.load(f)
           
           processor = DataProcessor("main_processor", config)
           processed = processor.process_data(input_data)
           metrics = analyze_metrics(processed, args.threshold)
           
           # Output with formatting
           output = {
               "processed_count": len(processed),
               "metrics": metrics,
               "config": processor.config,
               "timestamp": datetime.now().isoformat()
           }
           
           with open(args.output, 'w', encoding='utf-8') as f:
               json.dump(output, f, indent=2, ensure_ascii=False)
           
           print(f"✅ Processed {len(processed)} items -> {args.output}")
       
       except (FileNotFoundError, json.JSONDecodeError, PermissionError) as e:
           print(f"❌ Error: {e}")
           sys.exit(1)'''
       
       # Sample 2: JavaScript/TypeScript with special characters
       javascript_code = '''/**
    * JavaScript/TypeScript module with special character patterns
    * Tests vector search handling of JS-specific syntax
    */
   
   // ES6 imports with destructuring
   import { useState, useEffect, useCallback } from 'react';
   import type { FC, ReactNode, MouseEvent } from 'react';
   import * as utils from './utils';
   import config from '../config.json';
   
   // Type definitions with special characters
   interface User {
     id: string;
     name: string;
     email: string;
     roles: string[];
     metadata?: Record<string, unknown>;
     createdAt: Date;
     updatedAt?: Date;
   }
   
   interface ApiResponse<T> {
     data: T;
     status: 'success' | 'error';
     message?: string;
     pagination?: {
       page: number;
       limit: number;
       total: number;
       hasNext: boolean;
     };
   }
   
   // Generic function with complex type constraints
   const fetchData = async <T extends Record<string, any>>(
     endpoint: string,
     options: RequestInit = {}
   ): Promise<ApiResponse<T>> => {
     try {
       const response = await fetch(`${config.apiBase}/${endpoint}`, {
         ...options,
         headers: {
           'Content-Type': 'application/json',
           'Authorization': `Bearer ${localStorage.getItem('token')}`,
           ...options.headers,
         },
       });
   
       if (!response.ok) {
         throw new Error(`HTTP ${response.status}: ${response.statusText}`);
       }
   
       const data: ApiResponse<T> = await response.json();
       return data;
     } catch (error) {
       console.error('Fetch error:', error);
       throw error;
     }
   };
   
   // React component with hooks and special characters
   const UserManager: FC<{ userId?: string }> = ({ userId }) => {
     const [users, setUsers] = useState<User[]>([]);
     const [loading, setLoading] = useState(false);
     const [error, setError] = useState<string | null>(null);
   
     // Complex useEffect with cleanup
     useEffect(() => {
       let isMounted = true;
       const controller = new AbortController();
   
       const loadUsers = async () => {
         if (!userId) return;
   
         setLoading(true);
         setError(null);
   
         try {
           const response = await fetchData<User[]>('users', {
             signal: controller.signal,
           });
   
           if (isMounted) {
             setUsers(response.data);
           }
         } catch (err) {
           if (isMounted && err.name !== 'AbortError') {
             setError(err instanceof Error ? err.message : 'Unknown error');
           }
         } finally {
           if (isMounted) {
             setLoading(false);
           }
         }
       };
   
       loadUsers();
   
       return () => {
         isMounted = false;
         controller.abort();
       };
     }, [userId]);
   
     // Event handlers with type safety
     const handleUserClick = useCallback((event: MouseEvent<HTMLButtonElement>, user: User) => {
       event.preventDefault();
       
       // Complex object operations
       const userSummary = {
         ...user,
         displayName: `${user.name} <${user.email}>`,
         roleCount: user.roles?.length ?? 0,
         isAdmin: user.roles?.includes('admin') ?? false,
         lastActivity: user.updatedAt ?? user.createdAt,
       };
   
       console.log('User selected:', userSummary);
       
       // Optional chaining and nullish coalescing
       const metadata = user.metadata ?? {};
       const preferences = metadata.preferences as Record<string, any> ?? {};
       const theme = preferences.theme ?? 'light';
       
       document.body.setAttribute('data-theme', theme);
     }, []);
   
     // Array operations with special methods
     const filteredUsers = users
       .filter(user => user.roles.length > 0)
       .sort((a, b) => a.name.localeCompare(b.name))
       .map(user => ({
         ...user,
         initials: user.name
           .split(' ')
           .map(part => part[0])
           .join('')
           .toUpperCase(),
       }));
   
     // Conditional rendering with complex expressions
     if (loading) {
       return <div className="spinner">Loading users...</div>;
     }
   
     if (error) {
       return (
         <div className="error">
           <h3>Error loading users</h3>
           <p>{error}</p>
           <button onClick={() => window.location.reload()}>
             Try Again
           </button>
         </div>
       );
     }
   
     return (
       <div className="user-manager">
         <h2>Users ({filteredUsers.length})</h2>
         {filteredUsers.length === 0 ? (
           <p>No users found</p>
         ) : (
           <ul className="user-list">
             {filteredUsers.map(user => (
               <li key={user.id} className="user-item">
                 <button
                   type="button"
                   onClick={(e) => handleUserClick(e, user)}
                   className={`user-button ${user.roles.includes('admin') ? 'admin' : ''}`}
                 >
                   <span className="user-initials">{user.initials}</span>
                   <div className="user-info">
                     <div className="user-name">{user.name}</div>
                     <div className="user-email">{user.email}</div>
                     <div className="user-roles">
                       {user.roles.join(', ')}
                     </div>
                   </div>
                 </button>
               </li>
             ))}
           </ul>
         )}
       </div>
     );
   };
   
   // Async generator with special syntax
   async function* processUserBatch(users: User[]) {
     const batchSize = 10;
     
     for (let i = 0; i < users.length; i += batchSize) {
       const batch = users.slice(i, i + batchSize);
       
       // Parallel processing with Promise.all
       const processed = await Promise.all(
         batch.map(async (user) => {
           const profile = await fetchData<any>(`users/${user.id}/profile`);
           
           return {
             ...user,
             profile: profile.data,
             processed: true,
             batchNumber: Math.floor(i / batchSize) + 1,
           };
         })
       );
       
       yield processed;
     }
   }
   
   // Export with various patterns
   export default UserManager;
   export { fetchData, processUserBatch };
   export type { User, ApiResponse };'''
       
       # Sample 3: JSON with special characters and nested structures
       json_data = '''{
     "_metadata": {
       "version": "2.1.0",
       "created": "2024-08-04T10:30:00Z",
       "description": "Configuration file with special characters for testing"
     },
     "database": {
       "host": "localhost",
       "port": 5432,
       "name": "test_db",
       "credentials": {
         "username": "${DB_USER:-admin}",
         "password": "${DB_PASS}",
         "connection_string": "postgresql://${DB_USER}:${DB_PASS}@${DB_HOST}:${DB_PORT}/${DB_NAME}"
       },
       "pool": {
         "min_connections": 2,
         "max_connections": 20,
         "timeout": "30s"
       },
       "ssl": {
         "enabled": true,
         "cert_path": "/etc/ssl/certs/db.crt",
         "key_path": "/etc/ssl/private/db.key"
       }
     },
     "api": {
       "base_url": "https://api.example.com/v1",
       "endpoints": {
         "users": {
           "list": "/users",
           "create": "/users",
           "get": "/users/{id}",
           "update": "/users/{id}",
           "delete": "/users/{id}"
         },
         "auth": {
           "login": "/auth/login",
           "logout": "/auth/logout",
           "refresh": "/auth/refresh",
           "verify": "/auth/verify/{token}"
         }
       },
       "rate_limits": {
         "default": "100/hour",
         "authenticated": "1000/hour",
         "admin": "unlimited"
       },
       "cors": {
         "allowed_origins": ["https://app.example.com", "https://admin.example.com"],
         "allowed_methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
         "allowed_headers": ["Content-Type", "Authorization", "X-API-Key"],
         "max_age": 86400
       }
     },
     "logging": {
       "level": "info",
       "format": "{timestamp} [{level}] {message}",
       "outputs": [
         {
           "type": "console",
           "enabled": true,
           "colors": true
         },
         {
           "type": "file",
           "enabled": true,
           "path": "/var/log/app.log",
           "rotation": {
             "max_size": "100MB",
             "max_files": 5,
             "compress": true
           }
         },
         {
           "type": "syslog",
           "enabled": false,
           "facility": "local0",
           "tag": "myapp"
         }
       ]
     },
     "features": {
       "user_registration": {
         "enabled": true,
         "require_email_verification": true,
         "allowed_domains": ["example.com", "*.example.org"],
         "password_policy": {
           "min_length": 8,
           "require_uppercase": true,
           "require_lowercase": true,
           "require_numbers": true,
           "require_symbols": false,
           "banned_passwords": ["password123", "admin", "test"]
         }
       },
       "two_factor_auth": {
         "enabled": true,
         "methods": ["totp", "sms", "email"],
         "backup_codes": {
           "enabled": true,
           "count": 10,
           "length": 8
         }
       },
       "api_keys": {
         "enabled": true,
         "max_per_user": 5,
         "expiration": "1 year",
         "permissions": ["read", "write", "admin"]
       }
     },
     "monitoring": {
       "metrics": {
         "enabled": true,
         "endpoint": "/metrics",
         "format": "prometheus"
       },
       "health_checks": {
         "enabled": true,
         "endpoint": "/health",
         "components": ["database", "redis", "external_api"],
         "timeout": "5s"
       },
       "alerts": {
         "enabled": true,
         "channels": [
           {
             "type": "email",
             "recipients": ["admin@example.com", "alerts@example.com"],
             "conditions": ["error_rate > 5%", "response_time > 2s"]
           },
           {
             "type": "slack",
             "webhook": "https://hooks.slack.com/services/...",
             "channel": "#alerts",
             "conditions": ["service_down", "high_memory_usage"]
           }
         ]
       }
     },
     "cache": {
       "redis": {
         "host": "localhost",
         "port": 6379,
         "database": 0,
         "password": "${REDIS_PASSWORD}",
         "ssl": false,
         "pool_size": 10,
         "timeouts": {
           "connect": "5s",
           "read": "3s",
           "write": "3s"
         }
       },
       "policies": {
         "default_ttl": "1h",
         "max_ttl": "24h",
         "patterns": {
           "user_sessions": "30m",
           "api_responses": "5m",
           "static_content": "1d"
         }
       }
     },
     "security": {
       "jwt": {
         "secret": "${JWT_SECRET}",
         "algorithm": "HS256",
         "expiration": "1h",
         "refresh_expiration": "7d",
         "issuer": "api.example.com",
         "audience": "app.example.com"
       },
       "encryption": {
         "algorithm": "AES-256-GCM",
         "key_derivation": "PBKDF2",
         "iterations": 100000,
         "salt_length": 32
       },
       "headers": {
         "X-Content-Type-Options": "nosniff",
         "X-Frame-Options": "SAMEORIGIN",
         "X-XSS-Protection": "1; mode=block",
         "Strict-Transport-Security": "max-age=31536000; includeSubDomains"
       }
     }
   }'''
       
       # Generate code files
       samples = [
           ("python_special_chars.py", "Python code with special characters", python_code),
           ("javascript_special_chars.js", "JavaScript code with special characters", javascript_code),
           ("config_special_chars.json", "JSON configuration with special characters", json_data)
       ]
       
       generated_files = []
       for filename, pattern_focus, content in samples:
           # Use rust template for .py files, text template for others
           if filename.endswith('.py'):
               output_path = generator.generate_rust_file(
                   filename,
                   "code_samples",
                   pattern_focus,
                   content,
                   "code_samples"
               )
           elif filename.endswith('.js'):
               output_path = generator.generate_rust_file(
                   filename,
                   "code_samples", 
                   pattern_focus,
                   content,
                   "code_samples"
               )
           else:
               output_path = generator.generate_text_file(
                   filename,
                   "code_samples",
                   pattern_focus,
                   content,
                   "code_samples"
               )
           
           generated_files.append(output_path)
           print(f"Generated: {output_path}")
       
       return generated_files
   
   def analyze_special_characters(file_path):
       """Analyze special character usage in code files."""
       with open(file_path, 'r', encoding='utf-8') as f:
           content = f.read()
       
       # Count various special character categories
       brackets = content.count('[') + content.count(']') + content.count('{') + content.count('}') + content.count('(') + content.count(')')
       operators = content.count('=') + content.count('+') + content.count('-') + content.count('*') + content.count('/') + content.count('%')
       comparisons = content.count('==') + content.count('!=') + content.count('<=') + content.count('>=') + content.count('<') + content.count('>')
       logical = content.count('&&') + content.count('||') + content.count('!')
       arrows = content.count('->') + content.count('=>') + content.count('::')
       quotes = content.count('"') + content.count("'") + content.count('`')
       punctuation = content.count(',') + content.count(';') + content.count(':') + content.count('.')
       
       return {
           "total_chars": len(content),
           "brackets": brackets,
           "operators": operators,
           "comparisons": comparisons,
           "logical": logical,
           "arrows": arrows,
           "quotes": quotes,
           "punctuation": punctuation,
           "special_char_density": (brackets + operators + comparisons + logical + arrows + quotes + punctuation) / len(content) * 100 if content else 0
       }
   
   def main():
       """Main generation function."""
       print("Generating code samples with special characters...")
       
       # Ensure output directory exists
       os.makedirs("code_samples", exist_ok=True)
       
       try:
           files = generate_code_samples()
           print(f"\nSuccessfully generated {len(files)} code files:")
           
           # Analyze each file
           for file_path in files:
               print(f"\n  - {file_path}")
               stats = analyze_special_characters(file_path)
               print(f"    Total chars: {stats['total_chars']}")
               print(f"    Brackets: {stats['brackets']}")
               print(f"    Operators: {stats['operators']}")
               print(f"    Arrows: {stats['arrows']}")
               print(f"    Special char density: {stats['special_char_density']:.1f}%")
           
           print("\nCode generation with special characters completed successfully!")
           return 0
       
       except Exception as e:
           print(f"Error generating code files: {e}")
           return 1
   
   if __name__ == "__main__":
       sys.exit(main())
   ```
3. Run generation: `python generate_code_special_chars.py`
4. Return to root: `cd ..\..`
5. Commit: `git add data\test_files\generate_code_special_chars.py data\test_files\code_samples && git commit -m "task_089: Generate code files with special characters"`

## Expected Output
- Python code file with comprehensive special character usage
- JavaScript/TypeScript code file with modern syntax patterns
- JSON configuration file with nested special character structures
- Analysis of special character density and patterns
- Code samples directory with proper organization

## Success Criteria
- [ ] Python code file generated with complex patterns
- [ ] JavaScript code file generated with modern syntax
- [ ] JSON configuration file generated with nested structures
- [ ] Special character analysis performed
- [ ] All files use proper UTF-8 encoding
- [ ] Files follow template format

## Validation Commands
```cmd
cd data\test_files
python generate_code_special_chars.py
dir code_samples
```

## Next Task
task_090_generate_shell_script_patterns.md

## Notes
- Code files test vector search handling of programming syntax
- Special character density analysis helps evaluate complexity
- Multiple languages test cross-language pattern recognition
- Files provide realistic code examples for comprehensive testing