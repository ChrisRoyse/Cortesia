"""
File with intentional duplicate code blocks to test deduplication algorithms.
This simulates real-world scenarios where code is copy-pasted and modified slightly.
"""

import json
import time
from typing import Dict, List, Any, Optional
from datetime import datetime

# Duplicate Block 1 - Exact Copy
def calculate_user_score_v1(user_data: Dict[str, Any]) -> float:
    """Calculate user score based on activity metrics."""
    if not user_data:
        return 0.0
    
    base_score = 100.0
    posts_count = user_data.get('posts_count', 0)
    comments_count = user_data.get('comments_count', 0)
    likes_received = user_data.get('likes_received', 0)
    
    # Calculate score components
    post_score = posts_count * 5.0
    comment_score = comments_count * 2.0
    like_score = likes_received * 1.5
    
    total_score = base_score + post_score + comment_score + like_score
    
    # Apply multipliers
    if user_data.get('is_verified', False):
        total_score *= 1.2
    
    if user_data.get('account_age_days', 0) > 365:
        total_score *= 1.1
    
    return min(total_score, 1000.0)

# Duplicate Block 1 - Exact Copy (identical function)
def calculate_user_score_v2(user_data: Dict[str, Any]) -> float:
    """Calculate user score based on activity metrics."""
    if not user_data:
        return 0.0
    
    base_score = 100.0
    posts_count = user_data.get('posts_count', 0)
    comments_count = user_data.get('comments_count', 0)
    likes_received = user_data.get('likes_received', 0)
    
    # Calculate score components
    post_score = posts_count * 5.0
    comment_score = comments_count * 2.0
    like_score = likes_received * 1.5
    
    total_score = base_score + post_score + comment_score + like_score
    
    # Apply multipliers
    if user_data.get('is_verified', False):
        total_score *= 1.2
    
    if user_data.get('account_age_days', 0) > 365:
        total_score *= 1.1
    
    return min(total_score, 1000.0)

# Duplicate Block 2 - Minor Variations (variable names changed)
def calculate_user_score_v3(user_info: Dict[str, Any]) -> float:
    """Calculate user score based on activity metrics."""
    if not user_info:
        return 0.0
    
    initial_score = 100.0
    num_posts = user_info.get('posts_count', 0)
    num_comments = user_info.get('comments_count', 0)
    received_likes = user_info.get('likes_received', 0)
    
    # Calculate score components
    posts_points = num_posts * 5.0
    comments_points = num_comments * 2.0
    likes_points = received_likes * 1.5
    
    final_score = initial_score + posts_points + comments_points + likes_points
    
    # Apply multipliers
    if user_info.get('is_verified', False):
        final_score *= 1.2
    
    if user_info.get('account_age_days', 0) > 365:
        final_score *= 1.1
    
    return min(final_score, 1000.0)

# Duplicate Block 3 - Similar Logic, Different Implementation
def compute_user_rating(user_profile: Dict[str, Any]) -> float:
    """Compute user rating based on engagement."""
    if user_profile is None or len(user_profile) == 0:
        return 0.0
    
    starting_points = 100.0
    
    # Get metrics
    posts = user_profile.get('posts_count', 0)
    comments = user_profile.get('comments_count', 0) 
    likes = user_profile.get('likes_received', 0)
    
    # Score calculation
    post_contribution = posts * 5
    comment_contribution = comments * 2
    like_contribution = likes * 1.5
    
    score = starting_points + post_contribution + comment_contribution + like_contribution
    
    # Bonus calculations
    verified_bonus = 1.2 if user_profile.get('is_verified') else 1.0
    veteran_bonus = 1.1 if user_profile.get('account_age_days', 0) > 365 else 1.0
    
    score = score * verified_bonus * veteran_bonus
    
    return score if score <= 1000.0 else 1000.0

# Duplicate Block 4 - Data Validation (Exact Copy)
def validate_user_input(input_data: Dict[str, Any]) -> List[str]:
    """Validate user input data and return list of errors."""
    errors = []
    
    # Required fields validation
    required_fields = ['username', 'email', 'password']
    for field in required_fields:
        if field not in input_data or not input_data[field]:
            errors.append(f"Missing required field: {field}")
    
    # Username validation
    if 'username' in input_data:
        username = input_data['username']
        if len(username) < 3:
            errors.append("Username must be at least 3 characters long")
        if len(username) > 20:
            errors.append("Username must be no more than 20 characters long")
        if not username.replace('_', '').isalnum():
            errors.append("Username can only contain letters, numbers, and underscores")
    
    # Email validation
    if 'email' in input_data:
        email = input_data['email']
        if '@' not in email or '.' not in email:
            errors.append("Invalid email format")
    
    # Password validation
    if 'password' in input_data:
        password = input_data['password']
        if len(password) < 8:
            errors.append("Password must be at least 8 characters long")
        if not any(c.isupper() for c in password):
            errors.append("Password must contain at least one uppercase letter")
        if not any(c.islower() for c in password):
            errors.append("Password must contain at least one lowercase letter")
        if not any(c.isdigit() for c in password):
            errors.append("Password must contain at least one digit")
    
    return errors

# Duplicate Block 4 - Data Validation (Exact Copy with different function name)
def check_user_input(input_data: Dict[str, Any]) -> List[str]:
    """Validate user input data and return list of errors."""
    errors = []
    
    # Required fields validation
    required_fields = ['username', 'email', 'password']
    for field in required_fields:
        if field not in input_data or not input_data[field]:
            errors.append(f"Missing required field: {field}")
    
    # Username validation
    if 'username' in input_data:
        username = input_data['username']
        if len(username) < 3:
            errors.append("Username must be at least 3 characters long")
        if len(username) > 20:
            errors.append("Username must be no more than 20 characters long")
        if not username.replace('_', '').isalnum():
            errors.append("Username can only contain letters, numbers, and underscores")
    
    # Email validation
    if 'email' in input_data:
        email = input_data['email']
        if '@' not in email or '.' not in email:
            errors.append("Invalid email format")
    
    # Password validation
    if 'password' in input_data:
        password = input_data['password']
        if len(password) < 8:
            errors.append("Password must be at least 8 characters long")
        if not any(c.isupper() for c in password):
            errors.append("Password must contain at least one uppercase letter")
        if not any(c.islower() for c in password):
            errors.append("Password must contain at least one lowercase letter")
        if not any(c.isdigit() for c in password):
            errors.append("Password must contain at least one digit")
    
    return errors

# Duplicate Block 5 - Similar but with minor modifications
def validate_form_data(form_input: Dict[str, Any]) -> List[str]:
    """Validate form input and return validation errors."""
    validation_errors = []
    
    # Check required fields
    mandatory_fields = ['username', 'email', 'password']
    for field_name in mandatory_fields:
        if field_name not in form_input or not form_input[field_name]:
            validation_errors.append(f"Missing required field: {field_name}")
    
    # Validate username
    if 'username' in form_input:
        user_name = form_input['username']
        if len(user_name) < 3:
            validation_errors.append("Username must be at least 3 characters long")
        if len(user_name) > 20:
            validation_errors.append("Username must be no more than 20 characters long")
        if not user_name.replace('_', '').isalnum():
            validation_errors.append("Username can only contain letters, numbers, and underscores")
    
    # Validate email address
    if 'email' in form_input:
        email_address = form_input['email']
        if '@' not in email_address or '.' not in email_address:
            validation_errors.append("Invalid email format")
    
    # Validate password strength
    if 'password' in form_input:
        user_password = form_input['password']
        if len(user_password) < 8:
            validation_errors.append("Password must be at least 8 characters long")
        if not any(char.isupper() for char in user_password):
            validation_errors.append("Password must contain at least one uppercase letter")
        if not any(char.islower() for char in user_password):
            validation_errors.append("Password must contain at least one lowercase letter")
        if not any(char.isdigit() for char in user_password):
            validation_errors.append("Password must contain at least one digit")
    
    return validation_errors

# Duplicate Block 6 - Database Query Pattern (Exact Copy)
def get_user_posts(user_id: int, limit: int = 10, offset: int = 0) -> List[Dict[str, Any]]:
    """Retrieve user posts from database."""
    query = """
    SELECT 
        id,
        title,
        content,
        created_at,
        updated_at,
        view_count,
        like_count,
        comment_count
    FROM posts 
    WHERE user_id = ? 
        AND status = 'published'
        AND deleted_at IS NULL
    ORDER BY created_at DESC
    LIMIT ? OFFSET ?
    """
    
    # Simulate database connection and query execution
    try:
        # This would normally use a real database connection
        # cursor.execute(query, (user_id, limit, offset))
        # results = cursor.fetchall()
        
        # Simulated results
        results = []
        for i in range(min(limit, 5)):  # Simulate up to 5 posts
            results.append({
                'id': i + 1 + offset,
                'title': f'Post {i + 1 + offset}',
                'content': f'Content for post {i + 1 + offset}',
                'created_at': datetime.now(),
                'updated_at': datetime.now(),
                'view_count': i * 10,
                'like_count': i * 2,
                'comment_count': i
            })
        
        return results
        
    except Exception as e:
        print(f"Database error: {e}")
        return []

# Duplicate Block 6 - Database Query Pattern (Exact Copy with different name)
def fetch_user_posts(user_id: int, limit: int = 10, offset: int = 0) -> List[Dict[str, Any]]:
    """Retrieve user posts from database."""
    query = """
    SELECT 
        id,
        title,
        content,
        created_at,
        updated_at,
        view_count,
        like_count,
        comment_count
    FROM posts 
    WHERE user_id = ? 
        AND status = 'published'
        AND deleted_at IS NULL
    ORDER BY created_at DESC
    LIMIT ? OFFSET ?
    """
    
    # Simulate database connection and query execution
    try:
        # This would normally use a real database connection
        # cursor.execute(query, (user_id, limit, offset))
        # results = cursor.fetchall()
        
        # Simulated results
        results = []
        for i in range(min(limit, 5)):  # Simulate up to 5 posts
            results.append({
                'id': i + 1 + offset,
                'title': f'Post {i + 1 + offset}',
                'content': f'Content for post {i + 1 + offset}',
                'created_at': datetime.now(),
                'updated_at': datetime.now(),
                'view_count': i * 10,
                'like_count': i * 2,
                'comment_count': i
            })
        
        return results
        
    except Exception as e:
        print(f"Database error: {e}")
        return []

# Duplicate Block 7 - Similar query with modifications
def load_user_articles(user_id: int, max_results: int = 10, start_index: int = 0) -> List[Dict[str, Any]]:
    """Load user articles from the database."""
    sql_query = """
    SELECT 
        id,
        title,
        content,
        created_at,
        updated_at,
        view_count,
        like_count,
        comment_count
    FROM posts 
    WHERE user_id = ? 
        AND status = 'published'
        AND deleted_at IS NULL
    ORDER BY created_at DESC
    LIMIT ? OFFSET ?
    """
    
    # Database operations
    try:
        # Simulated database interaction
        post_list = []
        for index in range(min(max_results, 5)):
            post_list.append({
                'id': index + 1 + start_index,
                'title': f'Post {index + 1 + start_index}',
                'content': f'Content for post {index + 1 + start_index}',
                'created_at': datetime.now(),
                'updated_at': datetime.now(),
                'view_count': index * 10,
                'like_count': index * 2,
                'comment_count': index
            })
        
        return post_list
        
    except Exception as error:
        print(f"Database error: {error}")
        return []

# Duplicate Block 8 - Configuration Loading (Exact Copy)
def load_application_config(config_path: str = "config.json") -> Dict[str, Any]:
    """Load application configuration from JSON file."""
    default_config = {
        "database": {
            "host": "localhost",
            "port": 5432,
            "name": "myapp",
            "user": "admin",
            "password": "password"
        },
        "cache": {
            "enabled": True,
            "ttl": 3600,
            "max_size": 1000
        },
        "logging": {
            "level": "INFO",
            "file": "app.log",
            "max_size": "10MB",
            "backup_count": 5
        },
        "features": {
            "user_registration": True,
            "email_verification": True,
            "social_login": False,
            "api_rate_limiting": True
        }
    }
    
    try:
        with open(config_path, 'r') as config_file:
            user_config = json.load(config_file)
            
        # Merge with defaults
        final_config = default_config.copy()
        for key, value in user_config.items():
            if isinstance(value, dict) and key in final_config:
                final_config[key].update(value)
            else:
                final_config[key] = value
        
        return final_config
        
    except FileNotFoundError:
        print(f"Config file {config_path} not found, using defaults")
        return default_config
    except json.JSONDecodeError as e:
        print(f"Error parsing config file: {e}")
        return default_config

# Duplicate Block 8 - Configuration Loading (Exact Copy with different name)
def read_app_configuration(config_file_path: str = "config.json") -> Dict[str, Any]:
    """Load application configuration from JSON file."""
    default_config = {
        "database": {
            "host": "localhost",
            "port": 5432,
            "name": "myapp",
            "user": "admin",
            "password": "password"
        },
        "cache": {
            "enabled": True,
            "ttl": 3600,
            "max_size": 1000
        },
        "logging": {
            "level": "INFO",
            "file": "app.log",
            "max_size": "10MB",
            "backup_count": 5
        },
        "features": {
            "user_registration": True,
            "email_verification": True,
            "social_login": False,
            "api_rate_limiting": True
        }
    }
    
    try:
        with open(config_file_path, 'r') as config_file:
            user_config = json.load(config_file)
            
        # Merge with defaults
        final_config = default_config.copy()
        for key, value in user_config.items():
            if isinstance(value, dict) and key in final_config:
                final_config[key].update(value)
            else:
                final_config[key] = value
        
        return final_config
        
    except FileNotFoundError:
        print(f"Config file {config_file_path} not found, using defaults")
        return default_config
    except json.JSONDecodeError as e:
        print(f"Error parsing config file: {e}")
        return default_config

# Duplicate Block 9 - Nearly identical with whitespace differences
class UserManager:
    """Manage user operations and data."""
    
    def __init__(self, database_connection):
        self.db = database_connection
        self.cache = {}
    
    def create_user(self, user_data: Dict[str, Any]) -> bool:
        """Create a new user account."""
        # Validate input
        validation_errors = validate_user_input(user_data)
        if validation_errors:
            print(f"Validation errors: {validation_errors}")
            return False
        
        # Check if user already exists
        existing_user = self.get_user_by_email(user_data['email'])
        if existing_user:
            print("User with this email already exists")
            return False
        
        # Hash password
        password_hash = self.hash_password(user_data['password'])
        
        # Insert into database
        try:
            query = """
            INSERT INTO users (username, email, password_hash, created_at)
            VALUES (?, ?, ?, ?)
            """
            self.db.execute(query, (
                user_data['username'],
                user_data['email'],
                password_hash,
                datetime.now()
            ))
            self.db.commit()
            return True
        except Exception as e:
            print(f"Database error: {e}")
            return False
    
    def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """Get user by email address."""
        try:
            query = "SELECT * FROM users WHERE email = ?"
            result = self.db.execute(query, (email,)).fetchone()
            return dict(result) if result else None
        except Exception as e:
            print(f"Database error: {e}")
            return None
    
    def hash_password(self, password: str) -> str:
        """Hash password for storage."""
        import hashlib
        return hashlib.sha256(password.encode()).hexdigest()

# Duplicate Block 9 - Almost identical class with minor differences
class UserService:
    """Handle user operations and data management."""
    
    def __init__(self, db_connection):
        self.database = db_connection
        self.user_cache = {}
    
    def create_user(self, user_info: Dict[str, Any]) -> bool:
        """Create a new user account."""
        # Validate the input data
        errors = validate_user_input(user_info)
        if errors:
            print(f"Validation errors: {errors}")
            return False
        
        # Check for existing user
        existing = self.get_user_by_email(user_info['email'])
        if existing:
            print("User with this email already exists")
            return False
        
        # Hash the password
        hashed_password = self.hash_password(user_info['password'])
        
        # Insert user into database
        try:
            sql = """
            INSERT INTO users (username, email, password_hash, created_at)
            VALUES (?, ?, ?, ?)
            """
            self.database.execute(sql, (
                user_info['username'],
                user_info['email'],
                hashed_password,
                datetime.now()
            ))
            self.database.commit()
            return True
        except Exception as error:
            print(f"Database error: {error}")
            return False
    
    def get_user_by_email(self, email_address: str) -> Optional[Dict[str, Any]]:
        """Retrieve user by email address."""
        try:
            sql_query = "SELECT * FROM users WHERE email = ?"
            result = self.database.execute(sql_query, (email_address,)).fetchone()
            return dict(result) if result else None
        except Exception as error:
            print(f"Database error: {error}")
            return None
    
    def hash_password(self, password: str) -> str:
        """Create hash of password for secure storage."""
        import hashlib
        return hashlib.sha256(password.encode()).hexdigest()

# This file contains multiple examples of duplicate code:
# 1. Exact duplicates (same function, different name)
# 2. Near duplicates (same logic, different variable names)
# 3. Structural duplicates (same pattern, different implementation details)
# 4. Whitespace/formatting differences
# 
# These patterns test how well deduplication algorithms can identify:
# - Identical code blocks
# - Semantically equivalent code with different names
# - Similar algorithms with minor variations
# - Code that follows the same structural pattern