"""
Configuration settings for the calculator application.
"""

import json
import os
from typing import Dict, Any

class CalculatorConfig:
    """Configuration manager for calculator settings."""
    
    DEFAULT_CONFIG = {
        "precision": 6,
        "angle_mode": "degrees",  # degrees or radians
        "history_limit": 100,
        "auto_save_history": True,
        "history_file": "calculator_history.json",
        "gui_theme": "default",
        "window_size": "400x600",
        "scientific_notation_threshold": 1e6,
        "enable_sound": False,
        "recent_operations_limit": 10
    }
    
    def __init__(self, config_file="calculator_config.json"):
        self.config_file = config_file
        self.config = self.DEFAULT_CONFIG.copy()
        self.load_config()
    
    def load_config(self):
        """Load configuration from file."""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    file_config = json.load(f)
                    self.config.update(file_config)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load config file: {e}")
                print("Using default configuration.")
    
    def save_config(self):
        """Save current configuration to file."""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            return True
        except IOError as e:
            print(f"Error saving config: {e}")
            return False
    
    def get(self, key: str, default=None) -> Any:
        """Get configuration value."""
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any):
        """Set configuration value."""
        self.config[key] = value
    
    def reset_to_defaults(self):
        """Reset configuration to default values."""
        self.config = self.DEFAULT_CONFIG.copy()
    
    def get_precision(self) -> int:
        """Get current precision setting."""
        return self.config.get("precision", 6)
    
    def set_precision(self, precision: int):
        """Set precision for calculations."""
        if 0 <= precision <= 15:
            self.config["precision"] = precision
        else:
            raise ValueError("Precision must be between 0 and 15")
    
    def is_degrees_mode(self) -> bool:
        """Check if angle mode is degrees."""
        return self.config.get("angle_mode", "degrees") == "degrees"
    
    def set_angle_mode(self, mode: str):
        """Set angle mode (degrees or radians)."""
        if mode in ["degrees", "radians"]:
            self.config["angle_mode"] = mode
        else:
            raise ValueError("Angle mode must be 'degrees' or 'radians'")
    
    def get_history_limit(self) -> int:
        """Get maximum number of history entries."""
        return self.config.get("history_limit", 100)
    
    def should_auto_save_history(self) -> bool:
        """Check if history should be automatically saved."""
        return self.config.get("auto_save_history", True)
    
    def get_history_file(self) -> str:
        """Get history file path."""
        return self.config.get("history_file", "calculator_history.json")

# Global configuration instance
config = CalculatorConfig()