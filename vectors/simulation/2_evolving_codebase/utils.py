"""
Utility functions for the calculator application.
"""

def is_number(value):
    """Check if a value is a number."""
    try:
        float(value)
        return True
    except ValueError:
        return False

def format_result(result):
    """Format calculation result for display."""
    if isinstance(result, float) and result.is_integer():
        return f"Result: {int(result)}"
    elif isinstance(result, float):
        return f"Result: {result:.6f}".rstrip('0').rstrip('.')
    else:
        return f"Result: {result}"

def validate_input(value, input_type="number"):
    """Validate user input based on type."""
    if input_type == "number":
        try:
            return float(value)
        except ValueError:
            raise ValueError("Invalid number format")
    elif input_type == "integer":
        try:
            return int(float(value))
        except ValueError:
            raise ValueError("Invalid integer format")
    elif input_type == "positive":
        try:
            num = float(value)
            if num <= 0:
                raise ValueError("Number must be positive")
            return num
        except ValueError:
            raise ValueError("Invalid positive number format")
    else:
        raise ValueError("Unknown validation type")

def save_history_to_file(history, filename="calculator_history.txt"):
    """Save calculation history to a file."""
    try:
        with open(filename, 'w') as f:
            for item in history:
                f.write(item + '\n')
        return True
    except IOError:
        return False