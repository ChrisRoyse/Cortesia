# Calculator Application

A comprehensive calculator application that has evolved from a simple command-line tool to a full-featured GUI application with advanced mathematical operations.

## Features

- **Basic Operations**: Addition, subtraction, multiplication, division
- **Advanced Operations**: Power, square root, factorial, logarithm, trigonometric functions
- **Graphical Interface**: User-friendly tkinter GUI
- **Calculation History**: Track and review previous calculations
- **Configuration Management**: Customizable settings and preferences
- **Error Handling**: Robust error handling for invalid operations

## Evolution History

This application has undergone several iterations:

1. **Initial Version** (v1.0): Basic command-line calculator with add/subtract
2. **Enhanced Operations** (v1.1): Added multiply, divide, and advanced mathematical functions
3. **Improved Interface** (v1.2): Enhanced command-line interface with better error handling
4. **GUI Implementation** (v2.0): Complete rewrite with tkinter GUI and configuration system

## Usage

### GUI Mode (Recommended)

```python
python gui_calculator.py
```

### Using Calculator Classes

```python
from calculator import Calculator
from advanced_operations import AdvancedCalculator

# Basic calculator
calc = Calculator()
result = calc.add(5, 3)  # Returns 8

# Advanced operations
advanced = AdvancedCalculator()
result = advanced.power(2, 3)  # Returns 8
```

## Configuration

The application supports various configuration options through `config.py`:

- Precision settings
- Angle mode (degrees/radians)
- History limits
- Auto-save preferences
- GUI themes

## File Structure

- `calculator.py` - Core calculator class with basic operations
- `advanced_operations.py` - Advanced mathematical operations
- `gui_calculator.py` - GUI application interface
- `utils.py` - Utility functions for formatting and validation
- `config.py` - Configuration management
- `README.md` - This documentation file

## Dependencies

- Python 3.6+
- tkinter (included with Python)
- math (standard library)

## License

MIT License