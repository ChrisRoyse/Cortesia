"""
GUI Calculator using tkinter - Replacing command line interface.
"""

import tkinter as tk
from tkinter import ttk, messagebox
from calculator import Calculator
from advanced_operations import AdvancedCalculator
from utils import format_result

class CalculatorGUI:
    """GUI Calculator application."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Advanced Calculator")
        self.root.geometry("400x600")
        self.root.resizable(False, False)
        
        # Initialize calculators
        self.calc = Calculator()
        self.advanced_calc = AdvancedCalculator()
        
        # Variables
        self.display_var = tk.StringVar(value="0")
        self.current_operation = None
        self.first_number = None
        self.waiting_for_operand = False
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the user interface."""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Display
        display_frame = ttk.Frame(main_frame)
        display_frame.grid(row=0, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=(0, 10))
        
        display = ttk.Entry(display_frame, textvariable=self.display_var, 
                           font=("Arial", 16), justify="right", state="readonly")
        display.grid(row=0, column=0, sticky=(tk.W, tk.E))
        display_frame.columnconfigure(0, weight=1)
        
        # Buttons
        self.create_buttons(main_frame)
        
        # History frame
        history_frame = ttk.LabelFrame(main_frame, text="History", padding="5")
        history_frame.grid(row=6, column=0, columnspan=4, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
        
        self.history_listbox = tk.Listbox(history_frame, height=8)
        scrollbar = ttk.Scrollbar(history_frame, orient="vertical", command=self.history_listbox.yview)
        self.history_listbox.configure(yscrollcommand=scrollbar.set)
        
        self.history_listbox.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        history_frame.columnconfigure(0, weight=1)
        history_frame.rowconfigure(0, weight=1)
        
        # Clear history button
        ttk.Button(history_frame, text="Clear History", 
                  command=self.clear_history).grid(row=1, column=0, columnspan=2, pady=(5, 0))
        
        # Configure grid weights
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.columnconfigure(2, weight=1)
        main_frame.columnconfigure(3, weight=1)
        main_frame.rowconfigure(6, weight=1)
    
    def create_buttons(self, parent):
        """Create calculator buttons."""
        # Number buttons
        numbers = [
            ('7', 1, 0), ('8', 1, 1), ('9', 1, 2),
            ('4', 2, 0), ('5', 2, 1), ('6', 2, 2),
            ('1', 3, 0), ('2', 3, 1), ('3', 3, 2),
            ('0', 4, 1)
        ]
        
        for (text, row, col) in numbers:
            btn = ttk.Button(parent, text=text, width=8,
                           command=lambda t=text: self.on_number_click(t))
            btn.grid(row=row, column=col, padx=2, pady=2)
        
        # Operation buttons
        operations = [
            ('+', 1, 3, 'add'),
            ('-', 2, 3, 'subtract'), 
            ('*', 3, 3, 'multiply'),
            ('/', 4, 3, 'divide'),
            ('^', 5, 0, 'power'),
            ('√', 5, 1, 'sqrt'),
            ('!', 5, 2, 'factorial'),
            ('ln', 5, 3, 'log')
        ]
        
        for (text, row, col, op) in operations:
            btn = ttk.Button(parent, text=text, width=8,
                           command=lambda o=op: self.on_operation_click(o))
            btn.grid(row=row, column=col, padx=2, pady=2)
        
        # Special buttons
        ttk.Button(parent, text=".", width=8,
                  command=self.on_decimal_click).grid(row=4, column=0, padx=2, pady=2)
        
        ttk.Button(parent, text="=", width=8,
                  command=self.on_equals_click).grid(row=4, column=2, padx=2, pady=2)
        
        ttk.Button(parent, text="C", width=8,
                  command=self.on_clear_click).grid(row=0, column=3, padx=2, pady=2)
        
        ttk.Button(parent, text="CE", width=8,
                  command=self.on_clear_entry_click).grid(row=0, column=2, padx=2, pady=2)
    
    def on_number_click(self, number):
        """Handle number button clicks."""
        if self.waiting_for_operand:
            self.display_var.set(number)
            self.waiting_for_operand = False
        else:
            current = self.display_var.get()
            if current == "0":
                self.display_var.set(number)
            else:
                self.display_var.set(current + number)
    
    def on_decimal_click(self):
        """Handle decimal point button click."""
        current = self.display_var.get()
        if self.waiting_for_operand:
            self.display_var.set("0.")
            self.waiting_for_operand = False
        elif "." not in current:
            self.display_var.set(current + ".")
    
    def on_operation_click(self, operation):
        """Handle operation button clicks."""
        try:
            current_value = float(self.display_var.get())
            
            if operation in ['sqrt', 'factorial', 'log']:
                # Single operand operations
                if operation == 'sqrt':
                    result = self.advanced_calc.square_root(current_value)
                    self.calc.history.append(f"√{current_value} = {result}")
                elif operation == 'factorial':
                    result = self.advanced_calc.factorial(int(current_value))
                    self.calc.history.append(f"{int(current_value)}! = {result}")
                elif operation == 'log':
                    result = self.advanced_calc.logarithm(current_value)
                    self.calc.history.append(f"ln({current_value}) = {result}")
                
                self.display_var.set(str(result))
                self.update_history()
                self.waiting_for_operand = True
            else:
                # Two operand operations
                if self.current_operation and not self.waiting_for_operand:
                    self.on_equals_click()
                
                self.first_number = current_value
                self.current_operation = operation
                self.waiting_for_operand = True
                
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def on_equals_click(self):
        """Handle equals button click."""
        if self.current_operation and self.first_number is not None:
            try:
                second_number = float(self.display_var.get())
                
                if self.current_operation == 'add':
                    result = self.calc.add(self.first_number, second_number)
                elif self.current_operation == 'subtract':
                    result = self.calc.subtract(self.first_number, second_number)
                elif self.current_operation == 'multiply':
                    result = self.calc.multiply(self.first_number, second_number)
                elif self.current_operation == 'divide':
                    result = self.calc.divide(self.first_number, second_number)
                elif self.current_operation == 'power':
                    result = self.advanced_calc.power(self.first_number, second_number)
                    self.calc.history.append(f"{self.first_number} ^ {second_number} = {result}")
                
                self.display_var.set(str(result))
                self.update_history()
                self.current_operation = None
                self.first_number = None
                self.waiting_for_operand = True
                
            except Exception as e:
                messagebox.showerror("Error", str(e))
    
    def on_clear_click(self):
        """Handle clear button click."""
        self.display_var.set("0")
        self.current_operation = None
        self.first_number = None
        self.waiting_for_operand = False
    
    def on_clear_entry_click(self):
        """Handle clear entry button click."""
        self.display_var.set("0")
    
    def update_history(self):
        """Update the history display."""
        self.history_listbox.delete(0, tk.END)
        for item in self.calc.get_history():
            self.history_listbox.insert(tk.END, item)
        
        # Scroll to bottom
        self.history_listbox.see(tk.END)
    
    def clear_history(self):
        """Clear the calculation history."""
        self.calc.clear_history()
        self.history_listbox.delete(0, tk.END)
    
    def run(self):
        """Start the GUI application."""
        self.root.mainloop()

def main():
    """Main function to run the GUI calculator."""
    app = CalculatorGUI()
    app.run()

if __name__ == "__main__":
    main()