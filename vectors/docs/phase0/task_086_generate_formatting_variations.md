# Micro-Task 086: Generate Formatting Variations

## Objective
Generate text files with different formatting styles to test how vector search handles various text presentation formats.

## Context
Formatting variations test the vector search system's ability to extract meaningful content from differently formatted text while ignoring presentation markup that shouldn't affect semantic meaning.

## Prerequisites
- Task 085 completed (Paragraph variations generated)

## Time Estimate
9 minutes

## Instructions
1. Navigate to test data directory: `cd data\test_files`
2. Create formatting generation script `generate_formatting_variations.py`:
   ```python
   #!/usr/bin/env python3
   """
   Generate formatting variation samples for vector search presentation testing.
   """
   
   import os
   import sys
   from pathlib import Path
   sys.path.append('templates')
   from template_generator import TestFileGenerator
   
   def generate_formatting_variations():
       """Generate files with different text formatting styles."""
       generator = TestFileGenerator()
       
       # Sample 1: Mixed case and emphasis patterns
       case_emphasis = """Title Case Formatting Example
   
   This Document Tests Various Text FORMATTING Patterns
   including UPPERCASE words, lowercase sections, and Mixed Case Titles.
   
   SECTION 1: UPPERCASE EMPHASIS
   SOME DOCUMENTATION USES ALL CAPS FOR HEADERS AND IMPORTANT INFORMATION.
   THIS CAN AFFECT VECTOR SEARCH PROCESSING AND SIMILARITY CALCULATIONS.
   PROPER HANDLING OF CASE VARIATIONS IS ESSENTIAL FOR SEARCH ACCURACY.
   
   section 2: lowercase emphasis  
   conversely, some text uses all lowercase formatting for stylistic reasons.
   this tests whether the search system normalizes case appropriately.
   search queries should match content regardless of case differences.
   
   Section 3: Mixed Case Patterns
   Regular sentence case provides the most common formatting pattern.
   However, technical documents often Mix Cases in Various Ways.
   API names, variable names, and technical terms follow different conventions.
   
   Important Notes About Formatting:
   - CamelCase identifiers in programming contexts
   - snake_case naming conventions in some languages  
   - CONSTANT_VALUES often use all uppercase
   - File.extension.naming uses periods as separators
   
   Conclusion: Case handling affects search quality significantly."""
   
       # Sample 2: Special spacing and alignment
       spacing_alignment = """        Indented Content Example
   
   This file tests various spacing and alignment patterns:
   
        Indented paragraphs with 8 spaces
        Multiple lines maintaining alignment
        Consistent indentation levels
   
           Nested indentation with 11 spaces
           Deeper nesting for hierarchical content
           Maintaining readable structure
   
               Even deeper nesting at 15 spaces
               Testing extreme indentation handling
               Whitespace preservation requirements
   
   Normal paragraph returns to standard alignment.
   No leading spaces for comparison purposes.
   Standard text flow and formatting.
   
   Tabs	versus	spaces	create	different	alignment	patterns.
   Mixed	whitespace	can	cause	processing	issues.
   Consistent	formatting	improves	search	accuracy.
   
   Multiple


   blank


   lines


   test


   paragraph
   separation
   handling.
   
   Trailing spaces test boundary detection...    
   Lines with varying amounts of trailing whitespace...     
   Should be handled consistently by the search system...        """
   
       # Sample 3: Line length variations
       line_length = """Short lines.
   Medium length lines for comparison.
   This is a significantly longer line that tests how the vector search system handles content when it spans more characters than typical sentences and may require different processing approaches.
   
   Very short.
   A bit longer sentence here.
   This line contains a moderate amount of text that represents typical paragraph content and should process normally without any special considerations or handling requirements.
   An extremely long line that goes on and on with extensive content covering multiple topics and concepts to test how the system handles very long sequences of text that might exceed normal processing windows or chunk boundaries and could potentially cause issues with vector generation, similarity calculations, or search result ranking if not handled properly by the underlying algorithms and data structures used in the vector search implementation.
   
   Back to normal length sentences for comparison and validation.
   Testing the transition between different line lengths.
   Ensuring consistent processing across varying content structures.
   
   123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890
   Fixed width content tests character counting and boundary detection.
   Exact character limits help validate processing consistency.
   
   Word
   per
   line
   tests
   extreme
   formatting
   cases.
   
   Regular paragraph flow resumes here with normal sentence structures and typical content organization patterns that users would expect to find in standard documentation or text files."""
   
       # Sample 4: Punctuation and symbol variations
       punctuation_symbols = """Punctuation & Symbol Testing Document!!!
   
   Question marks test query-like content? How does the system handle interrogative sentences? What about multiple questions in sequence???
   
   Exclamation points add emphasis! Multiple exclamations create stronger emphasis!! Sometimes excessive punctuation appears in user content!!!
   
   Quotation marks "test content boundaries" and 'single quote handling' for different quotation styles. "Nested 'quotes' within quotes" create complex parsing scenarios.
   
   Parenthetical content (like this) interrupts normal flow. Multiple (nested (parentheses) structures) test boundary detection. [Square brackets] serve similar functions. {Curly braces} appear in some contexts.
   
   Mathematical symbols: 2 + 2 = 4, x < y, a > b, ratio 3:1, percentage 50%, temperature -10°C, approximately ~5 items.
   
   Hyphenated-words and compound-terms create token-boundary challenges. Multi-word-combinations with many-hyphens test tokenization.
   
   Ellipsis points... indicate continuation or pause... Multiple ellipses...... may appear in informal content...
   
   Special characters: @mentions, #hashtags, $currency symbols, % percentages, ^ exponents, & ampersands, * asterisks for emphasis.
   
   Email addresses like user@example.com and URLs like https://example.com/path?param=value test structured data within text.
   
   Punctuation combinations: Hello,world! (no space) vs. Hello, world! (with space). Consistency matters: word;another vs. word; another.
   
   Final sentence with mixed punctuation: "Is this correct?!" asked the user... (probably not).
   """
   
       # Sample 5: Number and date formatting
       numbers_dates = """Numerical Content and Date Formatting Test
   
   Integer Numbers:
   Small numbers: 1, 2, 3, 10, 25, 100
   Large numbers: 1,000, 10,000, 1,000,000, 999,999,999
   Very large: 1000000000000, 42000000000000000
   
   Decimal Numbers:
   Simple decimals: 3.14, 2.71, 1.41, 0.5, 0.25
   Precise values: 3.14159265359, 2.718281828, 1.414213562
   Scientific notation: 1.23e10, 4.56E-7, 9.87e+12
   
   Percentages and Ratios:
   Percentages: 50%, 33.33%, 0.1%, 100.0%
   Ratios: 16:9, 4:3, 1:1, 3:2:1
   Fractions: 1/2, 3/4, 22/7, 355/113
   
   Dates and Times:
   ISO dates: 2024-08-04, 2023-12-31, 2025-01-01
   US format: 08/04/2024, 12/31/2023, 01/01/2025
   European: 04/08/2024, 31/12/2023, 01/01/2025
   Long format: August 4, 2024, December 31st, 2023
   
   Time formats:
   24-hour: 14:30:00, 09:15:30, 23:59:59
   12-hour: 2:30 PM, 9:15 AM, 11:59 PM
   Timezone: 14:30 UTC, 9:15 EST, 23:59 PST
   
   Version Numbers:
   Software versions: v1.0.0, 2.1.3, 3.14.159, 0.9.8-beta
   Semantic versioning: 1.2.3-alpha.1, 2.0.0-rc.1
   
   Measurements:
   Distance: 5.5 km, 10 miles, 100 meters, 6 feet 2 inches
   Weight: 2.5 kg, 5 pounds, 150 grams, 1 ton
   Temperature: 25°C, 77°F, 298.15 K, -40° (same in C and F)
   
   Monetary values:
   USD: $100.00, $1,234.56, $0.99, $1,000,000
   EUR: €50.00, €999.99, €1.234.567,89
   Other: £25.50, ¥1000, ₹500, ₿0.001
   
   Phone numbers and codes:
   US format: (555) 123-4567, 555-123-4567, +1-555-123-4567
   International: +44 20 7946 0958, +33 1 42 86 83 26
   Zip codes: 12345, 12345-6789, SW1A 1AA, H0H 0H0"""
   
       # Generate all formatting variation files
       samples = [
           ("case_emphasis.txt", "Mixed case and emphasis patterns", case_emphasis),
           ("spacing_alignment.txt", "Spacing and alignment variations", spacing_alignment),  
           ("line_length.txt", "Line length variations", line_length),
           ("punctuation_symbols.txt", "Punctuation and symbol patterns", punctuation_symbols),
           ("numbers_dates.txt", "Number and date formatting", numbers_dates)
       ]
       
       generated_files = []
       for filename, pattern_focus, content in samples:
           output_path = generator.generate_text_file(
               filename,
               "basic_text",
               pattern_focus,
               content,
               "basic_text"
           )
           generated_files.append(output_path)
           print(f"Generated: {output_path}")
       
       return generated_files
   
   def analyze_formatting(file_path):
       """Analyze formatting characteristics in a file."""
       with open(file_path, 'r', encoding='utf-8') as f:
           content = f.read()
       
       # Basic statistics
       lines = content.split('\n')
       non_empty_lines = [line for line in lines if line.strip()]
       
       # Character analysis
       uppercase_chars = sum(1 for c in content if c.isupper())
       lowercase_chars = sum(1 for c in content if c.islower()) 
       digit_chars = sum(1 for c in content if c.isdigit())
       whitespace_chars = sum(1 for c in content if c.isspace())
       punctuation_chars = sum(1 for c in content if not c.isalnum() and not c.isspace())
       
       # Line analysis
       line_lengths = [len(line) for line in lines]
       
       return {
           "total_chars": len(content),
           "total_lines": len(lines),
           "non_empty_lines": len(non_empty_lines),
           "uppercase_chars": uppercase_chars,
           "lowercase_chars": lowercase_chars,
           "digit_chars": digit_chars,
           "whitespace_chars": whitespace_chars,
           "punctuation_chars": punctuation_chars,
           "avg_line_length": sum(line_lengths) // len(line_lengths) if line_lengths else 0,
           "max_line_length": max(line_lengths) if line_lengths else 0,
           "min_line_length": min(line_lengths) if line_lengths else 0
       }
   
   def main():
       """Main generation function."""
       print("Generating formatting variation samples...")
       
       # Ensure output directory exists
       os.makedirs("basic_text", exist_ok=True)
       
       try:
           files = generate_formatting_variations()
           print(f"\nSuccessfully generated {len(files)} formatting variation files:")
           
           # Analyze each file
           for file_path in files:
               print(f"\n  - {file_path}")
               stats = analyze_formatting(file_path)
               print(f"    Total chars: {stats['total_chars']}")
               print(f"    Lines: {stats['total_lines']} ({stats['non_empty_lines']} non-empty)")
               print(f"    Avg line length: {stats['avg_line_length']} chars")
               print(f"    Case ratio: {stats['uppercase_chars']}U/{stats['lowercase_chars']}L")
               print(f"    Digits: {stats['digit_chars']}, Punctuation: {stats['punctuation_chars']}")
           
           print("\nFormatting variation generation completed successfully!")
           return 0
       
       except Exception as e:
           print(f"Error generating formatting variations: {e}")
           return 1
   
   if __name__ == "__main__":
       sys.exit(main())
   ```
3. Run generation: `python generate_formatting_variations.py`
4. Return to root: `cd ..\..`
5. Commit: `git add data\test_files\generate_formatting_variations.py data\test_files\basic_text && git commit -m "task_086: Generate formatting variations for presentation testing"`

## Expected Output
- 5 text files with different formatting patterns
- Case and emphasis variations
- Spacing and alignment tests
- Punctuation and number handling
- Formatting analysis statistics

## Success Criteria
- [ ] Case emphasis file generated
- [ ] Spacing alignment file generated
- [ ] Line length variation file generated
- [ ] Punctuation symbols file generated
- [ ] Numbers and dates file generated
- [ ] All files analyzed for formatting statistics

## Validation Commands
```cmd
cd data\test_files
python generate_formatting_variations.py
dir basic_text
```

## Next Task
task_087_generate_content_type_variations.md

## Notes
- Formatting variations test content normalization capabilities
- Different presentation styles validate semantic extraction
- Statistics help understand formatting complexity
- Files ensure robust handling of diverse text formats