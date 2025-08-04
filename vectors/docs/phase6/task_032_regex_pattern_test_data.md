# Task 032: Generate Regex Pattern Test Data

## Context
You are implementing a comprehensive validation system for a Rust-based vector indexing system. This builds on Tasks 010-031. Regex pattern testing is essential for validating search functionality across different pattern types, edge cases, and performance scenarios that might cause catastrophic backtracking or other regex engine issues.

## Project Structure
```
src/
  validation/
    test_data.rs  <- Extend this file
  lib.rs
```

## Task Description
Implement the `generate_regex_pattern_tests()` method that creates test files with comprehensive regex patterns including basic patterns, complex quantifiers, lookarounds, character classes, and pathological cases that could cause performance issues.

## Requirements
1. Add to existing `src/validation/test_data.rs`
2. Generate files with basic regex patterns (literals, wildcards, character classes)
3. Include complex patterns with quantifiers, alternation, and grouping
4. Create pathological patterns that could cause catastrophic backtracking
5. Test Unicode-aware regex patterns and character classes
6. Include files with regex metacharacters that need escaping
7. Generate realistic code search patterns (function definitions, imports, etc.)

## Expected Code Structure to Add
```rust
impl TestDataGenerator {
    fn generate_regex_pattern_tests(&self) -> Result<Vec<GeneratedTestFile>> {
        let mut files = Vec::new();
        
        // Basic regex patterns
        let basic_patterns_content = self.generate_basic_regex_patterns()?;
        let mut basic_file = self.create_test_file("regex_basic_patterns.txt", &basic_patterns_content, TestFileType::RegexPatterns)?;
        basic_file.expected_matches = vec![
            "hello".to_string(),
            "world".to_string(),
            "test123".to_string(),
            "pattern_match".to_string(),
            "BASIC_REGEX_TEST".to_string(),
        ];
        files.push(basic_file);
        
        // Character classes and quantifiers
        let char_class_content = self.generate_character_class_patterns()?;
        let mut char_class_file = self.create_test_file("regex_character_classes.txt", &char_class_content, TestFileType::RegexPatterns)?;
        char_class_file.expected_matches = vec![
            "[0-9]+".to_string(),
            "[a-zA-Z]+".to_string(),
            "\\d+".to_string(),
            "\\w+".to_string(),
            "CHAR_CLASS_TEST".to_string(),
        ];
        files.push(char_class_file);
        
        // Complex quantifiers and alternation
        let complex_patterns_content = self.generate_complex_regex_patterns()?;
        let mut complex_file = self.create_test_file("regex_complex_patterns.txt", &complex_patterns_content, TestFileType::RegexPatterns)?;
        complex_file.expected_matches = vec![
            "complex_pattern_1".to_string(),
            "complex_pattern_2".to_string(),
            "alternation_test".to_string(),
            "quantifier_test".to_string(),
            "COMPLEX_REGEX_TEST".to_string(),
        ];
        files.push(complex_file);
        
        // Lookahead and lookbehind patterns
        let lookaround_content = self.generate_lookaround_patterns()?;
        let mut lookaround_file = self.create_test_file("regex_lookaround.txt", &lookaround_content, TestFileType::RegexPatterns)?;
        lookaround_file.expected_matches = vec![
            "positive_lookahead".to_string(),
            "negative_lookahead".to_string(),
            "positive_lookbehind".to_string(),
            "negative_lookbehind".to_string(),
            "LOOKAROUND_TEST".to_string(),
        ];
        files.push(lookaround_file);
        
        // Pathological patterns (potential performance issues)
        let pathological_content = self.generate_pathological_patterns()?;
        let mut pathological_file = self.create_test_file("regex_pathological.txt", &pathological_content, TestFileType::RegexPatterns)?;
        pathological_file.expected_matches = vec![
            "PATHOLOGICAL_START".to_string(),
            "PATHOLOGICAL_END".to_string(),
            "backtrack_test".to_string(),
            "catastrophic_pattern".to_string(),
        ];
        files.push(pathological_file);
        
        // Unicode and international patterns
        let unicode_patterns_content = self.generate_unicode_regex_patterns()?;
        let mut unicode_file = self.create_test_file("regex_unicode.txt", &unicode_patterns_content, TestFileType::RegexPatterns)?;
        unicode_file.expected_matches = vec![
            "unicode_test_한글".to_string(),
            "unicode_test_العربية".to_string(),
            "unicode_test_中文".to_string(),
            "UNICODE_REGEX_TEST".to_string(),
        ];
        files.push(unicode_file);
        
        // Code-specific regex patterns
        let code_patterns_content = self.generate_code_regex_patterns()?;
        let mut code_file = self.create_test_file("regex_code_patterns.rs", &code_patterns_content, TestFileType::RegexPatterns)?;
        code_file.expected_matches = vec![
            "fn main()".to_string(),
            "pub struct".to_string(),
            "impl<T>".to_string(),
            "use std::".to_string(),
            "CODE_REGEX_TEST".to_string(),
        ];
        files.push(code_file);
        
        // Escaped metacharacters
        let escaped_patterns_content = self.generate_escaped_metacharacter_patterns()?;
        let mut escaped_file = self.create_test_file("regex_escaped_chars.txt", &escaped_patterns_content, TestFileType::RegexPatterns)?;
        escaped_file.expected_matches = vec![
            "literal.dot".to_string(),
            "literal*asterisk".to_string(),
            "literal+plus".to_string(),
            "literal?question".to_string(),
            "ESCAPED_REGEX_TEST".to_string(),
        ];
        files.push(escaped_file);
        
        Ok(files)
    }
    
    /// Generate basic regex pattern test content
    fn generate_basic_regex_patterns(&self) -> Result<String> {
        Ok(r#"Basic Regex Pattern Test File
============================

BASIC_REGEX_TEST: Testing fundamental regex patterns

1. Literal matches:
   hello world
   test123
   pattern_match
   simple_string
   
2. Wildcard patterns (.):
   a.c matches: abc, aXc, a5c
   test.txt matches files: test1txt, testAtxt, test_txt
   
3. Start and end anchors (^ and $):
   ^start_of_line
   end_of_line$
   ^exact_match$
   
4. Word boundaries (\b):
   \bword\b matches: word, "word", word.
   boundary_test for \btest\b pattern
   
5. Basic escape sequences:
   \n for newline
   \t for tab
   \r for carriage return
   \\ for literal backslash
   
Sample text for basic pattern matching:

Line 1: hello world test
Line 2: pattern_match found here
Line 3: 123 numbers and ABC letters
Line 4: special.chars*here+now?maybe
Line 5: word boundary test word end

Multiple line test:
hello on line 1
world on line 2
test123 on line 3

Exact match tests:
exactly_this_pattern
exactly_this_pattern_not
not_exactly_this_pattern

Word boundary tests:
test word boundary
word in middle
boundary word
wordboundary (no spaces)

Tab	separated	values	here
Newline
separated
values
here

End of basic regex patterns section.
"#.to_string())
    }
    
    /// Generate character class pattern test content
    fn generate_character_class_patterns(&self) -> Result<String> {
        Ok(r#"Character Class Regex Pattern Tests
===================================

CHAR_CLASS_TEST: Testing character classes and shorthand patterns

1. Digit patterns:
   Numbers: 123, 456, 789, 0
   Mixed: abc123def, test456end
   Multiple digits: 12345, 67890
   
2. Letter patterns:
   Lowercase: abcdefghijklmnopqrstuvwxyz
   Uppercase: ABCDEFGHIJKLMNOPQRSTUVWXYZ
   Mixed case: AbCdEfGhIjKlMnOpQrStUvWxYz
   
3. Alphanumeric patterns:
   Mixed: abc123XYZ, test789END, start456middle789end
   
4. Whitespace patterns:
   Spaces: word1 word2 word3
   Tabs: item1	item2	item3
   Mixed:   spaced	and	tabbed   content
   
5. Word character patterns (\w):
   identifier_name_123
   variable_name
   function_call_2
   class_Name_V2
   
6. Non-word character patterns (\W):
   Special chars: !@#$%^&*()
   Punctuation: .,;:'"?!
   Brackets: []{}()<>
   
7. Custom character classes:
   Vowels [aeiouAEIOU]: beautiful, EDUCATION, programming
   Consonants [bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ]: strength, crypts
   Hex digits [0-9a-fA-F]: 1A2B, DEAD, BEEF, cafe, 123ABC
   
8. Negated character classes:
   Non-digits [^0-9]: hello, world, test
   Non-vowels [^aeiouAEIOU]: rhythm, fly, my
   Non-alphanumeric [^a-zA-Z0-9]: !@#, $%^, &*()
   
9. Quantifier patterns:
   One or more digits: \d+ matches 123, 4567, 89
   Zero or more letters: [a-z]* matches abc, hello, ""
   One or more word chars: \w+ matches test123, hello_world
   
10. Range patterns:
    Letters A-M: ABCDEFGHIJKLM and abcdefghijklm
    Letters N-Z: NOPQRSTUVWXYZ and nopqrstuvwxyz
    Digits 0-5: 012345 in various contexts
    Digits 6-9: 6789 in various contexts
    
Sample data for character class testing:

Email addresses:
user@example.com
test.email+tag@domain.co.uk
admin123@server.org

Phone numbers:
123-456-7890
(555) 123-4567
+1-800-555-0123

Identifiers:
variable_name_123
className
CONSTANT_VALUE
_private_var
$special_var

File extensions:
.txt .log .dat .csv .json .xml .html .css .js .py .rs .cpp .h

IP addresses:
192.168.1.1
10.0.0.1
255.255.255.0

Hexadecimal colors:
#FF0000 #00FF00 #0000FF
#FFFFFF #000000 #CCCCCC

End of character class patterns section.
"#.to_string())
    }
    
    /// Generate complex regex pattern test content
    fn generate_complex_regex_patterns(&self) -> Result<String> {
        Ok(r#"Complex Regex Pattern Tests
===========================

COMPLEX_REGEX_TEST: Testing advanced regex constructs

1. Alternation patterns (OR):
   Pattern: (cat|dog|bird)
   Matches: I have a cat, dog walker, bird watcher
   
   Pattern: (red|green|blue)
   Matches: red car, green tree, blue sky
   
   complex_pattern_1: choose between option1, option2, option3
   complex_pattern_2: select from choice1, choice2, choice3

2. Grouping and capturing:
   Pattern: (test)_(\d+)
   Matches: test_123, test_456, test_789
   
   Pattern: (hello|hi)_(\w+)
   Matches: hello_world, hi_there, hello_test

3. Non-capturing groups:
   Pattern: (?:cat|dog) food
   Matches: cat food, dog food (groups not captured)
   
   alternation_test: (?:option1|option2|option3) selected

4. Quantifier variations:
   Zero or one (?): colou?r matches color, colour
   Zero or more (*): ab*c matches ac, abc, abbc, abbbc
   One or more (+): ab+c matches abc, abbc, abbbc (not ac)
   Exact count {n}: a{3} matches aaa
   Range {n,m}: a{2,4} matches aa, aaa, aaaa
   
   quantifier_test patterns:
   test? matches tes, test
   test* matches "", tes, test, testt, testtt
   test+ matches test, testt, testtt (not "")
   test{3} matches testesttest
   test{2,4} matches testtest, testtesttest, testtestesttest

5. Greedy vs non-greedy quantifiers:
   Greedy: <.*> matches <tag>content</tag> (entire string)
   Non-greedy: <.*?> matches <tag> and </tag> separately
   
   Greedy example: "quote1" and "quote2" matched as one by ".*"
   Non-greedy example: "quote1" and "quote2" matched separately by ".*?"

6. Advanced grouping with alternation:
   Pattern: (urgent|important|critical)_(task|item|issue)
   Matches: urgent_task, important_item, critical_issue
   
   Pattern: (get|set|delete)_(\w+)_(\d+)
   Matches: get_user_123, set_config_456, delete_record_789

7. Nested groups:
   Pattern: ((hello|hi)_(world|there))
   Matches: hello_world, hi_there, hello_there, hi_world
   
   Pattern: (test_((sub1|sub2)_(\d+)))
   Matches: test_sub1_123, test_sub2_456

8. Case sensitivity tests:
   Case insensitive pattern: (?i)test
   Should match: test, Test, TEST, TeSt
   
   Mixed case text:
   Test Case, test case, TEST CASE, TeSt CaSe

9. Complex real-world patterns:
   
   URL pattern: https?://[\w\-\.]+(:\d+)?(/[\w\-\.]*)*(\?[\w=&]*)?
   Test URLs:
   http://example.com
   https://www.test.org:8080/path/to/resource?param=value
   http://sub.domain.com/folder/file.html
   
   Email pattern: [\w\-\.]+@[\w\-\.]+\.[a-zA-Z]{2,}
   Test emails:
   user@domain.com
   test.email@sub.domain.org
   admin+tag@company.co.uk
   
   IPv4 pattern: (\d{1,3}\.){3}\d{1,3}
   Test IPs:
   192.168.1.1
   10.0.0.1
   255.255.255.255

10. Complex alternation with quantifiers:
    Pattern: (red|green|blue){2,3}
    Matches: redred, greenblue, bluegreered
    
    Pattern: (cat|dog)+ (food|toy)
    Matches: cat food, dogdog toy, catdog food

Sample complex matching scenarios:

Log entry pattern: \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} (INFO|WARN|ERROR) (.+)
Sample logs:
2024-01-15 10:30:45 INFO Application started
2024-01-15 10:31:12 ERROR Database connection failed
2024-01-15 10:31:15 WARN Retrying connection

Function definition pattern: (public|private|protected)?\s*(static)?\s*(\w+)\s+(\w+)\s*\(([^)]*)\)
Sample functions:
public static void main(String[] args)
private int calculateSum(int a, int b)
protected String formatName(String first, String last)

End of complex regex patterns section.
"#.to_string())
    }
    
    /// Generate lookaround pattern test content
    fn generate_lookaround_patterns(&self) -> Result<String> {
        Ok(r#"Lookaround Regex Pattern Tests
==============================

LOOKAROUND_TEST: Testing positive and negative lookahead/lookbehind

1. Positive Lookahead (?=pattern):
   Pattern: test(?=ing)
   Text: testing, tested, test, testable
   Matches: "test" in "testing" only
   
   positive_lookahead examples:
   word(?=s) matches "word" in "words" but not "word"
   test(?=_case) matches "test" in "test_case" but not "test_other"
   
   Sample text:
   words and word
   test_case and test_other
   password and passwords

2. Negative Lookahead (?!pattern):
   Pattern: test(?!ing)
   Text: testing, tested, test, testable
   Matches: "test" in "tested", "test", "testable" but not "testing"
   
   negative_lookahead examples:
   word(?!s) matches "word" standalone but not in "words"
   test(?!_case) matches "test" in "test_other" but not "test_case"
   
   Sample text:
   standalone word vs words
   test_other vs test_case
   file vs files

3. Positive Lookbehind (?<=pattern):
   Pattern: (?<=pre_)test
   Text: pre_test, test, post_test
   Matches: "test" in "pre_test" only
   
   positive_lookbehind examples:
   (?<=sub_)task matches "task" in "sub_task" but not "main_task"
   (?<=get_)value matches "value" in "get_value" but not "set_value"
   
   Sample text:
   sub_task and main_task
   get_value and set_value
   pre_process and post_process

4. Negative Lookbehind (?<!pattern):
   Pattern: (?<!pre_)test
   Text: pre_test, test, post_test
   Matches: "test" in standalone "test" and "post_test" but not "pre_test"
   
   negative_lookbehind examples:
   (?<!sub_)task matches "task" in "main_task" but not "sub_task"
   (?<!get_)value matches "value" in "set_value" but not "get_value"
   
   Sample text:
   main_task vs sub_task
   set_value vs get_value
   post_test vs pre_test

5. Combined lookarounds:
   Pattern: (?<=start_)test(?=_end)
   Matches: "test" only when preceded by "start_" and followed by "_end"
   
   Sample text:
   start_test_end (matches)
   start_test_other (no match)
   other_test_end (no match)
   other_test_other (no match)
   
   Pattern: (?<!bad_)(?=good_)test
   Complex lookaround combining negative lookbehind and positive lookahead

6. Password validation examples:
   Strong password: (?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]
   
   Test passwords:
   Password123! (should match all criteria)
   password123 (missing uppercase and special char)
   PASSWORD123 (missing lowercase and special char)
   Password! (missing digit)
   
7. Word boundary with lookarounds:
   Pattern: \b(?=\w*test\w*)\w+
   Matches complete words containing "test"
   
   Sample text:
   testing tested contest protest
   test tests tester
   
8. Number validation with lookarounds:
   Pattern: (?=.*\d{3,})(?=.*[1-9])\d+
   Numbers with at least 3 digits and at least one non-zero
   
   Test numbers:
   123, 1000, 0001, 000, 42, 999

9. File extension lookarounds:
   Pattern: \w+(?=\.(txt|log|dat))
   Matches filename part before specific extensions
   
   Sample files:
   document.txt
   error.log
   data.dat
   image.jpg (should not match)
   
10. Context-sensitive matching:
    Pattern: (?<=function\s+)\w+(?=\s*\()
    Matches function names between "function " and "("
    
    Sample code:
    function testName() {}
    function calculateSum(a, b) {}
    var testName = function() {}
    
Complex lookaround scenarios:

Email domain validation:
(?<=@)[a-zA-Z0-9.-]+(?=\.[a-zA-Z]{2,})
Matches domain part of email addresses

SQL injection prevention:
(?<![\w'])(?=.*['";])\w+
Detects potentially dangerous SQL patterns

URL path extraction:
(?<=://)[\w.-]+(?=/|\?|$)
Extracts domain from URLs

Log level extraction:
(?<=\[)[A-Z]+(?=\])
Extracts log levels from [INFO], [ERROR], etc.

End of lookaround patterns section.
"#.to_string())
    }
    
    /// Generate pathological pattern test content
    fn generate_pathological_patterns(&self) -> Result<String> {
        Ok(r#"Pathological Regex Pattern Tests
=================================

PATHOLOGICAL_START: Warning - these patterns may cause performance issues

IMPORTANT: These patterns are designed to test regex engine limits and 
potential catastrophic backtracking scenarios. Use with caution in production.

1. Exponential backtracking patterns:
   Pattern: (a+)+b
   Problematic text: aaaaaaaaaaaaaaaaaaaa (no 'b' at end)
   
   backtrack_test_case_1:
   Text: aaaaaaaaaaaaaaaaaaaaX
   The pattern (a+)+ will try many combinations before failing
   
   Pattern: (a*)*b
   Similar issue with nested quantifiers
   
   backtrack_test_case_2:
   Text: aaaaaaaaaaaaaaaaaaaaY
   Causes exponential backtracking

2. Nested quantifier catastrophe:
   Pattern: (a+)+c
   Pattern: (a*)*c  
   Pattern: (a?)?c
   
   catastrophic_pattern examples:
   String of 20+ 'a's without trailing 'c': aaaaaaaaaaaaaaaaaaaZ
   String of 30+ 'a's without trailing 'c': aaaaaaaaaaaaaaaaaaaaaaaaaaaaaZ

3. Alternation with overlapping patterns:
   Pattern: (a|a)*b
   Pattern: (.|.)*b
   
   Overlapping alternation causes backtracking:
   aaaaaaaaaaaaaaaaaaaa without 'b'

4. Complex nested groups:
   Pattern: ((a+)*b+)*c
   Pattern: ((a*b*)*c*)*d
   
   Multiple levels of nesting create exponential possibilities:
   Text: aaabbbaaabbb without trailing 'c' or 'd'

5. Greedy quantifiers with alternation:
   Pattern: .*.*=.*
   Pattern: a.*a.*a.*b
   
   Multiple .* patterns can cause extensive backtracking:
   Long string without the final required character

6. Real-world pathological patterns:

   HTML/XML parsing (dangerous):
   Pattern: <.*>.*</.*>
   Problematic HTML: <div>lots of content without proper closing
   
   Email regex (complex):
   Pattern: ^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$
   Can be slow with malformed input
   
   URL validation (complex):
   Pattern: https?://[^\s/$.?#].[^\s]*
   May backtrack extensively on malformed URLs

7. Stress test patterns:

   Pattern that matches everything except one case:
   [^x]*x
   Test with long strings not ending in 'x'
   
   Optional everything pattern:
   (.*)?
   Technically matches but can be slow
   
   Multiple optional groups:
   (a?)*(b?)*(c?)*d
   Exponential combinations before final 'd'

8. Boundary condition tests:

   Very long alternation:
   (option1|option2|option3|...|option100)
   Large alternation groups can be slow
   
   Deep nesting:
   ((((((a+)+)+)+)+)+)+b
   Deep nesting multiplies backtracking

9. Unicode pathological patterns:

   Pattern: (\p{L}+)+\p{N}
   Unicode letter class with quantifiers
   Test with long strings of letters without numbers
   
   Pattern: (\p{Script=Latin}*)*\p{Script=Greek}
   Script-specific matching with nested quantifiers

10. Mitigation test cases:

    Atomic groups (?>pattern):
    (?>a+)b vs (a+)b
    Atomic groups prevent backtracking
    
    Possessive quantifiers:
    a++b vs a+b
    Possessive quantifiers don't backtrack
    
    More specific patterns:
    [a-z]+@[a-z]+\.[a-z]{2,4} vs .*@.*\..*
    Specific patterns avoid broad matching

Test strings for pathological patterns:

Short safe strings:
abc, test, hello, world

Medium strings (potential issues):
aaaaaaaaaaaaaaaaaaaaa (20 a's)
bbbbbbbbbbbbbbbbbbbbb (20 b's)
ccccccccccccccccccccc (20 c's)

Long problematic strings (use with caution):
aaaaaaaaaaaaaaaaaaaaaaaaaaaaaX (30 a's + X)
bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbY (30 b's + Y)
ccccccccccccccccccccccccccccccZ (30 c's + Z)

Very long strings (extreme caution):
String of 100+ characters without expected ending

HTML-like problematic content:
<div class="test">very long content without proper closing
<span>nested <div>content</span> improper nesting
<!-- comment without proper closing

URL-like problematic content:
http://very.long.domain.name/very/long/path/without/proper/format
https://malformed...url...with...many...dots

Email-like problematic content:
very.long.email.address.with.many.dots@domain.without.proper.format
user@domain...with...many...dots...com

PATHOLOGICAL_END: End of pathological pattern tests

Note: When implementing regex engines, consider:
1. Timeout mechanisms for long-running matches
2. Backtracking limits
3. Input length limits
4. Pattern complexity analysis
5. Alternative algorithms (finite automata)
"#.to_string())
    }
    
    /// Generate Unicode regex pattern test content
    fn generate_unicode_regex_patterns(&self) -> Result<String> {
        Ok(r#"Unicode Regex Pattern Tests
===========================

UNICODE_REGEX_TEST: Testing Unicode-aware regex patterns

1. Basic Unicode character matching:
   Korean text: unicode_test_한글 - Hangul characters
   Arabic text: unicode_test_العربية - Arabic script
   Chinese text: unicode_test_中文 - Chinese characters
   Japanese text: unicode_test_日本語 - Japanese characters
   Russian text: unicode_test_русский - Cyrillic script
   Greek text: unicode_test_ελληνικά - Greek script
   Hindi text: unicode_test_हिन्दी - Devanagari script

2. Unicode property classes:
   
   Letter class \p{L}:
   Latin: abcdefghijklmnopqrstuvwxyz
   Cyrillic: абвгдеёжзийклмнопрстуфхцчшщъыьэюя
   Greek: αβγδεζηθικλμνξοπρστυφχψω
   Arabic: ابتثجحخدذرزسشصضطظعغفقكلمنهوي
   
   Number class \p{N}:
   Arabic numerals: 0123456789
   Arabic-Indic: ٠١٢٣٤٥٦٧٨٩
   Devanagari: ०१२३४५६७८९
   Bengali: ০১২৩৪৫৬৭৮৯

3. Script-specific patterns:
   
   \p{Script=Latin}: 
   English, French, German, Spanish, Italian text
   Àáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿ
   
   \p{Script=Cyrillic}:
   Russian: Привет мир
   Bulgarian: Здравей свят
   Serbian: Здраво свете
   
   \p{Script=Arabic}:
   Arabic: مرحبا بالعالم
   Persian: سلام دنیا
   Urdu: ہیلو ورلڈ
   
   \p{Script=Han}:
   Simplified Chinese: 你好世界
   Traditional Chinese: 你好世界
   Japanese Kanji: 世界

4. Unicode categories:
   
   Uppercase letter \p{Lu}:
   ABCDEFGHIJKLMNOPQRSTUVWXYZ
   ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ
   АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ
   
   Lowercase letter \p{Ll}:
   abcdefghijklmnopqrstuvwxyz
   αβγδεζηθικλμνξοπρστυφχψω
   абвгдеёжзийклмнопрстуфхцчшщъыьэюя
   
   Decimal number \p{Nd}:
   0123456789 ٠١٢٣٤٥٦٧٨٩ ०१२३४५६७८९
   
   Punctuation \p{P}:
   !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
   ¡¿«»‚„‹›''""‛‟•‰‱′″‴‵‶‷‸‹›
   
   Symbol \p{S}:
   $+<=>^`|~¢£¤¥¦¨©«®°±²³´µ¶·¸¹»¼½¾¿×÷

5. Combining characters and normalization:
   
   Base + combining: e + ◌́ = é
   Multiple combining: e + ◌́ + ◌̂ = ế
   
   Café written as:
   - café (precomposed)
   - cafe´ (base + combining acute)
   - café (mixed)
   
   Naïve written as:
   - naïve (precomposed)
   - naive¨ (base + combining diaeresis)

6. Emoji and symbols:
   
   Basic emoji: 😀😃😄😁😆😅🤣😂🙂🙃😉😊😇
   Skin tone variants: 👋🏻👋🏼👋🏽👋🏾👋🏿
   Complex emoji: 👨‍👩‍👧‍👦 👩‍💻 🏳️‍🌈 🏳️‍⚧️
   
   Mathematical symbols: ∀∂∃∄∅∆∇∈∉∊∋∌∍∎∏∐∑−∓∔∕∖∗∘∙√∛∜∝∞∟∠∡∢∣∤∥∦∧∨∩∪∫∬∭∮∯∰∱∲∳∴∵∶∷∸∹∺∻∼∽∾∿≀≁≂≃≄≅≆≇≈≉≊≋≌≍≎≏≐≑≒≓≔≕≖≗≘≙≚≛≜≝≞≟≠≡≢≣≤≥≦≧≨≩≪≫≬≭≮≯≰≱≲≳≴≵≶≷≸≹≺≻≼≽≾≿⊀⊁⊂⊃⊄⊅⊆⊇⊈⊉⊊⊋⊌⊍⊎⊏⊐⊑⊒⊓⊔⊕⊖⊗⊘⊙⊚⊛⊜⊝⊞⊟⊠⊡⊢⊣⊤⊥⊦⊧⊨⊩⊪⊫⊬⊭⊮⊯⊰⊱⊲⊳⊴⊵⊶⊷⊸⊹⊺⊻⊼⊽⊾⊿⋀⋁⋂⋃⋄⋅⋆⋇⋈⋉⋊⋋⋌⋍⋎⋏⋐⋑⋒⋓⋔⋕⋖⋗⋘⋙⋚⋛⋜⋝⋞⋟⋠⋡⋢⋣⋤⋥⋦⋧⋨⋩⋪⋫⋬⋭⋮⋯⋰⋱⋲⋳⋴⋵⋶⋷⋸⋹⋺⋻⋼⋽⋾⋿

7. Case folding and normalization:
   
   Case folding examples:
   HELLO → hello
   İstanbul → i̇stanbul (Turkish)
   STRAßE → strasse (German)
   
   Unicode normalization forms:
   é as NFC: é (single codepoint)
   é as NFD: e + ́ (base + combining)

8. Direction and layout:
   
   Left-to-right: Hello World (English)
   Right-to-left: مرحبا بالعالم (Arabic)
   Mixed: Hello مرحبا World
   
   Bidirectional marks:
   LRM (Left-to-Right Mark): ‎
   RLM (Right-to-Left Mark): ‏
   
9. Zero-width characters:
   
   Zero-width space: ​ (U+200B)
   Zero-width non-joiner: ‌ (U+200C)
   Zero-width joiner: ‍ (U+200D)
   Zero-width no-break space: ﻿ (U+FEFF)
   
   Word with zero-width chars: test​ing‌test‍ing

10. Language-specific patterns:
    
    Email with Unicode domains:
    user@пример.рф (Russian)
    user@例え.テスト (Japanese)
    user@مثال.إختبار (Arabic)
    
    URLs with Unicode:
    https://пример.рф/путь
    https://例え.テスト/パス
    https://مثال.إختبار/مسار
    
    File names with Unicode:
    документ.txt (Russian)
    文档.txt (Chinese)
    文書.txt (Japanese)
    مستند.txt (Arabic)

11. Pattern matching challenges:
    
    Accent-insensitive matching:
    café should match cafe, café, cafe´
    
    Case-insensitive Unicode:
    İstanbul should match istanbul, ISTANBUL
    
    Script mixing:
    Mixed script identifiers: test_тест_テスト_测试
    
    Emoji in text:
    Text with emoji: Hello 👋 World 🌍
    
Real-world Unicode test cases:

International names:
José María González-Rodríguez
François Müller-Nørgård
Владимир Путин
محمد عبد الله
田中太郎
김철수

International addresses:
123 Main Street, New York, NY 10001, USA
1 Rue de la Paix, 75001 Paris, France
Красная площадь, 1, Москва, Россия
شارع التحرير، القاهرة، مصر
1-1-1 東京都千代田区、日本
서울특별시 강남구 테헤란로 123

Mixed content:
Programming variable: userName_用户名_اسم_المستخدم
Function name: calculateSum_計算和_حساب_المجموع
Error message: "无效的用户名" (Invalid username)
Log entry: [2024-01-15] INFO: Пользователь вошел в систему

End of Unicode regex pattern tests.
"#.to_string())
    }
    
    /// Generate code-specific regex pattern test content
    fn generate_code_regex_patterns(&self) -> Result<String> {
        Ok(r#"//! Code-Specific Regex Pattern Tests
//! Testing patterns commonly used for parsing source code

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tokio::sync::RwLock;

/// CODE_REGEX_TEST: Testing code-specific regex patterns

// 1. Function definitions
fn main() {
    println!("Hello, world!");
}

pub fn calculate_sum(a: i32, b: i32) -> i32 {
    a + b
}

async fn fetch_data() -> Result<String, Error> {
    // Async function implementation
    Ok("data".to_string())
}

pub async fn process_request<T: Serialize>(data: T) -> Result<Response, ProcessError> {
    // Generic async function
    unimplemented!()
}

// 2. Struct definitions
pub struct User {
    pub id: u64,
    pub name: String,
    pub email: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Config {
    database_url: String,
    api_key: String,
    timeout: Duration,
}

pub struct GenericContainer<T, U> 
where 
    T: Clone + Debug,
    U: Send + Sync,
{
    data: Vec<T>,
    metadata: U,
}

// 3. Implementation blocks
impl User {
    pub fn new(id: u64, name: String, email: String) -> Self {
        Self { id, name, email }
    }
    
    pub fn display_name(&self) -> &str {
        &self.name
    }
}

impl<T, U> GenericContainer<T, U> 
where 
    T: Clone + Debug,
    U: Send + Sync,
{
    pub fn new(metadata: U) -> Self {
        Self {
            data: Vec::new(),
            metadata,
        }
    }
}

// 4. Trait definitions
pub trait Processable {
    type Output;
    
    fn process(&self) -> Self::Output;
    fn validate(&self) -> bool;
}

pub trait AsyncProcessor<T>: Send + Sync {
    async fn process_async(&self, input: T) -> Result<String, ProcessError>;
}

// 5. Enum definitions
#[derive(Debug, Clone, PartialEq)]
pub enum Status {
    Pending,
    InProgress { started_at: DateTime<Utc> },
    Completed { result: String },
    Failed { error: String, retryable: bool },
}

#[derive(Error, Debug)]
pub enum ProcessError {
    #[error("Network error: {0}")]
    Network(String),
    #[error("Validation failed")]
    Validation,
    #[error("Timeout after {seconds} seconds")]
    Timeout { seconds: u64 },
}

// 6. Import statements
use std::collections::{HashMap, HashSet, BTreeMap};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, oneshot};
use serde::{Deserialize, Serialize};
use anyhow::{Result, Context, Error};
use thiserror::Error;
use regex::Regex;
use uuid::Uuid;

// 7. Macro definitions
macro_rules! create_struct {
    ($name:ident { $($field:ident: $type:ty),+ }) => {
        struct $name {
            $($field: $type,)+
        }
    };
}

macro_rules! log_debug {
    ($($arg:tt)*) => {
        #[cfg(debug_assertions)]
        println!("[DEBUG] {}", format!($($arg)*));
    };
}

// 8. Attribute usage
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ApiResponse {
    #[serde(rename = "responseId")]
    pub id: String,
    pub data: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_user_creation() {
        let user = User::new(1, "Test".to_string(), "test@example.com".to_string());
        assert_eq!(user.id, 1);
    }
    
    #[tokio::test]
    async fn test_async_function() {
        let result = fetch_data().await;
        assert!(result.is_ok());
    }
}

// 9. Generic type constraints
pub fn generic_function<T, U, V>(input: T) -> Result<V, ProcessError>
where
    T: Clone + Debug + Send + 'static,
    U: Serialize + DeserializeOwned,
    V: Default + PartialEq,
{
    // Generic function implementation
    Ok(V::default())
}

// 10. Complex type definitions
type DatabaseConnection = Arc<Mutex<Connection>>;
type UserCache = Arc<RwLock<HashMap<u64, User>>>;
type AsyncResult<T> = Pin<Box<dyn Future<Output = Result<T, ProcessError>> + Send>>;
type EventHandler = Box<dyn Fn(&Event) -> Result<(), HandleError> + Send + Sync>;

// 11. Constants and statics
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);
const MAX_RETRIES: u32 = 3;
const API_VERSION: &str = "v1.0.0";

static GLOBAL_CONFIG: OnceCell<Config> = OnceCell::new();
static USER_CACHE: Lazy<UserCache> = Lazy::new(|| {
    Arc::new(RwLock::new(HashMap::new()))
});

// 12. Module declarations
pub mod database {
    pub mod connection;
    pub mod models;
    pub mod queries;
}

pub mod api {
    pub use self::handlers::*;
    pub use self::middleware::*;
    
    mod handlers;
    mod middleware;
}

// 13. Conditional compilation
#[cfg(feature = "async")]
pub async fn async_operation() -> Result<String, Error> {
    // Async implementation
    Ok("result".to_string())
}

#[cfg(not(feature = "async"))]
pub fn sync_operation() -> Result<String, Error> {
    // Sync implementation
    Ok("result".to_string())
}

#[cfg(target_os = "windows")]
fn platform_specific_function() {
    // Windows-specific implementation
}

#[cfg(target_os = "linux")]
fn platform_specific_function() {
    // Linux-specific implementation
}

// 14. Closure and function pointer patterns
let closure = |x: i32, y: i32| -> i32 { x + y };
let async_closure = |data: String| async move {
    process_data(data).await
};

type SyncCallback = fn(i32) -> Result<String, Error>;
type AsyncCallback = Box<dyn Fn(i32) -> AsyncResult<String>>;

// 15. Pattern matching
match status {
    Status::Pending => println!("Waiting..."),
    Status::InProgress { started_at } => {
        println!("Started at: {}", started_at);
    }
    Status::Completed { result } => {
        println!("Result: {}", result);
    }
    Status::Failed { error, retryable: true } => {
        println!("Retryable error: {}", error);
    }
    Status::Failed { error, retryable: false } => {
        println!("Fatal error: {}", error);
    }
}

// 16. Complex expressions and method calls
let result = database
    .get_connection()
    .await?
    .query("SELECT * FROM users WHERE active = $1")
    .bind(true)
    .fetch_all()
    .await
    .context("Failed to fetch users")?;

let transformed = data
    .into_iter()
    .filter(|item| item.is_valid())
    .map(|item| item.transform())
    .collect::<Result<Vec<_>, _>>()?;

// 17. Lifetime annotations
struct DataProcessor<'a> {
    config: &'a Config,
    cache: &'a mut HashMap<String, String>,
}

impl<'a> DataProcessor<'a> {
    fn process_with_lifetime<'b>(&'b self, input: &'b str) -> &'b str 
    where 
        'a: 'b 
    {
        // Processing with lifetimes
        input
    }
}

// 18. Unsafe code blocks
unsafe fn unsafe_operation(ptr: *mut u8, len: usize) -> Result<(), Error> {
    if ptr.is_null() {
        return Err(Error::msg("Null pointer"));
    }
    
    let slice = std::slice::from_raw_parts_mut(ptr, len);
    // Unsafe operations
    Ok(())
}

// 19. FFI declarations
extern "C" {
    fn external_function(input: *const c_char) -> c_int;
    fn another_external(data: *mut c_void, size: size_t) -> c_int;
}

#[no_mangle]
pub extern "C" fn exported_function(value: c_int) -> c_int {
    value * 2
}

// 20. Documentation comments
/// This function calculates the factorial of a number
/// 
/// # Arguments
/// 
/// * `n` - The number to calculate factorial for
/// 
/// # Examples
/// 
/// ```
/// let result = factorial(5);
/// assert_eq!(result, 120);
/// ```
/// 
/// # Panics
/// 
/// This function will panic if `n` is greater than 20
/// 
/// # Errors
/// 
/// Returns `CalculateError` if the calculation fails
pub fn factorial(n: u64) -> Result<u64, CalculateError> {
    if n > 20 {
        panic!("Number too large");
    }
    
    if n == 0 {
        return Err(CalculateError::InvalidInput);
    }
    
    Ok((1..=n).product())
}
"#.to_string())
    }
    
    /// Generate escaped metacharacter pattern test content
    fn generate_escaped_metacharacter_patterns(&self) -> Result<String> {
        Ok(r#"Escaped Metacharacter Regex Tests
==================================

ESCAPED_REGEX_TEST: Testing literal matching of regex metacharacters

1. Literal dot (.) matching:
   Pattern: \.
   
   Matches literal dots in:
   file.txt
   version.2.1.0
   www.example.com
   literal.dot in text
   3.14159 (pi)
   
   Should NOT match other characters:
   fileXtxt (X instead of dot)
   version_2_1_0 (underscores instead of dots)

2. Literal asterisk (*) matching:
   Pattern: \*
   
   Matches literal asterisks in:
   wildcard*pattern
   emphasis*text*emphasis
   literal*asterisk
   multiply: 5 * 6 = 30
   
   SQL patterns:
   SELECT * FROM table
   UPDATE table SET * WHERE condition

3. Literal plus (+) matching:
   Pattern: \+
   
   Matches literal plus signs in:
   addition: 2 + 3 = 5
   phone: +1-555-123-4567
   email: user+tag@domain.com
   literal+plus
   
   URL encoding:
   space%20encoded
   plus+sign+encoded

4. Literal question mark (?) matching:
   Pattern: \?
   
   Matches literal question marks in:
   What is this? A question
   URL parameters: ?param=value&other=data
   Ternary operator: condition ? true : false
   literal?question
   
   HTTP query strings:
   /search?q=term&page=1

5. Literal caret (^) matching:
   Pattern: \^
   
   Matches literal carets in:
   XOR operator: a ^ b
   Exponentiation: 2^8 = 256
   regex anchor: ^start
   literal^caret
   
   Mathematical expressions:
   x^2 + y^2 = z^2

6. Literal dollar sign ($) matching:
   Pattern: \$
   
   Matches literal dollar signs in:
   Price: $19.99
   Variable: $variable_name
   Shell script: echo $HOME
   End anchor: pattern$
   literal$dollar
   
   Currency amounts:
   $1,000.00
   Total: $45.67

7. Literal parentheses matching:
   Pattern: \( and \)
   
   Matches literal parentheses in:
   Function call: func(arg1, arg2)
   Mathematical: (a + b) * c
   Group notation: (123) 456-7890
   literal(parentheses)
   
   Code patterns:
   if (condition) { }
   for (int i = 0; i < n; i++)

8. Literal square brackets matching:
   Pattern: \[ and \]
   
   Matches literal brackets in:
   Array access: array[index]
   Character class: [a-z]
   JSON array: ["item1", "item2"]
   literal[brackets]
   
   Configuration:
   [section]
   key=value

9. Literal curly braces matching:
   Pattern: \{ and \}
   
   Matches literal braces in:
   Code blocks: if (true) { code }
   JSON object: {"key": "value"}
   Template: ${variable}
   literal{braces}
   
   CSS rules:
   .class { property: value; }

10. Literal backslash matching:
    Pattern: \\
    
    Matches literal backslashes in:
    Windows path: C:\Windows\System32
    Escape sequence: \n \t \r
    Regex escape: \d \w \s
    literal\backslash
    
    File paths:
    C:\Program Files\Application
    \\server\share\file

11. Literal pipe (|) matching:
    Pattern: \|
    
    Matches literal pipes in:
    Shell command: cat file | grep pattern
    Table separator: column1 | column2 | column3
    Logical OR: condition1 | condition2
    literal|pipe
    
    Markdown tables:
    | Header 1 | Header 2 |
    | Value 1  | Value 2  |

12. Complex escaped pattern combinations:
    
    File extensions with dots:
    \.txt \.log \.json \.xml
    
    Version numbers:
    v\d+\.\d+\.\d+
    Example: v1.2.3, v10.5.0
    
    IP addresses with dots:
    \d+\.\d+\.\d+\.\d+
    Example: 192.168.1.1
    
    Email with plus and dot:
    \w+\+\w+@\w+\.\w+
    Example: user+tag@domain.com
    
    URL with various metacharacters:
    https://example\.com/path\?param=value&other=data
    
    Mathematical expressions:
    \d+\s*\+\s*\d+\s*=\s*\d+
    Example: 2 + 3 = 5
    
    Phone numbers with parentheses and dashes:
    \(\d{3}\)\s*\d{3}-\d{4}
    Example: (555) 123-4567

13. Mixed literal and special characters:
    
    Pattern: hello\.world\*test
    Matches: hello.world*test
    Does not match: helloXworld*test or hello.worldXtest
    
    Pattern: user\+\d+@domain\.com
    Matches: user+123@domain.com
    Does not match: user123@domain.com or user+abc@domain.com
    
    Pattern: \$\d+\.\d{2}
    Matches: $19.99, $100.00, $5.25
    Does not match: 19.99, $19.9, $100.0

14. Real-world escaped pattern examples:
    
    Log file names:
    application\.log\.\d{4}-\d{2}-\d{2}
    Matches: application.log.2024-01-15
    
    Config file sections:
    \[database\]
    Matches: [database]
    
    CSS selectors:
    \.class-name\s*\{
    Matches: .class-name {
    
    Command line options:
    --\w+\s*=\s*\w+
    Matches: --option=value
    
    Windows registry paths:
    HKEY_LOCAL_MACHINE\\SOFTWARE\\Company
    
    SQL LIKE patterns:
    column LIKE '%pattern\_%'
    Matches records with literal underscore

Sample text for testing escaped patterns:

Configuration files:
[database]
host=localhost
port=5432

[logging]
level=INFO
file=app.log

Code snippets:
function calculate(a, b) {
    return a + b;
}

if (condition) {
    process();
}

URLs and emails:
https://www.example.com/path?param=value
user+tag@domain.com
support@company.co.uk

File paths:
C:\Program Files\Application\bin\app.exe
/usr/local/bin/script.sh
./relative/path/file.txt

Mathematical expressions:
2 + 3 = 5
x^2 + y^2 = z^2
f(x) = 2*x + 1

Prices and currency:
Item 1: $19.99
Item 2: $45.67
Total: $65.66

Phone numbers:
(555) 123-4567
+1-800-555-0123
555.123.4567

End of escaped metacharacter tests.
"#.to_string())
    }
}
```

## Success Criteria
- Method generates 8 comprehensive regex pattern test files
- Files cover basic patterns, character classes, complex quantifiers, and lookarounds
- Pathological patterns that could cause performance issues are included
- Unicode-aware patterns with international character support
- Code-specific patterns for parsing source code (functions, structs, imports)
- Escaped metacharacter patterns for literal matching
- All files include appropriate expected_matches arrays for validation
- Patterns range from simple to complex, testing regex engine capabilities

## Time Limit
10 minutes maximum