"""
文件包含各种Unicode字符和多语言内容，用于测试编码处理。
Файл содержащий различные Unicode символы и многоязычный контент для тестирования кодировки.
ファイルにはさまざまなUnicode文字と多言語コンテンツが含まれており、エンコーディング処理をテストします。
"""

import re
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

class MultilingualProcessor:
    """处理多语言内容的处理器 - Processor for handling multilingual content."""
    
    def __init__(self):
        # Unicode characters and symbols
        self.symbols = {
            "currency": ["$", "€", "£", "¥", "₹", "₽", "₩", "₦", "₪", "₨"],
            "math": ["∑", "∏", "∫", "∂", "∇", "∞", "≈", "≠", "≤", "≥", "±", "×", "÷"],
            "arrows": ["→", "←", "↑", "↓", "↔", "⇒", "⇐", "⇑", "⇓", "⇔"],
            "greek": ["α", "β", "γ", "δ", "ε", "ζ", "η", "θ", "ι", "κ", "λ", "μ", "ν", "ξ", "ο", "π", "ρ", "σ", "τ", "υ", "φ", "χ", "ψ", "ω"],
            "emoji": ["😀", "😂", "🤔", "👍", "❤️", "🌟", "🚀", "💡", "🎉", "🔥", "💯", "🌈"],
            "chinese_traditional": ["龍", "鳳", "麒", "麟", "飛", "鶴", "仙", "境", "蒼", "穹"],
            "chinese_simplified": ["龙", "凤", "麒", "麟", "飞", "鹤", "仙", "境", "苍", "穹"],
            "japanese_hiragana": ["あ", "い", "う", "え", "お", "か", "き", "く", "け", "こ", "さ", "し", "す", "せ", "そ"],
            "japanese_katakana": ["ア", "イ", "ウ", "エ", "オ", "カ", "キ", "ク", "ケ", "コ", "サ", "シ", "ス", "セ", "ソ"],
            "korean": ["가", "나", "다", "라", "마", "바", "사", "아", "자", "차", "카", "타", "파", "하"],
            "arabic": ["ا", "ب", "ت", "ث", "ج", "ح", "خ", "د", "ذ", "ر", "ز", "س", "ش", "ص", "ض"],
            "hebrew": ["א", "ב", "ג", "ד", "ה", "ו", "ז", "ח", "ט", "י", "כ", "ל", "מ", "נ", "ס"],
            "cyrillic": ["а", "б", "в", "г", "д", "е", "ё", "ж", "з", "и", "й", "к", "л", "м", "н"],
            "devanagari": ["अ", "आ", "इ", "ई", "उ", "ऊ", "ए", "ऐ", "ओ", "औ", "क", "ख", "ग", "घ", "ङ"]
        }
        
        # Multilingual greetings and phrases
        self.greetings = {
            "english": "Hello, welcome to our application!",
            "spanish": "¡Hola, bienvenido a nuestra aplicación!",
            "french": "Bonjour, bienvenue dans notre application !",
            "german": "Hallo, willkommen in unserer Anwendung!",
            "italian": "Ciao, benvenuto nella nostra applicazione!",
            "portuguese": "Olá, bem-vindo ao nosso aplicativo!",
            "russian": "Привет, добро пожаловать в наше приложение!",
            "chinese_simplified": "您好，欢迎使用我们的应用程序！",
            "chinese_traditional": "您好，歡迎使用我們的應用程序！",
            "japanese": "こんにちは、私たちのアプリケーションへようこそ！",
            "korean": "안녕하세요, 저희 애플리케이션에 오신 것을 환영합니다!",
            "arabic": "مرحبا، أهلا بك في تطبيقنا!",
            "hebrew": "שלום, ברוכים הבאים לאפליקציה שלנו!",
            "hindi": "नमस्ते, हमारे एप्लिकेशन में आपका स्वागत है!",
            "thai": "สวัสดี ยินดีต้อนรับสู่แอปพลิเคชันของเรา!",
            "vietnamese": "Xin chào, chào mừng bạn đến với ứng dụng của chúng tôi!",
            "turkish": "Merhaba, uygulamamıza hoş geldiniz!",
            "dutch": "Hallo, welkom bij onze applicatie!",
            "swedish": "Hej, välkommen till vår applikation!",
            "norwegian": "Hei, velkommen til vår applikasjon!",
            "finnish": "Hei, tervetuloa sovellukseemme!",
            "polish": "Cześć, witamy w naszej aplikacji!",
            "czech": "Ahoj, vítejte v naší aplikaci!",
            "hungarian": "Helló, üdvözöljük alkalmazásunkban!"
        }
        
        # Technical terms in various languages
        self.tech_terms = {
            "database": {
                "english": "database",
                "spanish": "base de datos",
                "french": "base de données",
                "german": "Datenbank",
                "russian": "база данных",
                "chinese": "数据库",
                "japanese": "データベース",
                "korean": "데이터베이스"
            },
            "algorithm": {
                "english": "algorithm",
                "spanish": "algoritmo",
                "french": "algorithme",
                "german": "Algorithmus",
                "russian": "алгоритм",
                "chinese": "算法",
                "japanese": "アルゴリズム",
                "korean": "알고리즘"
            },
            "function": {
                "english": "function",
                "spanish": "función",
                "french": "fonction",
                "german": "Funktion",
                "russian": "функция",
                "chinese": "函数",
                "japanese": "関数",
                "korean": "함수"
            }
        }
    
    def process_unicode_text(self, text: str) -> Dict[str, Any]:
        """
        处理包含Unicode字符的文本
        Process text containing Unicode characters
        Unicodeを含むテキストを処理する
        """
        
        # Character classification
        char_stats = {
            "total_chars": len(text),
            "ascii_chars": 0,
            "latin_extended": 0,
            "cyrillic": 0,
            "chinese_japanese": 0,
            "arabic": 0,
            "devanagari": 0,
            "emoji": 0,
            "symbols": 0,
            "whitespace": 0,
            "digits": 0,
            "punctuation": 0
        }
        
        # Language detection patterns
        language_patterns = {
            "chinese": r'[\u4e00-\u9fff]',
            "japanese_hiragana": r'[\u3040-\u309f]',
            "japanese_katakana": r'[\u30a0-\u30ff]',
            "korean": r'[\uac00-\ud7af]',
            "arabic": r'[\u0600-\u06ff]',
            "hebrew": r'[\u0590-\u05ff]',
            "cyrillic": r'[\u0400-\u04ff]',
            "devanagari": r'[\u0900-\u097f]',
            "thai": r'[\u0e00-\u0e7f]',
            "emoji": r'[\U0001f600-\U0001f64f\U0001f300-\U0001f5ff\U0001f680-\U0001f6ff\U0001f1e0-\U0001f1ff]'
        }
        
        detected_languages = []
        
        for char in text:
            codepoint = ord(char)
            
            # Basic ASCII
            if 0 <= codepoint <= 127:
                if char.isspace():
                    char_stats["whitespace"] += 1
                elif char.isdigit():
                    char_stats["digits"] += 1
                elif char.isalpha():
                    char_stats["ascii_chars"] += 1
                else:
                    char_stats["punctuation"] += 1
            
            # Extended Latin (European languages)
            elif 128 <= codepoint <= 591:
                char_stats["latin_extended"] += 1
            
            # Cyrillic
            elif 1024 <= codepoint <= 1279:
                char_stats["cyrillic"] += 1
                if "russian" not in detected_languages:
                    detected_languages.append("russian")
            
            # Chinese/Japanese/Korean (CJK)
            elif 19968 <= codepoint <= 40959:  # CJK Unified Ideographs
                char_stats["chinese_japanese"] += 1
                if "chinese" not in detected_languages:
                    detected_languages.append("chinese")
            
            # Japanese Hiragana
            elif 12352 <= codepoint <= 12447:
                char_stats["chinese_japanese"] += 1
                if "japanese" not in detected_languages:
                    detected_languages.append("japanese")
            
            # Japanese Katakana
            elif 12448 <= codepoint <= 12543:
                char_stats["chinese_japanese"] += 1
                if "japanese" not in detected_languages:
                    detected_languages.append("japanese")
            
            # Korean Hangul
            elif 44032 <= codepoint <= 55215:
                char_stats["chinese_japanese"] += 1
                if "korean" not in detected_languages:
                    detected_languages.append("korean")
            
            # Arabic
            elif 1536 <= codepoint <= 1791:
                char_stats["arabic"] += 1
                if "arabic" not in detected_languages:
                    detected_languages.append("arabic")
            
            # Devanagari (Hindi, Sanskrit)
            elif 2304 <= codepoint <= 2431:
                char_stats["devanagari"] += 1
                if "hindi" not in detected_languages:
                    detected_languages.append("hindi")
            
            # Emoji ranges
            elif (127744 <= codepoint <= 128511 or  # Miscellaneous Symbols and Pictographs
                  128512 <= codepoint <= 128591 or  # Emoticons
                  128640 <= codepoint <= 128767):   # Transport and Map Symbols
                char_stats["emoji"] += 1
            
            # Other symbols
            else:
                char_stats["symbols"] += 1
        
        # Additional language detection using regex
        for lang, pattern in language_patterns.items():
            if re.search(pattern, text):
                if lang not in detected_languages:
                    detected_languages.append(lang)
        
        # Text normalization examples
        normalized_variants = {
            "nfc": text,  # Canonical decomposition followed by canonical composition
            "nfd": text,  # Canonical decomposition
            "nfkc": text, # Compatibility decomposition followed by canonical composition
            "nfkd": text  # Compatibility decomposition
        }
        
        # Try to normalize using unicodedata if available
        try:
            import unicodedata
            normalized_variants = {
                "nfc": unicodedata.normalize('NFC', text),
                "nfd": unicodedata.normalize('NFD', text),
                "nfkc": unicodedata.normalize('NFKC', text),
                "nfkd": unicodedata.normalize('NFKD', text)
            }
        except ImportError:
            pass
        
        return {
            "character_statistics": char_stats,
            "detected_languages": detected_languages,
            "text_length": len(text),
            "byte_length": len(text.encode('utf-8')),
            "normalized_forms": normalized_variants,
            "sample_characters": {
                "first_10": text[:10],
                "last_10": text[-10:],
                "middle_10": text[len(text)//2-5:len(text)//2+5] if len(text) > 10 else text
            }
        }
    
    def create_multilingual_content(self) -> str:
        """
        创建多语言内容示例
        Create multilingual content example
        多言語コンテンツの例を作成
        """
        
        content_parts = []
        
        # Add header with various scripts
        content_parts.append("=== 多语言内容示例 ===")
        content_parts.append("=== Multilingual Content Example ===")
        content_parts.append("=== Beispiel für mehrsprachige Inhalte ===")
        content_parts.append("=== 多言語コンテンツの例 ===")
        content_parts.append("=== 다국어 콘텐츠 예제 ===")
        content_parts.append("=== مثال على المحتوى متعدد اللغات ===")
        content_parts.append("=== Пример многоязычного контента ===")
        content_parts.append("")
        
        # Programming concepts in different languages
        content_parts.append("# Programming Concepts / 编程概念 / プログラミングの概念")
        content_parts.append("")
        
        programming_concepts = [
            "变量 (Variable) - переменная - 変数 - 변수 - متغير",
            "函数 (Function) - функция - 関数 - 함수 - دالة", 
            "类 (Class) - класс - クラス - 클래스 - فئة",
            "对象 (Object) - объект - オブジェクト - 객체 - كائن",
            "算法 (Algorithm) - алгоритм - アルゴリズム - 알고리즘 - خوارزمية",
            "数据结构 (Data Structure) - структура данных - データ構造 - 데이터 구조 - هيكل البيانات"
        ]
        
        for concept in programming_concepts:
            content_parts.append(f"  • {concept}")
        
        content_parts.append("")
        
        # Mathematical expressions with Unicode
        content_parts.append("# Mathematical Expressions / 数学表达式 / 数式")
        content_parts.append("")
        content_parts.append("∫₀^∞ e^(-x²) dx = √π/2")
        content_parts.append("∑ᵢ₌₁ⁿ i² = n(n+1)(2n+1)/6")
        content_parts.append("lim(x→∞) (1 + 1/x)ˣ = e ≈ 2.71828...")
        content_parts.append("∇²φ = ∂²φ/∂x² + ∂²φ/∂y² + ∂²φ/∂z² = 0 (Laplace equation)")
        content_parts.append("")
        
        # Currency and financial data
        content_parts.append("# Currency Examples / 货币示例 / 通貨の例")
        content_parts.append("")
        currencies = [
            "$1,234.56 (US Dollar)", 
            "€987.65 (Euro)",
            "£543.21 (British Pound)",
            "¥123,456 (Japanese Yen)",
            "₹12,345.67 (Indian Rupee)",
            "₽8,765.43 (Russian Ruble)",
            "₩1,234,567 (Korean Won)",
            "₦45,678.90 (Nigerian Naira)"
        ]
        
        for currency in currencies:
            content_parts.append(f"  • {currency}")
        
        content_parts.append("")
        
        # Emoji and modern Unicode
        content_parts.append("# Emoji and Symbols / 表情符号和符号 / 絵文字と記号")
        content_parts.append("")
        content_parts.append("👨‍💻 Developer working on 🚀 rocket science with 💡 brilliant ideas")
        content_parts.append("🌍🌎🌏 Global application supporting 🇺🇸🇨🇳🇯🇵🇰🇷🇷🇺🇩🇪🇫🇷🇪🇸 languages")
        content_parts.append("Performance metrics: 📈 95% ⬆️, Load time: ⚡ 0.5s, Users: 👥 10K+")
        content_parts.append("Status: ✅ Online, 🔒 Secure, 🛡️ Protected, 🔄 Auto-updating")
        content_parts.append("")
        
        # Code snippets with comments in different languages
        content_parts.append("# Code Examples / 代码示例 / コードの例")
        content_parts.append("")
        content_parts.append("```python")
        content_parts.append("# 这是一个函数 / This is a function / これは関数です")
        content_parts.append("def calculate_平均值(numbers: list) -> float:")
        content_parts.append('    """计算平均值 Calculate average 平均を計算する"""')
        content_parts.append("    if not numbers:")
        content_parts.append("        raise ValueError('空列表 Empty list 空のリスト')")
        content_parts.append("    return sum(numbers) / len(numbers)")
        content_parts.append("")
        content_parts.append("# Usage example / 使用示例 / 使用例")
        content_parts.append("数据 = [1, 2, 3, 4, 5]  # データ / data")
        content_parts.append("结果 = calculate_平均值(数据)  # 結果 / result")
        content_parts.append("print(f'平均值是: {结果}')  # 平均値は: / Average is:")
        content_parts.append("```")
        content_parts.append("")
        
        # Text in various writing systems
        content_parts.append("# Various Writing Systems / 各种文字系统 / 様々な文字体系")
        content_parts.append("")
        
        writing_samples = [
            "Latin: The quick brown fox jumps over the lazy dog",
            "Cyrillic: Съешь же ещё этих мягких французских булок да выпей чаю",
            "Chinese: 天地玄黄，宇宙洪荒。日月盈昃，辰宿列张。",
            "Japanese: いろはにほへと ちりぬるを わかよたれそ つねならむ",
            "Korean: 다람쥐 헌 쳇바퀴에 타고파",
            "Arabic: أبجد هوز حطي كلمن سعفص قرشت ثخذ ضظغ",
            "Hebrew: אבגד הוזח טיכל מנסע פצקר שתץף ץףך",
            "Devanagari: अकारादि हकारान्त वर्णमाला शिक्षणम्",
            "Thai: กขฃคฅฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮ"
        ]
        
        for sample in writing_samples:
            content_parts.append(f"  • {sample}")
        
        content_parts.append("")
        
        # Special Unicode characters and combining marks
        content_parts.append("# Special Characters / 特殊字符 / 特殊文字")
        content_parts.append("")
        special_chars = [
            "Combining marks: e̊ (e + combining ring above)",
            "Ligatures: ﬀ ﬁ ﬂ ﬃ ﬄ ﬆ",
            "Fractions: ½ ⅓ ¼ ⅕ ⅙ ⅛ ⅔ ¾ ⅖ ⅗ ⅘ ⅜ ⅝ ⅞",
            "Superscripts: x² y³ z⁴ a⁵ b⁶ c⁷ d⁸ e⁹ f¹⁰",
            "Subscripts: H₂O CO₂ NH₃ CH₄ C₆H₁₂O₆",
            "Arrows: ← → ↑ ↓ ↔ ↕ ↖ ↗ ↘ ↙ ⇒ ⇐ ⇑ ⇓ ⇔",
            "Box drawing: ┌─┐ │ │ └─┘ ╔═╗ ║ ║ ╚═╝",
            "Musical notes: ♪ ♫ ♬ ♭ ♮ ♯ 𝄞 𝄢 𝄡",
            "Chess pieces: ♔ ♕ ♖ ♗ ♘ ♙ ♚ ♛ ♜ ♝ ♞ ♟"
        ]
        
        for char_group in special_chars:
            content_parts.append(f"  • {char_group}")
        
        content_parts.append("")
        
        # Regional indicator symbols (flag emojis)
        content_parts.append("# Country Flags / 国旗 / 国旗")
        content_parts.append("")
        content_parts.append("🇺🇸🇨🇳🇯🇵🇰🇷🇷🇺🇩🇪🇫🇷🇬🇧🇮🇹🇪🇸🇧🇷🇮🇳🇨🇦🇦🇺🇲🇽🇳🇱🇸🇪🇳🇴🇩🇰🇫🇮🇵🇱🇨🇿🇭🇺🇷🇴🇬🇷🇹🇷")
        content_parts.append("")
        
        # Technical symbols
        content_parts.append("# Technical Symbols / 技术符号 / 技術記号")
        content_parts.append("")
        content_parts.append("⚙️ Settings ⚡ Performance 🔧 Tools 📊 Analytics 🛡️ Security")
        content_parts.append("⭐ Rating 🔍 Search 📝 Edit ✏️ Write 🗑️ Delete 💾 Save")
        content_parts.append("🌐 Network 📡 Signal 🔗 Link 🔒 Lock 🔓 Unlock")
        content_parts.append("")
        
        # Mathematical and scientific notation
        content_parts.append("# Scientific Notation / 科学记数法 / 科学記法")
        content_parts.append("")
        content_parts.append("Avogadro's number: Nₐ = 6.02214076 × 10²³ mol⁻¹")
        content_parts.append("Planck constant: h = 6.62607015 × 10⁻³⁴ J⋅s")
        content_parts.append("Speed of light: c = 2.99792458 × 10⁸ m/s")
        content_parts.append("Elementary charge: e = 1.602176634 × 10⁻¹⁹ C")
        content_parts.append("")
        
        return "\n".join(content_parts)
    
    def generate_unicode_test_cases(self) -> List[Dict[str, Any]]:
        """生成Unicode测试用例 Generate Unicode test cases."""
        
        test_cases = [
            {
                "name": "basic_multilingual",
                "content": "Hello 你好 こんにちは 안녕하세요 مرحبا שלום Привет",
                "expected_languages": ["english", "chinese", "japanese", "korean", "arabic", "hebrew", "russian"],
                "description": "Basic greetings in multiple languages"
            },
            {
                "name": "emoji_mixed",
                "content": "Great work! 👍 Keep it up! 🚀 Amazing results! 💯",
                "expected_languages": ["english"],
                "description": "English text mixed with emoji"
            },
            {
                "name": "mathematical_notation",
                "content": "∫₀^∞ e^(-x²)dx = √π/2 ∧ ∀x∈ℝ: x² ≥ 0",
                "expected_languages": [],
                "description": "Mathematical notation with Unicode symbols"
            },
            {
                "name": "programming_variables",
                "content": "let π = 3.14159; let Δt = 0.01; let α = π/4;",
                "expected_languages": ["english"],
                "description": "Programming code with Greek letters as variables"
            },
            {
                "name": "currency_symbols",
                "content": "Price: $100.50, €85.30, £72.40, ¥11,500, ₹7,500.25",
                "expected_languages": ["english"],
                "description": "Various currency symbols with numbers"
            },
            {
                "name": "chinese_traditional_simplified",
                "content": "繁體中文：資訊科技發展 简体中文：信息技术发展",
                "expected_languages": ["chinese"],
                "description": "Traditional and simplified Chinese characters"
            },
            {
                "name": "japanese_mixed_scripts",
                "content": "日本語のテキスト：ひらがな、カタカナ、漢字が混在している。",
                "expected_languages": ["japanese"],
                "description": "Japanese text with hiragana, katakana, and kanji"
            },
            {
                "name": "arabic_with_numbers",
                "content": "النص العربي مع الأرقام: ١٢٣٤٥٦٧٨٩٠ و 123456789",
                "expected_languages": ["arabic"],
                "description": "Arabic text with Arabic-Indic and Western numerals"
            },
            {
                "name": "right_to_left_mixed",
                "content": "This is English text. هذا نص عربي. This is English again.",
                "expected_languages": ["english", "arabic"],
                "description": "Mixed left-to-right and right-to-left text"
            },
            {
                "name": "combining_characters",
                "content": "Café naïve résumé Zürich Москва́",
                "expected_languages": ["english", "russian"],
                "description": "Text with combining diacritical marks"
            },
            {
                "name": "zero_width_characters",
                "content": "Word\u200BBreak Zero\u200CWidth\u200DJoiner",
                "expected_languages": ["english"],
                "description": "Text with zero-width characters"
            },
            {
                "name": "ethiopic_script",
                "content": "ሰላም አለም! እንዴት ነህ? (Hello World! How are you? in Amharic)",
                "expected_languages": ["english"],
                "description": "Ethiopic script (Amharic) with English translation"
            }
        ]
        
        return test_cases
    
    def validate_encoding_handling(self, text: str) -> Dict[str, Any]:
        """验证编码处理 Validate encoding handling."""
        
        encoding_tests = {}
        
        # Test various encodings
        encodings_to_test = ['utf-8', 'utf-16', 'utf-32', 'latin1', 'cp1252', 'ascii']
        
        for encoding in encodings_to_test:
            try:
                encoded = text.encode(encoding)
                decoded = encoded.decode(encoding)
                encoding_tests[encoding] = {
                    "success": True,
                    "byte_length": len(encoded),
                    "round_trip_success": decoded == text,
                    "encoding_errors": None
                }
            except UnicodeEncodeError as e:
                encoding_tests[encoding] = {
                    "success": False,
                    "byte_length": None,
                    "round_trip_success": False,
                    "encoding_errors": str(e)
                }
            except UnicodeDecodeError as e:
                encoding_tests[encoding] = {
                    "success": False,
                    "byte_length": None,
                    "round_trip_success": False,
                    "encoding_errors": str(e)
                }
        
        return {
            "original_text": text,
            "text_length": len(text),
            "encoding_tests": encoding_tests,
            "utf8_byte_length": len(text.encode('utf-8')),
            "contains_non_ascii": any(ord(c) > 127 for c in text),
            "contains_emoji": any(ord(c) >= 0x1F600 for c in text)
        }

# Example usage and test data
if __name__ == "__main__":
    processor = MultilingualProcessor()
    
    # Create multilingual content
    multilingual_text = processor.create_multilingual_content()
    print("=== Multilingual Content Created ===")
    print(f"Content length: {len(multilingual_text)} characters")
    print(f"UTF-8 byte length: {len(multilingual_text.encode('utf-8'))} bytes")
    print()
    
    # Process the content
    analysis = processor.process_unicode_text(multilingual_text)
    print("=== Unicode Analysis Results ===")
    print(f"Detected languages: {analysis['detected_languages']}")
    print(f"Character statistics: {analysis['character_statistics']}")
    print()
    
    # Test various Unicode strings
    test_cases = processor.generate_unicode_test_cases()
    print("=== Unicode Test Cases ===")
    for i, test_case in enumerate(test_cases[:3], 1):  # Show first 3 test cases
        print(f"{i}. {test_case['name']}: {test_case['content']}")
        case_analysis = processor.process_unicode_text(test_case['content'])
        print(f"   Languages detected: {case_analysis['detected_languages']}")
        print(f"   Byte length: {case_analysis['byte_length']}")
        print()
    
    # Encoding validation
    sample_text = "Hello 世界 🌍 Тест עולם"
    encoding_results = processor.validate_encoding_handling(sample_text)
    print("=== Encoding Validation ===")
    print(f"Sample text: {sample_text}")
    for encoding, result in encoding_results['encoding_tests'].items():
        status = "✅" if result['success'] else "❌"
        print(f"{status} {encoding}: {result}")
    
    # This content tests various aspects of Unicode handling:
    # 1. Multiple scripts and languages
    # 2. Emoji and symbols
    # 3. Mathematical notation
    # 4. Currency symbols
    # 5. Combining characters
    # 6. Right-to-left text
    # 7. Various number systems
    # 8. Special Unicode ranges