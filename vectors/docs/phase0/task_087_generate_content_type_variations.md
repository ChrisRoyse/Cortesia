# Micro-Task 087: Generate Content Type Variations

## Objective
Generate text files representing different content types and domains to test vector search performance across diverse subject matter.

## Context
Content type variations ensure the vector search system performs well across different domains and writing styles, from technical documentation to creative writing to formal business communication.

## Prerequisites
- Task 086 completed (Formatting variations generated)

## Time Estimate
10 minutes

## Instructions
1. Navigate to test data directory: `cd data\test_files`
2. Create content type generation script `generate_content_types.py`:
   ```python
   #!/usr/bin/env python3
   """
   Generate content type variations for vector search domain testing.
   """
   
   import os
   import sys
   from pathlib import Path
   sys.path.append('templates')
   from template_generator import TestFileGenerator
   
   def generate_content_type_variations():
       """Generate files with different content types and domains."""
       generator = TestFileGenerator()
       
       # Sample 1: Technical documentation
       technical_content = """API Documentation: User Management Service
   
   Overview
   The User Management Service provides RESTful endpoints for user account operations including registration, authentication, profile management, and access control.
   
   Authentication
   All endpoints require bearer token authentication except for registration and login.
   Include the Authorization header with format: Bearer <token>
   
   Endpoints
   
   POST /api/users/register
   Creates a new user account with email verification.
   
   Request Body:
   {
     "email": "user@example.com",
     "password": "SecurePassword123!",
     "firstName": "John",
     "lastName": "Doe"
   }
   
   Response (201 Created):
   {
     "userId": "uuid-string",
     "email": "user@example.com", 
     "status": "pending_verification",
     "created": "2024-08-04T10:30:00Z"
   }
   
   GET /api/users/{userId}
   Retrieves user profile information.
   
   Response (200 OK):
   {
     "userId": "uuid-string",
     "email": "user@example.com",
     "firstName": "John",
     "lastName": "Doe",
     "created": "2024-08-04T10:30:00Z",
     "lastLogin": "2024-08-04T15:45:00Z"
   }
   
   Error Responses
   400 Bad Request: Invalid input data
   401 Unauthorized: Missing or invalid authentication
   404 Not Found: User does not exist
   500 Internal Server Error: Server processing error
   
   Rate Limiting
   All endpoints are limited to 100 requests per minute per user.
   Exceed this limit and receive 429 Too Many Requests response.
   
   Security Considerations
   - Passwords require minimum 8 characters with mixed case, numbers, and symbols
   - Email verification required before account activation
   - Failed login attempts trigger temporary account lockout
   - All communication must use HTTPS in production environments"""
   
       # Sample 2: Business communication
       business_content = """Quarterly Business Review - Q3 2024
   
   Executive Summary
   Q3 2024 demonstrated strong performance across key metrics with revenue growth of 15% year-over-year and successful expansion into new market segments. Strategic initiatives delivered measurable results while operational efficiency improvements reduced costs by 8%.
   
   Financial Performance
   Revenue: $2.4M (up 15% from Q3 2023)
   Gross Margin: 68% (improved from 64% previous quarter)
   Operating Expenses: $1.1M (down 8% through efficiency gains)
   Net Income: $520K (up 28% year-over-year)
   Customer Acquisition Cost: $145 (down from $180)
   
   Market Analysis
   The competitive landscape evolved significantly with two major competitors announcing new product launches. Our market share remained stable at 23% despite increased competition. Customer satisfaction scores improved to 4.2/5.0 from 3.8/5.0 following service improvements.
   
   Product Development
   Three major features launched successfully:
   - Advanced analytics dashboard (75% user adoption)
   - Mobile application v2.0 (4.6 star rating)
   - API platform for enterprise customers (12 initial integrations)
   
   Sales and Marketing
   Lead generation increased 35% through improved digital marketing campaigns. Conversion rate optimization resulted in 18% higher close rates. Sales team expansion from 8 to 12 representatives supported territory growth.
   
   Operational Highlights
   - Reduced customer support response time from 4 hours to 90 minutes
   - Implemented automated testing reducing deployment time by 60%
   - Achieved 99.8% uptime across all services
   - Completed SOC 2 Type II compliance audit successfully
   
   Challenges and Mitigation
   Supply chain disruptions affected third-party integrations, resolved through alternative vendor partnerships. Talent acquisition remained competitive, addressed through enhanced benefits package and remote work options.
   
   Q4 2024 Outlook
   Focus areas include international expansion, enterprise customer onboarding improvements, and advanced security feature development. Revenue target: $2.8M with continued margin improvement goals."""
   
       # Sample 3: Scientific/academic content
       academic_content = """Research Paper: Machine Learning Applications in Natural Language Processing
   
   Abstract
   This study examines the effectiveness of transformer-based architectures for semantic understanding tasks in natural language processing. We evaluate performance across multiple benchmarks including question answering, text classification, and sentiment analysis. Results demonstrate significant improvements over traditional approaches with attention mechanisms providing superior context modeling capabilities.
   
   Introduction
   Natural language processing has evolved rapidly with the introduction of transformer architectures (Vaswani et al., 2017). These models utilize self-attention mechanisms to capture long-range dependencies and contextual relationships more effectively than recurrent neural networks.
   
   The primary research questions addressed in this work are:
   1. How do transformer models compare to LSTM architectures on semantic understanding tasks?
   2. What impact does model size have on performance across different domains?
   3. How does fine-tuning affect task-specific performance?
   
   Methodology
   We conducted experiments using three datasets: SQuAD 2.0 for question answering, IMDB for sentiment analysis, and AG News for text classification. Model architectures included BERT-base, BERT-large, and RoBERTa for comparison with LSTM baselines.
   
   Training procedures followed standard practices with learning rates of 2e-5, batch sizes of 16, and 3 epochs of fine-tuning. Evaluation metrics included accuracy, F1-score, and exact match for question answering tasks.
   
   Results
   Transformer models consistently outperformed LSTM baselines:
   - Question Answering: BERT-large achieved 87.4% F1 vs 72.1% for LSTM
   - Sentiment Analysis: RoBERTa reached 94.2% accuracy vs 89.7% for LSTM  
   - Text Classification: BERT-base obtained 92.8% accuracy vs 85.3% for LSTM
   
   Statistical significance was confirmed using paired t-tests (p < 0.001 for all comparisons).
   
   Discussion
   The superior performance of transformer models can be attributed to their ability to model bidirectional context and capture complex linguistic relationships. Attention visualizations reveal that models learn to focus on relevant tokens for each task.
   
   Limitations include computational requirements during training and inference, making deployment challenging in resource-constrained environments. Future work should explore model compression techniques and efficient architectures.
   
   Conclusion
   Transformer-based models represent a significant advancement in natural language processing capabilities. The experimental results support their adoption for semantic understanding tasks, though practical considerations around computational efficiency remain important.
   
   References
   Vaswani, A., et al. (2017). Attention is all you need. Advances in Neural Information Processing Systems, 30."""
   
       # Sample 4: Creative/narrative content
       creative_content = """The Last Library
   
   Chapter 1: Discovery
   
   Sarah's flashlight carved a narrow beam through the dust-laden air as she descended the forgotten staircase. The smell of old paper and leather bindings grew stronger with each step, a scent she hadn't encountered in decades. In the world above, digital screens had long replaced the printed word, but here, in this hidden chamber beneath the ruins of the old university, something extraordinary waited.
   
   The beam fell upon towering shelves that stretched into darkness, packed with volumes that should have been destroyed in the Great Digitization of 2089. Her heart raced as she realized what she had discovered: the last physical library on Earth.
   
   "Impossible," she whispered, her voice barely audible in the cavernous space. She approached the nearest shelf and carefully withdrew a book, its spine cracked but still legible: "Pride and Prejudice" by Jane Austen. The pages felt foreign beneath her fingertips, textured and substantial in ways that holographic text could never replicate.
   
   Chapter 2: The Guardian
   
   "You shouldn't be here," a voice echoed from the shadows.
   
   Sarah spun around, nearly dropping the precious book. An elderly man emerged from between the stacks, his eyes reflecting a lifetime of reading and remembering. He wore the faded robes of an academic, a relic from an age when knowledge was tangible.
   
   "I'm Professor Chen," he said, adjusting wire-rimmed glasses that seemed as ancient as the books surrounding them. "I've been the guardian of this place for thirty years, waiting for someone like you to find it."
   
   "Someone like me?" Sarah asked, clutching the Austen novel to her chest.
   
   "Someone who still remembers what we've lost," he replied, gesturing to the countless volumes. "Someone who understands that knowledge isn't just information—it's wisdom, imagination, and the human experience captured in ink and paper."
   
   Chapter 3: The Choice
   
   Professor Chen led Sarah deeper into the library, past sections dedicated to science, philosophy, poetry, and histories of civilizations long forgotten. In the center stood a reading room with comfortable chairs and wooden tables scarred by generations of students.
   
   "The authorities believe all physical books were destroyed," Chen explained. "But a few of us saved what we could, hiding them here until the world was ready to remember."
   
   Sarah sat in one of the ancient chairs and opened a volume of Shakespeare's sonnets. The words seemed to breathe with life, connecting her to centuries of readers who had touched these same pages.
   
   "What happens now?" she asked.
   
   Chen smiled sadly. "That choice is yours. You can leave and forget this place ever existed, or you can help me preserve what remains of our literary heritage for future generations who might one day rediscover the magic of the written word."
   
   Sarah looked around at the treasure trove of human knowledge and creativity. In a world that had chosen efficiency over beauty, speed over contemplation, she knew her answer.
   
   "I'll stay," she said, and began her new role as guardian of the last library."""
   
       # Sample 5: Conversational/informal content
       conversational_content = """Hey there! Welcome to TechTalk Daily 🎙️
   
   Episode 247: "Why Your Code Probably Sucks (And How to Fix It)"
   
   [INTRO MUSIC]
   
   HOST: What's up, code warriors! I'm Jake, and you're listening to TechTalk Daily, the podcast where we dive deep into the messy reality of software development. No corporate BS, no marketing fluff—just real talk about real code.
   
   Today we're talking about something that hits close to home for all of us: bad code. Yeah, I said it. Your code probably sucks. Mine does too sometimes. But here's the thing—it's totally fixable!
   
   I got this email from Lisa in Portland, and she's like, "Jake, I've been coding for three years, but when I look at code I wrote six months ago, I want to delete everything and start over. Is this normal?"
   
   Lisa, oh my friend, this is SO normal! If you're not occasionally horrified by your old code, you're probably not growing as a developer. It's like looking at photos from high school—embarrassing but necessary for personal development!
   
   So let's talk about the top five ways your code probably sucks:
   
   Number 1: Variable names that make no sense
   
   Come on, we've all been there. You're in the zone, cranking out code at 2 AM, and you name your variables things like "x", "temp", "thing", or my personal favorite—"stuffDoer". 
   
   Future you (and your teammates) will hate past you for this. Spend the extra 10 seconds to write meaningful names. Your code should read like a story, not like ancient hieroglyphics.
   
   Number 2: Functions that do everything except make coffee
   
   I see these monster functions that are like 200 lines long and handle user authentication, database queries, email sending, AND somehow also calculate the meaning of life. Break it down, people! One function, one responsibility. 
   
   If you can't explain what your function does in one sentence without using the word "and", it's doing too much.
   
   Number 3: Comments that lie
   
   Either your comments are completely wrong because you changed the code but forgot to update them, or they're stating the obvious like "// increment i by 1" next to "i++". 
   
   Good comments explain WHY, not WHAT. The code already shows what it does—tell me why you chose this approach instead of the other five ways you could have done it.
   
   [MUSIC BREAK]
   
   Number 4: Error handling? What's that?
   
   Your code just assumes everything will work perfectly all the time. The database will always be available, the network will never fail, users will always input valid data, and unicorns will deliver your HTTP responses.
   
   Reality check: things break. Handle your errors gracefully, log useful information, and give your users meaningful feedback instead of cryptic error codes.
   
   Number 5: Copy-paste programming
   
   You found some code on Stack Overflow that almost does what you need, so you just copy-paste it in fifteen different places with slight modifications. Now you have the same bug in fifteen places, and fixing anything is a nightmare.
   
   Abstract common functionality, create reusable components, and please, PLEASE understand the code you're copying before you paste it into your production system.
   
   Alright, that's our show for today! Remember, writing bad code doesn't make you a bad developer—it makes you human. The goal isn't perfection; it's continuous improvement.
   
   Hit me up on Twitter @JakeTechTalk with your worst code horror stories. I read every message, and the best ones might make it into a future episode!
   
   Until next time, keep coding, keep learning, and remember—your code is probably better than you think it is!
   
   [OUTRO MUSIC]"""
   
       # Generate all content type files
       samples = [
           ("technical_documentation.txt", "Technical API documentation", technical_content),
           ("business_communication.txt", "Business quarterly review content", business_content),
           ("academic_research.txt", "Scientific/academic research paper", academic_content),
           ("creative_narrative.txt", "Creative fiction writing", creative_content),
           ("conversational_informal.txt", "Informal conversational content", conversational_content)
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
   
   def analyze_content_type(file_path):
       """Analyze content type characteristics."""
       with open(file_path, 'r', encoding='utf-8') as f:
           content = f.read()
       
       # Word analysis
       words = content.split()
       word_count = len(words)
       unique_words = len(set(word.lower().strip('.,!?";:()[]{}') for word in words))
       
       # Sentence analysis
       sentence_endings = content.count('.') + content.count('!') + content.count('?')
       
       # Technical indicators
       code_indicators = content.count('{') + content.count('}') + content.count('[]') + content.count('()')
       url_indicators = content.count('http') + content.count('www') + content.count('.com')
       
       # Formal vs informal indicators  
       formal_words = ['therefore', 'however', 'furthermore', 'consequently', 'nevertheless']
       informal_words = ['yeah', 'ok', 'hey', 'awesome', 'cool', 'stuff']
       
       formal_count = sum(content.lower().count(word) for word in formal_words)
       informal_count = sum(content.lower().count(word) for word in informal_words)
       
       return {
           "word_count": word_count,
           "unique_words": unique_words,
           "vocabulary_richness": unique_words / word_count if word_count > 0 else 0,
           "sentence_count": sentence_endings,
           "avg_words_per_sentence": word_count / sentence_endings if sentence_endings > 0 else 0,
           "code_indicators": code_indicators,
           "url_indicators": url_indicators,
           "formal_indicators": formal_count,
           "informal_indicators": informal_count,
           "formality_ratio": formal_count / (formal_count + informal_count + 1)
       }
   
   def main():
       """Main generation function."""
       print("Generating content type variations...")
       
       # Ensure output directory exists
       os.makedirs("basic_text", exist_ok=True)
       
       try:
           files = generate_content_type_variations()
           print(f"\nSuccessfully generated {len(files)} content type files:")
           
           # Analyze each file
           for file_path in files:
               print(f"\n  - {file_path}")
               stats = analyze_content_type(file_path)
               print(f"    Words: {stats['word_count']} ({stats['unique_words']} unique)")
               print(f"    Vocabulary richness: {stats['vocabulary_richness']:.2f}")
               print(f"    Avg words/sentence: {stats['avg_words_per_sentence']:.1f}")
               print(f"    Formality ratio: {stats['formality_ratio']:.2f}")
               print(f"    Technical indicators: {stats['code_indicators']}")
           
           print("\nContent type variation generation completed successfully!")
           return 0
       
       except Exception as e:
           print(f"Error generating content type variations: {e}")
           return 1
   
   if __name__ == "__main__":
       sys.exit(main())
   ```
3. Run generation: `python generate_content_types.py`
4. Return to root: `cd ..\..`
5. Commit: `git add data\test_files\generate_content_types.py data\test_files\basic_text && git commit -m "task_087: Generate content type variations for domain testing"`

## Expected Output
- 5 text files representing different content domains
- Technical documentation sample
- Business communication sample
- Academic/scientific content sample
- Creative narrative sample
- Conversational/informal sample

## Success Criteria
- [ ] Technical documentation file generated
- [ ] Business communication file generated
- [ ] Academic research file generated
- [ ] Creative narrative file generated
- [ ] Conversational content file generated
- [ ] Content analysis statistics calculated

## Validation Commands
```cmd
cd data\test_files
python generate_content_types.py
dir basic_text
```

## Next Task
task_088_validate_basic_text_generation.md

## Notes
- Different content types test domain-specific performance
- Vocabulary richness varies significantly across domains
- Formality indicators help classify content styles
- Files ensure robust cross-domain vector search capability