# SmolLM-135M-Instruct Integration Plan

## Model Overview

**Model ID**: `HuggingFaceTB/SmolLM-135M-Instruct`  
**Parameters**: 135,000,000 (135M)  
**Architecture**: Transformer (Llama-based) + Instruction Tuning  
**Size Category**: Small  
**Priority**: Tier 1 - Chat & Instructions  
**Use Case**: Chat, instruction following, conversational AI

## Key Differences from Base Model

This model builds on SmolLM-135M with additional instruction-following capabilities:

- **Instruction Tuning**: Fine-tuned for following user instructions
- **Chat Format**: Supports conversational context
- **Safety Alignment**: Additional safety training
- **Enhanced Reasoning**: Better instruction understanding

## Technical Specifications

### Model Details
- **Base Model**: SmolLM-135M
- **Training**: Base pre-training + instruction tuning
- **Chat Template**: Uses specific format for instructions
- **Context Length**: 2048 tokens
- **Special Tokens**: System, user, assistant tokens

### Performance Targets
- **Same as base model**: 80-120 tokens/second (CPU)
- **Memory Usage**: 250-350MB peak (slightly higher due to chat context)
- **First Token Latency**: <35ms
- **Bundle Size**: ~65MB quantized (Int4)

## Implementation Plan

### Phase 1: Extend Base Model (Day 1)

#### Step 1.1: Chat Template Integration
```rust
// src/models/smollm/smollm_135m_instruct.rs
use super::SmolLM135M;

pub struct SmolLM135MInstruct {
    base_model: SmolLM135M,
    chat_template: ChatTemplate,
    conversation_history: Vec<ChatMessage>,
    system_prompt: String,
}

#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ChatRole {
    System,
    User,
    Assistant,
}

pub struct ChatTemplate {
    system_token: String,
    user_token: String,
    assistant_token: String,
    end_token: String,
}

impl ChatTemplate {
    pub fn smollm_instruct() -> Self {
        Self {
            system_token: "<|system|>".to_string(),
            user_token: "<|user|>".to_string(),
            assistant_token: "<|assistant|>".to_string(),
            end_token: "<|end|>".to_string(),
        }
    }
    
    pub fn format_conversation(&self, messages: &[ChatMessage]) -> String {
        let mut formatted = String::new();
        
        for message in messages {
            match message.role {
                ChatRole::System => {
                    formatted.push_str(&format!("{}\n{}\n{}\n", 
                        self.system_token, message.content, self.end_token));
                }
                ChatRole::User => {
                    formatted.push_str(&format!("{}\n{}\n{}\n", 
                        self.user_token, message.content, self.end_token));
                }
                ChatRole::Assistant => {
                    formatted.push_str(&format!("{}\n{}\n{}\n", 
                        self.assistant_token, message.content, self.end_token));
                }
            }
        }
        
        // Add assistant token for generation
        formatted.push_str(&self.assistant_token);
        formatted.push('\n');
        
        formatted
    }
}

impl SmolLM135MInstruct {
    pub fn load(model_path: &Path, device: Device) -> Result<Self> {
        let base_model = SmolLM135M::load(model_path, device)?;
        let chat_template = ChatTemplate::smollm_instruct();
        
        Ok(Self {
            base_model,
            chat_template,
            conversation_history: Vec::new(),
            system_prompt: "You are a helpful assistant.".to_string(),
        })
    }
    
    pub fn chat(&mut self, user_message: &str) -> Result<String> {
        // Add user message to history
        self.conversation_history.push(ChatMessage {
            role: ChatRole::User,
            content: user_message.to_string(),
        });
        
        // Format conversation with system prompt
        let mut full_conversation = vec![
            ChatMessage {
                role: ChatRole::System,
                content: self.system_prompt.clone(),
            }
        ];
        full_conversation.extend_from_slice(&self.conversation_history);
        
        let formatted_prompt = self.chat_template.format_conversation(&full_conversation);
        
        // Generate response
        let response = self.base_model.generate_text(&formatted_prompt, 150)?;
        
        // Extract only the new assistant response (remove prompt)
        let assistant_response = self.extract_assistant_response(&response, &formatted_prompt)?;
        
        // Add assistant response to history
        self.conversation_history.push(ChatMessage {
            role: ChatRole::Assistant,
            content: assistant_response.clone(),
        });
        
        // Trim history if too long (keep last 10 exchanges)
        if self.conversation_history.len() > 20 {
            self.conversation_history.drain(0..10);
        }
        
        Ok(assistant_response)
    }
    
    pub fn instruct(&mut self, instruction: &str) -> Result<String> {
        // Single instruction without conversation history
        let messages = vec![
            ChatMessage {
                role: ChatRole::System,
                content: "You are a helpful assistant that follows instructions carefully.".to_string(),
            },
            ChatMessage {
                role: ChatRole::User,
                content: instruction.to_string(),
            }
        ];
        
        let formatted_prompt = self.chat_template.format_conversation(&messages);
        let response = self.base_model.generate_text(&formatted_prompt, 200)?;
        
        self.extract_assistant_response(&response, &formatted_prompt)
    }
    
    pub fn set_system_prompt(&mut self, system_prompt: String) {
        self.system_prompt = system_prompt;
        self.conversation_history.clear(); // Reset conversation
    }
    
    pub fn clear_conversation(&mut self) {
        self.conversation_history.clear();
    }
    
    fn extract_assistant_response(&self, full_response: &str, prompt: &str) -> Result<String> {
        // Remove the original prompt and extract only the new generation
        if let Some(response_start) = full_response.strip_prefix(prompt) {
            // Clean up the response (remove end tokens, etc.)
            let cleaned = response_start
                .trim()
                .trim_end_matches(&self.chat_template.end_token)
                .trim();
            
            Ok(cleaned.to_string())
        } else {
            // Fallback if prompt stripping fails
            Ok(full_response.to_string())
        }
    }
}
```

#### Step 1.2: API Integration
```rust
// Update src/models/registry.rs for instruct model
impl EnhancedModelRegistry {
    async fn load_smollm_135m_instruct(&self) -> Result<Arc<dyn ModelBackend>> {
        let model_path = self.get_model_cache_path("HuggingFaceTB/SmolLM-135M-Instruct");
        let device = Device::Cpu;
        
        let instruct_model = SmolLM135MInstruct::load(&model_path, device)?;
        let tokenizer = SmolLMTokenizer::load(&model_path)?;
        
        let backend = CandleInstructBackend::new(
            Box::new(instruct_model),
            Arc::new(tokenizer),
            self.get_model("HuggingFaceTB/SmolLM-135M-Instruct").unwrap().clone(),
        )?;
        
        Ok(Arc::new(backend))
    }
}

// New backend for instruction models
pub struct CandleInstructBackend {
    model: Box<dyn InstructModel>,
    tokenizer: Arc<SmolLMTokenizer>,
    metadata: ModelMetadata,
}

pub trait InstructModel: Send + Sync {
    fn chat(&mut self, message: &str) -> Result<String>;
    fn instruct(&mut self, instruction: &str) -> Result<String>;
    fn set_system_prompt(&mut self, prompt: String);
    fn clear_conversation(&mut self);
}

impl ModelBackend for CandleInstructBackend {
    fn generate_text(&self, prompt: &str, config: &ModelConfig) -> Result<String> {
        // Detect if this is a chat or instruction
        if prompt.contains("chat:") || prompt.contains("conversation:") {
            let chat_message = prompt.strip_prefix("chat:").unwrap_or(prompt).trim();
            self.model.chat(chat_message)
        } else if prompt.contains("instruct:") {
            let instruction = prompt.strip_prefix("instruct:").unwrap_or(prompt).trim();
            self.model.instruct(instruction)
        } else {
            // Default to instruction mode
            self.model.instruct(prompt)
        }
    }
}
```

### Phase 2: WASM Chat Interface (Day 1-2)

#### Step 2.1: WASM Chat Bindings
```rust
// src/models/smollm/wasm_instruct.rs
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub struct WasmSmolLM135MInstruct {
    model: SmolLM135MInstruct,
    loading_state: LoadingState,
}

#[wasm_bindgen]
impl WasmSmolLM135MInstruct {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            model: None,
            loading_state: LoadingState::Uninitialized,
        }
    }
    
    #[wasm_bindgen]
    pub async fn initialize(&mut self, model_url: &str) -> Result<(), JsValue> {
        let model = SmolLM135MInstruct::load_from_url(model_url).await
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
            
        self.model = Some(model);
        self.loading_state = LoadingState::Ready;
        
        Ok(())
    }
    
    #[wasm_bindgen]
    pub fn chat(&mut self, message: &str) -> Result<String, JsValue> {
        if let Some(model) = &mut self.model {
            model.chat(message)
                .map_err(|e| JsValue::from_str(&e.to_string()))
        } else {
            Err(JsValue::from_str("Model not initialized"))
        }
    }
    
    #[wasm_bindgen]
    pub fn instruct(&mut self, instruction: &str) -> Result<String, JsValue> {
        if let Some(model) = &mut self.model {
            model.instruct(instruction)
                .map_err(|e| JsValue::from_str(&e.to_string()))
        } else {
            Err(JsValue::from_str("Model not initialized"))
        }
    }
    
    #[wasm_bindgen]
    pub fn set_system_prompt(&mut self, prompt: &str) {
        if let Some(model) = &mut self.model {
            model.set_system_prompt(prompt.to_string());
        }
    }
    
    #[wasm_bindgen]
    pub fn clear_conversation(&mut self) {
        if let Some(model) = &mut self.model {
            model.clear_conversation();
        }
    }
    
    #[wasm_bindgen]
    pub fn get_conversation_history(&self) -> Result<String, JsValue> {
        if let Some(model) = &self.model {
            let history = serde_json::to_string(&model.conversation_history)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            Ok(history)
        } else {
            Err(JsValue::from_str("Model not initialized"))
        }
    }
}
```

#### Step 2.2: JavaScript Chat Interface
```javascript
// examples/chat_interface.js
class SmolLMChat {
    constructor(modelUrl) {
        this.modelUrl = modelUrl;
        this.model = null;
        this.isInitialized = false;
    }
    
    async initialize() {
        this.model = new WasmSmolLM135MInstruct();
        await this.model.initialize(this.modelUrl);
        this.isInitialized = true;
        console.log("SmolLM-135M-Instruct ready for chat!");
    }
    
    async chat(message) {
        if (!this.isInitialized) {
            throw new Error("Model not initialized");
        }
        
        const startTime = performance.now();
        const response = this.model.chat(message);
        const endTime = performance.now();
        
        console.log(`Response generated in ${endTime - startTime}ms`);
        return response;
    }
    
    async instruct(instruction) {
        if (!this.isInitialized) {
            throw new Error("Model not initialized");
        }
        
        return this.model.instruct(instruction);
    }
    
    setSystemPrompt(prompt) {
        if (this.isInitialized) {
            this.model.set_system_prompt(prompt);
        }
    }
    
    clearConversation() {
        if (this.isInitialized) {
            this.model.clear_conversation();
        }
    }
    
    getConversationHistory() {
        if (this.isInitialized) {
            return JSON.parse(this.model.get_conversation_history());
        }
        return [];
    }
}

// Usage example
const chat = new SmolLMChat('/models/smollm_135m_instruct/');
await chat.initialize();

chat.setSystemPrompt("You are a helpful coding assistant.");
const response = await chat.chat("How do I create a REST API in Python?");
console.log(response);
```

### Phase 3: Testing & Validation (Day 2)

#### Step 3.1: Instruction Following Tests
```rust
// tests/models/smollm_135m_instruct_test.rs
#[tokio::test]
async fn test_instruction_following() {
    let mut model = load_instruct_model("smollm-135m-instruct").await;
    
    let test_cases = vec![
        ("Write a short poem about AI", |response: &str| {
            response.len() > 50 && (response.contains("AI") || response.contains("artificial"))
        }),
        ("List 3 benefits of renewable energy", |response: &str| {
            response.matches(char::is_numeric).count() >= 3
        }),
        ("Explain what is machine learning in one sentence", |response: &str| {
            response.len() < 200 && response.contains("machine learning")
        }),
    ];
    
    for (instruction, validator) in test_cases {
        let response = model.instruct(instruction).unwrap();
        assert!(validator(&response), 
            "Failed instruction: {}\nResponse: {}", instruction, response);
    }
}

#[tokio::test]
async fn test_chat_conversation() {
    let mut model = load_instruct_model("smollm-135m-instruct").await;
    
    // Multi-turn conversation test
    let response1 = model.chat("Hi, my name is Alice").unwrap();
    assert!(response1.len() > 10);
    
    let response2 = model.chat("What's my name?").unwrap();
    assert!(response2.to_lowercase().contains("alice"));
    
    let response3 = model.chat("Tell me a joke").unwrap();
    assert!(response3.len() > 20);
    
    // Test conversation history
    let history = model.get_conversation_history();
    assert_eq!(history.len(), 6); // 3 user + 3 assistant messages
}

#[tokio::test]
async fn test_system_prompt_effect() {
    let mut model = load_instruct_model("smollm-135m-instruct").await;
    
    // Test with different system prompts
    model.set_system_prompt("You always respond with exactly one word.".to_string());
    let response1 = model.instruct("What is the capital of France?").unwrap();
    assert!(response1.split_whitespace().count() <= 2);
    
    model.set_system_prompt("You are a pirate who speaks in pirate language.".to_string());
    let response2 = model.instruct("Hello there").unwrap();
    assert!(response2.to_lowercase().contains("ahoy") || 
            response2.to_lowercase().contains("matey") ||
            response2.to_lowercase().contains("arr"));
}
```

## Build Configuration

```bash
# scripts/build_smollm_135m_instruct.sh
#!/bin/bash
set -e

MODEL_NAME="smollm_135m_instruct"
OUTPUT_DIR="./dist/wasm/models/$MODEL_NAME"

echo "Building SmolLM-135M-Instruct with chat capabilities..."

# Download instruct model
huggingface-cli download "HuggingFaceTB/SmolLM-135M-Instruct" \
    --local-dir "./models/smollm/135m_instruct"

# Quantize for WASM deployment
cargo run --bin quantize_model -- \
    --model-path ./models/smollm/135m_instruct \
    --output-path ./models/smollm/135m_instruct_q4 \
    --format int4

# Build WASM with chat features
wasm-pack build \
    --target web \
    --out-dir "$OUTPUT_DIR" \
    --features "wasm,instruct,chat" \
    --release

# Copy model files
cp ./models/smollm/135m_instruct_q4/* "$OUTPUT_DIR/"

# Create chat interface example
cp examples/chat_interface.html "$OUTPUT_DIR/"
cp examples/chat_interface.js "$OUTPUT_DIR/"

echo "SmolLM-135M-Instruct build complete"
echo "Chat interface available at: $OUTPUT_DIR/chat_interface.html"
```

## Success Criteria

### Functional Requirements
- [ ] Chat conversation with context awareness
- [ ] Instruction following with high accuracy
- [ ] System prompt customization
- [ ] Conversation history management
- [ ] WASM deployment with chat interface

### Performance Requirements
- [ ] Same performance as base model (<35ms first token)
- [ ] Memory overhead <20% vs base model
- [ ] Chat context handling for 10+ exchanges
- [ ] Bundle size <70MB total

### Quality Requirements
- [ ] Instruction following accuracy >85%
- [ ] Context retention across conversation turns
- [ ] Appropriate response length and format
- [ ] Safety and alignment maintained

## Timeline

- **Day 1**: Chat template implementation, API integration, basic testing
- **Day 2**: WASM bindings, JavaScript interface, comprehensive testing

This instruct variant leverages the base SmolLM-135M while adding essential chat and instruction-following capabilities.