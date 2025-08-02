# Micro Task 27: LLM Explanation Generation

**Priority**: CRITICAL  
**Estimated Time**: 50 minutes  
**Dependencies**: 26_reasoning_extraction.md completed  
**Skills Required**: LLM integration, natural language generation, explanation frameworks

## Objective

Implement LLM-powered explanation generation that uses extracted reasoning chains and activation data to produce natural, human-readable explanations of the brain-inspired system's decision-making process.

## Context

While templates provide structured explanations, LLMs can generate more natural, contextual, and nuanced explanations that adapt to different audiences and explanation needs. This task integrates local LLM capabilities to enhance explanation quality and naturalness.

## Specifications

### Core LLM Components

1. **LLMExplainer struct**
   - Local LLM integration
   - Prompt engineering system
   - Context-aware generation
   - Explanation personalization

2. **ExplanationPrompt struct**
   - Structured prompt templates
   - Variable injection system
   - Context building
   - Output formatting control

3. **LLMExplanationEngine struct**
   - Multi-model support
   - Explanation caching
   - Quality assessment
   - Fallback mechanisms

4. **ExplanationPersonalization struct**
   - Audience adaptation
   - Complexity adjustment
   - Domain specialization
   - Style customization

### Performance Requirements

- LLM explanation generation < 500ms
- Context preparation < 50ms
- Support for offline/local LLMs
- Explanation quality consistency
- Graceful degradation without LLM

## Implementation Guide

### Step 1: Core LLM Explanation Types

```rust
// File: src/cognitive/explanation/llm_explanation.rs

use std::collections::HashMap;
use std::time::{Duration, Instant};
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;
use crate::cognitive::explanation::reasoning_extraction::{ReasoningChain, ReasoningAnalysis};
use crate::cognitive::explanation::templates::{ExplanationContext, Evidence};
use crate::core::types::{NodeId, ActivationLevel};
use crate::enhanced_knowledge_storage::model_management::ModelManager;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplanationPrompt {
    pub prompt_id: PromptId,
    pub template: String,
    pub context_fields: Vec<ContextField>,
    pub output_format: OutputFormat,
    pub audience_level: AudienceLevel,
    pub explanation_style: ExplanationStyle,
    pub max_tokens: usize,
    pub temperature: f32,
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub struct PromptId(pub u64);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextField {
    pub field_name: String,
    pub field_type: ContextFieldType,
    pub required: bool,
    pub formatting: Option<FieldFormatting>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ContextFieldType {
    Query,
    ReasoningChain,
    ActivationData,
    Evidence,
    Confidence,
    Metadata,
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldFormatting {
    pub max_length: Option<usize>,
    pub truncation_strategy: TruncationStrategy,
    pub emphasis_level: EmphasisLevel,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TruncationStrategy {
    Simple,
    PreserveMost Important,
    PreserveEnds,
    Summarize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EmphasisLevel {
    Minimal,
    Moderate,
    High,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OutputFormat {
    PlainText,
    Markdown,
    Structured,
    Conversational,
    Technical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AudienceLevel {
    Beginner,
    Intermediate,
    Expert,
    Technical,
    General,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExplanationStyle {
    Concise,
    Detailed,
    StepByStep,
    Narrative,
    Analytical,
    Interactive,
}

#[derive(Debug)]
pub struct LLMExplainer {
    model_manager: ModelManager,
    prompt_registry: PromptRegistry,
    explanation_cache: Mutex<HashMap<String, CachedExplanation>>,
    personalization: ExplanationPersonalization,
    config: LLMExplanationConfig,
}

#[derive(Debug, Clone)]
pub struct LLMExplanationConfig {
    pub default_model: String,
    pub fallback_models: Vec<String>,
    pub max_context_length: usize,
    pub cache_enabled: bool,
    pub cache_ttl: Duration,
    pub quality_threshold: f32,
    pub timeout: Duration,
}

#[derive(Debug, Clone)]
pub struct CachedExplanation {
    pub explanation: String,
    pub quality_score: f32,
    pub timestamp: Instant,
    pub context_hash: u64,
}

#[derive(Debug)]
pub struct PromptRegistry {
    prompts: HashMap<PromptId, ExplanationPrompt>,
    style_prompts: HashMap<(AudienceLevel, ExplanationStyle), PromptId>,
    next_prompt_id: u64,
}

#[derive(Debug)]
pub struct ExplanationPersonalization {
    audience_profiles: HashMap<AudienceLevel, AudienceProfile>,
    domain_specializations: HashMap<String, DomainProfile>,
    style_configurations: HashMap<ExplanationStyle, StyleConfig>,
}

#[derive(Debug, Clone)]
pub struct AudienceProfile {
    pub vocabulary_level: VocabularyLevel,
    pub technical_depth: TechnicalDepth,
    pub preferred_examples: Vec<String>,
    pub avoid_concepts: Vec<String>,
    pub explanation_preferences: ExplanationPreferences,
}

#[derive(Debug, Clone)]
pub enum VocabularyLevel {
    Simple,
    Intermediate,
    Advanced,
    Professional,
}

#[derive(Debug, Clone)]
pub enum TechnicalDepth {
    Minimal,
    Conceptual,
    Implementation,
    Mathematical,
}

#[derive(Debug, Clone)]
pub struct ExplanationPreferences {
    pub include_uncertainty: bool,
    pub show_reasoning_steps: bool,
    pub provide_alternatives: bool,
    pub include_confidence: bool,
    pub use_analogies: bool,
}

#[derive(Debug, Clone)]
pub struct DomainProfile {
    pub domain_terminology: HashMap<String, String>,
    pub common_concepts: Vec<String>,
    pub specialized_knowledge: Vec<String>,
    pub context_requirements: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct StyleConfig {
    pub sentence_structure: SentenceStructure,
    pub paragraph_organization: ParagraphOrganization,
    pub transition_style: TransitionStyle,
    pub emphasis_markers: Vec<String>,
}

#[derive(Debug, Clone)]
pub enum SentenceStructure {
    Simple,
    Compound,
    Complex,
    Mixed,
}

#[derive(Debug, Clone)]
pub enum ParagraphOrganization {
    Sequential,
    Hierarchical,
    Thematic,
    Comparative,
}

#[derive(Debug, Clone)]
pub enum TransitionStyle {
    Formal,
    Conversational,
    Logical,
    Narrative,
}
```

### Step 2: LLM Explainer Implementation

```rust
impl LLMExplainer {
    pub async fn new(model_manager: ModelManager) -> Result<Self, LLMExplanationError> {
        let config = LLMExplanationConfig {
            default_model: "local_explanation_model".to_string(),
            fallback_models: vec!["gpt2_local".to_string()],
            max_context_length: 4096,
            cache_enabled: true,
            cache_ttl: Duration::from_hours(1),
            quality_threshold: 0.7,
            timeout: Duration::from_secs(30),
        };
        
        let mut prompt_registry = PromptRegistry::new();
        prompt_registry.load_default_prompts();
        
        let personalization = ExplanationPersonalization::new();
        
        Ok(Self {
            model_manager,
            prompt_registry,
            explanation_cache: Mutex::new(HashMap::new()),
            personalization,
            config,
        })
    }
    
    pub async fn generate_explanation(
        &self,
        reasoning_chain: &ReasoningChain,
        query: &str,
        audience: AudienceLevel,
        style: ExplanationStyle,
    ) -> Result<String, LLMExplanationError> {
        // Check cache first
        let cache_key = self.create_cache_key(reasoning_chain, query, &audience, &style);
        
        if let Some(cached) = self.check_cache(&cache_key).await {
            if cached.quality_score >= self.config.quality_threshold {
                return Ok(cached.explanation);
            }
        }
        
        // Build context for LLM
        let context = self.build_explanation_context(reasoning_chain, query, &audience, &style)?;
        
        // Get appropriate prompt
        let prompt = self.prompt_registry.get_prompt_for_style(&audience, &style)
            .ok_or(LLMExplanationError::NoSuitablePrompt)?;
        
        // Generate explanation with LLM
        let explanation = self.call_llm_with_context(&prompt, &context).await?;
        
        // Assess explanation quality
        let quality_score = self.assess_explanation_quality(&explanation, reasoning_chain).await?;
        
        // Cache result if quality is sufficient
        if quality_score >= self.config.quality_threshold {
            self.cache_explanation(cache_key, explanation.clone(), quality_score).await;
        }
        
        Ok(explanation)
    }
    
    async fn call_llm_with_context(
        &self,
        prompt: &ExplanationPrompt,
        context: &HashMap<String, String>,
    ) -> Result<String, LLMExplanationError> {
        // Prepare full prompt with context substitution
        let full_prompt = self.substitute_context_in_prompt(&prompt.template, context)?;
        
        // Try primary model first
        match self.try_model(&self.config.default_model, &full_prompt, prompt).await {
            Ok(response) => return Ok(response),
            Err(e) => {
                log::warn!("Primary model failed: {:?}, trying fallback", e);
            }
        }
        
        // Try fallback models
        for fallback_model in &self.config.fallback_models {
            match self.try_model(fallback_model, &full_prompt, prompt).await {
                Ok(response) => return Ok(response),
                Err(e) => {
                    log::warn!("Fallback model {} failed: {:?}", fallback_model, e);
                }
            }
        }
        
        Err(LLMExplanationError::AllModelsFailed)
    }
    
    async fn try_model(
        &self,
        model_name: &str,
        prompt: &str,
        prompt_config: &ExplanationPrompt,
    ) -> Result<String, LLMExplanationError> {
        // Load model through model manager
        let model = self.model_manager.get_model(model_name).await
            .map_err(|e| LLMExplanationError::ModelLoadError(e.to_string()))?;
        
        // Prepare generation parameters
        let generation_params = GenerationParameters {
            max_tokens: prompt_config.max_tokens,
            temperature: prompt_config.temperature,
            top_p: 0.9,
            stop_tokens: vec!["</explanation>".to_string()],
            timeout: self.config.timeout,
        };
        
        // Generate text
        let response = tokio::time::timeout(
            self.config.timeout,
            model.generate_text(prompt, generation_params)
        ).await
        .map_err(|_| LLMExplanationError::Timeout)?
        .map_err(|e| LLMExplanationError::GenerationError(e.to_string()))?;
        
        // Post-process response
        self.post_process_response(&response, &prompt_config.output_format)
    }
    
    fn build_explanation_context(
        &self,
        reasoning_chain: &ReasoningChain,
        query: &str,
        audience: &AudienceLevel,
        style: &ExplanationStyle,
    ) -> Result<HashMap<String, String>, LLMExplanationError> {
        let mut context = HashMap::new();
        
        // Basic context
        context.insert("query".to_string(), query.to_string());
        context.insert("audience_level".to_string(), format!("{:?}", audience));
        context.insert("explanation_style".to_string(), format!("{:?}", style));
        
        // Reasoning chain context
        context.insert("step_count".to_string(), reasoning_chain.steps.len().to_string());
        context.insert("confidence_score".to_string(), format!("{:.2}", reasoning_chain.confidence_score));
        context.insert("coherence_score".to_string(), format!("{:.2}", reasoning_chain.coherence_score));
        
        // Format reasoning steps for context
        let steps_text = self.format_reasoning_steps_for_llm(reasoning_chain, audience)?;
        context.insert("reasoning_steps".to_string(), steps_text);
        
        // Format connections
        let connections_text = self.format_connections_for_llm(reasoning_chain, audience)?;
        context.insert("logical_connections".to_string(), connections_text);
        
        // Add audience-specific context
        if let Some(profile) = self.personalization.audience_profiles.get(audience) {
            context.insert("vocabulary_level".to_string(), format!("{:?}", profile.vocabulary_level));
            context.insert("technical_depth".to_string(), format!("{:?}", profile.technical_depth));
            
            if profile.explanation_preferences.include_uncertainty {
                context.insert("include_uncertainty".to_string(), "true".to_string());
            }
            
            if profile.explanation_preferences.show_reasoning_steps {
                context.insert("show_detailed_steps".to_string(), "true".to_string());
            }
        }
        
        // Add style-specific context
        if let Some(style_config) = self.personalization.style_configurations.get(style) {
            context.insert("sentence_structure".to_string(), format!("{:?}", style_config.sentence_structure));
            context.insert("organization".to_string(), format!("{:?}", style_config.paragraph_organization));
        }
        
        Ok(context)
    }
    
    fn format_reasoning_steps_for_llm(
        &self,
        reasoning_chain: &ReasoningChain,
        audience: &AudienceLevel,
    ) -> Result<String, LLMExplanationError> {
        let mut formatted_steps = Vec::new();
        
        for (i, step) in reasoning_chain.steps.iter().enumerate() {
            let step_text = match audience {
                AudienceLevel::Beginner | AudienceLevel::General => {
                    format!("{}. {} → {}", 
                           i + 1, 
                           self.simplify_text(&step.premise), 
                           self.simplify_text(&step.conclusion))
                },
                AudienceLevel::Technical | AudienceLevel::Expert => {
                    format!("{}. [{}] {} → {} (confidence: {:.2}, operation: {:?})", 
                           i + 1,
                           format!("{:?}", step.step_type),
                           step.premise, 
                           step.conclusion,
                           step.confidence,
                           step.logical_operation)
                },
                _ => {
                    format!("{}. {} → {} (confidence: {:.2})", 
                           i + 1, 
                           step.premise, 
                           step.conclusion,
                           step.confidence)
                }
            };
            
            formatted_steps.push(step_text);
        }
        
        Ok(formatted_steps.join("\n"))
    }
    
    fn format_connections_for_llm(
        &self,
        reasoning_chain: &ReasoningChain,
        audience: &AudienceLevel,
    ) -> Result<String, LLMExplanationError> {
        if reasoning_chain.connections.is_empty() {
            return Ok("No explicit logical connections identified.".to_string());
        }
        
        let mut formatted_connections = Vec::new();
        
        for connection in &reasoning_chain.connections {
            let connection_text = match audience {
                AudienceLevel::Beginner | AudienceLevel::General => {
                    format!("Step {} leads to Step {} (strength: {:.1}/1.0)",
                           connection.source_step.0,
                           connection.target_step.0,
                           connection.strength)
                },
                AudienceLevel::Technical | AudienceLevel::Expert => {
                    format!("Step {} --[{:?}]--> Step {} (strength: {:.2}, validity: {:.2})",
                           connection.source_step.0,
                           connection.connection_type,
                           connection.target_step.0,
                           connection.strength,
                           connection.logical_validity)
                },
                _ => {
                    format!("Step {} connects to Step {} via {} (strength: {:.2})",
                           connection.source_step.0,
                           connection.target_step.0,
                           format!("{:?}", connection.connection_type).to_lowercase(),
                           connection.strength)
                }
            };
            
            formatted_connections.push(connection_text);
        }
        
        Ok(formatted_connections.join("\n"))
    }
    
    fn simplify_text(&self, text: &str) -> String {
        // Simple text simplification for general audiences
        text.replace("activation", "signal")
            .replace("node", "concept")
            .replace("inference", "reasoning")
            .replace("propagation", "spreading")
            .replace("convergence", "coming together")
    }
    
    fn substitute_context_in_prompt(
        &self,
        template: &str,
        context: &HashMap<String, String>,
    ) -> Result<String, LLMExplanationError> {
        let mut result = template.to_string();
        
        // Replace {{variable}} patterns
        for (key, value) in context {
            let placeholder = format!("{{{{{}}}}}", key);
            result = result.replace(&placeholder, value);
        }
        
        // Check for unresolved placeholders
        if result.contains("{{") && result.contains("}}") {
            return Err(LLMExplanationError::UnresolvedPlaceholders);
        }
        
        Ok(result)
    }
    
    async fn assess_explanation_quality(
        &self,
        explanation: &str,
        reasoning_chain: &ReasoningChain,
    ) -> Result<f32, LLMExplanationError> {
        let mut quality_score = 0.0;
        
        // Length appropriateness (not too short, not too long)
        let length_score = if explanation.len() > 50 && explanation.len() < 2000 {
            1.0
        } else if explanation.len() < 20 {
            0.2
        } else {
            0.7
        };
        quality_score += length_score * 0.2;
        
        // Mention of key reasoning elements
        let step_mention_score = if explanation.contains("step") || explanation.contains("reason") {
            1.0
        } else {
            0.5
        };
        quality_score += step_mention_score * 0.3;
        
        // Confidence mention if chain has low confidence
        let confidence_score = if reasoning_chain.confidence_score < 0.6 {
            if explanation.contains("uncertain") || explanation.contains("confidence") || explanation.contains("might") {
                1.0
            } else {
                0.3
            }
        } else {
            0.8 // Default good score for high confidence chains
        };
        quality_score += confidence_score * 0.2;
        
        // Coherence (simple heuristic: complete sentences)
        let sentence_count = explanation.matches('.').count() + explanation.matches('!').count() + explanation.matches('?').count();
        let coherence_score = if sentence_count >= 2 && sentence_count <= 10 {
            1.0
        } else if sentence_count == 1 {
            0.7
        } else {
            0.4
        };
        quality_score += coherence_score * 0.3;
        
        Ok(quality_score.min(1.0))
    }
    
    fn post_process_response(&self, response: &str, format: &OutputFormat) -> Result<String, LLMExplanationError> {
        let mut processed = response.trim().to_string();
        
        // Remove any system prompts or unwanted prefixes
        if processed.starts_with("Assistant:") {
            processed = processed.strip_prefix("Assistant:").unwrap_or(&processed).trim().to_string();
        }
        
        // Ensure proper formatting based on output format
        match format {
            OutputFormat::Markdown => {
                if !processed.starts_with('#') && !processed.contains("**") {
                    // Add some basic markdown formatting
                    processed = processed.replace('\n', "\n\n");
                }
            },
            OutputFormat::Structured => {
                // Ensure structured format with clear sections
                if !processed.contains('\n') {
                    processed = format!("## Explanation\n\n{}", processed);
                }
            },
            _ => {} // No special processing for other formats
        }
        
        Ok(processed)
    }
    
    fn create_cache_key(
        &self,
        reasoning_chain: &ReasoningChain,
        query: &str,
        audience: &AudienceLevel,
        style: &ExplanationStyle,
    ) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        query.hash(&mut hasher);
        reasoning_chain.chain_id.hash(&mut hasher);
        format!("{:?}", audience).hash(&mut hasher);
        format!("{:?}", style).hash(&mut hasher);
        reasoning_chain.steps.len().hash(&mut hasher);
        (reasoning_chain.confidence_score * 1000.0) as u64.hash(&mut hasher);
        
        format!("llm_explanation_{}", hasher.finish())
    }
    
    async fn check_cache(&self, cache_key: &str) -> Option<CachedExplanation> {
        let cache = self.explanation_cache.lock().await;
        
        if let Some(cached) = cache.get(cache_key) {
            // Check if cache entry is still valid
            if cached.timestamp.elapsed() < self.config.cache_ttl {
                return Some(cached.clone());
            }
        }
        
        None
    }
    
    async fn cache_explanation(&self, cache_key: String, explanation: String, quality_score: f32) {
        if !self.config.cache_enabled {
            return;
        }
        
        let mut cache = self.explanation_cache.lock().await;
        
        let cached_entry = CachedExplanation {
            explanation,
            quality_score,
            timestamp: Instant::now(),
            context_hash: 0, // Would implement proper hashing
        };
        
        cache.insert(cache_key, cached_entry);
        
        // Simple cache size management
        if cache.len() > 1000 {
            // Remove oldest entries
            let old_keys: Vec<String> = cache.iter()
                .filter(|(_, entry)| entry.timestamp.elapsed() > Duration::from_hours(2))
                .map(|(key, _)| key.clone())
                .collect();
            
            for key in old_keys {
                cache.remove(&key);
            }
        }
    }
    
    pub async fn explain_with_alternatives(
        &self,
        reasoning_analysis: &ReasoningAnalysis,
        query: &str,
        audience: AudienceLevel,
    ) -> Result<MultiExplanation, LLMExplanationError> {
        let mut explanations = Vec::new();
        
        // Primary explanation
        if let Some(primary_chain) = &reasoning_analysis.primary_chain {
            let primary_explanation = self.generate_explanation(
                primary_chain,
                query,
                audience.clone(),
                ExplanationStyle::Detailed,
            ).await?;
            
            explanations.push(ExplanationVariant {
                explanation_type: ExplanationType::Primary,
                content: primary_explanation,
                confidence: primary_chain.confidence_score,
                reasoning_quality: reasoning_analysis.reasoning_quality.overall_quality,
            });
        }
        
        // Alternative explanations from alternative chains
        for alt_chain in &reasoning_analysis.alternative_chains {
            if alt_chain.confidence_score > 0.3 {
                let alt_explanation = self.generate_explanation(
                    alt_chain,
                    query,
                    audience.clone(),
                    ExplanationStyle::Concise,
                ).await?;
                
                explanations.push(ExplanationVariant {
                    explanation_type: ExplanationType::Alternative,
                    content: alt_explanation,
                    confidence: alt_chain.confidence_score,
                    reasoning_quality: alt_chain.coherence_score,
                });
            }
        }
        
        // Uncertainty explanation if needed
        if reasoning_analysis.reasoning_quality.overall_quality < 0.6 {
            let uncertainty_explanation = self.generate_uncertainty_explanation(
                reasoning_analysis,
                query,
                audience,
            ).await?;
            
            explanations.push(ExplanationVariant {
                explanation_type: ExplanationType::Uncertainty,
                content: uncertainty_explanation,
                confidence: reasoning_analysis.reasoning_quality.overall_quality,
                reasoning_quality: 0.8, // Uncertainty explanations are usually clear
            });
        }
        
        Ok(MultiExplanation {
            explanations,
            query: query.to_string(),
            audience,
            generation_time: Instant::now(),
        })
    }
    
    async fn generate_uncertainty_explanation(
        &self,
        reasoning_analysis: &ReasoningAnalysis,
        query: &str,
        audience: AudienceLevel,
    ) -> Result<String, LLMExplanationError> {
        let mut context = HashMap::new();
        context.insert("query".to_string(), query.to_string());
        context.insert("quality_score".to_string(), format!("{:.2}", reasoning_analysis.reasoning_quality.overall_quality));
        context.insert("gap_count".to_string(), reasoning_analysis.logical_gaps.len().to_string());
        context.insert("contradiction_count".to_string(), reasoning_analysis.contradiction_points.len().to_string());
        
        let uncertainty_prompt = ExplanationPrompt {
            prompt_id: PromptId(999),
            template: "For the query '{{query}}', I found some uncertainty in my reasoning process. The overall reasoning quality is {{quality_score}}, with {{gap_count}} logical gaps and {{contradiction_count}} contradictions identified. This suggests that while I can provide an answer, there may be alternative interpretations or missing information that could lead to different conclusions.".to_string(),
            context_fields: vec![],
            output_format: OutputFormat::Conversational,
            audience_level: audience,
            explanation_style: ExplanationStyle::Detailed,
            max_tokens: 200,
            temperature: 0.3,
        };
        
        self.call_llm_with_context(&uncertainty_prompt, &context).await
    }
    
    pub fn clear_cache(&self) {
        tokio::spawn(async move {
            // Would clear cache in actual implementation
        });
    }
}

#[derive(Debug, Clone)]
pub struct MultiExplanation {
    pub explanations: Vec<ExplanationVariant>,
    pub query: String,
    pub audience: AudienceLevel,
    pub generation_time: Instant,
}

#[derive(Debug, Clone)]
pub struct ExplanationVariant {
    pub explanation_type: ExplanationType,
    pub content: String,
    pub confidence: f32,
    pub reasoning_quality: f32,
}

#[derive(Debug, Clone)]
pub enum ExplanationType {
    Primary,
    Alternative,
    Uncertainty,
    Summary,
}

#[derive(Debug, Clone)]
pub struct GenerationParameters {
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub stop_tokens: Vec<String>,
    pub timeout: Duration,
}
```

### Step 3: Prompt Registry Implementation

```rust
impl PromptRegistry {
    pub fn new() -> Self {
        Self {
            prompts: HashMap::new(),
            style_prompts: HashMap::new(),
            next_prompt_id: 1,
        }
    }
    
    pub fn load_default_prompts(&mut self) {
        // Detailed explanation prompt
        let detailed_prompt = ExplanationPrompt {
            prompt_id: PromptId(self.next_prompt_id),
            template: r#"
You are explaining how an AI reasoning system arrived at an answer. Please provide a clear, detailed explanation.

Query: {{query}}
Audience Level: {{audience_level}}

Reasoning Steps:
{{reasoning_steps}}

Logical Connections:
{{logical_connections}}

System Confidence: {{confidence_score}}

Please explain in {{explanation_style}} style for a {{audience_level}} audience:
1. How the system processed this query
2. The key reasoning steps it followed
3. Why it arrived at this conclusion
{{#if include_uncertainty}}4. Any uncertainties or limitations in the reasoning{{/if}}

Make your explanation natural and easy to understand while being accurate about the AI's reasoning process.
"#.to_string(),
            context_fields: vec![
                ContextField {
                    field_name: "query".to_string(),
                    field_type: ContextFieldType::Query,
                    required: true,
                    formatting: None,
                },
                ContextField {
                    field_name: "reasoning_steps".to_string(),
                    field_type: ContextFieldType::ReasoningChain,
                    required: true,
                    formatting: Some(FieldFormatting {
                        max_length: Some(1000),
                        truncation_strategy: TruncationStrategy::PreserveMostImportant,
                        emphasis_level: EmphasisLevel::Moderate,
                    }),
                },
            ],
            output_format: OutputFormat::Conversational,
            audience_level: AudienceLevel::General,
            explanation_style: ExplanationStyle::Detailed,
            max_tokens: 500,
            temperature: 0.7,
        };
        
        self.register_prompt(detailed_prompt, AudienceLevel::General, ExplanationStyle::Detailed);
        
        // Concise explanation prompt
        let concise_prompt = ExplanationPrompt {
            prompt_id: PromptId(self.next_prompt_id),
            template: r#"
Briefly explain how the AI reasoned through this query:

Query: {{query}}
Key Steps: {{reasoning_steps}}
Confidence: {{confidence_score}}

Provide a concise, clear explanation in 2-3 sentences for a {{audience_level}} audience.
"#.to_string(),
            context_fields: vec![],
            output_format: OutputFormat::PlainText,
            audience_level: AudienceLevel::General,
            explanation_style: ExplanationStyle::Concise,
            max_tokens: 150,
            temperature: 0.5,
        };
        
        self.register_prompt(concise_prompt, AudienceLevel::General, ExplanationStyle::Concise);
        
        // Technical explanation prompt
        let technical_prompt = ExplanationPrompt {
            prompt_id: PromptId(self.next_prompt_id),
            template: r#"
Technical Analysis of AI Reasoning Process:

Query: {{query}}
System Configuration: Neural-inspired spreading activation with pathway tracing

Detailed Reasoning Trace:
{{reasoning_steps}}

Logical Connection Analysis:
{{logical_connections}}

Quality Metrics:
- Confidence Score: {{confidence_score}}
- Coherence Score: {{coherence_score}}
{{#if show_detailed_steps}}
- Reasoning Quality: {{reasoning_quality}}
{{/if}}

Provide a technical explanation suitable for AI researchers and developers, including:
1. Activation propagation analysis
2. Pathway significance assessment
3. Logical validity evaluation
4. Potential limitations and edge cases
"#.to_string(),
            context_fields: vec![],
            output_format: OutputFormat::Technical,
            audience_level: AudienceLevel::Technical,
            explanation_style: ExplanationStyle::Analytical,
            max_tokens: 800,
            temperature: 0.3,
        };
        
        self.register_prompt(technical_prompt, AudienceLevel::Technical, ExplanationStyle::Analytical);
    }
    
    fn register_prompt(
        &mut self,
        mut prompt: ExplanationPrompt,
        audience: AudienceLevel,
        style: ExplanationStyle,
    ) -> PromptId {
        let prompt_id = PromptId(self.next_prompt_id);
        self.next_prompt_id += 1;
        
        prompt.prompt_id = prompt_id;
        prompt.audience_level = audience.clone();
        prompt.explanation_style = style.clone();
        
        self.prompts.insert(prompt_id, prompt);
        self.style_prompts.insert((audience, style), prompt_id);
        
        prompt_id
    }
    
    pub fn get_prompt_for_style(
        &self,
        audience: &AudienceLevel,
        style: &ExplanationStyle,
    ) -> Option<&ExplanationPrompt> {
        // Try exact match first
        if let Some(prompt_id) = self.style_prompts.get(&(audience.clone(), style.clone())) {
            return self.prompts.get(prompt_id);
        }
        
        // Fallback to general audience with same style
        if let Some(prompt_id) = self.style_prompts.get(&(AudienceLevel::General, style.clone())) {
            return self.prompts.get(prompt_id);
        }
        
        // Fallback to detailed explanation for any audience
        if let Some(prompt_id) = self.style_prompts.get(&(audience.clone(), ExplanationStyle::Detailed)) {
            return self.prompts.get(prompt_id);
        }
        
        // Final fallback to any available prompt
        self.prompts.values().next()
    }
    
    pub fn add_custom_prompt(
        &mut self,
        prompt: ExplanationPrompt,
        audience: AudienceLevel,
        style: ExplanationStyle,
    ) -> PromptId {
        self.register_prompt(prompt, audience, style)
    }
}
```

### Step 4: Integration and Error Handling

```rust
#[derive(Debug, Clone)]
pub enum LLMExplanationError {
    ModelLoadError(String),
    GenerationError(String),
    NoSuitablePrompt,
    UnresolvedPlaceholders,
    ContextBuildError(String),
    QualityAssessmentError,
    AllModelsFailed,
    Timeout,
    CacheError(String),
}

impl std::fmt::Display for LLMExplanationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LLMExplanationError::ModelLoadError(msg) => write!(f, "Model load error: {}", msg),
            LLMExplanationError::GenerationError(msg) => write!(f, "Generation error: {}", msg),
            LLMExplanationError::NoSuitablePrompt => write!(f, "No suitable prompt found"),
            LLMExplanationError::UnresolvedPlaceholders => write!(f, "Unresolved placeholders in prompt"),
            LLMExplanationError::ContextBuildError(msg) => write!(f, "Context build error: {}", msg),
            LLMExplanationError::QualityAssessmentError => write!(f, "Quality assessment error"),
            LLMExplanationError::AllModelsFailed => write!(f, "All models failed to generate explanation"),
            LLMExplanationError::Timeout => write!(f, "Generation timeout"),
            LLMExplanationError::CacheError(msg) => write!(f, "Cache error: {}", msg),
        }
    }
}

impl std::error::Error for LLMExplanationError {}

// Personalization implementations
impl ExplanationPersonalization {
    pub fn new() -> Self {
        let mut personalization = Self {
            audience_profiles: HashMap::new(),
            domain_specializations: HashMap::new(),
            style_configurations: HashMap::new(),
        };
        
        personalization.load_default_profiles();
        personalization
    }
    
    fn load_default_profiles(&mut self) {
        // Beginner profile
        self.audience_profiles.insert(
            AudienceLevel::Beginner,
            AudienceProfile {
                vocabulary_level: VocabularyLevel::Simple,
                technical_depth: TechnicalDepth::Minimal,
                preferred_examples: vec!["everyday analogies".to_string()],
                avoid_concepts: vec!["technical jargon".to_string(), "mathematical notation".to_string()],
                explanation_preferences: ExplanationPreferences {
                    include_uncertainty: false,
                    show_reasoning_steps: true,
                    provide_alternatives: false,
                    include_confidence: false,
                    use_analogies: true,
                },
            },
        );
        
        // Technical profile
        self.audience_profiles.insert(
            AudienceLevel::Technical,
            AudienceProfile {
                vocabulary_level: VocabularyLevel::Professional,
                technical_depth: TechnicalDepth::Implementation,
                preferred_examples: vec!["code examples".to_string(), "algorithmic details".to_string()],
                avoid_concepts: vec!["oversimplification".to_string()],
                explanation_preferences: ExplanationPreferences {
                    include_uncertainty: true,
                    show_reasoning_steps: true,
                    provide_alternatives: true,
                    include_confidence: true,
                    use_analogies: false,
                },
            },
        );
        
        // Style configurations
        self.style_configurations.insert(
            ExplanationStyle::Concise,
            StyleConfig {
                sentence_structure: SentenceStructure::Simple,
                paragraph_organization: ParagraphOrganization::Sequential,
                transition_style: TransitionStyle::Logical,
                emphasis_markers: vec!["importantly".to_string(), "therefore".to_string()],
            },
        );
        
        self.style_configurations.insert(
            ExplanationStyle::Detailed,
            StyleConfig {
                sentence_structure: SentenceStructure::Complex,
                paragraph_organization: ParagraphOrganization::Hierarchical,
                transition_style: TransitionStyle::Narrative,
                emphasis_markers: vec!["furthermore".to_string(), "specifically".to_string(), "in particular".to_string()],
            },
        );
    }
}
```

## File Locations

- `src/cognitive/explanation/llm_explanation.rs` - Main implementation
- `src/cognitive/explanation/mod.rs` - Module exports and integration
- `tests/cognitive/explanation/llm_explanation_tests.rs` - Test implementation

## Success Criteria

- [ ] LLM explanation generation functional
- [ ] Context building and prompt substitution accurate
- [ ] Multiple audience levels supported
- [ ] Explanation quality assessment working
- [ ] Caching system improves performance
- [ ] Graceful fallback when LLM unavailable
- [ ] All tests pass:
  - Basic explanation generation
  - Prompt template system
  - Audience adaptation
  - Quality assessment
  - Error handling and fallbacks

## Test Requirements

```rust
#[test]
async fn test_basic_llm_explanation() {
    let model_manager = MockModelManager::new();
    let explainer = LLMExplainer::new(model_manager).await.unwrap();
    
    let reasoning_chain = create_test_reasoning_chain();
    
    let explanation = explainer.generate_explanation(
        &reasoning_chain,
        "What is artificial intelligence?",
        AudienceLevel::General,
        ExplanationStyle::Detailed,
    ).await.unwrap();
    
    assert!(!explanation.is_empty());
    assert!(explanation.len() > 50);
    assert!(explanation.contains("reasoning") || explanation.contains("AI"));
}

#[test]
fn test_prompt_context_substitution() {
    let explainer = LLMExplainer::new(MockModelManager::new()).await.unwrap();
    
    let template = "Query: {{query}}, Steps: {{step_count}}";
    let mut context = HashMap::new();
    context.insert("query".to_string(), "test query".to_string());
    context.insert("step_count".to_string(), "3".to_string());
    
    let result = explainer.substitute_context_in_prompt(template, &context).unwrap();
    
    assert_eq!(result, "Query: test query, Steps: 3");
    assert!(!result.contains("{{"));
}

#[test]
async fn test_audience_adaptation() {
    let explainer = LLMExplainer::new(MockModelManager::new()).await.unwrap();
    let reasoning_chain = create_test_reasoning_chain();
    
    // Test beginner explanation
    let beginner_explanation = explainer.generate_explanation(
        &reasoning_chain,
        "How does this work?",
        AudienceLevel::Beginner,
        ExplanationStyle::Concise,
    ).await.unwrap();
    
    // Test technical explanation
    let technical_explanation = explainer.generate_explanation(
        &reasoning_chain,
        "How does this work?",
        AudienceLevel::Technical,
        ExplanationStyle::Detailed,
    ).await.unwrap();
    
    // Technical explanation should be longer and more detailed
    assert!(technical_explanation.len() > beginner_explanation.len());
}

#[test]
async fn test_explanation_quality_assessment() {
    let explainer = LLMExplainer::new(MockModelManager::new()).await.unwrap();
    let reasoning_chain = create_test_reasoning_chain();
    
    // Good explanation
    let good_explanation = "This AI system processed your query through multiple reasoning steps. First, it identified the key concepts. Then, it found relevant connections in its knowledge base. Finally, it synthesized the information to provide an answer with 85% confidence.";
    
    let quality = explainer.assess_explanation_quality(good_explanation, &reasoning_chain).await.unwrap();
    assert!(quality > 0.7);
    
    // Poor explanation
    let poor_explanation = "Yes.";
    let quality = explainer.assess_explanation_quality(poor_explanation, &reasoning_chain).await.unwrap();
    assert!(quality < 0.5);
}

#[test]
async fn test_explanation_caching() {
    let explainer = LLMExplainer::new(MockModelManager::new()).await.unwrap();
    let reasoning_chain = create_test_reasoning_chain();
    
    // First generation
    let start_time = Instant::now();
    let explanation1 = explainer.generate_explanation(
        &reasoning_chain,
        "test query",
        AudienceLevel::General,
        ExplanationStyle::Detailed,
    ).await.unwrap();
    let first_duration = start_time.elapsed();
    
    // Second generation (should use cache)
    let start_time = Instant::now();
    let explanation2 = explainer.generate_explanation(
        &reasoning_chain,
        "test query",
        AudienceLevel::General,
        ExplanationStyle::Detailed,
    ).await.unwrap();
    let second_duration = start_time.elapsed();
    
    assert_eq!(explanation1, explanation2);
    // Second call should be faster (cached)
    // Note: This might not always be true in tests, but shows the intent
    assert!(second_duration <= first_duration);
}

#[test]
async fn test_alternative_explanations() {
    let explainer = LLMExplainer::new(MockModelManager::new()).await.unwrap();
    
    let reasoning_analysis = create_test_reasoning_analysis_with_alternatives();
    
    let multi_explanation = explainer.explain_with_alternatives(
        &reasoning_analysis,
        "complex query",
        AudienceLevel::Intermediate,
    ).await.unwrap();
    
    assert!(!multi_explanation.explanations.is_empty());
    
    // Should have primary explanation
    let has_primary = multi_explanation.explanations.iter()
        .any(|e| matches!(e.explanation_type, ExplanationType::Primary));
    assert!(has_primary);
}

fn create_test_reasoning_chain() -> ReasoningChain {
    ReasoningChain {
        chain_id: ChainId(1),
        steps: vec![
            ReasoningStep {
                step_id: StepId(1),
                step_type: StepType::EntityRecognition,
                premise: "Identify query entities".to_string(),
                conclusion: "Found AI concept".to_string(),
                evidence: vec![],
                confidence: 0.9,
                activation_nodes: vec![NodeId(1)],
                logical_operation: LogicalOperation::DirectReference,
                timestamp: Instant::now(),
            },
            ReasoningStep {
                step_id: StepId(2),
                step_type: StepType::LogicalDeduction,
                premise: "AI definition lookup".to_string(),
                conclusion: "AI is computational intelligence".to_string(),
                evidence: vec![],
                confidence: 0.8,
                activation_nodes: vec![NodeId(2)],
                logical_operation: LogicalOperation::Implication,
                timestamp: Instant::now(),
            },
        ],
        connections: vec![],
        source_pathways: vec![],
        confidence_score: 0.85,
        completeness_score: 0.8,
        coherence_score: 0.9,
        start_time: Instant::now(),
        end_time: Some(Instant::now()),
    }
}
```

## Quality Gates

- [ ] LLM explanation generation < 500ms
- [ ] Context preparation < 50ms
- [ ] Explanation quality score correlation with human ratings
- [ ] Memory usage < 50MB for explanation cache
- [ ] Successful fallback without LLM available
- [ ] Thread-safe concurrent explanation generation

## Next Task

Upon completion, proceed to **28_evidence_collection.md**