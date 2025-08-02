# Micro Task 25: Explanation Templates

**Priority**: CRITICAL  
**Estimated Time**: 40 minutes  
**Dependencies**: 24_pathway_tests.md completed  
**Skills Required**: Template systems, natural language generation, explanation frameworks

## Objective

Implement a template-based explanation system that generates human-interpretable explanations for query results using activation traces and pathway information from the brain-inspired cognitive system.

## Context

Effective explanations are crucial for user trust and system transparency. This task creates a flexible template system that can generate various types of explanations based on activation patterns, reasoning pathways, and evidence gathered during query processing.

## Specifications

### Core Template Components

1. **ExplanationTemplate struct**
   - Template pattern definitions
   - Variable substitution system
   - Conditional logic support
   - Multi-modal output formats

2. **TemplateRegistry struct**
   - Template management and lookup
   - Category-based organization
   - Dynamic template loading
   - Performance optimization

3. **TemplateRenderer struct**
   - Variable substitution engine
   - Conditional rendering logic
   - Output formatting
   - Multi-language support

4. **ExplanationContext struct**
   - Query information
   - Activation data
   - Pathway traces
   - Evidence collections

### Performance Requirements

- Template rendering < 5ms per explanation
- Support for 100+ concurrent explanations
- Memory efficient template storage
- Hot-reload capability for template updates
- Caching for frequently used templates

## Implementation Guide

### Step 1: Core Template Types

```rust
// File: src/cognitive/explanation/templates.rs

use std::collections::HashMap;
use std::fmt;
use serde::{Deserialize, Serialize};
use crate::core::types::{NodeId, EntityId, ActivationLevel};
use crate::cognitive::learning::pathway_tracing::ActivationPathway;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplanationTemplate {
    pub template_id: TemplateId,
    pub name: String,
    pub category: TemplateCategory,
    pub pattern: String,
    pub variables: Vec<TemplateVariable>,
    pub conditions: Vec<TemplateCondition>,
    pub output_format: OutputFormat,
    pub priority: i32,
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub struct TemplateId(pub u64);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TemplateCategory {
    FactualAnswer,
    ReasoningChain,
    PathwayExplanation,
    ActivationTrace,
    EvidenceSupport,
    Uncertainty,
    Comparison,
    Summary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateVariable {
    pub name: String,
    pub var_type: VariableType,
    pub required: bool,
    pub default_value: Option<String>,
    pub formatting: Option<VariableFormatting>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VariableType {
    Text,
    Number,
    Entity,
    ActivationLevel,
    PathwayList,
    EvidenceList,
    Timestamp,
    Boolean,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariableFormatting {
    pub precision: Option<usize>,
    pub units: Option<String>,
    pub date_format: Option<String>,
    pub list_separator: Option<String>,
    pub max_length: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateCondition {
    pub variable: String,
    pub operator: ConditionOperator,
    pub value: String,
    pub action: ConditionAction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConditionOperator {
    Equals,
    NotEquals,
    GreaterThan,
    LessThan,
    Contains,
    NotContains,
    Exists,
    NotExists,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConditionAction {
    Include(String),
    Exclude(String),
    Replace(String),
    Modify(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OutputFormat {
    PlainText,
    Markdown,
    Html,
    Json,
    Structured,
}

#[derive(Debug, Clone)]
pub struct ExplanationContext {
    pub query: String,
    pub query_type: String,
    pub activation_data: HashMap<NodeId, ActivationLevel>,
    pub pathways: Vec<ActivationPathway>,
    pub entities: Vec<EntityId>,
    pub evidence: Vec<Evidence>,
    pub confidence: f32,
    pub processing_time: f32,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub struct Evidence {
    pub source: String,
    pub content: String,
    pub confidence: f32,
    pub relevance: f32,
    pub timestamp: std::time::Instant,
}
```

### Step 2: Template Registry Implementation

```rust
#[derive(Debug)]
pub struct TemplateRegistry {
    templates: HashMap<TemplateId, ExplanationTemplate>,
    category_index: HashMap<TemplateCategory, Vec<TemplateId>>,
    name_index: HashMap<String, TemplateId>,
    next_template_id: u64,
    default_templates: Vec<TemplateId>,
}

impl TemplateRegistry {
    pub fn new() -> Self {
        let mut registry = Self {
            templates: HashMap::new(),
            category_index: HashMap::new(),
            name_index: HashMap::new(),
            next_template_id: 1,
            default_templates: Vec::new(),
        };
        
        registry.load_default_templates();
        registry
    }
    
    pub fn register_template(&mut self, template: ExplanationTemplate) -> TemplateId {
        let template_id = if template.template_id.0 == 0 {
            let id = TemplateId(self.next_template_id);
            self.next_template_id += 1;
            id
        } else {
            template.template_id
        };
        
        let mut template = template;
        template.template_id = template_id;
        
        // Update indices
        self.category_index
            .entry(template.category.clone())
            .or_insert_with(Vec::new)
            .push(template_id);
        
        self.name_index.insert(template.name.clone(), template_id);
        
        self.templates.insert(template_id, template);
        template_id
    }
    
    pub fn get_template(&self, template_id: TemplateId) -> Option<&ExplanationTemplate> {
        self.templates.get(&template_id)
    }
    
    pub fn get_template_by_name(&self, name: &str) -> Option<&ExplanationTemplate> {
        self.name_index.get(name)
            .and_then(|id| self.templates.get(id))
    }
    
    pub fn get_templates_by_category(&self, category: &TemplateCategory) -> Vec<&ExplanationTemplate> {
        self.category_index.get(category)
            .map(|ids| {
                ids.iter()
                    .filter_map(|id| self.templates.get(id))
                    .collect()
            })
            .unwrap_or_default()
    }
    
    pub fn select_best_template(
        &self,
        category: &TemplateCategory,
        context: &ExplanationContext,
    ) -> Option<&ExplanationTemplate> {
        let candidates = self.get_templates_by_category(category);
        
        // Score templates based on context match and priority
        let mut scored_templates: Vec<(&ExplanationTemplate, f32)> = candidates
            .into_iter()
            .map(|template| {
                let score = self.calculate_template_score(template, context);
                (template, score)
            })
            .collect();
        
        // Sort by score (descending) and priority (descending)
        scored_templates.sort_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                .then(b.0.priority.cmp(&a.0.priority))
        });
        
        scored_templates.first().map(|(template, _)| *template)
    }
    
    fn calculate_template_score(&self, template: &ExplanationTemplate, context: &ExplanationContext) -> f32 {
        let mut score = 0.0;
        
        // Check variable availability
        let available_vars = self.get_available_variables(context);
        let required_vars: Vec<&String> = template.variables.iter()
            .filter(|v| v.required)
            .map(|v| &v.name)
            .collect();
        
        let satisfied_requirements = required_vars.iter()
            .filter(|var| available_vars.contains(*var))
            .count();
        
        if required_vars.is_empty() {
            score += 0.5;
        } else {
            score += (satisfied_requirements as f32) / (required_vars.len() as f32);
        }
        
        // Bonus for optional variables that are available
        let optional_vars: Vec<&String> = template.variables.iter()
            .filter(|v| !v.required)
            .map(|v| &v.name)
            .collect();
        
        let available_optional = optional_vars.iter()
            .filter(|var| available_vars.contains(*var))
            .count();
        
        if !optional_vars.is_empty() {
            score += 0.3 * (available_optional as f32) / (optional_vars.len() as f32);
        }
        
        // Priority bonus
        score += (template.priority as f32) / 100.0;
        
        score
    }
    
    fn get_available_variables(&self, context: &ExplanationContext) -> Vec<String> {
        let mut vars = vec![
            "query".to_string(),
            "query_type".to_string(),
            "confidence".to_string(),
            "processing_time".to_string(),
        ];
        
        if !context.activation_data.is_empty() {
            vars.push("activation_data".to_string());
            vars.push("activated_nodes".to_string());
        }
        
        if !context.pathways.is_empty() {
            vars.push("pathways".to_string());
            vars.push("pathway_count".to_string());
        }
        
        if !context.entities.is_empty() {
            vars.push("entities".to_string());
            vars.push("entity_count".to_string());
        }
        
        if !context.evidence.is_empty() {
            vars.push("evidence".to_string());
            vars.push("evidence_count".to_string());
        }
        
        // Add metadata variables
        for key in context.metadata.keys() {
            vars.push(format!("metadata.{}", key));
        }
        
        vars
    }
    
    fn load_default_templates(&mut self) {
        // Factual Answer Template
        let factual_template = ExplanationTemplate {
            template_id: TemplateId(0),
            name: "Simple Factual Answer".to_string(),
            category: TemplateCategory::FactualAnswer,
            pattern: "Based on the query \"{{query}}\", I found {{entity_count}} relevant entities. {{#if confidence > 0.8}}I'm confident that{{else}}I believe that{{/if}} {{primary_answer}}.".to_string(),
            variables: vec![
                TemplateVariable {
                    name: "query".to_string(),
                    var_type: VariableType::Text,
                    required: true,
                    default_value: None,
                    formatting: None,
                },
                TemplateVariable {
                    name: "entity_count".to_string(),
                    var_type: VariableType::Number,
                    required: true,
                    default_value: Some("0".to_string()),
                    formatting: None,
                },
                TemplateVariable {
                    name: "confidence".to_string(),
                    var_type: VariableType::Number,
                    required: true,
                    default_value: Some("0.0".to_string()),
                    formatting: Some(VariableFormatting {
                        precision: Some(2),
                        units: None,
                        date_format: None,
                        list_separator: None,
                        max_length: None,
                    }),
                },
                TemplateVariable {
                    name: "primary_answer".to_string(),
                    var_type: VariableType::Text,
                    required: true,
                    default_value: Some("the answer is uncertain".to_string()),
                    formatting: None,
                },
            ],
            conditions: vec![],
            output_format: OutputFormat::PlainText,
            priority: 50,
        };
        
        self.register_template(factual_template);
        
        // Reasoning Chain Template
        let reasoning_template = ExplanationTemplate {
            template_id: TemplateId(0),
            name: "Reasoning Chain Explanation".to_string(),
            category: TemplateCategory::ReasoningChain,
            pattern: "To answer \"{{query}}\", I followed this reasoning:\n\n{{#each pathways}}{{step_number}}. {{pathway_description}}\n{{/each}}\n\nThis led me to conclude: {{conclusion}}".to_string(),
            variables: vec![
                TemplateVariable {
                    name: "query".to_string(),
                    var_type: VariableType::Text,
                    required: true,
                    default_value: None,
                    formatting: None,
                },
                TemplateVariable {
                    name: "pathways".to_string(),
                    var_type: VariableType::PathwayList,
                    required: true,
                    default_value: None,
                    formatting: Some(VariableFormatting {
                        precision: None,
                        units: None,
                        date_format: None,
                        list_separator: Some("\n".to_string()),
                        max_length: Some(5),
                    }),
                },
                TemplateVariable {
                    name: "conclusion".to_string(),
                    var_type: VariableType::Text,
                    required: true,
                    default_value: None,
                    formatting: None,
                },
            ],
            conditions: vec![
                TemplateCondition {
                    variable: "pathway_count".to_string(),
                    operator: ConditionOperator::GreaterThan,
                    value: "1".to_string(),
                    action: ConditionAction::Include("multiple reasoning steps".to_string()),
                },
            ],
            output_format: OutputFormat::Markdown,
            priority: 75,
        };
        
        self.register_template(reasoning_template);
        
        // Pathway Explanation Template
        let pathway_template = ExplanationTemplate {
            template_id: TemplateId(0),
            name: "Activation Pathway Explanation".to_string(),
            category: TemplateCategory::PathwayExplanation,
            pattern: "The neural-like activation spread through {{pathway_count}} pathways:\n\n{{#each pathways}}**Pathway {{@index}}**: {{source_node}} → {{target_nodes}} (strength: {{activation_strength}})\n{{/each}}\n\nTotal activation: {{total_activation}}".to_string(),
            variables: vec![
                TemplateVariable {
                    name: "pathway_count".to_string(),
                    var_type: VariableType::Number,
                    required: true,
                    default_value: Some("0".to_string()),
                    formatting: None,
                },
                TemplateVariable {
                    name: "pathways".to_string(),
                    var_type: VariableType::PathwayList,
                    required: true,
                    default_value: None,
                    formatting: None,
                },
                TemplateVariable {
                    name: "total_activation".to_string(),
                    var_type: VariableType::Number,
                    required: true,
                    default_value: Some("0.0".to_string()),
                    formatting: Some(VariableFormatting {
                        precision: Some(3),
                        units: None,
                        date_format: None,
                        list_separator: None,
                        max_length: None,
                    }),
                },
            ],
            conditions: vec![],
            output_format: OutputFormat::Markdown,
            priority: 60,
        };
        
        self.register_template(pathway_template);
    }
    
    pub fn list_templates(&self) -> Vec<&ExplanationTemplate> {
        self.templates.values().collect()
    }
    
    pub fn remove_template(&mut self, template_id: TemplateId) -> bool {
        if let Some(template) = self.templates.remove(&template_id) {
            // Clean up indices
            if let Some(category_ids) = self.category_index.get_mut(&template.category) {
                category_ids.retain(|&id| id != template_id);
            }
            self.name_index.remove(&template.name);
            self.default_templates.retain(|&id| id != template_id);
            true
        } else {
            false
        }
    }
}
```

### Step 3: Template Renderer Implementation

```rust
#[derive(Debug)]
pub struct TemplateRenderer {
    registry: TemplateRegistry,
    cache: HashMap<(TemplateId, u64), String>, // Template + context hash -> rendered
    cache_enabled: bool,
    max_cache_size: usize,
}

impl TemplateRenderer {
    pub fn new() -> Self {
        Self {
            registry: TemplateRegistry::new(),
            cache: HashMap::new(),
            cache_enabled: true,
            max_cache_size: 1000,
        }
    }
    
    pub fn with_registry(registry: TemplateRegistry) -> Self {
        Self {
            registry,
            cache: HashMap::new(),
            cache_enabled: true,
            max_cache_size: 1000,
        }
    }
    
    pub fn render_explanation(
        &mut self,
        category: &TemplateCategory,
        context: &ExplanationContext,
    ) -> Result<String, RenderError> {
        let template = self.registry.select_best_template(category, context)
            .ok_or(RenderError::NoSuitableTemplate)?;
        
        self.render_with_template(template, context)
    }
    
    pub fn render_with_template(
        &mut self,
        template: &ExplanationTemplate,
        context: &ExplanationContext,
    ) -> Result<String, RenderError> {
        // Check cache
        let context_hash = self.calculate_context_hash(context);
        let cache_key = (template.template_id, context_hash);
        
        if self.cache_enabled {
            if let Some(cached_result) = self.cache.get(&cache_key) {
                return Ok(cached_result.clone());
            }
        }
        
        // Validate required variables
        self.validate_variables(template, context)?;
        
        // Build variable map
        let variables = self.build_variable_map(template, context)?;
        
        // Apply conditions
        let effective_pattern = self.apply_conditions(template, &variables)?;
        
        // Render template
        let result = self.substitute_variables(&effective_pattern, &variables)?;
        
        // Format output
        let formatted_result = self.format_output(&result, &template.output_format)?;
        
        // Cache result
        if self.cache_enabled {
            self.manage_cache_size();
            self.cache.insert(cache_key, formatted_result.clone());
        }
        
        Ok(formatted_result)
    }
    
    fn validate_variables(
        &self,
        template: &ExplanationTemplate,
        context: &ExplanationContext,
    ) -> Result<(), RenderError> {
        let available_vars = self.registry.get_available_variables(context);
        
        for variable in &template.variables {
            if variable.required && !available_vars.contains(&variable.name) {
                if variable.default_value.is_none() {
                    return Err(RenderError::MissingRequiredVariable(variable.name.clone()));
                }
            }
        }
        
        Ok(())
    }
    
    fn build_variable_map(
        &self,
        template: &ExplanationTemplate,
        context: &ExplanationContext,
    ) -> Result<HashMap<String, String>, RenderError> {
        let mut variables = HashMap::new();
        
        // Basic context variables
        variables.insert("query".to_string(), context.query.clone());
        variables.insert("query_type".to_string(), context.query_type.clone());
        variables.insert("confidence".to_string(), format!("{:.2}", context.confidence));
        variables.insert("processing_time".to_string(), format!("{:.2}", context.processing_time));
        
        // Entity variables
        variables.insert("entity_count".to_string(), context.entities.len().to_string());
        
        // Pathway variables
        variables.insert("pathway_count".to_string(), context.pathways.len().to_string());
        if !context.pathways.is_empty() {
            let total_activation: f32 = context.pathways.iter()
                .map(|p| p.total_activation)
                .sum();
            variables.insert("total_activation".to_string(), format!("{:.3}", total_activation));
        }
        
        // Evidence variables
        variables.insert("evidence_count".to_string(), context.evidence.len().to_string());
        
        // Activation variables
        variables.insert("activated_nodes".to_string(), context.activation_data.len().to_string());
        
        // Metadata variables
        for (key, value) in &context.metadata {
            variables.insert(format!("metadata.{}", key), value.clone());
        }
        
        // Apply formatting to variables based on template requirements
        for template_var in &template.variables {
            if let Some(value) = variables.get(&template_var.name) {
                if let Some(formatting) = &template_var.formatting {
                    let formatted_value = self.format_variable(value, &template_var.var_type, formatting)?;
                    variables.insert(template_var.name.clone(), formatted_value);
                }
            } else if let Some(default) = &template_var.default_value {
                variables.insert(template_var.name.clone(), default.clone());
            }
        }
        
        Ok(variables)
    }
    
    fn format_variable(
        &self,
        value: &str,
        var_type: &VariableType,
        formatting: &VariableFormatting,
    ) -> Result<String, RenderError> {
        match var_type {
            VariableType::Number => {
                if let Ok(num) = value.parse::<f64>() {
                    if let Some(precision) = formatting.precision {
                        let formatted = format!("{:.1$}", num, precision);
                        if let Some(units) = &formatting.units {
                            Ok(format!("{} {}", formatted, units))
                        } else {
                            Ok(formatted)
                        }
                    } else {
                        Ok(value.to_string())
                    }
                } else {
                    Ok(value.to_string())
                }
            },
            VariableType::Text => {
                let mut result = value.to_string();
                if let Some(max_len) = formatting.max_length {
                    if result.len() > max_len {
                        result.truncate(max_len);
                        result.push_str("...");
                    }
                }
                Ok(result)
            },
            _ => Ok(value.to_string()),
        }
    }
    
    fn apply_conditions(
        &self,
        template: &ExplanationTemplate,
        variables: &HashMap<String, String>,
    ) -> Result<String, RenderError> {
        let mut pattern = template.pattern.clone();
        
        for condition in &template.conditions {
            if let Some(var_value) = variables.get(&condition.variable) {
                let condition_met = self.evaluate_condition(
                    var_value,
                    &condition.operator,
                    &condition.value,
                )?;
                
                if condition_met {
                    pattern = self.apply_condition_action(&pattern, &condition.action)?;
                }
            }
        }
        
        Ok(pattern)
    }
    
    fn evaluate_condition(
        &self,
        var_value: &str,
        operator: &ConditionOperator,
        condition_value: &str,
    ) -> Result<bool, RenderError> {
        match operator {
            ConditionOperator::Equals => Ok(var_value == condition_value),
            ConditionOperator::NotEquals => Ok(var_value != condition_value),
            ConditionOperator::GreaterThan => {
                if let (Ok(var_num), Ok(cond_num)) = (var_value.parse::<f64>(), condition_value.parse::<f64>()) {
                    Ok(var_num > cond_num)
                } else {
                    Ok(false)
                }
            },
            ConditionOperator::LessThan => {
                if let (Ok(var_num), Ok(cond_num)) = (var_value.parse::<f64>(), condition_value.parse::<f64>()) {
                    Ok(var_num < cond_num)
                } else {
                    Ok(false)
                }
            },
            ConditionOperator::Contains => Ok(var_value.contains(condition_value)),
            ConditionOperator::NotContains => Ok(!var_value.contains(condition_value)),
            ConditionOperator::Exists => Ok(!var_value.is_empty()),
            ConditionOperator::NotExists => Ok(var_value.is_empty()),
        }
    }
    
    fn apply_condition_action(
        &self,
        pattern: &str,
        action: &ConditionAction,
    ) -> Result<String, RenderError> {
        match action {
            ConditionAction::Include(text) => Ok(format!("{} {}", pattern, text)),
            ConditionAction::Exclude(text) => Ok(pattern.replace(text, "")),
            ConditionAction::Replace(replacement) => Ok(replacement.clone()),
            ConditionAction::Modify(modification) => Ok(format!("{} ({})", pattern, modification)),
        }
    }
    
    fn substitute_variables(
        &self,
        pattern: &str,
        variables: &HashMap<String, String>,
    ) -> Result<String, RenderError> {
        let mut result = pattern.to_string();
        
        // Simple variable substitution {{variable_name}}
        for (name, value) in variables {
            let placeholder = format!("{{{{{}}}}}", name);
            result = result.replace(&placeholder, value);
        }
        
        // Handle conditional blocks {{#if condition}}...{{else}}...{{/if}}
        result = self.process_conditional_blocks(&result, variables)?;
        
        // Handle loops {{#each collection}}...{{/each}}
        result = self.process_loop_blocks(&result, variables)?;
        
        Ok(result)
    }
    
    fn process_conditional_blocks(
        &self,
        text: &str,
        variables: &HashMap<String, String>,
    ) -> Result<String, RenderError> {
        // Simplified conditional processing - would need more robust parser in production
        let mut result = text.to_string();
        
        // Find {{#if condition}} blocks
        while let Some(start) = result.find("{{#if ") {
            if let Some(end_if) = result.find("{{/if}}") {
                let full_block = &result[start..end_if + 7];
                let condition_end = result[start..].find("}}").unwrap_or(0) + start + 2;
                let condition = &result[start + 6..condition_end - 2].trim();
                
                let condition_met = self.evaluate_simple_condition(condition, variables)?;
                
                // Extract then and else parts
                let content = &result[condition_end..end_if];
                let replacement = if let Some(else_pos) = content.find("{{else}}") {
                    let then_part = &content[..else_pos];
                    let else_part = &content[else_pos + 8..];
                    if condition_met { then_part } else { else_part }
                } else {
                    if condition_met { content } else { "" }
                };
                
                result = result.replace(full_block, replacement);
            } else {
                break;
            }
        }
        
        Ok(result)
    }
    
    fn process_loop_blocks(
        &self,
        text: &str,
        _variables: &HashMap<String, String>,
    ) -> Result<String, RenderError> {
        // Simplified loop processing - would need more robust implementation
        let mut result = text.to_string();
        
        // For now, just remove loop syntax and keep content
        result = result.replace("{{#each pathways}}", "");
        result = result.replace("{{/each}}", "");
        result = result.replace("{{@index}}", "1"); // Simplified index
        
        Ok(result)
    }
    
    fn evaluate_simple_condition(
        &self,
        condition: &str,
        variables: &HashMap<String, String>,
    ) -> Result<bool, RenderError> {
        // Simple condition evaluation: "variable > value"
        if let Some(gt_pos) = condition.find(" > ") {
            let var_name = condition[..gt_pos].trim();
            let value = condition[gt_pos + 3..].trim();
            
            if let (Some(var_value), Ok(condition_value)) = (variables.get(var_name), value.parse::<f64>()) {
                if let Ok(var_num) = var_value.parse::<f64>() {
                    return Ok(var_num > condition_value);
                }
            }
        }
        
        // Default to checking if variable exists and is non-empty
        if let Some(var_value) = variables.get(condition) {
            Ok(!var_value.is_empty() && var_value != "0" && var_value != "false")
        } else {
            Ok(false)
        }
    }
    
    fn format_output(&self, text: &str, format: &OutputFormat) -> Result<String, RenderError> {
        match format {
            OutputFormat::PlainText => Ok(text.to_string()),
            OutputFormat::Markdown => Ok(text.to_string()), // Already formatted as markdown
            OutputFormat::Html => {
                // Simple markdown to HTML conversion
                let html = text
                    .replace("\n\n", "</p><p>")
                    .replace("**", "<strong>")
                    .replace("*", "<em>");
                Ok(format!("<p>{}</p>", html))
            },
            OutputFormat::Json => {
                let json_obj = serde_json::json!({
                    "explanation": text,
                    "format": "rendered"
                });
                Ok(json_obj.to_string())
            },
            OutputFormat::Structured => Ok(text.to_string()),
        }
    }
    
    fn calculate_context_hash(&self, context: &ExplanationContext) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        context.query.hash(&mut hasher);
        context.query_type.hash(&mut hasher);
        context.entities.len().hash(&mut hasher);
        context.pathways.len().hash(&mut hasher);
        (context.confidence * 1000.0) as u64.hash(&mut hasher);
        hasher.finish()
    }
    
    fn manage_cache_size(&mut self) {
        if self.cache.len() >= self.max_cache_size {
            // Simple LRU-like: remove half the entries
            let keys_to_remove: Vec<_> = self.cache.keys()
                .take(self.max_cache_size / 2)
                .cloned()
                .collect();
            
            for key in keys_to_remove {
                self.cache.remove(&key);
            }
        }
    }
    
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }
    
    pub fn set_cache_enabled(&mut self, enabled: bool) {
        self.cache_enabled = enabled;
        if !enabled {
            self.clear_cache();
        }
    }
    
    pub fn get_registry(&self) -> &TemplateRegistry {
        &self.registry
    }
    
    pub fn get_registry_mut(&mut self) -> &mut TemplateRegistry {
        &mut self.registry
    }
}

#[derive(Debug, Clone)]
pub enum RenderError {
    NoSuitableTemplate,
    MissingRequiredVariable(String),
    InvalidCondition(String),
    VariableFormatError(String),
    TemplateParseError(String),
}

impl fmt::Display for RenderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RenderError::NoSuitableTemplate => write!(f, "No suitable template found"),
            RenderError::MissingRequiredVariable(var) => write!(f, "Missing required variable: {}", var),
            RenderError::InvalidCondition(cond) => write!(f, "Invalid condition: {}", cond),
            RenderError::VariableFormatError(err) => write!(f, "Variable formatting error: {}", err),
            RenderError::TemplateParseError(err) => write!(f, "Template parse error: {}", err),
        }
    }
}

impl std::error::Error for RenderError {}
```

### Step 4: Integration Interface

```rust
// File: src/cognitive/explanation/mod.rs

pub mod templates;

pub use templates::{
    ExplanationTemplate, TemplateRegistry, TemplateRenderer, ExplanationContext,
    TemplateCategory, OutputFormat, Evidence, RenderError,
};

use crate::cognitive::learning::pathway_tracing::ActivationPathway;
use crate::core::types::{NodeId, EntityId, ActivationLevel};

pub struct ExplanationGenerator {
    renderer: TemplateRenderer,
}

impl ExplanationGenerator {
    pub fn new() -> Self {
        Self {
            renderer: TemplateRenderer::new(),
        }
    }
    
    pub fn generate_explanation(
        &mut self,
        query: &str,
        query_type: &str,
        activation_data: HashMap<NodeId, ActivationLevel>,
        pathways: Vec<ActivationPathway>,
        entities: Vec<EntityId>,
        evidence: Vec<Evidence>,
        confidence: f32,
    ) -> Result<String, RenderError> {
        let context = ExplanationContext {
            query: query.to_string(),
            query_type: query_type.to_string(),
            activation_data,
            pathways,
            entities,
            evidence,
            confidence,
            processing_time: 0.0, // Would be measured in actual implementation
            metadata: HashMap::new(),
        };
        
        // Determine appropriate explanation category based on query type
        let category = match query_type {
            "factual" => TemplateCategory::FactualAnswer,
            "reasoning" => TemplateCategory::ReasoningChain,
            "pathway" => TemplateCategory::PathwayExplanation,
            _ => TemplateCategory::FactualAnswer,
        };
        
        self.renderer.render_explanation(&category, &context)
    }
    
    pub fn add_custom_template(&mut self, template: ExplanationTemplate) -> TemplateId {
        self.renderer.get_registry_mut().register_template(template)
    }
    
    pub fn get_available_templates(&self) -> Vec<&ExplanationTemplate> {
        self.renderer.get_registry().list_templates()
    }
}
```

## File Locations

- `src/cognitive/explanation/templates.rs` - Main implementation
- `src/cognitive/explanation/mod.rs` - Module exports and integration
- `tests/cognitive/explanation/template_tests.rs` - Test implementation

## Success Criteria

- [ ] ExplanationTemplate system compiles and runs
- [ ] TemplateRegistry manages templates correctly
- [ ] TemplateRenderer produces human-readable explanations
- [ ] Variable substitution works accurately
- [ ] Conditional logic functions properly
- [ ] Template caching improves performance
- [ ] All tests pass:
  - Template registration and retrieval
  - Variable substitution
  - Conditional rendering
  - Output formatting
  - Performance benchmarks

## Test Requirements

```rust
#[test]
fn test_template_registration() {
    let mut registry = TemplateRegistry::new();
    
    let template = ExplanationTemplate {
        template_id: TemplateId(0),
        name: "Test Template".to_string(),
        category: TemplateCategory::FactualAnswer,
        pattern: "Test: {{value}}".to_string(),
        variables: vec![
            TemplateVariable {
                name: "value".to_string(),
                var_type: VariableType::Text,
                required: true,
                default_value: None,
                formatting: None,
            }
        ],
        conditions: vec![],
        output_format: OutputFormat::PlainText,
        priority: 50,
    };
    
    let template_id = registry.register_template(template);
    assert!(template_id.0 > 0);
    
    let retrieved = registry.get_template(template_id);
    assert!(retrieved.is_some());
    assert_eq!(retrieved.unwrap().name, "Test Template");
}

#[test]
fn test_variable_substitution() {
    let mut renderer = TemplateRenderer::new();
    
    let template = ExplanationTemplate {
        template_id: TemplateId(1),
        name: "Variable Test".to_string(),
        category: TemplateCategory::FactualAnswer,
        pattern: "Query: {{query}}, Confidence: {{confidence}}".to_string(),
        variables: vec![
            TemplateVariable {
                name: "query".to_string(),
                var_type: VariableType::Text,
                required: true,
                default_value: None,
                formatting: None,
            },
            TemplateVariable {
                name: "confidence".to_string(),
                var_type: VariableType::Number,
                required: true,
                default_value: None,
                formatting: Some(VariableFormatting {
                    precision: Some(2),
                    units: None,
                    date_format: None,
                    list_separator: None,
                    max_length: None,
                }),
            },
        ],
        conditions: vec![],
        output_format: OutputFormat::PlainText,
        priority: 50,
    };
    
    let context = ExplanationContext {
        query: "What is AI?".to_string(),
        query_type: "factual".to_string(),
        activation_data: HashMap::new(),
        pathways: vec![],
        entities: vec![],
        evidence: vec![],
        confidence: 0.85,
        processing_time: 0.0,
        metadata: HashMap::new(),
    };
    
    let result = renderer.render_with_template(&template, &context).unwrap();
    assert!(result.contains("What is AI?"));
    assert!(result.contains("0.85"));
}

#[test]
fn test_conditional_rendering() {
    let mut renderer = TemplateRenderer::new();
    
    let template = ExplanationTemplate {
        template_id: TemplateId(2),
        name: "Conditional Test".to_string(),
        category: TemplateCategory::FactualAnswer,
        pattern: "{{#if confidence > 0.8}}High confidence{{else}}Low confidence{{/if}} answer".to_string(),
        variables: vec![
            TemplateVariable {
                name: "confidence".to_string(),
                var_type: VariableType::Number,
                required: true,
                default_value: None,
                formatting: None,
            },
        ],
        conditions: vec![],
        output_format: OutputFormat::PlainText,
        priority: 50,
    };
    
    // Test high confidence
    let high_confidence_context = ExplanationContext {
        query: "test".to_string(),
        query_type: "factual".to_string(),
        activation_data: HashMap::new(),
        pathways: vec![],
        entities: vec![],
        evidence: vec![],
        confidence: 0.9,
        processing_time: 0.0,
        metadata: HashMap::new(),
    };
    
    let result = renderer.render_with_template(&template, &high_confidence_context).unwrap();
    assert!(result.contains("High confidence"));
    
    // Test low confidence
    let low_confidence_context = ExplanationContext {
        confidence: 0.3,
        ..high_confidence_context
    };
    
    let result = renderer.render_with_template(&template, &low_confidence_context).unwrap();
    assert!(result.contains("Low confidence"));
}

#[test]
fn test_template_selection() {
    let registry = TemplateRegistry::new();
    
    let context = ExplanationContext {
        query: "test query".to_string(),
        query_type: "factual".to_string(),
        activation_data: HashMap::new(),
        pathways: vec![],
        entities: vec![EntityId(1), EntityId(2)],
        evidence: vec![],
        confidence: 0.8,
        processing_time: 0.0,
        metadata: HashMap::new(),
    };
    
    let template = registry.select_best_template(&TemplateCategory::FactualAnswer, &context);
    assert!(template.is_some());
    assert_eq!(template.unwrap().category, TemplateCategory::FactualAnswer);
}

#[test]
fn test_template_caching() {
    let mut renderer = TemplateRenderer::new();
    
    let template = ExplanationTemplate {
        template_id: TemplateId(3),
        name: "Cache Test".to_string(),
        category: TemplateCategory::FactualAnswer,
        pattern: "Result: {{query}}".to_string(),
        variables: vec![
            TemplateVariable {
                name: "query".to_string(),
                var_type: VariableType::Text,
                required: true,
                default_value: None,
                formatting: None,
            }
        ],
        conditions: vec![],
        output_format: OutputFormat::PlainText,
        priority: 50,
    };
    
    let context = ExplanationContext {
        query: "cached test".to_string(),
        query_type: "factual".to_string(),
        activation_data: HashMap::new(),
        pathways: vec![],
        entities: vec![],
        evidence: vec![],
        confidence: 0.5,
        processing_time: 0.0,
        metadata: HashMap::new(),
    };
    
    // First render
    let result1 = renderer.render_with_template(&template, &context).unwrap();
    
    // Second render (should use cache)
    let result2 = renderer.render_with_template(&template, &context).unwrap();
    
    assert_eq!(result1, result2);
    assert!(result1.contains("cached test"));
}
```

## Quality Gates

- [ ] Template rendering < 5ms per explanation
- [ ] Memory usage < 10MB for 1000 templates
- [ ] Cache hit rate > 70% for repeated explanations
- [ ] Support for concurrent template rendering
- [ ] No memory leaks during continuous operation
- [ ] Accurate variable substitution and formatting

## Next Task

Upon completion, proceed to **26_reasoning_extraction.md**