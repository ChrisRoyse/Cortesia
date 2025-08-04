# Task 12: Generate Chunk Boundary Test Files

## Context
You are continuing the test data generation phase (Phase 0, Task 12). Tasks 10-11 created special character and edge case test files. Now you need to create specific test files designed to validate semantic chunking behavior at boundaries.

## Objective
Generate test files specifically designed to test tree-sitter semantic chunking at various boundaries, ensuring that code elements are not incorrectly split and that semantic units remain intact.

## Requirements
1. Create files that test function boundary preservation
2. Create files that test struct/enum boundary preservation  
3. Create files that test impl block boundary preservation
4. Create files that test module boundary preservation
5. Create files that test generic parameter boundary preservation
6. Create files that challenge chunking algorithms with edge cases

## Implementation for test_data.rs (extend existing)
```rust
impl EdgeCaseTestGenerator {
    // ... existing methods ...
}

pub struct ChunkBoundaryTestGenerator;

impl ChunkBoundaryTestGenerator {
    /// Generate chunk boundary test files for semantic parsing validation
    pub fn generate_chunk_boundary_tests() -> Result<()> {
        info!("Generating chunk boundary test files");
        
        // Create test data directory
        std::fs::create_dir_all("test_data/chunk_boundaries")?;
        
        // Generate different boundary test scenarios
        Self::generate_function_boundary_tests()?;
        Self::generate_struct_boundary_tests()?;
        Self::generate_impl_boundary_tests()?;
        Self::generate_module_boundary_tests()?;
        Self::generate_generic_boundary_tests()?;
        Self::generate_macro_boundary_tests()?;
        Self::generate_comment_boundary_tests()?;
        Self::generate_complex_boundary_tests()?;
        
        info!("Chunk boundary test files generated successfully");
        Ok(())
    }
    
    fn generate_function_boundary_tests() -> Result<()> {
        debug!("Generating function boundary test files");
        
        // Test case: Function with exact chunk size boundaries
        let chunk_size = 1000; // Simulated chunk size
        
        // Function that should not be split
        let function_content = format!(
            r#"
{}
/// This function should remain as a single chunk
/// It contains important functionality that must stay together
pub fn important_function_with_complex_signature<T, E>(
    param1: &str,
    param2: Option<T>,
    param3: Result<Vec<String>, E>,
) -> Result<HashMap<String, T>, Box<dyn std::error::Error + Send + Sync>>
where
    T: Clone + Send + Sync + std::fmt::Debug + 'static,
    E: std::error::Error + Send + Sync + 'static,
{{
    // Function implementation that must stay with its signature
    let mut result = HashMap::new();
    
    if let Some(value) = param2 {{
        result.insert(param1.to_string(), value);
    }}
    
    match param3 {{
        Ok(strings) => {{
            for (i, s) in strings.iter().enumerate() {{
                // This processing logic belongs with the function
                println!("Processing: {{}}", s);
                if i > 10 {{
                    break;
                }}
            }}
        }}
        Err(e) => {{
            eprintln!("Error processing: {{}}", e);
            return Err(Box::new(e));
        }}
    }}
    
    Ok(result)
}}
            "#,
            "// ".repeat(chunk_size / 3) // Padding to approach chunk boundary
        );
        
        let path = Path::new("test_data/chunk_boundaries").join("function_preservation.rs");
        fs::write(path, function_content)?;
        
        // Multiple functions at boundary
        let multi_function_content = format!(
            r#"
{}
pub fn first_function() -> i32 {{
    // This function should be in its own chunk
    42
}}

pub fn second_function_at_boundary() -> String {{
    // This function starts near a chunk boundary
    "boundary test".to_string()
}}

pub fn third_function() -> Result<(), Error> {{
    // This function should also be preserved as a unit
    Ok(())
}}
            "#,
            "// ".repeat(chunk_size / 4)
        );
        
        let path = Path::new("test_data/chunk_boundaries").join("multiple_functions.rs");
        fs::write(path, multi_function_content)?;
        
        // Async function boundary test
        let async_function_content = format!(
            r#"
{}
/// Async function that must be kept together
pub async fn async_function_with_complex_logic<T>(
    input: Vec<T>,
) -> Result<Vec<T>, Box<dyn std::error::Error + Send + Sync>>
where
    T: Clone + Send + Sync + 'static,
{{
    // The await points and error handling must stay with the function
    let mut processed = Vec::new();
    
    for item in input {{
        tokio::time::sleep(tokio::time::Duration::from_millis(1)).await;
        processed.push(item.clone());
        
        if processed.len() > 1000 {{
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::Other,
                "Too many items"
            )));
        }}
    }}
    
    Ok(processed)
}}
            "#,
            "// ".repeat(chunk_size / 3)
        );
        
        let path = Path::new("test_data/chunk_boundaries").join("async_function_boundary.rs");
        fs::write(path, async_function_content)?;
        
        debug!("Created function boundary test files");
        Ok(())
    }
    
    fn generate_struct_boundary_tests() -> Result<()> {
        debug!("Generating struct boundary test files");
        
        let struct_content = format!(
            r#"
{}
/// This struct definition must remain intact across chunk boundaries
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ComplexStructAtBoundary<T, U, V> 
where
    T: Clone + Send + Sync + std::fmt::Debug + 'static,
    U: std::hash::Hash + Eq + Clone + Send + Sync + 'static,
    V: Default + Clone + Send + Sync + 'static,
{{
    /// Primary identifier field
    #[serde(rename = "id")]
    pub identifier: String,
    
    /// Generic data field
    pub data: HashMap<U, Vec<T>>,
    
    /// Optional configuration
    #[serde(skip_serializing_if = "Option::is_none")]
    pub config: Option<V>,
    
    /// Metadata that belongs with this struct
    pub metadata: StructMetadata,
    
    /// Creation timestamp
    #[serde(with = "timestamp_format")]
    pub created_at: SystemTime,
    
    /// Update history
    pub updates: Vec<UpdateRecord<T>>,
}}

/// Supporting types that belong with the main struct
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructMetadata {{
    pub version: u32,
    pub author: String,
    pub tags: HashSet<String>,
}}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateRecord<T> {{
    pub timestamp: SystemTime,
    pub changes: Vec<FieldChange<T>>,
    pub reason: String,
}}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FieldChange<T> {{
    Added {{ field: String, value: T }},
    Modified {{ field: String, old_value: T, new_value: T }},
    Removed {{ field: String, old_value: T }},
}}
            "#,
            "// ".repeat(500) // Padding
        );
        
        let path = Path::new("test_data/chunk_boundaries").join("struct_preservation.rs");
        fs::write(path, struct_content)?;
        
        // Enum boundary test
        let enum_content = format!(
            r#"
{}
/// Complex enum that should not be split across chunks
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
pub enum ComplexEnumAtBoundary<T, E> 
where
    T: Clone + Send + Sync + std::fmt::Debug + 'static,
    E: std::error::Error + Send + Sync + 'static,
{{
    /// Success variant with complex data
    Success {{
        data: T,
        metadata: HashMap<String, String>,
        processed_at: SystemTime,
    }},
    
    /// Error variant with context
    Error {{
        error: E,
        context: Vec<String>,
        retry_count: u32,
        last_attempt: SystemTime,
    }},
    
    /// Pending variant with progress tracking
    Pending {{
        progress: f64,
        estimated_completion: Option<SystemTime>,
        current_stage: ProcessingStage,
        stages_completed: Vec<ProcessingStage>,
    }},
    
    /// Complex variant with nested enum
    Complex {{
        primary: Box<ComplexEnumAtBoundary<T, E>>,
        alternatives: Vec<ComplexEnumAtBoundary<T, E>>,
        selection_criteria: SelectionCriteria,
    }},
}}

/// Supporting enum that belongs with the main enum
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ProcessingStage {{
    Initialization,
    DataCollection {{ source: String, progress: f64 }},
    Processing {{ algorithm: String, iteration: u32 }},
    Validation {{ rules_applied: Vec<String> }},
    Finalization {{ output_format: String }},
}}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SelectionCriteria {{
    pub priority: i32,
    pub confidence: f64,
    pub factors: HashMap<String, f64>,
}}
            "#,
            "// ".repeat(300)
        );
        
        let path = Path::new("test_data/chunk_boundaries").join("enum_preservation.rs");
        fs::write(path, enum_content)?;
        
        debug!("Created struct boundary test files");
        Ok(())
    }
    
    fn generate_impl_boundary_tests() -> Result<()> {
        debug!("Generating impl boundary test files");
        
        let impl_content = format!(
            r#"
{}
impl<T, U, V> ComplexStructAtBoundary<T, U, V>
where
    T: Clone + Send + Sync + std::fmt::Debug + 'static,
    U: std::hash::Hash + Eq + Clone + Send + Sync + 'static,
    V: Default + Clone + Send + Sync + 'static,
{{
    /// Constructor that must stay with the impl block
    pub fn new(identifier: String) -> Self {{
        Self {{
            identifier,
            data: HashMap::new(),
            config: Some(V::default()),
            metadata: StructMetadata {{
                version: 1,
                author: "system".to_string(),
                tags: HashSet::new(),
            }},
            created_at: SystemTime::now(),
            updates: Vec::new(),
        }}
    }}
    
    /// Method that performs complex operations
    pub async fn process_data(&mut self, new_data: HashMap<U, Vec<T>>) -> Result<(), ProcessingError> {{
        // This method's implementation must stay together
        let start_time = SystemTime::now();
        
        for (key, values) in new_data {{
            if let Some(existing) = self.data.get_mut(&key) {{
                existing.extend(values);
            }} else {{
                self.data.insert(key, values);
            }}
        }}
        
        // Record the update
        self.updates.push(UpdateRecord {{
            timestamp: start_time,
            changes: vec![FieldChange::Modified {{
                field: "data".to_string(),
                old_value: format!("{{}} entries", self.data.len() - new_data.len()),
                new_value: format!("{{}} entries", self.data.len()),
            }}],
            reason: "Data processing update".to_string(),
        }});
        
        // Simulate async processing
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
        
        Ok(())
    }}
    
    /// Validation method that belongs with the struct
    pub fn validate(&self) -> ValidationResult {{
        let mut issues = Vec::new();
        
        if self.identifier.is_empty() {{
            issues.push("Identifier cannot be empty".to_string());
        }}
        
        if self.data.is_empty() {{
            issues.push("Data cannot be empty".to_string());
        }}
        
        if self.metadata.version == 0 {{
            issues.push("Version must be greater than 0".to_string());
        }}
        
        if issues.is_empty() {{
            ValidationResult::Valid
        }} else {{
            ValidationResult::Invalid(issues)
        }}
    }}
}}

/// Trait implementation that should stay together
impl<T, U, V> std::fmt::Display for ComplexStructAtBoundary<T, U, V>
where
    T: Clone + Send + Sync + std::fmt::Debug + 'static,
    U: std::hash::Hash + Eq + Clone + Send + Sync + std::fmt::Debug + 'static,
    V: Default + Clone + Send + Sync + std::fmt::Debug + 'static,
{{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {{
        write!(f, "ComplexStruct {{ id: {{}}, data_entries: {{}}, version: {{}} }}", 
               self.identifier, 
               self.data.len(), 
               self.metadata.version)
    }}
}}

/// Error types that belong with the impl
#[derive(Debug, thiserror::Error)]
pub enum ProcessingError {{
    #[error("Invalid data format: {{0}}")]
    InvalidFormat(String),
    
    #[error("Processing timeout after {{0}} seconds")]
    Timeout(u64),
    
    #[error("Resource limit exceeded: {{0}}")]
    ResourceLimit(String),
}}

#[derive(Debug, Clone)]
pub enum ValidationResult {{
    Valid,
    Invalid(Vec<String>),
}}
            "#,
            "// ".repeat(200)
        );
        
        let path = Path::new("test_data/chunk_boundaries").join("impl_preservation.rs");
        fs::write(path, impl_content)?;
        
        debug!("Created impl boundary test files");
        Ok(())
    }
    
    fn generate_module_boundary_tests() -> Result<()> {
        debug!("Generating module boundary test files");
        
        let module_content = format!(
            r#"
{}
/// Module that should be kept as a coherent unit
pub mod boundary_test_module {{
    use std::collections::{{HashMap, HashSet}};
    use std::time::SystemTime;
    
    /// Module-level documentation that belongs with the module
    /// This entire module should be chunked as a logical unit
    
    pub struct ModuleLocalStruct {{
        pub data: String,
        pub references: Vec<String>,
    }}
    
    impl ModuleLocalStruct {{
        pub fn new(data: String) -> Self {{
            Self {{
                data,
                references: Vec::new(),
            }}
        }}
        
        pub fn add_reference(&mut self, reference: String) {{
            self.references.push(reference);
        }}
    }}
    
    /// Function that uses module-local types
    pub fn process_module_data(input: &str) -> ModuleLocalStruct {{
        let mut result = ModuleLocalStruct::new(input.to_string());
        
        // Add some processing logic
        let words: Vec<&str> = input.split_whitespace().collect();
        for word in words {{
            if word.len() > 3 {{
                result.add_reference(format!("ref_to_{{}}", word));
            }}
        }}
        
        result
    }}
    
    /// Nested module that should stay with its parent
    pub mod nested {{
        use super::ModuleLocalStruct;
        
        pub struct NestedStruct {{
            pub parent_ref: Option<ModuleLocalStruct>,
            pub nested_data: Vec<i32>,
        }}
        
        impl NestedStruct {{
            pub fn with_parent(parent: ModuleLocalStruct) -> Self {{
                Self {{
                    parent_ref: Some(parent),
                    nested_data: Vec::new(),
                }}
            }}
            
            pub fn add_data(&mut self, value: i32) {{
                self.nested_data.push(value);
            }}
        }}
        
        /// Function that bridges nested and parent module
        pub fn bridge_function(nested: &NestedStruct) -> Option<String> {{
            nested.parent_ref.as_ref().map(|p| p.data.clone())
        }}
    }}
    
    /// Module constants that belong with the module
    pub const MODULE_VERSION: u32 = 1;
    pub const MAX_REFERENCES: usize = 1000;
    
    /// Module-level type aliases
    pub type ModuleResult<T> = Result<T, ModuleError>;
    pub type ReferenceMap = HashMap<String, Vec<String>>;
    
    /// Module-specific error type
    #[derive(Debug, thiserror::Error)]
    pub enum ModuleError {{
        #[error("Reference limit exceeded: {{0}} > {{1}}")]
        ReferenceLimitExceeded(usize, usize),
        
        #[error("Invalid module state: {{0}}")]
        InvalidState(String),
    }}
}}

/// Second module to test module boundaries
pub mod another_boundary_module {{
    /// This module should be separate from the first one
    /// but its contents should stay together
    
    pub fn standalone_function() -> &'static str {{
        "This function belongs to the second module"
    }}
    
    pub struct SecondModuleStruct {{
        pub value: i32,
    }}
    
    impl SecondModuleStruct {{
        pub fn new(value: i32) -> Self {{
            Self {{ value }}
        }}
    }}
}}
            "#,
            "// ".repeat(100)
        );
        
        let path = Path::new("test_data/chunk_boundaries").join("module_preservation.rs");
        fs::write(path, module_content)?;
        
        debug!("Created module boundary test files");
        Ok(())
    }
    
    fn generate_generic_boundary_tests() -> Result<()> {
        debug!("Generating generic boundary test files");
        
        let generic_content = format!(
            r#"
{}
/// Generic definitions that must be kept together with their constraints
pub trait ComplexTraitWithManyGenerics<T, U, V, W, X, Y, Z>
where
    T: Clone + Send + Sync + std::fmt::Debug + 'static,
    U: std::hash::Hash + Eq + Clone + Send + Sync + 'static,
    V: Default + Clone + Send + Sync + std::fmt::Display + 'static,
    W: std::error::Error + Send + Sync + 'static,
    X: Iterator<Item = T> + Send + 'static,
    Y: Future<Output = Result<U, W>> + Send + 'static,
    Z: Fn(T) -> Result<V, W> + Send + Sync + 'static,
{{
    type AssociatedType1: Send + Sync + Clone;
    type AssociatedType2<'a>: Send + Sync + 'a where Self: 'a;
    type AssociatedType3<K>: Send + Sync + From<K> where K: Send + Sync;
    
    /// Method with complex generic constraints
    async fn complex_method<A, B, C>(
        &self,
        input1: A,
        input2: B,
        transformer: C,
    ) -> Result<Self::AssociatedType1, W>
    where
        A: IntoIterator<Item = T> + Send + 'static,
        B: Stream<Item = U> + Send + Unpin + 'static,
        C: Fn(T, U) -> Self::AssociatedType1 + Send + Sync + 'static;
    
    /// Another method with different constraints
    fn transform<M, N>(&self, mapper: M) -> N
    where
        M: Fn(&Self::AssociatedType1) -> N + Send + Sync,
        N: Send + Sync + 'static;
}}

/// Implementation that must stay with the trait
pub struct ConcreteImplementation<T, U> 
where
    T: Clone + Send + Sync + std::fmt::Debug + 'static,
    U: std::hash::Hash + Eq + Clone + Send + Sync + 'static,
{{
    pub data: HashMap<U, Vec<T>>,
    pub processor: Box<dyn Fn(&T) -> String + Send + Sync>,
    pub validator: Box<dyn Fn(&U) -> bool + Send + Sync>,
}}

/// Very complex implementation block
impl<T, U, V, W, X, Y, Z> ComplexTraitWithManyGenerics<T, U, V, W, X, Y, Z> 
for ConcreteImplementation<T, U>
where
    T: Clone + Send + Sync + std::fmt::Debug + 'static,
    U: std::hash::Hash + Eq + Clone + Send + Sync + 'static,
    V: Default + Clone + Send + Sync + std::fmt::Display + 'static,
    W: std::error::Error + Send + Sync + 'static,
    X: Iterator<Item = T> + Send + 'static,
    Y: Future<Output = Result<U, W>> + Send + 'static,
    Z: Fn(T) -> Result<V, W> + Send + Sync + 'static,
{{
    type AssociatedType1 = V;
    type AssociatedType2<'a> = &'a str where Self: 'a;
    type AssociatedType3<K> = HashMap<K, V> where K: Send + Sync;
    
    async fn complex_method<A, B, C>(
        &self,
        input1: A,
        input2: B,
        transformer: C,
    ) -> Result<Self::AssociatedType1, W>
    where
        A: IntoIterator<Item = T> + Send + 'static,
        B: Stream<Item = U> + Send + Unpin + 'static,
        C: Fn(T, U) -> Self::AssociatedType1 + Send + Sync + 'static,
    {{
        // Implementation that must stay with the method signature
        let mut result = V::default();
        
        // Process input1
        for item in input1 {{
            println!("Processing: {{:?}}", item);
        }}
        
        // This complex implementation belongs together
        tokio::time::sleep(tokio::time::Duration::from_millis(1)).await;
        
        Ok(result)
    }}
    
    fn transform<M, N>(&self, mapper: M) -> N
    where
        M: Fn(&Self::AssociatedType1) -> N + Send + Sync,
        N: Send + Sync + 'static,
    {{
        let default_value = V::default();
        mapper(&default_value)
    }}
}}
            "#,
            "// ".repeat(150)
        );
        
        let path = Path::new("test_data/chunk_boundaries").join("generic_preservation.rs");
        fs::write(path, generic_content)?;
        
        debug!("Created generic boundary test files");
        Ok(())
    }
    
    fn generate_macro_boundary_tests() -> Result<()> {
        debug!("Generating macro boundary test files");
        
        let macro_content = format!(
            r#"
{}
/// Macro definitions that must be kept as complete units
macro_rules! complex_macro_at_boundary {{
    // First arm - simple case
    ($name:ident) => {{
        pub struct $name {{
            pub value: i32,
        }}
        
        impl $name {{
            pub fn new() -> Self {{
                Self {{ value: 0 }}
            }}
        }}
    }};
    
    // Second arm - with type parameter
    ($name:ident, $type:ty) => {{
        pub struct $name {{
            pub value: $type,
        }}
        
        impl $name {{
            pub fn new(value: $type) -> Self {{
                Self {{ value }}
            }}
            
            pub fn get_value(&self) -> &$type {{
                &self.value
            }}
        }}
    }};
    
    // Third arm - complex case with multiple parameters
    ($name:ident, $type:ty, $($field:ident: $field_type:ty),+) => {{
        pub struct $name {{
            pub value: $type,
            $(pub $field: $field_type,)+
        }}
        
        impl $name {{
            pub fn new(value: $type, $($field: $field_type,)+) -> Self {{
                Self {{
                    value,
                    $($field,)+
                }}
            }}
            
            $(
                paste::paste! {{
                    pub fn [<get_ $field>](&self) -> &$field_type {{
                        &self.$field
                    }}
                    
                    pub fn [<set_ $field>](&mut self, value: $field_type) {{
                        self.$field = value;
                    }}
                }}
            )+
        }}
    }};
}}

/// Usage of the macro that should stay together
complex_macro_at_boundary!(SimpleStruct);
complex_macro_at_boundary!(TypedStruct, String);
complex_macro_at_boundary!(ComplexStruct, i32, name: String, active: bool);

/// Procedural macro attribute definition
#[proc_macro_attribute]
pub fn boundary_test(args: TokenStream, input: TokenStream) -> TokenStream {{
    // The entire macro implementation must stay together
    let args_str = args.to_string();
    let input_str = input.to_string();
    
    let output = if args_str.contains("debug") {{
        format!(
            r#"
            #[derive(Debug)]
            {{}}
            
            impl std::fmt::Display for {{}} {{
                fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {{
                    write!(f, "BoundaryTest: {{:?}}", self)
                }}
            }}
            "#,
            input_str, 
            extract_struct_name(&input_str).unwrap_or("Unknown".to_string())
        )
    }} else {{
        input_str
    }};
    
    output.parse().unwrap()
}}

/// Helper function for the macro
fn extract_struct_name(input: &str) -> Option<String> {{
    // This helper belongs with the macro
    input.lines()
        .find(|line| line.trim_start().starts_with("struct"))
        .and_then(|line| {{
            line.split_whitespace()
                .nth(1)
                .map(|name| name.trim_end_matches('{{').to_string())
        }})
}}

/// Function-like macro
macro_rules! generate_test_functions {{
    ($($name:ident => $value:expr),* $(,)?) => {{
        $(
            pub fn $name() -> i32 {{
                println!("Function {{}}: {{}}", stringify!($name), $value);
                $value
            }}
        )*
        
        pub fn all_test_functions() -> Vec<(&'static str, i32)> {{
            vec![
                $((stringify!($name), $name()),)*
            ]
        }}
    }};
}}

/// Usage that demonstrates the macro must stay together
generate_test_functions! {{
    test_one => 1,
    test_two => 2,
    test_three => 3,
    boundary_test_function => 42,
}}

/// Declarative macro with complex pattern matching
macro_rules! state_machine {{
    (
        $name:ident {{
            states: [ $($state:ident),* $(,)? ]
            transitions: [
                $($from:ident -> $to:ident on $event:ident),* $(,)?
            ]
        }}
    ) => {{
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub enum [<$name State>] {{
            $($state,)*
        }}
        
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub enum [<$name Event>] {{
            $($event,)*
        }}
        
        pub struct $name {{
            current_state: [<$name State>],
        }}
        
        impl $name {{
            pub fn new() -> Self {{
                Self {{
                    current_state: [<$name State>]::$($state)*, // First state
                }}
            }}
            
            pub fn transition(&mut self, event: [<$name Event>]) -> bool {{
                match (self.current_state, event) {{
                    $(([<$name State>]::$from, [<$name Event>]::$event) => {{
                        self.current_state = [<$name State>]::$to;
                        true
                    }},)*
                    _ => false,
                }}
            }}
            
            pub fn current_state(&self) -> [<$name State>] {{
                self.current_state
            }}
        }}
    }};
}}

/// Complex state machine usage
state_machine! {{
    TrafficLight {{
        states: [Red, Yellow, Green]
        transitions: [
            Red -> Green on Go,
            Green -> Yellow on Caution,
            Yellow -> Red on Stop
        ]
    }}
}}
            "#,
            "// ".repeat(100)
        );
        
        let path = Path::new("test_data/chunk_boundaries").join("macro_preservation.rs");
        fs::write(path, macro_content)?;
        
        debug!("Created macro boundary test files");
        Ok(())
    }
    
    fn generate_comment_boundary_tests() -> Result<()> {
        debug!("Generating comment boundary test files");
        
        let comment_content = format!(
            r#"
{}
/*!
 * This is a large module-level documentation comment that should be kept
 * together as a unit. It provides important context about the module's
 * purpose, usage, and implementation details.
 * 
 * # Overview
 * 
 * This module demonstrates how documentation comments should be preserved
 * across chunk boundaries. The documentation is an integral part of the
 * code and should not be separated from the items it documents.
 * 
 * # Examples
 * 
 * ```rust
 * use crate::BoundaryTestStruct;
 * 
 * let instance = BoundaryTestStruct::new("test");
 * assert_eq!(instance.get_name(), "test");
 * ```
 * 
 * # Implementation Notes
 * 
 * The implementation uses several advanced Rust features:
 * - Generic type parameters with complex bounds
 * - Associated types and lifetimes
 * - Async/await functionality
 * - Error handling with Result types
 * 
 * # Safety
 * 
 * This code is safe and does not use any unsafe blocks. All memory
 * management is handled by Rust's ownership system.
 */

/// This struct documentation should stay with the struct definition
/// 
/// The struct represents a boundary test case where the documentation
/// is extensive and must be preserved as part of the semantic unit.
/// 
/// # Type Parameters
/// 
/// - `T`: The type of data stored in the struct
/// - `E`: The error type used in operations
/// 
/// # Examples
/// 
/// ```rust
/// let mut instance = BoundaryTestStruct::<String, std::io::Error>::new();
/// instance.set_data("Hello, world!".to_string())?;
/// ```
#[derive(Debug, Clone)]
pub struct BoundaryTestStruct<T, E> 
where
    T: Clone + Send + Sync + std::fmt::Debug + 'static,
    E: std::error::Error + Send + Sync + 'static,
{{
    /// The main data field
    /// 
    /// This field stores the primary data for the struct and is accessed
    /// through the getter and setter methods defined in the implementation.
    data: Option<T>,
    
    /// Error history
    /// 
    /// This field maintains a history of errors encountered during
    /// operations on this struct instance.
    errors: Vec<E>,
    
    /// Creation timestamp
    /// 
    /// Records when this struct instance was created, useful for
    /// debugging and logging purposes.
    created_at: std::time::SystemTime,
}}

impl<T, E> BoundaryTestStruct<T, E>
where
    T: Clone + Send + Sync + std::fmt::Debug + 'static,
    E: std::error::Error + Send + Sync + 'static,
{{
    /// Creates a new instance
    /// 
    /// This constructor initializes all fields to their default values
    /// and records the creation timestamp.
    /// 
    /// # Returns
    /// 
    /// A new `BoundaryTestStruct` instance with empty data and error history.
    /// 
    /// # Examples
    /// 
    /// ```rust
    /// let instance = BoundaryTestStruct::<String, std::io::Error>::new();
    /// assert!(instance.get_data().is_none());
    /// ```
    pub fn new() -> Self {{
        Self {{
            data: None,
            errors: Vec::new(),
            created_at: std::time::SystemTime::now(),
        }}
    }}
    
    /// Sets the data field
    /// 
    /// This method updates the data field with the provided value.
    /// If an error occurs during validation, it is recorded in the
    /// error history.
    /// 
    /// # Parameters
    /// 
    /// - `data`: The new data value to store
    /// 
    /// # Returns
    /// 
    /// `Ok(())` if the operation succeeds, or an error if validation fails.
    /// 
    /// # Examples
    /// 
    /// ```rust
    /// let mut instance = BoundaryTestStruct::new();
    /// instance.set_data("test data".to_string())?;
    /// ```
    pub fn set_data(&mut self, data: T) -> Result<(), &'static str> {{
        // Validation logic would go here
        self.data = Some(data);
        Ok(())
    }}
    
    /// Gets a reference to the data
    /// 
    /// Returns an optional reference to the stored data. The reference
    /// is valid for the lifetime of the struct instance.
    /// 
    /// # Returns
    /// 
    /// `Some(&T)` if data is present, `None` otherwise.
    /// 
    /// # Examples
    /// 
    /// ```rust
    /// let instance = BoundaryTestStruct::new();
    /// assert!(instance.get_data().is_none());
    /// ```
    pub fn get_data(&self) -> Option<&T> {{
        self.data.as_ref()
    }}
}}

/// Free function with extensive documentation
/// 
/// This function demonstrates how standalone function documentation
/// should be preserved with the function definition across chunk boundaries.
/// 
/// The function performs a complex operation that requires detailed
/// explanation of its behavior, parameters, and return values.
/// 
/// # Parameters
/// 
/// - `input`: The input string to process
/// - `options`: Configuration options for processing
/// 
/// # Returns
/// 
/// A `Result` containing the processed string or an error description.
/// 
/// # Errors
/// 
/// This function returns an error if:
/// - The input string is empty
/// - The input contains invalid characters
/// - Processing fails due to resource constraints
/// 
/// # Examples
/// 
/// ```rust
/// let result = process_with_documentation("hello", &ProcessingOptions::default())?;
/// assert_eq!(result, "HELLO");
/// ```
/// 
/// # Implementation Details
/// 
/// The function uses a multi-stage processing pipeline:
/// 1. Input validation
/// 2. Character transformation
/// 3. Result verification
/// 
/// Each stage can potentially fail, and errors are propagated using
/// the `?` operator for clean error handling.
pub fn process_with_documentation(
    input: &str, 
    options: &ProcessingOptions
) -> Result<String, ProcessingError> {{
    if input.is_empty() {{
        return Err(ProcessingError::EmptyInput);
    }}
    
    // Processing implementation
    let result = input.to_uppercase();
    
    // Validate result
    if result.len() != input.len() {{
        return Err(ProcessingError::ProcessingFailed);
    }}
    
    Ok(result)
}}

/// Supporting types with their own documentation
#[derive(Debug, Default)]
pub struct ProcessingOptions {{
    /// Whether to preserve whitespace
    pub preserve_whitespace: bool,
    /// Maximum output length
    pub max_length: Option<usize>,
}}

/// Error type for processing operations
#[derive(Debug, thiserror::Error)]
pub enum ProcessingError {{
    /// Input was empty
    #[error("Input string cannot be empty")]
    EmptyInput,
    /// Processing failed
    #[error("Failed to process input")]
    ProcessingFailed,
}}
            "#,
            "// ".repeat(50)
        );
        
        let path = Path::new("test_data/chunk_boundaries").join("comment_preservation.rs");
        fs::write(path, comment_content)?;
        
        debug!("Created comment boundary test files");
        Ok(())
    }
    
    fn generate_complex_boundary_tests() -> Result<()> {
        debug!("Generating complex boundary test files");
        
        let complex_content = format!(
            r#"
{}
/// Real-world complex scenario that tests multiple boundary conditions
/// This file combines several challenging cases that could break chunking
pub mod complex_boundary_scenarios {{
    use std::{{
        collections::{{HashMap, HashSet, BTreeMap, VecDeque}},
        sync::{{Arc, Mutex, RwLock}},
        time::{{SystemTime, Duration}},
        future::Future,
        pin::Pin,
        task::{{Context, Poll}},
    }};
    
    /// Complex trait that spans multiple boundaries
    #[async_trait::async_trait]
    pub trait ComplexAsyncRepository<Entity, Key, Error>: Send + Sync
    where
        Entity: Send + Sync + Clone + std::fmt::Debug + 'static,
        Key: Send + Sync + Clone + std::hash::Hash + Eq + std::fmt::Debug + 'static,
        Error: std::error::Error + Send + Sync + 'static,
    {{
        type Connection: Send + Sync;
        type Transaction<'a>: Send + Sync where Self: 'a;
        type QueryBuilder: Send + Sync;
        type ResultSet: Send + Sync;
        
        /// Complex method with many parameters and constraints
        async fn find_with_complex_criteria<'a, F, P, O>(
            &'a self,
            connection: &'a mut Self::Connection,
            filter: F,
            pagination: P,
            ordering: O,
        ) -> Result<Self::ResultSet, Error>
        where
            F: Fn(&Entity) -> bool + Send + Sync + 'a,
            P: PaginationCriteria + Send + Sync + 'a,
            O: OrderingCriteria<Entity> + Send + Sync + 'a;
        
        /// Method that should not be split from the trait
        async fn batch_operation<'a, I, R>(
            &'a self,
            transaction: &'a mut Self::Transaction<'a>,
            operations: I,
        ) -> Result<Vec<R>, Error>
        where
            I: IntoIterator<Item = BatchOperation<Entity, Key>> + Send + 'a,
            R: From<BatchResult<Entity, Key>> + Send + Sync + 'a;
    }}
    
    /// Supporting types that belong with the trait
    pub trait PaginationCriteria {{
        fn offset(&self) -> usize;
        fn limit(&self) -> usize;
        fn has_more(&self) -> bool;
    }}
    
    pub trait OrderingCriteria<T> {{
        fn compare(&self, a: &T, b: &T) -> std::cmp::Ordering;
        fn field_name(&self) -> &str;
        fn is_ascending(&self) -> bool;
    }}
    
    #[derive(Debug, Clone)]
    pub enum BatchOperation<Entity, Key> {{
        Insert(Entity),
        Update {{ key: Key, entity: Entity }},
        Delete(Key),
        Upsert {{ key: Key, entity: Entity }},
    }}
    
    #[derive(Debug, Clone)]
    pub enum BatchResult<Entity, Key> {{
        Inserted(Key),
        Updated {{ key: Key, old_entity: Entity, new_entity: Entity }},
        Deleted {{ key: Key, entity: Entity }},
        Upserted {{ key: Key, entity: Entity, was_insert: bool }},
        Failed {{ operation: BatchOperation<Entity, Key>, error: String }},
    }}
    
    /// Implementation that demonstrates complex boundary preservation
    pub struct DatabaseRepository<Entity, Key> 
    where
        Entity: Send + Sync + Clone + std::fmt::Debug + 'static,
        Key: Send + Sync + Clone + std::hash::Hash + Eq + std::fmt::Debug + 'static,
    {{
        connection_pool: Arc<RwLock<Vec<DatabaseConnection>>>,
        cache: Arc<Mutex<HashMap<Key, (Entity, SystemTime)>>>,
        metrics: Arc<Mutex<RepositoryMetrics>>,
        config: RepositoryConfig,
        _phantom: std::marker::PhantomData<(Entity, Key)>,
    }}
    
    /// Complex implementation block that must stay together
    impl<Entity, Key> DatabaseRepository<Entity, Key>
    where
        Entity: Send + Sync + Clone + std::fmt::Debug + 'static,
        Key: Send + Sync + Clone + std::hash::Hash + Eq + std::fmt::Debug + 'static,
    {{
        pub fn new(config: RepositoryConfig) -> Self {{
            Self {{
                connection_pool: Arc::new(RwLock::new(Vec::new())),
                cache: Arc::new(Mutex::new(HashMap::new())),
                metrics: Arc::new(Mutex::new(RepositoryMetrics::default())),
                config,
                _phantom: std::marker::PhantomData,
            }}
        }}
        
        /// Complex method with intricate logic
        pub async fn initialize_connections(&self) -> Result<(), RepositoryError> {{
            let mut pool = self.connection_pool.write().unwrap();
            
            for i in 0..self.config.max_connections {{
                let connection = DatabaseConnection::new(
                    &self.config.connection_string,
                    Duration::from_secs(self.config.timeout_seconds),
                ).await?;
                
                // Test the connection
                connection.ping().await?;
                
                pool.push(connection);
                
                // Update metrics
                {{
                    let mut metrics = self.metrics.lock().unwrap();
                    metrics.connections_created += 1;
                    metrics.last_connection_time = SystemTime::now();
                }}
                
                // Prevent overwhelming the database
                tokio::time::sleep(Duration::from_millis(100)).await;
            }}
            
            Ok(())
        }}
        
        /// Another complex method that belongs with the struct
        pub async fn execute_with_retry<F, R>(&self, operation: F) -> Result<R, RepositoryError>
        where
            F: Fn() -> Pin<Box<dyn Future<Output = Result<R, RepositoryError>> + Send>> + Send + Sync,
            R: Send + 'static,
        {{
            let mut attempts = 0;
            let max_attempts = self.config.max_retry_attempts;
            
            loop {{
                attempts += 1;
                
                match operation().await {{
                    Ok(result) => {{
                        // Update success metrics
                        {{
                            let mut metrics = self.metrics.lock().unwrap();
                            metrics.successful_operations += 1;
                            if attempts > 1 {{
                                metrics.retried_operations += 1;
                            }}
                        }}
                        return Ok(result);
                    }}
                    Err(e) if attempts >= max_attempts => {{
                        // Update failure metrics
                        {{
                            let mut metrics = self.metrics.lock().unwrap();
                            metrics.failed_operations += 1;
                        }}
                        return Err(e);
                    }}
                    Err(_) => {{
                        // Wait before retry with exponential backoff
                        let delay = Duration::from_millis(100 * 2_u64.pow(attempts - 1));
                        tokio::time::sleep(delay).await;
                    }}
                }}
            }}
        }}
    }}
    
    /// Supporting structures that belong with the repository
    #[derive(Debug, Clone)]
    pub struct RepositoryConfig {{
        pub connection_string: String,
        pub max_connections: usize,
        pub timeout_seconds: u64,
        pub max_retry_attempts: usize,
        pub cache_ttl_seconds: u64,
        pub enable_metrics: bool,
    }}
    
    #[derive(Debug, Default)]
    pub struct RepositoryMetrics {{
        pub connections_created: u64,
        pub successful_operations: u64,
        pub failed_operations: u64,
        pub retried_operations: u64,
        pub cache_hits: u64,
        pub cache_misses: u64,
        pub last_connection_time: SystemSystemTime,
        pub last_operation_time: SystemTime,
    }}
    
    #[derive(Debug)]
    pub struct DatabaseConnection {{
        id: uuid::Uuid,
        created_at: SystemTime,
        last_used: SystemTime,
        is_active: bool,
    }}
    
    impl DatabaseConnection {{
        pub async fn new(connection_string: &str, timeout: Duration) -> Result<Self, RepositoryError> {{
            // Simulate connection creation
            tokio::time::sleep(Duration::from_millis(50)).await;
            
            Ok(Self {{
                id: uuid::Uuid::new_v4(),
                created_at: SystemTime::now(),
                last_used: SystemTime::now(),
                is_active: true,
            }})
        }}
        
        pub async fn ping(&self) -> Result<(), RepositoryError> {{
            // Simulate ping
            tokio::time::sleep(Duration::from_millis(10)).await;
            Ok(())
        }}
    }}
    
    /// Error types that complete the module
    #[derive(Debug, thiserror::Error)]
    pub enum RepositoryError {{
        #[error("Connection failed: {{0}}")]
        ConnectionFailed(String),
        
        #[error("Operation timeout after {{0}} seconds")]
        Timeout(u64),
        
        #[error("Invalid configuration: {{0}}")]
        InvalidConfig(String),
        
        #[error("Database error: {{0}}")]
        DatabaseError(String),
    }}
}}
            "#,
            "// ".repeat(75)
        );
        
        let path = Path::new("test_data/chunk_boundaries").join("complex_boundary_scenario.rs");
        fs::write(path, complex_content)?;
        
        debug!("Created complex boundary test files");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_generate_chunk_boundary_tests() {
        ChunkBoundaryTestGenerator::generate_chunk_boundary_tests().unwrap();
        
        // Verify key files were created
        assert!(Path::new("test_data/chunk_boundaries/function_preservation.rs").exists());
        assert!(Path::new("test_data/chunk_boundaries/struct_preservation.rs").exists());
        assert!(Path::new("test_data/chunk_boundaries/impl_preservation.rs").exists());
        assert!(Path::new("test_data/chunk_boundaries/module_preservation.rs").exists());
        assert!(Path::new("test_data/chunk_boundaries/generic_preservation.rs").exists());
        assert!(Path::new("test_data/chunk_boundaries/macro_preservation.rs").exists());
        assert!(Path::new("test_data/chunk_boundaries/comment_preservation.rs").exists());
        assert!(Path::new("test_data/chunk_boundaries/complex_boundary_scenario.rs").exists());
    }
    
    #[test]
    fn test_boundary_file_structure() {
        ChunkBoundaryTestGenerator::generate_chunk_boundary_tests().unwrap();
        
        // Check that files contain expected boundary test patterns
        let function_content = std::fs::read_to_string(
            "test_data/chunk_boundaries/function_preservation.rs"
        ).unwrap();
        
        assert!(function_content.contains("important_function_with_complex_signature"));
        assert!(function_content.contains("where"));
        assert!(function_content.contains("Result<HashMap<String, T>"));
    }
}
```

## Implementation Steps
1. Add ChunkBoundaryTestGenerator struct to test_data.rs
2. Implement generate_function_boundary_tests() for function preservation
3. Implement generate_struct_boundary_tests() for struct/enum preservation
4. Implement generate_impl_boundary_tests() for implementation blocks
5. Implement generate_module_boundary_tests() for module coherence
6. Implement generate_generic_boundary_tests() for complex generics
7. Implement generate_macro_boundary_tests() for macro definitions
8. Implement generate_comment_boundary_tests() for documentation preservation
9. Implement generate_complex_boundary_tests() for real-world scenarios

## Success Criteria
- [ ] ChunkBoundaryTestGenerator struct implemented and compiling
- [ ] Function boundary tests preserve complete function definitions
- [ ] Struct boundary tests keep struct definitions with their implementations
- [ ] Impl boundary tests preserve implementation blocks as units
- [ ] Module boundary tests maintain module coherence
- [ ] Generic boundary tests handle complex type constraints
- [ ] Macro boundary tests keep macro definitions complete
- [ ] Comment boundary tests preserve documentation with code
- [ ] Complex scenarios test multiple boundary conditions together

## Test Command
```bash
cargo test test_generate_chunk_boundary_tests
cargo test test_boundary_file_structure
ls test_data/chunk_boundaries/
```

## Generated Files
After completion, these files should exist:
- `function_preservation.rs` - Function boundary tests
- `multiple_functions.rs` - Multiple function boundaries
- `async_function_boundary.rs` - Async function preservation
- `struct_preservation.rs` - Struct definition preservation
- `enum_preservation.rs` - Enum definition preservation
- `impl_preservation.rs` - Implementation block preservation
- `module_preservation.rs` - Module boundary preservation
- `generic_preservation.rs` - Complex generic preservation
- `macro_preservation.rs` - Macro definition preservation
- `comment_preservation.rs` - Documentation preservation
- `complex_boundary_scenario.rs` - Real-world complex scenario

## Validation Points
These files test that chunking preserves:
- Function signatures with their implementations
- Struct definitions with their fields and constraints
- Implementation blocks with all their methods
- Module contents as coherent units
- Generic type parameters with their bounds
- Macro definitions with all their arms
- Documentation comments with their associated items

## Time Estimate
10 minutes

## Next Task
Task 13: Create ground truth validation data structure for testing accuracy.