# Task 029: Generate Synthetic Rust Code Test Data

## Context
You are implementing a comprehensive validation system for a Rust-based vector indexing system. This builds on Tasks 010-028. Synthetic Rust code generation is essential for creating realistic test scenarios that mirror real-world usage patterns across different Rust codebases and styles.

## Project Structure
```
src/
  validation/
    test_data.rs  <- Extend this file
  lib.rs
```

## Task Description
Implement the `generate_synthetic_rust_tests()` method that creates diverse, realistic Rust code files with varying complexity levels, different architectural patterns, and comprehensive search targets.

## Requirements
1. Add to existing `src/validation/test_data.rs`
2. Generate Rust files with different complexity levels (simple, medium, complex)
3. Include various Rust patterns: async/await, generics, macros, traits, modules
4. Create files with different architectural styles (procedural, OOP-style, functional)
5. Include realistic dependencies and imports
6. Add comprehensive test cases with expected search matches
7. Generate files that test Rust-specific syntax patterns

## Expected Code Structure to Add
```rust
impl TestDataGenerator {
    fn generate_synthetic_rust_tests(&self) -> Result<Vec<GeneratedTestFile>> {
        let mut files = Vec::new();
        
        // Simple Rust file - basic structs and functions
        let simple_rust = self.generate_simple_rust_file()?;
        let mut simple_file = self.create_test_file("synthetic_simple.rs", &simple_rust, TestFileType::RustCode)?;
        simple_file.expected_matches = vec![
            "pub struct".to_string(),
            "impl".to_string(),
            "fn new(".to_string(),
            "Result<".to_string(),
            "#[derive(".to_string(),
        ];
        files.push(simple_file);
        
        // Medium complexity - async/await with error handling
        let async_rust = self.generate_async_rust_file()?;
        let mut async_file = self.create_test_file("synthetic_async.rs", &async_rust, TestFileType::RustCode)?;
        async_file.expected_matches = vec![
            "async fn".to_string(),
            ".await".to_string(),
            "tokio::".to_string(),
            "Arc<Mutex<".to_string(),
            "match result".to_string(),
        ];
        files.push(async_file);
        
        // Complex generic and trait-heavy code
        let generic_rust = self.generate_generic_rust_file()?;
        let mut generic_file = self.create_test_file("synthetic_generics.rs", &generic_rust, TestFileType::RustCode)?;
        generic_file.expected_matches = vec![
            "trait ".to_string(),
            "<T: ".to_string(),
            "where".to_string(),
            "Box<dyn".to_string(),
            "PhantomData".to_string(),
        ];
        files.push(generic_file);
        
        // Macro-heavy file
        let macro_rust = self.generate_macro_rust_file()?;
        let mut macro_file = self.create_test_file("synthetic_macros.rs", &macro_rust, TestFileType::RustCode)?;
        macro_file.expected_matches = vec![
            "macro_rules!".to_string(),
            "($($".to_string(),
            "stringify!".to_string(),
            "vec![".to_string(),
            "#[cfg(".to_string(),
        ];
        files.push(macro_file);
        
        // Module system and visibility
        let module_rust = self.generate_module_rust_file()?;
        let mut module_file = self.create_test_file("synthetic_modules.rs", &module_rust, TestFileType::RustCode)?;
        module_file.expected_matches = vec![
            "pub mod".to_string(),
            "use crate::".to_string(),
            "pub(crate)".to_string(),
            "super::".to_string(),
            "self::".to_string(),
        ];
        files.push(module_file);
        
        // Error handling patterns
        let error_rust = self.generate_error_handling_rust_file()?;
        let mut error_file = self.create_test_file("synthetic_errors.rs", &error_rust, TestFileType::RustCode)?;
        error_file.expected_matches = vec![
            "anyhow::Result".to_string(),
            ".context(".to_string(),
            "thiserror::Error".to_string(),
            "match error".to_string(),
            "bail!(".to_string(),
        ];
        files.push(error_file);
        
        Ok(files)
    }
    
    /// Generate simple Rust file with basic patterns
    fn generate_simple_rust_file(&self) -> Result<String> {
        Ok(r#"//! Simple Rust module demonstrating basic patterns
//! This file contains fundamental Rust constructs for testing

use std::collections::HashMap;
use std::fmt::{self, Display};
use serde::{Deserialize, Serialize};

/// A simple user data structure
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct User {
    pub id: u64,
    pub name: String,
    pub email: String,
    pub active: bool,
}

/// User management system
#[derive(Debug)]
pub struct UserManager {
    users: HashMap<u64, User>,
    next_id: u64,
}

impl UserManager {
    /// Create a new user manager
    pub fn new() -> Self {
        Self {
            users: HashMap::new(),
            next_id: 1,
        }
    }
    
    /// Add a new user to the system
    pub fn add_user(&mut self, name: String, email: String) -> Result<u64, String> {
        if name.is_empty() {
            return Err("Name cannot be empty".to_string());
        }
        
        if email.is_empty() {
            return Err("Email cannot be empty".to_string());
        }
        
        let user = User {
            id: self.next_id,
            name,
            email,
            active: true,
        };
        
        self.users.insert(self.next_id, user);
        let id = self.next_id;
        self.next_id += 1;
        
        Ok(id)
    }
    
    /// Get user by ID
    pub fn get_user(&self, id: u64) -> Option<&User> {
        self.users.get(&id)
    }
    
    /// Update user status
    pub fn set_user_active(&mut self, id: u64, active: bool) -> Result<(), String> {
        match self.users.get_mut(&id) {
            Some(user) => {
                user.active = active;
                Ok(())
            }
            None => Err(format!("User with ID {} not found", id)),
        }
    }
    
    /// Get all active users
    pub fn active_users(&self) -> Vec<&User> {
        self.users.values().filter(|user| user.active).collect()
    }
    
    /// Get user count
    pub fn user_count(&self) -> usize {
        self.users.len()
    }
}

impl Display for User {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "User {} ({}) - {}", self.name, self.email, 
               if self.active { "Active" } else { "Inactive" })
    }
}

impl Default for UserManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_user_creation() {
        let mut manager = UserManager::new();
        let id = manager.add_user("John Doe".to_string(), "john@example.com".to_string()).unwrap();
        
        assert_eq!(id, 1);
        assert_eq!(manager.user_count(), 1);
        
        let user = manager.get_user(id).unwrap();
        assert_eq!(user.name, "John Doe");
        assert_eq!(user.email, "john@example.com");
        assert!(user.active);
    }
    
    #[test]
    fn test_user_status_update() {
        let mut manager = UserManager::new();
        let id = manager.add_user("Jane Smith".to_string(), "jane@example.com".to_string()).unwrap();
        
        manager.set_user_active(id, false).unwrap();
        let user = manager.get_user(id).unwrap();
        assert!(!user.active);
        
        assert_eq!(manager.active_users().len(), 0);
    }
}
"#.to_string())
    }
    
    /// Generate async Rust file with tokio patterns
    fn generate_async_rust_file(&self) -> Result<String> {
        Ok(r#"//! Async Rust patterns for concurrent processing
//! Demonstrates async/await with real-world patterns

use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{Mutex, RwLock, mpsc, oneshot};
use tokio::time::{sleep, timeout};
use anyhow::{Result, Context};
use serde::{Deserialize, Serialize};

/// Async task manager for handling concurrent operations
#[derive(Debug)]
pub struct AsyncTaskManager {
    active_tasks: Arc<RwLock<Vec<String>>>,
    task_results: Arc<Mutex<Vec<TaskResult>>>,
    shutdown_sender: Option<oneshot::Sender<()>>,
}

/// Result of an async task execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskResult {
    pub task_id: String,
    pub duration_ms: u64,
    pub success: bool,
    pub data: Option<String>,
    pub error: Option<String>,
}

/// Configuration for async operations
#[derive(Debug, Clone)]
pub struct AsyncConfig {
    pub max_concurrent_tasks: usize,
    pub task_timeout: Duration,
    pub retry_attempts: u32,
}

impl AsyncTaskManager {
    /// Create a new async task manager
    pub fn new() -> Self {
        Self {
            active_tasks: Arc::new(RwLock::new(Vec::new())),
            task_results: Arc::new(Mutex::new(Vec::new())),
            shutdown_sender: None,
        }
    }
    
    /// Start the task manager with background processing
    pub async fn start(&mut self, config: AsyncConfig) -> Result<()> {
        let (shutdown_tx, mut shutdown_rx) = oneshot::channel();
        self.shutdown_sender = Some(shutdown_tx);
        
        let active_tasks = Arc::clone(&self.active_tasks);
        let task_results = Arc::clone(&self.task_results);
        
        // Background task processor
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_millis(100));
            
            loop {
                tokio::select! {
                    _ = interval.tick() => {
                        Self::process_background_tasks(&active_tasks, &task_results).await;
                    }
                    _ = &mut shutdown_rx => {
                        println!("Task manager shutting down...");
                        break;
                    }
                }
            }
        });
        
        Ok(())
    }
    
    /// Execute a task with timeout and error handling
    pub async fn execute_task(&self, task_id: String, task_fn: impl std::future::Future<Output = Result<String>>) -> Result<TaskResult> {
        // Add to active tasks
        {
            let mut active = self.active_tasks.write().await;
            active.push(task_id.clone());
        }
        
        let start_time = std::time::Instant::now();
        
        // Execute with timeout
        let result = match timeout(Duration::from_secs(30), task_fn).await {
            Ok(task_result) => task_result,
            Err(_) => Err(anyhow::anyhow!("Task timed out")),
        };
        
        let duration = start_time.elapsed();
        
        // Remove from active tasks
        {
            let mut active = self.active_tasks.write().await;
            active.retain(|id| id != &task_id);
        }
        
        // Create result
        let task_result = match result {
            Ok(data) => TaskResult {
                task_id: task_id.clone(),
                duration_ms: duration.as_millis() as u64,
                success: true,
                data: Some(data),
                error: None,
            },
            Err(e) => TaskResult {
                task_id: task_id.clone(),
                duration_ms: duration.as_millis() as u64,
                success: false,
                data: None,
                error: Some(e.to_string()),
            },
        };
        
        // Store result
        {
            let mut results = self.task_results.lock().await;
            results.push(task_result.clone());
        }
        
        Ok(task_result)
    }
    
    /// Execute multiple tasks concurrently
    pub async fn execute_batch(&self, tasks: Vec<(String, Box<dyn std::future::Future<Output = Result<String>> + Send>)>) -> Result<Vec<TaskResult>> {
        let mut handles = Vec::new();
        
        for (task_id, task_future) in tasks {
            let manager = self;
            let handle = tokio::spawn(manager.execute_task(task_id, task_future));
            handles.push(handle);
        }
        
        let mut results = Vec::new();
        for handle in handles {
            match handle.await {
                Ok(result) => match result {
                    Ok(task_result) => results.push(task_result),
                    Err(e) => return Err(e.context("Task execution failed")),
                },
                Err(e) => return Err(anyhow::anyhow!("Task join failed: {}", e)),
            }
        }
        
        Ok(results)
    }
    
    /// Get current active tasks
    pub async fn get_active_tasks(&self) -> Vec<String> {
        let active = self.active_tasks.read().await;
        active.clone()
    }
    
    /// Get task results
    pub async fn get_results(&self) -> Vec<TaskResult> {
        let results = self.task_results.lock().await;
        results.clone()
    }
    
    /// Background processing of tasks
    async fn process_background_tasks(
        active_tasks: &Arc<RwLock<Vec<String>>>,
        _task_results: &Arc<Mutex<Vec<TaskResult>>>,
    ) {
        let active = active_tasks.read().await;
        if !active.is_empty() {
            println!("Processing {} active tasks", active.len());
        }
    }
}

/// Example async function that simulates work
pub async fn simulate_async_work(work_type: &str, duration_ms: u64) -> Result<String> {
    println!("Starting {} work for {}ms", work_type, duration_ms);
    
    sleep(Duration::from_millis(duration_ms)).await;
    
    // Simulate occasional failures
    if duration_ms > 5000 {
        return Err(anyhow::anyhow!("Work took too long and failed"));
    }
    
    Ok(format!("{} work completed successfully", work_type))
}

/// Channel-based communication example
pub async fn channel_communication_example() -> Result<()> {
    let (tx, mut rx) = mpsc::channel::<String>(100);
    
    // Producer task
    let producer = tokio::spawn(async move {
        for i in 0..10 {
            let message = format!("Message {}", i);
            if tx.send(message).await.is_err() {
                break;
            }
            sleep(Duration::from_millis(100)).await;
        }
    });
    
    // Consumer task
    let consumer = tokio::spawn(async move {
        let mut messages = Vec::new();
        while let Some(message) = rx.recv().await {
            messages.push(message);
        }
        messages
    });
    
    // Wait for both tasks
    let (_, received_messages) = tokio::try_join!(producer, consumer)?;
    
    println!("Received {} messages", received_messages.len());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_async_task_execution() {
        let manager = AsyncTaskManager::new();
        
        let result = manager.execute_task(
            "test_task".to_string(),
            simulate_async_work("test", 100)
        ).await.unwrap();
        
        assert!(result.success);
        assert!(result.duration_ms >= 100);
        assert_eq!(result.task_id, "test_task");
    }
    
    #[tokio::test]
    async fn test_task_timeout() {
        let manager = AsyncTaskManager::new();
        
        let result = manager.execute_task(
            "timeout_task".to_string(),
            simulate_async_work("slow", 6000)
        ).await.unwrap();
        
        assert!(!result.success);
        assert!(result.error.is_some());
    }
}
"#.to_string())
    }
    
    /// Generate generic and trait-heavy Rust code
    fn generate_generic_rust_file(&self) -> Result<String> {
        Ok(r#"//! Advanced generics and trait system demonstrations
//! Complex type system features for comprehensive testing

use std::marker::PhantomData;
use std::ops::{Add, Mul};
use std::fmt::{Debug, Display};
use serde::{Deserialize, Serialize};

/// Generic trait for processable items
pub trait Processable<T> {
    type Output;
    type Error;
    
    fn process(&self, input: T) -> Result<Self::Output, Self::Error>;
    fn batch_process(&self, inputs: Vec<T>) -> Vec<Result<Self::Output, Self::Error>> {
        inputs.into_iter().map(|input| self.process(input)).collect()
    }
}

/// Generic container with type constraints
#[derive(Debug, Clone)]
pub struct Container<T, U> 
where
    T: Clone + Debug,
    U: Display + Send,
{
    data: Vec<T>,
    metadata: U,
    _phantom: PhantomData<(T, U)>,
}

impl<T, U> Container<T, U>
where
    T: Clone + Debug + Default,
    U: Display + Send + Clone,
{
    pub fn new(metadata: U) -> Self {
        Self {
            data: Vec::new(),
            metadata,
            _phantom: PhantomData,
        }
    }
    
    pub fn add(&mut self, item: T) {
        self.data.push(item);
    }
    
    pub fn get_all(&self) -> &[T] {
        &self.data
    }
    
    pub fn transform<V, F>(&self, f: F) -> Container<V, U>
    where
        V: Clone + Debug + Default,
        F: Fn(&T) -> V,
    {
        let transformed_data = self.data.iter().map(f).collect();
        Container {
            data: transformed_data,
            metadata: self.metadata.clone(),
            _phantom: PhantomData,
        }
    }
}

/// Higher-kinded type simulation with associated types
pub trait Functor<A> {
    type Wrapped<B>;
    
    fn map<B, F>(self, f: F) -> Self::Wrapped<B>
    where
        F: FnOnce(A) -> B;
}

/// Complex trait with multiple type parameters and bounds
pub trait ComplexProcessor<Input, Output, Context>
where
    Input: Clone + Send,
    Output: Debug + Send,
    Context: Display,
{
    type Intermediate: Debug;
    type ProcessingError: std::error::Error;
    
    fn preprocess(&self, input: Input, context: &Context) -> Result<Self::Intermediate, Self::ProcessingError>;
    fn process_intermediate(&self, intermediate: Self::Intermediate) -> Result<Output, Self::ProcessingError>;
    
    fn full_process(&self, input: Input, context: &Context) -> Result<Output, Self::ProcessingError> {
        let intermediate = self.preprocess(input, context)?;
        self.process_intermediate(intermediate)
    }
}

/// Generic mathematical operations with trait bounds
#[derive(Debug, Clone)]
pub struct MathProcessor<T>
where
    T: Add<Output = T> + Mul<Output = T> + Copy + Debug,
{
    multiplier: T,
    addend: T,
}

impl<T> MathProcessor<T>
where
    T: Add<Output = T> + Mul<Output = T> + Copy + Debug,
{
    pub fn new(multiplier: T, addend: T) -> Self {
        Self { multiplier, addend }
    }
    
    pub fn process(&self, value: T) -> T {
        value * self.multiplier + self.addend
    }
    
    pub fn process_batch(&self, values: &[T]) -> Vec<T> {
        values.iter().map(|&v| self.process(v)).collect()
    }
}

/// Dynamic trait objects
pub trait DynamicProcessor: Send + Sync + Debug {
    fn process_dynamic(&self, input: &str) -> Box<dyn std::any::Any>;
    fn name(&self) -> &'static str;
}

/// Storage for dynamic processors
pub struct ProcessorRegistry {
    processors: Vec<Box<dyn DynamicProcessor>>,
}

impl ProcessorRegistry {
    pub fn new() -> Self {
        Self {
            processors: Vec::new(),
        }
    }
    
    pub fn register<P>(&mut self, processor: P)
    where
        P: DynamicProcessor + 'static,
    {
        self.processors.push(Box::new(processor));
    }
    
    pub fn process_all(&self, input: &str) -> Vec<Box<dyn std::any::Any>> {
        self.processors
            .iter()
            .map(|p| p.process_dynamic(input))
            .collect()
    }
    
    pub fn list_processors(&self) -> Vec<&'static str> {
        self.processors.iter().map(|p| p.name()).collect()
    }
}

/// Concrete implementation of complex traits
#[derive(Debug)]
pub struct StringProcessor {
    prefix: String,
    suffix: String,
}

impl StringProcessor {
    pub fn new(prefix: String, suffix: String) -> Self {
        Self { prefix, suffix }
    }
}

impl Processable<String> for StringProcessor {
    type Output = String;
    type Error = String;
    
    fn process(&self, input: String) -> Result<Self::Output, Self::Error> {
        if input.is_empty() {
            return Err("Input cannot be empty".to_string());
        }
        
        Ok(format!("{}{}{}", self.prefix, input, self.suffix))
    }
}

impl DynamicProcessor for StringProcessor {
    fn process_dynamic(&self, input: &str) -> Box<dyn std::any::Any> {
        let result = self.process(input.to_string()).unwrap_or_else(|e| e);
        Box::new(result)
    }
    
    fn name(&self) -> &'static str {
        "StringProcessor"
    }
}

/// Generic builder pattern
#[derive(Debug, Default)]
pub struct Builder<T> {
    value: Option<T>,
    validators: Vec<Box<dyn Fn(&T) -> bool>>,
}

impl<T> Builder<T>
where
    T: Debug + Clone,
{
    pub fn new() -> Self {
        Self {
            value: None,
            validators: Vec::new(),
        }
    }
    
    pub fn with_value(mut self, value: T) -> Self {
        self.value = Some(value);
        self
    }
    
    pub fn with_validator<F>(mut self, validator: F) -> Self
    where
        F: Fn(&T) -> bool + 'static,
    {
        self.validators.push(Box::new(validator));
        self
    }
    
    pub fn build(self) -> Result<T, &'static str> {
        let value = self.value.ok_or("No value provided")?;
        
        for validator in &self.validators {
            if !validator(&value) {
                return Err("Validation failed");
            }
        }
        
        Ok(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_generic_container() {
        let mut container: Container<i32, String> = Container::new("Test metadata".to_string());
        container.add(42);
        container.add(100);
        
        assert_eq!(container.get_all().len(), 2);
        
        let string_container = container.transform(|&x| x.to_string());
        assert_eq!(string_container.get_all()[0], "42");
    }
    
    #[test]
    fn test_math_processor() {
        let processor = MathProcessor::new(2, 5);
        assert_eq!(processor.process(10), 25); // 10 * 2 + 5
        
        let batch_result = processor.process_batch(&[1, 2, 3]);
        assert_eq!(batch_result, vec![7, 9, 11]); // (1*2+5), (2*2+5), (3*2+5)
    }
    
    #[test]
    fn test_dynamic_processor() {
        let mut registry = ProcessorRegistry::new();
        let processor = StringProcessor::new("[".to_string(), "]".to_string());
        
        registry.register(processor);
        let results = registry.process_all("test");
        
        assert_eq!(results.len(), 1);
        assert_eq!(registry.list_processors(), vec!["StringProcessor"]);
    }
}
"#.to_string())
    }
    
    /// Generate macro-heavy Rust code
    fn generate_macro_rust_file(&self) -> Result<String> {
        Ok(r#"//! Macro system demonstrations and complex procedural patterns
//! Testing macro_rules!, proc macros, and conditional compilation

use std::collections::HashMap;

/// Declarative macro for creating hash maps
macro_rules! hashmap {
    ($($key:expr => $value:expr),* $(,)?) => {
        {
            let mut map = HashMap::new();
            $(
                map.insert($key, $value);
            )*
            map
        }
    };
}

/// Macro for creating test structures
macro_rules! create_test_struct {
    (
        $name:ident {
            $($field:ident: $field_type:ty),*
        }
    ) => {
        #[derive(Debug, Clone, PartialEq)]
        pub struct $name {
            $(pub $field: $field_type,)*
        }
        
        impl $name {
            pub fn new($($field: $field_type,)*) -> Self {
                Self {
                    $($field,)*
                }
            }
        }
    };
}

/// Macro for generating repetitive code patterns
macro_rules! generate_methods {
    ($struct_name:ident, $($method_name:ident -> $return_type:ty),+) => {
        impl $struct_name {
            $(
                pub fn $method_name(&self) -> $return_type {
                    stringify!($method_name).len() as $return_type
                }
            )+
        }
    };
}

/// Complex macro with nested patterns
macro_rules! implement_processor {
    (
        $processor_name:ident<$generic:ident> 
        where $generic:ident: $($bound:tt)+
        {
            process: |$input:ident: $input_type:ty| -> $output_type:ty $process_body:block
            validate: |$validation_input:ident: $validation_type:ty| -> bool $validate_body:block
        }
    ) => {
        pub struct $processor_name<$generic>
        where
            $generic: $($bound)+
        {
            _phantom: std::marker::PhantomData<$generic>,
        }
        
        impl<$generic> $processor_name<$generic>
        where
            $generic: $($bound)+
        {
            pub fn new() -> Self {
                Self {
                    _phantom: std::marker::PhantomData,
                }
            }
            
            pub fn process(&self, $input: $input_type) -> $output_type $process_body
            
            pub fn validate(&self, $validation_input: $validation_type) -> bool $validate_body
        }
    };
}

/// Conditional compilation based on features
#[cfg(feature = "advanced")]
macro_rules! advanced_feature {
    ($($item:item)*) => {
        $(
            #[cfg(feature = "advanced")]
            $item
        )*
    };
}

#[cfg(not(feature = "advanced"))]
macro_rules! advanced_feature {
    ($($item:item)*) => {};
}

/// Create test structures using macros
create_test_struct!(
    TestUser {
        id: u64,
        name: String,
        active: bool
    }
);

create_test_struct!(
    TestProduct {
        sku: String,
        price: f64,
        in_stock: bool
    }
);

/// Generate methods for test structures
generate_methods!(
    TestUser,
    get_id_length -> usize,
    get_name_length -> usize,
    get_status_length -> usize
);

/// Implement processor using complex macro
implement_processor!(
    StringLengthProcessor<T>
    where T: AsRef<str> + Clone
    {
        process: |input: T| -> usize {
            input.as_ref().len()
        }
        validate: |input: T| -> bool {
            !input.as_ref().is_empty()
        }
    }
);

/// Advanced feature code (conditionally compiled)
advanced_feature! {
    pub struct AdvancedProcessor {
        config: HashMap<String, String>,
    }
    
    impl AdvancedProcessor {
        pub fn new() -> Self {
            Self {
                config: hashmap! {
                    "version".to_string() => "1.0.0".to_string(),
                    "mode".to_string() => "advanced".to_string(),
                },
            }
        }
        
        pub fn process_advanced(&self, input: &str) -> String {
            format!("Advanced processing: {}", input)
        }
    }
}

/// Macro for debugging with conditional compilation
macro_rules! debug_print {
    ($($arg:tt)*) => {
        #[cfg(debug_assertions)]
        println!("[DEBUG] {}", format!($($arg)*));
    };
}

/// Macro for creating enum variants
macro_rules! create_status_enum {
    ($name:ident { $($variant:ident),+ }) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub enum $name {
            $($variant,)+
        }
        
        impl $name {
            pub fn all_variants() -> Vec<Self> {
                vec![$(Self::$variant,)+]
            }
            
            pub fn as_str(&self) -> &'static str {
                match self {
                    $(Self::$variant => stringify!($variant),)+
                }
            }
        }
    };
}

create_status_enum!(ProcessingStatus {
    Pending,
    InProgress,
    Completed,
    Failed,
    Cancelled
});

/// Function demonstrating macro usage
pub fn demonstrate_macros() {
    let user_map = hashmap! {
        1 => "Alice",
        2 => "Bob",
        3 => "Charlie",
    };
    
    debug_print!("Created user map with {} entries", user_map.len());
    
    let user = TestUser::new(1, "Alice".to_string(), true);
    debug_print!("Created user: {:?}", user);
    
    let processor = StringLengthProcessor::<String>::new();
    let length = processor.process("Hello, world!".to_string());
    debug_print!("String length: {}", length);
    
    let all_statuses = ProcessingStatus::all_variants();
    debug_print!("All status variants: {:?}", all_statuses);
    
    for status in all_statuses {
        debug_print!("Status: {}", status.as_str());
    }
}

/// Macro for creating simple test cases
macro_rules! test_case {
    ($test_name:ident: $input:expr => $expected:expr) => {
        #[test]
        fn $test_name() {
            let result = $input;
            assert_eq!(result, $expected);
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    
    test_case!(test_simple_math: 2 + 2 => 4);
    test_case!(test_string_length: "hello".len() => 5);
    
    #[test]
    fn test_hashmap_macro() {
        let map = hashmap! {
            "key1" => "value1",
            "key2" => "value2",
        };
        
        assert_eq!(map.len(), 2);
        assert_eq!(map.get("key1"), Some(&"value1"));
    }
    
    #[test]
    fn test_created_struct() {
        let user = TestUser::new(1, "Test User".to_string(), true);
        assert_eq!(user.id, 1);
        assert_eq!(user.name, "Test User");
        assert!(user.active);
    }
    
    #[test]
    fn test_string_processor() {
        let processor = StringLengthProcessor::<String>::new();
        assert_eq!(processor.process("test".to_string()), 4);
        assert!(processor.validate("non-empty".to_string()));
        assert!(!processor.validate("".to_string()));
    }
    
    #[test]
    fn test_status_enum() {
        let status = ProcessingStatus::InProgress;
        assert_eq!(status.as_str(), "InProgress");
        
        let all = ProcessingStatus::all_variants();
        assert!(all.contains(&ProcessingStatus::Pending));
        assert!(all.contains(&ProcessingStatus::Completed));
    }
    
    #[cfg(feature = "advanced")]
    #[test]
    fn test_advanced_processor() {
        let processor = AdvancedProcessor::new();
        let result = processor.process_advanced("test input");
        assert!(result.contains("Advanced processing"));
    }
}
"#.to_string())
    }
    
    /// Generate module system and visibility patterns
    fn generate_module_rust_file(&self) -> Result<String> {
        Ok(r#"//! Module system and visibility patterns
//! Comprehensive demonstration of Rust's module system

use std::collections::BTreeMap;

/// Public module for external API
pub mod api {
    use super::internal;
    
    /// Public API structure
    #[derive(Debug)]
    pub struct ApiClient {
        base_url: String,
        client: internal::HttpClient,
    }
    
    impl ApiClient {
        pub fn new(base_url: String) -> Self {
            Self {
                base_url,
                client: internal::HttpClient::new(),
            }
        }
        
        pub async fn get(&self, endpoint: &str) -> Result<String, String> {
            let url = format!("{}/{}", self.base_url, endpoint);
            self.client.get(&url).await
        }
        
        pub async fn post(&self, endpoint: &str, data: &str) -> Result<String, String> {
            let url = format!("{}/{}", self.base_url, endpoint);
            self.client.post(&url, data).await
        }
    }
    
    /// Re-export commonly used types
    pub use crate::models::{User, Product};
    pub use super::utils::format_response;
}

/// Internal module (crate-visible)
pub(crate) mod internal {
    use std::time::Duration;
    
    /// HTTP client for internal use
    #[derive(Debug)]
    pub(crate) struct HttpClient {
        timeout: Duration,
        headers: std::collections::HashMap<String, String>,
    }
    
    impl HttpClient {
        pub(crate) fn new() -> Self {
            let mut headers = std::collections::HashMap::new();
            headers.insert("User-Agent".to_string(), "RustClient/1.0".to_string());
            
            Self {
                timeout: Duration::from_secs(30),
                headers,
            }
        }
        
        pub(crate) async fn get(&self, url: &str) -> Result<String, String> {
            // Simulate HTTP GET
            tokio::time::sleep(Duration::from_millis(100)).await;
            Ok(format!("GET response from {}", url))
        }
        
        pub(crate) async fn post(&self, url: &str, data: &str) -> Result<String, String> {
            // Simulate HTTP POST
            tokio::time::sleep(Duration::from_millis(150)).await;
            Ok(format!("POST response from {} with data: {}", url, data))
        }
        
        pub(crate) fn set_timeout(&mut self, timeout: Duration) {
            self.timeout = timeout;
        }
    }
}

/// Private module (module-scoped)
mod private {
    /// This struct is only visible within this module
    struct PrivateConfig {
        secret_key: String,
        debug_mode: bool,
    }
    
    impl PrivateConfig {
        fn new(secret_key: String) -> Self {
            Self {
                secret_key,
                debug_mode: false,
            }
        }
        
        fn is_debug(&self) -> bool {
            self.debug_mode
        }
    }
    
    /// Function to create config (visible to parent module)
    pub(super) fn create_config() -> String {
        let config = PrivateConfig::new("secret123".to_string());
        format!("Config created, debug: {}", config.is_debug())
    }
}

/// Models module
pub mod models {
    use serde::{Deserialize, Serialize};
    
    /// User model with various visibility levels
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct User {
        pub id: u64,
        pub username: String,
        pub(crate) email: String,        // Crate-visible
        active: bool,                     // Private
    }
    
    impl User {
        pub fn new(id: u64, username: String, email: String) -> Self {
            Self {
                id,
                username,
                email,
                active: true,
            }
        }
        
        pub fn is_active(&self) -> bool {
            self.active
        }
        
        pub(crate) fn set_active(&mut self, active: bool) {
            self.active = active;
        }
        
        /// This method can access the crate-visible email field
        pub(crate) fn get_email(&self) -> &str {
            &self.email
        }
    }
    
    /// Product model
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct Product {
        pub id: String,
        pub name: String,
        pub price: f64,
        pub(super) category: String,      // Parent module visible
    }
    
    impl Product {
        pub fn new(id: String, name: String, price: f64, category: String) -> Self {
            Self {
                id,
                name,
                price,
                category,
            }
        }
        
        /// Accessor for category (available to parent module)
        pub(super) fn category(&self) -> &str {
            &self.category
        }
    }
}

/// Utilities module with mixed visibility
pub mod utils {
    use crate::models::Product;
    
    /// Public utility function
    pub fn format_response(status: u16, message: &str) -> String {
        format!("HTTP {}: {}", status, message)
    }
    
    /// Crate-visible utility
    pub(crate) fn validate_product(product: &Product) -> bool {
        !product.name.is_empty() && product.price > 0.0
    }
    
    /// Module-private helper
    fn internal_helper(input: &str) -> String {
        format!("Processed: {}", input)
    }
    
    /// Super-visible function that uses private helper
    pub(super) fn process_with_helper(input: &str) -> String {
        internal_helper(input)
    }
}

/// Database module demonstrating nested modules
pub mod database {
    /// Connection module
    pub mod connection {
        use std::sync::Arc;
        use tokio::sync::Mutex;
        
        /// Database connection pool
        #[derive(Debug)]
        pub struct ConnectionPool {
            connections: Arc<Mutex<Vec<Connection>>>,
            max_size: usize,
        }
        
        /// Individual database connection
        #[derive(Debug)]
        pub struct Connection {
            id: u32,
            connected: bool,
        }
        
        impl ConnectionPool {
            pub fn new(max_size: usize) -> Self {
                Self {
                    connections: Arc::new(Mutex::new(Vec::new())),
                    max_size,
                }
            }
            
            pub async fn get_connection(&self) -> Option<Connection> {
                let mut connections = self.connections.lock().await;
                if connections.is_empty() && connections.len() < self.max_size {
                    let new_conn = Connection {
                        id: connections.len() as u32,
                        connected: true,
                    };
                    connections.push(new_conn);
                }
                connections.pop()
            }
            
            pub async fn return_connection(&self, connection: Connection) {
                let mut connections = self.connections.lock().await;
                connections.push(connection);
            }
        }
    }
    
    /// Query module
    pub mod query {
        use super::connection::Connection;
        use crate::models::{User, Product};
        
        /// Query builder
        pub struct QueryBuilder {
            table: String,
            conditions: Vec<String>,
        }
        
        impl QueryBuilder {
            pub fn new(table: &str) -> Self {
                Self {
                    table: table.to_string(),
                    conditions: Vec::new(),
                }
            }
            
            pub fn where_clause(mut self, condition: &str) -> Self {
                self.conditions.push(condition.to_string());
                self
            }
            
            pub fn build(&self) -> String {
                let mut query = format!("SELECT * FROM {}", self.table);
                if !self.conditions.is_empty() {
                    query.push_str(&format!(" WHERE {}", self.conditions.join(" AND ")));
                }
                query
            }
            
            pub async fn execute(self, _connection: &Connection) -> Result<Vec<String>, String> {
                // Simulate query execution
                tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                Ok(vec![self.build()])
            }
        }
        
        /// User-specific queries
        pub mod user_queries {
            use super::*;
            
            pub async fn find_active_users(connection: &Connection) -> Result<Vec<User>, String> {
                let query = QueryBuilder::new("users")
                    .where_clause("active = true")
                    .execute(connection)
                    .await?;
                
                // Simulate returning users
                Ok(vec![
                    User::new(1, "alice".to_string(), "alice@example.com".to_string()),
                    User::new(2, "bob".to_string(), "bob@example.com".to_string()),
                ])
            }
        }
        
        /// Product-specific queries
        pub mod product_queries {
            use super::*;
            
            pub async fn find_products_by_category(
                connection: &Connection,
                category: &str,
            ) -> Result<Vec<Product>, String> {
                let query = QueryBuilder::new("products")
                    .where_clause(&format!("category = '{}'", category))
                    .execute(connection)
                    .await?;
                
                // Simulate returning products
                Ok(vec![
                    Product::new("p1".to_string(), "Product 1".to_string(), 29.99, category.to_string()),
                ])
            }
        }
    }
}

/// Main application logic using all modules
pub async fn demonstrate_module_system() -> Result<(), String> {
    // Use public API
    let client = api::ApiClient::new("https://api.example.com".to_string());
    let response = client.get("users").await?;
    println!("API Response: {}", response);
    
    // Use models
    let user = models::User::new(1, "testuser".to_string(), "test@example.com".to_string());
    println!("Created user: {:?}", user);
    
    let product = models::Product::new(
        "prod1".to_string(),
        "Test Product".to_string(),
        19.99,
        "electronics".to_string(),
    );
    
    // Use utilities (public function)
    let formatted = utils::format_response(200, "Success");
    println!("Formatted response: {}", formatted);
    
    // Use database modules
    let pool = database::connection::ConnectionPool::new(10);
    if let Some(conn) = pool.get_connection().await {
        let users = database::query::user_queries::find_active_users(&conn).await?;
        println!("Found {} active users", users.len());
        
        let products = database::query::product_queries::find_products_by_category(&conn, "electronics").await?;
        println!("Found {} products in electronics", products.len());
        
        pool.return_connection(conn).await;
    }
    
    Ok(())
}

/// Function using private module
fn use_private_config() -> String {
    private::create_config()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_public_api() {
        let client = api::ApiClient::new("https://test.com".to_string());
        // Test basic construction
        assert_eq!(format!("{:?}", client).contains("ApiClient"), true);
    }
    
    #[test]
    fn test_models() {
        let user = models::User::new(1, "test".to_string(), "test@test.com".to_string());
        assert_eq!(user.id, 1);
        assert!(user.is_active());
        
        let product = models::Product::new(
            "p1".to_string(),
            "Product".to_string(),
            10.0,
            "category".to_string(),
        );
        assert_eq!(product.price, 10.0);
    }
    
    #[test]
    fn test_utils() {
        let formatted = utils::format_response(404, "Not Found");
        assert_eq!(formatted, "HTTP 404: Not Found");
    }
    
    #[test]
    fn test_private_config() {
        let config = use_private_config();
        assert!(config.contains("Config created"));
    }
    
    #[test]
    fn test_query_builder() {
        let query = database::query::QueryBuilder::new("users")
            .where_clause("active = true")
            .where_clause("age > 18")
            .build();
        
        assert!(query.contains("SELECT * FROM users"));
        assert!(query.contains("WHERE active = true AND age > 18"));
    }
}
"#.to_string())
    }
    
    /// Generate error handling patterns
    fn generate_error_handling_rust_file(&self) -> Result<String> {
        Ok(r#"//! Error handling patterns and comprehensive error management
//! Demonstrates thiserror, anyhow, and custom error types

use std::fmt;
use std::io;
use std::num::ParseIntError;
use anyhow::{Context, Result, bail, ensure};
use thiserror::Error;
use serde::{Deserialize, Serialize};

/// Custom error types using thiserror
#[derive(Error, Debug)]
pub enum ValidationError {
    #[error("Field '{field}' is required but was empty")]
    EmptyField { field: String },
    
    #[error("Value '{value}' is out of range [{min}, {max}]")]
    OutOfRange { value: i32, min: i32, max: i32 },
    
    #[error("Invalid format for field '{field}': expected {expected}, got '{actual}'")]
    InvalidFormat {
        field: String,
        expected: String,
        actual: String,
    },
    
    #[error("Multiple validation errors occurred")]
    Multiple(#[from] Vec<ValidationError>),
}

/// Database-related errors
#[derive(Error, Debug)]
pub enum DatabaseError {
    #[error("Connection failed: {message}")]
    ConnectionFailed { message: String },
    
    #[error("Query execution failed")]
    QueryFailed(#[from] sqlx::Error),
    
    #[error("Record not found: {table}.{id}")]
    RecordNotFound { table: String, id: String },
    
    #[error("Constraint violation: {constraint}")]
    ConstraintViolation { constraint: String },
    
    #[error("Transaction failed")]
    TransactionFailed(#[source] Box<dyn std::error::Error + Send + Sync>),
}

/// Network-related errors
#[derive(Error, Debug)]
pub enum NetworkError {
    #[error("HTTP request failed with status {status}")]
    HttpError { status: u16 },
    
    #[error("Timeout occurred after {seconds} seconds")]
    Timeout { seconds: u64 },
    
    #[error("DNS resolution failed for host '{host}'")]
    DnsResolution { host: String },
    
    #[error("Network I/O error")]
    IoError(#[from] io::Error),
    
    #[error("JSON parsing error")]
    JsonError(#[from] serde_json::Error),
}

/// Application-level errors that can contain multiple error types
#[derive(Error, Debug)]
pub enum AppError {
    #[error("Validation failed")]
    Validation(#[from] ValidationError),
    
    #[error("Database operation failed")]
    Database(#[from] DatabaseError),
    
    #[error("Network operation failed")]
    Network(#[from] NetworkError),
    
    #[error("Configuration error: {message}")]
    Configuration { message: String },
    
    #[error("Internal error: {message}")]
    Internal { message: String },
}

/// Result type alias for convenience
pub type AppResult<T> = Result<T, AppError>;

/// Error context for tracking error chains
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorContext {
    pub operation: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub context_data: std::collections::HashMap<String, String>,
}

impl ErrorContext {
    pub fn new(operation: &str) -> Self {
        Self {
            operation: operation.to_string(),
            timestamp: chrono::Utc::now(),
            context_data: std::collections::HashMap::new(),
        }
    }
    
    pub fn with_data(mut self, key: &str, value: &str) -> Self {
        self.context_data.insert(key.to_string(), value.to_string());
        self
    }
}

/// Validator for demonstrating error handling patterns
pub struct DataValidator;

impl DataValidator {
    pub fn validate_user_input(
        name: &str,
        age: i32,
        email: &str,
    ) -> Result<(), ValidationError> {
        let mut errors = Vec::new();
        
        // Validate name
        if name.is_empty() {
            errors.push(ValidationError::EmptyField {
                field: "name".to_string(),
            });
        }
        
        // Validate age
        if age < 0 || age > 150 {
            errors.push(ValidationError::OutOfRange {
                value: age,
                min: 0,
                max: 150,
            });
        }
        
        // Validate email format
        if !email.contains('@') {
            errors.push(ValidationError::InvalidFormat {
                field: "email".to_string(),
                expected: "user@domain.com".to_string(),
                actual: email.to_string(),
            });
        }
        
        if !errors.is_empty() {
            return Err(ValidationError::Multiple(errors));
        }
        
        Ok(())
    }
    
    pub fn validate_range(value: i32, min: i32, max: i32) -> Result<i32, ValidationError> {
        if value < min || value > max {
            Err(ValidationError::OutOfRange { value, min, max })
        } else {
            Ok(value)
        }
    }
}

/// Database service demonstrating error propagation
pub struct DatabaseService {
    connection_string: String,
}

impl DatabaseService {
    pub fn new(connection_string: String) -> Self {
        Self { connection_string }
    }
    
    pub async fn connect(&self) -> Result<(), DatabaseError> {
        // Simulate connection attempt
        if self.connection_string.is_empty() {
            return Err(DatabaseError::ConnectionFailed {
                message: "Empty connection string".to_string(),
            });
        }
        
        // Simulate connection failure
        if self.connection_string.contains("invalid") {
            return Err(DatabaseError::ConnectionFailed {
                message: "Invalid connection parameters".to_string(),
            });
        }
        
        Ok(())
    }
    
    pub async fn find_user(&self, id: &str) -> Result<User, DatabaseError> {
        self.connect().await?;
        
        // Simulate record not found
        if id == "nonexistent" {
            return Err(DatabaseError::RecordNotFound {
                table: "users".to_string(),
                id: id.to_string(),
            });
        }
        
        // Return dummy user
        Ok(User {
            id: id.to_string(),
            name: "Test User".to_string(),
            email: "test@example.com".to_string(),
        })
    }
    
    pub async fn create_user(&self, user: &User) -> Result<String, DatabaseError> {
        self.connect().await?;
        
        // Validate input
        DataValidator::validate_user_input(&user.name, 25, &user.email)
            .map_err(|e| DatabaseError::ConstraintViolation {
                constraint: format!("User validation failed: {}", e),
            })?;
        
        // Simulate constraint violation
        if user.name == "duplicate" {
            return Err(DatabaseError::ConstraintViolation {
                constraint: "unique_username".to_string(),
            });
        }
        
        Ok(user.id.clone())
    }
}

/// Network service demonstrating error chaining
pub struct NetworkService {
    base_url: String,
    timeout_seconds: u64,
}

impl NetworkService {
    pub fn new(base_url: String, timeout_seconds: u64) -> Self {
        Self {
            base_url,
            timeout_seconds,
        }
    }
    
    pub async fn fetch_data(&self, endpoint: &str) -> Result<String, NetworkError> {
        // Simulate DNS resolution failure
        if self.base_url.contains("invalid-domain") {
            return Err(NetworkError::DnsResolution {
                host: self.base_url.clone(),
            });
        }
        
        // Simulate timeout
        if endpoint.contains("slow") {
            return Err(NetworkError::Timeout {
                seconds: self.timeout_seconds,
            });
        }
        
        // Simulate HTTP error
        if endpoint.contains("error") {
            return Err(NetworkError::HttpError { status: 404 });
        }
        
        // Return successful response
        Ok(format!("Data from {}/{}", self.base_url, endpoint))
    }
    
    pub async fn post_json<T: Serialize>(&self, endpoint: &str, data: &T) -> Result<String, NetworkError> {
        // Simulate JSON serialization
        let _json = serde_json::to_string(data)?;
        
        self.fetch_data(endpoint).await
    }
}

/// User model for demonstration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct User {
    pub id: String,
    pub name: String,
    pub email: String,
}

/// Application service that combines multiple error-prone operations
pub struct ApplicationService {
    db: DatabaseService,
    network: NetworkService,
}

impl ApplicationService {
    pub fn new(db_connection: String, api_url: String) -> Self {
        Self {
            db: DatabaseService::new(db_connection),
            network: NetworkService::new(api_url, 30),
        }
    }
    
    /// Complex operation demonstrating error propagation and context
    pub async fn create_user_with_verification(&self, user: User) -> AppResult<String> {
        let context = ErrorContext::new("create_user_with_verification")
            .with_data("user_id", &user.id)
            .with_data("user_name", &user.name);
        
        // Validate input first
        DataValidator::validate_user_input(&user.name, 25, &user.email)
            .context("Input validation failed")?;
        
        // Check if user already exists
        match self.db.find_user(&user.id).await {
            Ok(_) => {
                bail!("User with ID '{}' already exists", user.id);
            }
            Err(DatabaseError::RecordNotFound { .. }) => {
                // This is expected, continue
            }
            Err(e) => {
                return Err(e.into());
            }
        }
        
        // Verify email with external service
        let verification_result = self
            .network
            .fetch_data(&format!("verify?email={}", user.email))
            .await
            .context("Email verification failed")?;
        
        ensure!(
            verification_result.contains("valid"),
            "Email verification returned invalid result: {}",
            verification_result
        );
        
        // Create user in database
        let user_id = self
            .db
            .create_user(&user)
            .await
            .context("Failed to create user in database")?;
        
        // Send welcome email
        let welcome_data = serde_json::json!({
            "user_id": user_id,
            "email": user.email,
            "name": user.name
        });
        
        self.network
            .post_json("send-welcome", &welcome_data)
            .await
            .context("Failed to send welcome email")?;
        
        Ok(user_id)
    }
    
    /// Operation demonstrating error recovery
    pub async fn get_user_with_fallback(&self, user_id: &str) -> AppResult<User> {
        // Try primary database first
        match self.db.find_user(user_id).await {
            Ok(user) => Ok(user),
            Err(DatabaseError::RecordNotFound { .. }) => {
                // Try to fetch from external API as fallback
                let user_data = self
                    .network
                    .fetch_data(&format!("users/{}", user_id))
                    .await
                    .context("Failed to fetch user from external API")?;
                
                // Parse the response (simplified)
                let user: User = serde_json::from_str(&user_data)
                    .context("Failed to parse user data from external API")?;
                
                Ok(user)
            }
            Err(e) => Err(e.into()),
        }
    }
}

/// Error handling utilities
pub mod error_utils {
    use super::*;
    
    /// Extract root cause from error chain
    pub fn get_root_cause(error: &dyn std::error::Error) -> &dyn std::error::Error {
        let mut current = error;
        while let Some(source) = current.source() {
            current = source;
        }
        current
    }
    
    /// Format error chain for logging
    pub fn format_error_chain(error: &dyn std::error::Error) -> String {
        let mut chain = Vec::new();
        let mut current = Some(error);
        
        while let Some(err) = current {
            chain.push(err.to_string());
            current = err.source();
        }
        
        chain.join(" -> ")
    }
    
    /// Check if error chain contains specific error type
    pub fn contains_error_type<T: std::error::Error + 'static>(error: &dyn std::error::Error) -> bool {
        let mut current = Some(error);
        
        while let Some(err) = current {
            if err.downcast_ref::<T>().is_some() {
                return true;
            }
            current = err.source();
        }
        
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_validation_error() {
        let result = DataValidator::validate_user_input("", -5, "invalid-email");
        
        assert!(result.is_err());
        
        if let Err(ValidationError::Multiple(errors)) = result {
            assert_eq!(errors.len(), 3);
        } else {
            panic!("Expected Multiple validation errors");
        }
    }
    
    #[test]
    fn test_range_validation() {
        assert!(DataValidator::validate_range(50, 0, 100).is_ok());
        assert!(DataValidator::validate_range(-1, 0, 100).is_err());
        assert!(DataValidator::validate_range(101, 0, 100).is_err());
    }
    
    #[tokio::test]
    async fn test_database_connection_error() {
        let db = DatabaseService::new("invalid://connection".to_string());
        let result = db.connect().await;
        
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), DatabaseError::ConnectionFailed { .. }));
    }
    
    #[tokio::test]
    async fn test_network_timeout_error() {
        let network = NetworkService::new("https://example.com".to_string(), 1);
        let result = network.fetch_data("slow-endpoint").await;
        
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), NetworkError::Timeout { .. }));
    }
    
    #[tokio::test]
    async fn test_user_not_found() {
        let db = DatabaseService::new("valid://connection".to_string());
        let result = db.find_user("nonexistent").await;
        
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), DatabaseError::RecordNotFound { .. }));
    }
    
    #[test]
    fn test_error_utilities() {
        let validation_error = ValidationError::EmptyField {
            field: "test".to_string(),
        };
        
        let chain = error_utils::format_error_chain(&validation_error);
        assert!(chain.contains("Field 'test' is required"));
        
        let root = error_utils::get_root_cause(&validation_error);
        assert_eq!(root.to_string(), validation_error.to_string());
    }
    
    #[tokio::test]
    async fn test_application_service_error_propagation() {
        let app = ApplicationService::new(
            "valid://connection".to_string(),
            "https://api.example.com".to_string(),
        );
        
        let invalid_user = User {
            id: "test".to_string(),
            name: "".to_string(), // This will cause validation error
            email: "test@example.com".to_string(),
        };
        
        let result = app.create_user_with_verification(invalid_user).await;
        assert!(result.is_err());
        
        // Check that the error contains validation context
        let error_string = result.unwrap_err().to_string();
        assert!(error_string.contains("Input validation failed"));
    }
}
"#.to_string())
    }
}
```

## Success Criteria
- All 6 synthetic Rust files generate with distinct complexity levels
- Each file demonstrates different Rust patterns and idioms
- Files include comprehensive expected_matches for search validation
- Code is realistic and follows Rust best practices
- Test coverage includes various Rust-specific syntax patterns
- Files compile and tests pass when integrated

## Time Limit
10 minutes maximum