//! Test Registry System
//! 
//! Provides test discovery, registration, and management capabilities.

use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::time::Duration;
use uuid::Uuid;

/// Categories of tests in the simulation framework
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TestCategory {
    Unit,
    Integration,
    Simulation,
    Performance,
    Stress,
    EndToEnd,
    Regression,
}

/// Priority levels for test execution
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TestPriority {
    Critical,
    High, 
    Medium,
    Low,
}

/// Expected outcomes for deterministic testing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpectedOutcomes {
    /// Expected return status
    pub success: bool,
    /// Expected performance metrics
    pub performance_expectations: PerformanceExpectations,
    /// Expected output checksums for validation
    pub output_checksums: HashMap<String, String>,
    /// Expected side effects
    pub side_effects: Vec<SideEffect>,
}

/// Performance expectations for tests
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceExpectations {
    /// Maximum allowed duration
    pub max_duration: Duration,
    /// Maximum memory usage in bytes
    pub max_memory_bytes: u64,
    /// Minimum performance thresholds
    pub min_throughput: Option<f64>,
    /// Accuracy requirements
    pub min_accuracy: Option<f64>,
}

/// Expected side effects from test execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SideEffect {
    pub effect_type: String,
    pub description: String,
    pub validation_criteria: String,
}

/// Data requirements for test execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataRequirements {
    /// Required datasets
    pub datasets: Vec<String>,
    /// Minimum data size requirements
    pub min_data_size: u64,
    /// Required data properties
    pub data_properties: HashMap<String, String>,
    /// Temporary storage requirements
    pub temp_storage_mb: u64,
}

/// Resource requirements for test execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRequirements {
    /// Minimum CPU cores required
    pub min_cpu_cores: u32,
    /// Minimum memory in MB
    pub min_memory_mb: u64,
    /// Required GPU memory in MB (if applicable)
    pub gpu_memory_mb: Option<u64>,
    /// Network requirements
    pub network_required: bool,
}

/// Complete test descriptor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestDescriptor {
    /// Unique test identifier
    pub id: Uuid,
    /// Human-readable test name
    pub name: String,
    /// Test category
    pub category: TestCategory,
    /// Test priority
    pub priority: TestPriority,
    /// Test dependencies (must run before this test)
    pub dependencies: Vec<String>,
    /// Maximum execution timeout
    pub timeout: Duration,
    /// Expected test outcomes
    pub expected_outcomes: ExpectedOutcomes,
    /// Data requirements
    pub data_requirements: DataRequirements,
    /// Resource requirements
    pub resource_requirements: ResourceRequirements,
    /// Test description
    pub description: String,
    /// Test author/maintainer
    pub author: String,
    /// Test version
    pub version: String,
    /// Tags for filtering
    pub tags: HashSet<String>,
    /// Test function path
    pub function_path: String,
    /// Configuration parameters
    pub config_params: HashMap<String, String>,
}

impl TestDescriptor {
    /// Create a new test descriptor
    pub fn new(name: String, category: TestCategory) -> Self {
        Self {
            id: Uuid::new_v4(),
            name,
            category,
            priority: TestPriority::Medium,
            dependencies: Vec::new(),
            timeout: Duration::from_secs(300), // 5 minutes default
            expected_outcomes: ExpectedOutcomes {
                success: true,
                performance_expectations: PerformanceExpectations {
                    max_duration: Duration::from_secs(60),
                    max_memory_bytes: 100 * 1024 * 1024, // 100MB
                    min_throughput: None,
                    min_accuracy: None,
                },
                output_checksums: HashMap::new(),
                side_effects: Vec::new(),
            },
            data_requirements: DataRequirements {
                datasets: Vec::new(),
                min_data_size: 0,
                data_properties: HashMap::new(),
                temp_storage_mb: 10,
            },
            resource_requirements: ResourceRequirements {
                min_cpu_cores: 1,
                min_memory_mb: 512,
                gpu_memory_mb: None,
                network_required: false,
            },
            description: String::new(),
            author: String::new(),
            version: "1.0.0".to_string(),
            tags: HashSet::new(),
            function_path: String::new(),
            config_params: HashMap::new(),
        }
    }

    /// Builder pattern for test configuration
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    pub fn with_dependencies(mut self, deps: Vec<String>) -> Self {
        self.dependencies = deps;
        self
    }

    pub fn with_priority(mut self, priority: TestPriority) -> Self {
        self.priority = priority;
        self
    }

    pub fn with_tags(mut self, tags: HashSet<String>) -> Self {
        self.tags = tags;
        self
    }

    pub fn with_description(mut self, description: String) -> Self {
        self.description = description;
        self
    }

    pub fn with_author(mut self, author: String) -> Self {
        self.author = author;
        self
    }

    /// Validate test descriptor
    pub fn validate(&self) -> Result<()> {
        if self.name.is_empty() {
            return Err(anyhow!("Test name cannot be empty"));
        }

        if self.timeout.as_secs() == 0 {
            return Err(anyhow!("Test timeout must be greater than 0"));
        }

        if self.function_path.is_empty() {
            return Err(anyhow!("Function path cannot be empty"));
        }

        // Validate dependencies don't create cycles
        if self.dependencies.contains(&self.name) {
            return Err(anyhow!("Test cannot depend on itself"));
        }

        Ok(())
    }
}

/// Main test registry for managing all tests
pub struct TestRegistry {
    /// All registered tests by ID
    tests: HashMap<Uuid, TestDescriptor>,
    /// Tests organized by category
    tests_by_category: HashMap<TestCategory, Vec<Uuid>>,
    /// Tests organized by name for quick lookup
    tests_by_name: HashMap<String, Uuid>,
    /// Dependency graph for execution ordering
    dependency_graph: HashMap<String, Vec<String>>,
}

impl TestRegistry {
    /// Create a new test registry
    pub fn new() -> Result<Self> {
        Ok(Self {
            tests: HashMap::new(),
            tests_by_category: HashMap::new(),
            tests_by_name: HashMap::new(),
            dependency_graph: HashMap::new(),
        })
    }

    /// Register a new test
    pub fn register_test(&mut self, test: TestDescriptor) -> Result<()> {
        test.validate()?;

        let test_id = test.id;
        let test_name = test.name.clone();
        let test_category = test.category.clone();

        // Check for duplicate names
        if self.tests_by_name.contains_key(&test_name) {
            return Err(anyhow!("Test with name '{}' already exists", test_name));
        }

        // Update dependency graph
        self.dependency_graph.insert(test_name.clone(), test.dependencies.clone());

        // Register test
        self.tests.insert(test_id, test);
        self.tests_by_name.insert(test_name, test_id);
        
        // Update category index
        self.tests_by_category
            .entry(test_category)
            .or_insert_with(Vec::new)
            .push(test_id);

        Ok(())
    }

    /// Get test by ID
    pub fn get_test(&self, id: &Uuid) -> Option<&TestDescriptor> {
        self.tests.get(id)
    }

    /// Get test by name
    pub fn get_test_by_name(&self, name: &str) -> Option<&TestDescriptor> {
        self.tests_by_name.get(name)
            .and_then(|id| self.tests.get(id))
    }

    /// Get all tests in a category
    pub fn get_tests_by_category(&self, category: &TestCategory) -> Vec<&TestDescriptor> {
        self.tests_by_category
            .get(category)
            .map(|ids| ids.iter().filter_map(|id| self.tests.get(id)).collect())
            .unwrap_or_default()
    }

    /// Get all tests with specific tags
    pub fn get_tests_by_tags(&self, tags: &HashSet<String>) -> Vec<&TestDescriptor> {
        self.tests.values()
            .filter(|test| !test.tags.is_disjoint(tags))
            .collect()
    }

    /// Get tests by priority
    pub fn get_tests_by_priority(&self, priority: &TestPriority) -> Vec<&TestDescriptor> {
        self.tests.values()
            .filter(|test| &test.priority == priority)
            .collect()
    }

    /// Get all tests
    pub fn get_all_tests(&self) -> Vec<&TestDescriptor> {
        self.tests.values().collect()
    }

    /// Resolve test execution order based on dependencies
    pub fn resolve_execution_order(&self, test_names: &[String]) -> Result<Vec<String>> {
        let mut resolved = Vec::new();
        let mut visited = HashSet::new();
        let mut visiting = HashSet::new();

        for test_name in test_names {
            if !visited.contains(test_name) {
                self.topological_sort(test_name, &mut resolved, &mut visited, &mut visiting)?;
            }
        }

        Ok(resolved)
    }

    /// Topological sort for dependency resolution
    fn topological_sort(
        &self,
        test_name: &str,
        resolved: &mut Vec<String>,
        visited: &mut HashSet<String>,
        visiting: &mut HashSet<String>,
    ) -> Result<()> {
        if visiting.contains(test_name) {
            return Err(anyhow!("Circular dependency detected involving test: {}", test_name));
        }

        if visited.contains(test_name) {
            return Ok(());
        }

        visiting.insert(test_name.to_string());

        if let Some(dependencies) = self.dependency_graph.get(test_name) {
            for dep in dependencies {
                // Verify dependency exists
                if !self.tests_by_name.contains_key(dep) {
                    return Err(anyhow!("Unknown dependency '{}' for test '{}'", dep, test_name));
                }
                self.topological_sort(dep, resolved, visited, visiting)?;
            }
        }

        visiting.remove(test_name);
        visited.insert(test_name.to_string());
        resolved.push(test_name.to_string());

        Ok(())
    }

    /// Discover tests automatically (placeholder for proc macro integration)
    pub async fn discover_tests(&self) -> Result<Vec<TestDescriptor>> {
        // In a real implementation, this would use proc macros to automatically
        // discover test functions marked with custom attributes
        // For now, return all registered tests
        Ok(self.tests.values().cloned().collect())
    }

    /// Get test statistics
    pub fn get_statistics(&self) -> TestRegistryStats {
        let mut stats = TestRegistryStats::default();
        
        stats.total_tests = self.tests.len();
        
        for test in self.tests.values() {
            match test.category {
                TestCategory::Unit => stats.unit_tests += 1,
                TestCategory::Integration => stats.integration_tests += 1,
                TestCategory::Simulation => stats.simulation_tests += 1,
                TestCategory::Performance => stats.performance_tests += 1,
                TestCategory::Stress => stats.stress_tests += 1,
                TestCategory::EndToEnd => stats.end_to_end_tests += 1,
                TestCategory::Regression => stats.regression_tests += 1,
            }

            match test.priority {
                TestPriority::Critical => stats.critical_priority += 1,
                TestPriority::High => stats.high_priority += 1,
                TestPriority::Medium => stats.medium_priority += 1,
                TestPriority::Low => stats.low_priority += 1,
            }
        }

        stats
    }

    /// Export test registry to JSON
    pub fn export_to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(&self.tests.values().collect::<Vec<_>>())
            .map_err(|e| anyhow!("Failed to serialize test registry: {}", e))
    }

    /// Import test registry from JSON
    pub fn import_from_json(&mut self, json: &str) -> Result<()> {
        let tests: Vec<TestDescriptor> = serde_json::from_str(json)
            .map_err(|e| anyhow!("Failed to deserialize test registry: {}", e))?;

        for test in tests {
            self.register_test(test)?;
        }

        Ok(())
    }
}

/// Statistics about the test registry
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct TestRegistryStats {
    pub total_tests: usize,
    pub unit_tests: usize,
    pub integration_tests: usize,
    pub simulation_tests: usize,
    pub performance_tests: usize,
    pub stress_tests: usize,
    pub end_to_end_tests: usize,
    pub regression_tests: usize,
    pub critical_priority: usize,
    pub high_priority: usize,
    pub medium_priority: usize,
    pub low_priority: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_test_descriptor_creation() {
        let test = TestDescriptor::new("test_example".to_string(), TestCategory::Unit);
        assert_eq!(test.name, "test_example");
        assert_eq!(test.category, TestCategory::Unit);
        assert!(test.validate().is_ok());
    }

    #[test]
    fn test_registry_registration() {
        let mut registry = TestRegistry::new().unwrap();
        let test = TestDescriptor::new("test_example".to_string(), TestCategory::Unit);
        
        assert!(registry.register_test(test).is_ok());
        assert_eq!(registry.get_statistics().total_tests, 1);
    }

    #[test]
    fn test_dependency_resolution() {
        let mut registry = TestRegistry::new().unwrap();
        
        let test1 = TestDescriptor::new("test1".to_string(), TestCategory::Unit);
        let test2 = TestDescriptor::new("test2".to_string(), TestCategory::Unit)
            .with_dependencies(vec!["test1".to_string()]);
        
        registry.register_test(test1).unwrap();
        registry.register_test(test2).unwrap();
        
        let order = registry.resolve_execution_order(&["test2".to_string()]).unwrap();
        assert_eq!(order, vec!["test1", "test2"]);
    }

    #[test]
    fn test_circular_dependency_detection() {
        let mut registry = TestRegistry::new().unwrap();
        
        let test1 = TestDescriptor::new("test1".to_string(), TestCategory::Unit)
            .with_dependencies(vec!["test2".to_string()]);
        let test2 = TestDescriptor::new("test2".to_string(), TestCategory::Unit)
            .with_dependencies(vec!["test1".to_string()]);
        
        registry.register_test(test1).unwrap();
        registry.register_test(test2).unwrap();
        
        let result = registry.resolve_execution_order(&["test1".to_string()]);
        assert!(result.is_err());
    }
}