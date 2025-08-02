/*!
LLMKG Codebase Analysis Collector
Real-time static code analysis and structure monitoring
*/

use crate::monitoring::metrics::MetricRegistry;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use serde::{Serialize, Deserialize};
use syn::{ItemFn, ItemStruct, ItemEnum, visit::Visit};
use std::sync::{Arc, RwLock};
use walkdir::WalkDir;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodebaseMetrics {
    pub total_files: usize,
    pub total_lines: usize,
    pub total_functions: usize,
    pub total_structs: usize,
    pub total_enums: usize,
    pub total_modules: usize,
    pub file_structure: FileStructure,
    pub function_map: HashMap<String, FunctionInfo>,
    pub dependency_graph: DependencyGraph,
    pub complexity_metrics: ComplexityMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileStructure {
    pub path: String,
    pub file_type: FileType,
    pub size_bytes: u64,
    pub line_count: usize,
    pub children: Vec<FileStructure>,
    pub functions: Vec<String>,
    pub structs: Vec<String>,
    pub enums: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FileType {
    Directory,
    RustFile,
    TypeScriptFile,
    JsonFile,
    TomlFile,
    MarkdownFile,
    Other(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionInfo {
    pub name: String,
    pub file_path: String,
    pub line_number: usize,
    pub parameters: Vec<String>,
    pub return_type: Option<String>,
    pub complexity: usize,
    pub is_public: bool,
    pub is_async: bool,
    pub calls: Vec<String>,
    pub called_by: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyGraph {
    pub modules: HashMap<String, ModuleInfo>,
    pub edges: Vec<DependencyEdge>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleInfo {
    pub name: String,
    pub path: String,
    pub exports: Vec<String>,
    pub imports: Vec<String>,
    pub internal_calls: usize,
    pub external_calls: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyEdge {
    pub from: String,
    pub to: String,
    pub dependency_type: DependencyType,
    pub strength: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DependencyType {
    Import,
    FunctionCall,
    StructUsage,
    TraitImplementation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplexityMetrics {
    pub cyclomatic_complexity: HashMap<String, usize>,
    pub cognitive_complexity: HashMap<String, usize>,
    pub coupling_metrics: HashMap<String, f32>,
    pub cohesion_metrics: HashMap<String, f32>,
}

pub struct CodebaseAnalyzer {
    root_path: PathBuf,
    metrics: Arc<RwLock<CodebaseMetrics>>,
    file_watcher: Option<notify::RecommendedWatcher>,
}

impl CodebaseAnalyzer {
    fn should_skip_path(&self, path: &Path) -> bool {
        // Skip problematic directories that can cause deep recursion or are not relevant for analysis
        if let Some(dir_name) = path.file_name().and_then(|n| n.to_str()) {
            match dir_name {
                ".git" | "target" | "node_modules" | ".next" | "dist" | "build" | ".cache" | 
                ".vscode" | ".idea" | "__pycache__" | ".pytest_cache" | "coverage" => {
                    return true;
                }
                _ => {}
            }
        }
        
        // Skip any path that contains these problematic directory names
        let path_str = path.to_string_lossy();
        if path_str.contains("/.git/") || path_str.contains("\\target\\") || 
           path_str.contains("/node_modules/") || path_str.contains("\\node_modules\\") {
            return true;
        }
        
        false
    }

    pub fn new(root_path: PathBuf) -> Self {
        let initial_metrics = CodebaseMetrics {
            total_files: 0,
            total_lines: 0,
            total_functions: 0,
            total_structs: 0,
            total_enums: 0,
            total_modules: 0,
            file_structure: FileStructure {
                path: root_path.to_string_lossy().to_string(),
                file_type: FileType::Directory,
                size_bytes: 0,
                line_count: 0,
                children: Vec::new(),
                functions: Vec::new(),
                structs: Vec::new(),
                enums: Vec::new(),
            },
            function_map: HashMap::new(),
            dependency_graph: DependencyGraph {
                modules: HashMap::new(),
                edges: Vec::new(),
            },
            complexity_metrics: ComplexityMetrics {
                cyclomatic_complexity: HashMap::new(),
                cognitive_complexity: HashMap::new(),
                coupling_metrics: HashMap::new(),
                cohesion_metrics: HashMap::new(),
            },
        };

        Self {
            root_path,
            metrics: Arc::new(RwLock::new(initial_metrics)),
            file_watcher: None,
        }
    }

    pub async fn analyze_codebase(&self) -> Result<(), Box<dyn std::error::Error>> {
        let mut metrics = self.metrics.write().unwrap();
        
        // Reset metrics
        *metrics = self.create_initial_metrics();
        
        // Analyze file structure
        let file_structure = self.analyze_directory(&self.root_path)?;
        metrics.file_structure = file_structure;
        
        // Analyze Rust files
        self.analyze_rust_files(&mut metrics)?;
        
        // Analyze TypeScript files
        self.analyze_typescript_files(&mut metrics)?;
        
        // Build dependency graph
        self.build_dependency_graph(&mut metrics)?;
        
        // Calculate complexity metrics
        self.calculate_complexity_metrics(&mut metrics)?;
        
        Ok(())
    }

    fn create_initial_metrics(&self) -> CodebaseMetrics {
        CodebaseMetrics {
            total_files: 0,
            total_lines: 0,
            total_functions: 0,
            total_structs: 0,
            total_enums: 0,
            total_modules: 0,
            file_structure: FileStructure {
                path: self.root_path.to_string_lossy().to_string(),
                file_type: FileType::Directory,
                size_bytes: 0,
                line_count: 0,
                children: Vec::new(),
                functions: Vec::new(),
                structs: Vec::new(),
                enums: Vec::new(),
            },
            function_map: HashMap::new(),
            dependency_graph: DependencyGraph {
                modules: HashMap::new(),
                edges: Vec::new(),
            },
            complexity_metrics: ComplexityMetrics {
                cyclomatic_complexity: HashMap::new(),
                cognitive_complexity: HashMap::new(),
                coupling_metrics: HashMap::new(),
                cohesion_metrics: HashMap::new(),
            },
        }
    }

    fn analyze_directory(&self, path: &Path) -> Result<FileStructure, Box<dyn std::error::Error>> {
        self.analyze_directory_with_depth(path, 0, 10) // Max depth of 10 to prevent stack overflow
    }
    
    fn analyze_directory_with_depth(&self, path: &Path, current_depth: usize, max_depth: usize) -> Result<FileStructure, Box<dyn std::error::Error>> {
        let mut structure = FileStructure {
            path: path.to_string_lossy().to_string(),
            file_type: FileType::Directory,
            size_bytes: 0,
            line_count: 0,
            children: Vec::new(),
            functions: Vec::new(),
            structs: Vec::new(),
            enums: Vec::new(),
        };

        // Prevent stack overflow by limiting depth
        if current_depth >= max_depth {
            return Ok(structure);
        }

        // Skip problematic directories that can cause deep recursion
        if self.should_skip_path(path) {
            return Ok(structure);
        }

        // Use WalkDir with max_depth(1) to get immediate children only
        for entry in WalkDir::new(path).max_depth(1) {
            let entry = entry?;
            let child_path = entry.path();
            
            // Skip the directory itself
            if child_path == path {
                continue;
            }

            let child_structure = if child_path.is_dir() {
                self.analyze_directory_with_depth(child_path, current_depth + 1, max_depth)?
            } else {
                self.analyze_file(child_path)?
            };

            structure.children.push(child_structure);
        }

        Ok(structure)
    }

    fn analyze_file(&self, path: &Path) -> Result<FileStructure, Box<dyn std::error::Error>> {
        let metadata = fs::metadata(path)?;
        let content = fs::read_to_string(path).unwrap_or_default();
        let line_count = content.lines().count();

        let file_type = match path.extension().and_then(|s| s.to_str()) {
            Some("rs") => FileType::RustFile,
            Some("ts") | Some("tsx") => FileType::TypeScriptFile,
            Some("json") => FileType::JsonFile,
            Some("toml") => FileType::TomlFile,
            Some("md") => FileType::MarkdownFile,
            Some(ext) => FileType::Other(ext.to_string()),
            None => FileType::Other("no_extension".to_string()),
        };

        let mut structure = FileStructure {
            path: path.to_string_lossy().to_string(),
            file_type,
            size_bytes: metadata.len(),
            line_count,
            children: Vec::new(),
            functions: Vec::new(),
            structs: Vec::new(),
            enums: Vec::new(),
        };

        // Analyze Rust file content if applicable
        if matches!(structure.file_type, FileType::RustFile) {
            if let Ok(parsed) = syn::parse_file(&content) {
                let mut visitor = RustAnalysisVisitor::new();
                visitor.visit_file(&parsed);
                
                structure.functions = visitor.functions;
                structure.structs = visitor.structs;
                structure.enums = visitor.enums;
            }
        }

        Ok(structure)
    }

    fn analyze_rust_files(&self, metrics: &mut CodebaseMetrics) -> Result<(), Box<dyn std::error::Error>> {
        for entry in WalkDir::new(&self.root_path) {
            let entry = entry?;
            let path = entry.path();
            
            // Skip problematic paths
            if self.should_skip_path(path) {
                continue;
            }
            
            if path.extension().and_then(|s| s.to_str()) == Some("rs") {
                if let Ok(content) = fs::read_to_string(path) {
                    if let Ok(parsed) = syn::parse_file(&content) {
                        let mut visitor = RustAnalysisVisitor::new();
                        visitor.visit_file(&parsed);
                        
                        metrics.total_functions += visitor.functions.len();
                        metrics.total_structs += visitor.structs.len();
                        metrics.total_enums += visitor.enums.len();
                        
                        // Add functions to function map
                        for (i, func_name) in visitor.functions.iter().enumerate() {
                            let function_info = FunctionInfo {
                                name: func_name.clone(),
                                file_path: path.to_string_lossy().to_string(),
                                line_number: i + 1, // Simplified
                                parameters: Vec::new(), // TODO: Extract from AST
                                return_type: None, // TODO: Extract from AST
                                complexity: 1, // TODO: Calculate cyclomatic complexity
                                is_public: true, // TODO: Extract visibility
                                is_async: false, // TODO: Extract async status
                                calls: Vec::new(), // TODO: Extract function calls
                                called_by: Vec::new(),
                            };
                            metrics.function_map.insert(func_name.clone(), function_info);
                        }
                    }
                }
                metrics.total_files += 1;
                metrics.total_lines += fs::read_to_string(path)?.lines().count();
            }
        }
        Ok(())
    }

    fn analyze_typescript_files(&self, metrics: &mut CodebaseMetrics) -> Result<(), Box<dyn std::error::Error>> {
        for entry in WalkDir::new(&self.root_path) {
            let entry = entry?;
            let path = entry.path();
            
            // Skip problematic paths
            if self.should_skip_path(path) {
                continue;
            }
            
            if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
                if ext == "ts" || ext == "tsx" {
                    metrics.total_files += 1;
                    metrics.total_lines += fs::read_to_string(path)?.lines().count();
                    
                    // TODO: Parse TypeScript AST for detailed analysis
                }
            }
        }
        Ok(())
    }

    fn build_dependency_graph(&self, metrics: &mut CodebaseMetrics) -> Result<(), Box<dyn std::error::Error>> {
        // Parse all Rust files to build real dependency graph
        for entry in WalkDir::new(&self.root_path) {
            let entry = entry?;
            let path = entry.path();
            
            // Skip problematic paths
            if self.should_skip_path(path) {
                continue;
            }
            
            if path.extension().and_then(|s| s.to_str()) == Some("rs") {
                self.analyze_file_dependencies(path, metrics)?;
            }
        }
        
        // Build edges based on imports
        self.build_dependency_edges(metrics)?;
        
        Ok(())
    }
    
    fn analyze_file_dependencies(&self, path: &Path, metrics: &mut CodebaseMetrics) -> Result<(), Box<dyn std::error::Error>> {
        let content = fs::read_to_string(path)?;
        let module_path = self.path_to_module_name(path);
        
        let mut imports = Vec::new();
        let mut exports = Vec::new();
        
        // Parse use statements
        for line in content.lines() {
            let line = line.trim();
            
            // Handle use crate:: imports
            if line.starts_with("use crate::") {
                if let Some(import) = self.parse_crate_import(line) {
                    imports.push(import);
                }
            }
            
            // Handle external crate imports
            if line.starts_with("use ") && !line.starts_with("use crate::") && !line.starts_with("use std::") && !line.starts_with("use self::") {
                if let Some(import) = self.parse_external_import(line) {
                    imports.push(import);
                }
            }
            
            // Handle pub mod declarations (exports)
            if line.starts_with("pub mod ") {
                if let Some(export) = self.parse_pub_mod(line) {
                    exports.push(export);
                }
            }
            
            // Handle pub use re-exports
            if line.starts_with("pub use ") {
                if let Some(export) = self.parse_pub_use(line) {
                    exports.push(export);
                }
            }
        }
        
        // Count function calls for complexity
        let (internal_calls, external_calls) = self.count_function_calls(&content, &module_path);
        
        let module_info = ModuleInfo {
            name: module_path.clone(),
            path: path.to_string_lossy().to_string(),
            exports,
            imports,
            internal_calls,
            external_calls,
        };
        
        metrics.dependency_graph.modules.insert(module_path, module_info);
        Ok(())
    }
    
    fn path_to_module_name(&self, path: &Path) -> String {
        let relative_path = path.strip_prefix(&self.root_path).unwrap_or(path);
        let mut components = Vec::new();
        
        for component in relative_path.components() {
            if let std::path::Component::Normal(comp) = component {
                if let Some(comp_str) = comp.to_str() {
                    if comp_str.ends_with(".rs") {
                        if comp_str == "mod.rs" || comp_str == "lib.rs" {
                            continue;
                        }
                        components.push(comp_str.trim_end_matches(".rs").to_string());
                    } else {
                        components.push(comp_str.to_string());
                    }
                }
            }
        }
        
        components.join("::")
    }
    
    fn parse_crate_import(&self, line: &str) -> Option<String> {
        // Parse "use crate::module::submodule::Item;" -> "module::submodule"
        let without_use = line.strip_prefix("use crate::")?;
        let without_semicolon = without_use.trim_end_matches(';').trim();
        
        // Handle nested imports like use crate::core::{types, entity};
        if without_semicolon.contains('{') {
            let base_path = without_semicolon.split("::").collect::<Vec<_>>();
            if base_path.len() >= 2 {
                return Some(base_path[..base_path.len()-1].join("::"));
            }
        }
        
        // Handle simple imports like use crate::core::types::EntityKey;
        let parts: Vec<&str> = without_semicolon.split("::").collect();
        if parts.len() >= 2 {
            // Return module path without the final item
            Some(parts[..parts.len()-1].join("::"))
        } else {
            Some(parts[0].to_string())
        }
    }
    
    fn parse_external_import(&self, line: &str) -> Option<String> {
        // Parse external crate imports like "use serde::{Serialize, Deserialize};"
        let without_use = line.strip_prefix("use ")?;
        let without_semicolon = without_use.trim_end_matches(';').trim();
        
        let first_part = without_semicolon.split("::").next()?.split('{').next()?.trim();
        Some(format!("external::{first_part}"))
    }
    
    fn parse_pub_mod(&self, line: &str) -> Option<String> {
        // Parse "pub mod module_name;" -> "module_name"
        let without_pub_mod = line.strip_prefix("pub mod ")?;
        let without_semicolon = without_pub_mod.trim_end_matches(';').trim();
        Some(without_semicolon.to_string())
    }
    
    fn parse_pub_use(&self, line: &str) -> Option<String> {
        // Parse "pub use crate::core::types::EntityKey;" -> "re-export: EntityKey"
        let without_pub_use = line.strip_prefix("pub use ")?;
        let without_semicolon = without_pub_use.trim_end_matches(';').trim();
        
        if let Some(item) = without_semicolon.split("::").last() {
            Some(format!("re-export: {item}"))
        } else {
            Some(format!("re-export: {without_semicolon}"))
        }
    }
    
    fn count_function_calls(&self, content: &str, _module_path: &str) -> (usize, usize) {
        let internal_calls = content.matches("self.").count() + content.matches("Self::").count();
        let total_double_colon = content.matches("::").count();
        
        // Prevent underflow by using saturating subtraction
        let external_calls = total_double_colon.saturating_sub(internal_calls);
        (internal_calls, external_calls)
    }
    
    fn build_dependency_edges(&self, metrics: &mut CodebaseMetrics) -> Result<(), Box<dyn std::error::Error>> {
        let modules = metrics.dependency_graph.modules.clone();
        
        for (module_name, module_info) in &modules {
            for import in &module_info.imports {
                if let Some(target_module) = self.resolve_import_to_module(import, &modules) {
                    let edge = DependencyEdge {
                        from: module_name.clone(),
                        to: target_module,
                        dependency_type: if import.starts_with("external::") {
                            DependencyType::Import
                        } else {
                            DependencyType::Import
                        },
                        strength: self.calculate_dependency_strength(module_info, import),
                    };
                    metrics.dependency_graph.edges.push(edge);
                }
            }
        }
        
        Ok(())
    }
    
    fn resolve_import_to_module(&self, import: &str, modules: &HashMap<String, ModuleInfo>) -> Option<String> {
        // Handle external imports
        if import.starts_with("external::") {
            return Some(import.to_string());
        }
        
        // Find matching module
        for module_name in modules.keys() {
            if import.starts_with(module_name) {
                return Some(module_name.clone());
            }
        }
        
        // Try partial matches
        for module_name in modules.keys() {
            let import_parts: Vec<&str> = import.split("::").collect();
            let module_parts: Vec<&str> = module_name.split("::").collect();
            
            if import_parts.len() >= module_parts.len() {
                let matches = import_parts.iter()
                    .zip(module_parts.iter())
                    .all(|(a, b)| a == b);
                
                if matches {
                    return Some(module_name.clone());
                }
            }
        }
        
        None
    }
    
    fn calculate_dependency_strength(&self, _module_info: &ModuleInfo, _import: &str) -> f32 {
        // Simple heuristic: more internal calls = stronger dependency
        // This could be enhanced with more sophisticated analysis
        0.5 // Default strength
    }

    fn calculate_complexity_metrics(&self, metrics: &mut CodebaseMetrics) -> Result<(), Box<dyn std::error::Error>> {
        // TODO: Implement proper complexity calculations
        for func_name in metrics.function_map.keys() {
            metrics.complexity_metrics.cyclomatic_complexity.insert(func_name.clone(), 1);
            metrics.complexity_metrics.cognitive_complexity.insert(func_name.clone(), 1);
        }
        Ok(())
    }

    pub fn get_metrics(&self) -> CodebaseMetrics {
        self.metrics.read().unwrap().clone()
    }

    pub async fn start_watching(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        use notify::{Watcher, RecursiveMode};
        use std::sync::mpsc;
        
        let (tx, rx) = mpsc::channel();
        let mut watcher = notify::recommended_watcher(tx)?;
        watcher.watch(&self.root_path, RecursiveMode::Recursive)?;
        
        let _metrics = self.metrics.clone();
        let root_path = self.root_path.clone();
        
        tokio::spawn(async move {
            let analyzer = CodebaseAnalyzer::new(root_path);
            
            while let Ok(_event) = rx.recv() {
                // File system event occurred, re-analyze
                if let Err(e) = analyzer.analyze_codebase().await {
                    eprintln!("Error during codebase analysis: {e}");
                }
            }
        });
        
        self.file_watcher = Some(watcher);
        Ok(())
    }
}

struct RustAnalysisVisitor {
    functions: Vec<String>,
    structs: Vec<String>,
    enums: Vec<String>,
}

impl RustAnalysisVisitor {
    fn new() -> Self {
        Self {
            functions: Vec::new(),
            structs: Vec::new(),
            enums: Vec::new(),
        }
    }
}

impl<'ast> Visit<'ast> for RustAnalysisVisitor {
    fn visit_item_fn(&mut self, node: &'ast ItemFn) {
        self.functions.push(node.sig.ident.to_string());
        syn::visit::visit_item_fn(self, node);
    }

    fn visit_item_struct(&mut self, node: &'ast ItemStruct) {
        self.structs.push(node.ident.to_string());
        syn::visit::visit_item_struct(self, node);
    }

    fn visit_item_enum(&mut self, node: &'ast ItemEnum) {
        self.enums.push(node.ident.to_string());
        syn::visit::visit_item_enum(self, node);
    }
}

impl super::MetricsCollector for CodebaseAnalyzer {
    fn collect(&self, registry: &MetricRegistry) -> Result<(), Box<dyn std::error::Error>> {
        let metrics = self.get_metrics();
        
        // Register codebase metrics
        let files_gauge = registry.gauge("codebase_total_files", std::collections::HashMap::new());
        files_gauge.set(metrics.total_files as f64);
        
        let lines_gauge = registry.gauge("codebase_total_lines", std::collections::HashMap::new());
        lines_gauge.set(metrics.total_lines as f64);
        
        let functions_gauge = registry.gauge("codebase_total_functions", std::collections::HashMap::new());
        functions_gauge.set(metrics.total_functions as f64);
        
        let structs_gauge = registry.gauge("codebase_total_structs", std::collections::HashMap::new());
        structs_gauge.set(metrics.total_structs as f64);
        
        let enums_gauge = registry.gauge("codebase_total_enums", std::collections::HashMap::new());
        enums_gauge.set(metrics.total_enums as f64);
        
        let modules_gauge = registry.gauge("codebase_total_modules", std::collections::HashMap::new());
        modules_gauge.set(metrics.total_modules as f64);
        
        Ok(())
    }
    
    fn name(&self) -> &str {
        "codebase_analyzer"
    }
    
    fn is_enabled(&self, config: &super::MetricsCollectionConfig) -> bool {
        config.enabled_collectors.contains(&"codebase_analyzer".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use std::fs;
    use tempfile::TempDir;

    #[tokio::test] 
    async fn test_codebase_analyzer_with_limited_scope() {
        // Create a temporary directory with a controlled structure to prevent stack overflow
        let temp_dir = TempDir::new().unwrap();
        let temp_path = temp_dir.path();
        
        // Create some test files
        fs::write(temp_path.join("test.rs"), "fn main() {}").unwrap();
        fs::write(temp_path.join("README.md"), "# Test").unwrap();
        
        // Create a subdirectory with a file
        let sub_dir = temp_path.join("src");
        fs::create_dir(&sub_dir).unwrap();
        fs::write(sub_dir.join("lib.rs"), "pub mod test;").unwrap();
        
        let analyzer = CodebaseAnalyzer::new(temp_path.to_path_buf());
        
        // This should not cause a stack overflow anymore
        assert!(analyzer.analyze_codebase().await.is_ok());
        
        let metrics = analyzer.get_metrics();
        assert!(metrics.total_files > 0);
        assert!(metrics.total_lines > 0);
    }
    
    #[test]
    fn test_should_skip_path() {
        let temp_dir = TempDir::new().unwrap();
        let analyzer = CodebaseAnalyzer::new(temp_dir.path().to_path_buf());
        
        // Test various paths that should be skipped
        assert!(analyzer.should_skip_path(&temp_dir.path().join(".git")));
        assert!(analyzer.should_skip_path(&temp_dir.path().join("target")));
        assert!(analyzer.should_skip_path(&temp_dir.path().join("node_modules")));
        assert!(analyzer.should_skip_path(&temp_dir.path().join(".cache")));
        
        // Test paths that should not be skipped
        assert!(!analyzer.should_skip_path(&temp_dir.path().join("src")));
        assert!(!analyzer.should_skip_path(&temp_dir.path().join("main.rs")));
        assert!(!analyzer.should_skip_path(&temp_dir.path().join("docs")));
    }
    
    #[test]
    fn test_directory_depth_limiting() {
        let temp_dir = TempDir::new().unwrap();
        let analyzer = CodebaseAnalyzer::new(temp_dir.path().to_path_buf());
        
        // This should return early due to depth limiting, not cause stack overflow
        let result = analyzer.analyze_directory_with_depth(temp_dir.path(), 15, 10);
        assert!(result.is_ok());
        
        let structure = result.unwrap();
        assert!(structure.children.is_empty()); // Should be empty due to depth limit
    }

    #[tokio::test]
    async fn test_codebase_analyzer() {
        let current_dir = env::current_dir().unwrap();
        let analyzer = CodebaseAnalyzer::new(current_dir);
        
        assert!(analyzer.analyze_codebase().await.is_ok());
        
        let metrics = analyzer.get_metrics();
        assert!(metrics.total_files > 0);
        assert!(metrics.total_lines > 0);
    }
}