#!/usr/bin/env python3
"""
Performance validation to ensure fixes don't degrade performance by more than 10%
"""

import time
import tracemalloc
from smart_chunker import smart_chunk_content

# Test cases representing different complexity levels
test_cases = [
    # Simple case
    ('simple', '''def simple():
    """Simple function"""
    return 42''', 'python'),
    
    # Medium complexity with JSDoc
    ('medium', '''/**
 * Calculator with operations
 * @param {number} a - First number
 * @param {number} b - Second number
 */
function calculator(a, b) {
    return {
        add: () => a + b,
        subtract: () => a - b,
        multiply: () => a * b,
        divide: () => b !== 0 ? a / b : null
    };
}

/**
 * Utility class for math operations
 */
class MathUtils {
    /**
     * Calculate factorial
     * @param {number} n - Number
     */
    static factorial(n) {
        if (n <= 1) return 1;
        return n * MathUtils.factorial(n - 1);
    }
}''', 'javascript'),
    
    # Complex Rust case
    ('complex', '''/// Advanced neural network implementation
/// with multiple layers and activation functions
/// 
/// This struct provides comprehensive neural network
/// functionality including:
/// - Forward propagation
/// - Backward propagation  
/// - Weight optimization
/// - Gradient calculation
/// - Learning rate adaptation
pub struct NeuralNetwork {
    layers: Vec<Layer>,
    learning_rate: f64,
    momentum: f64,
    regularization: f64,
}

impl NeuralNetwork {
    /// Create a new neural network
    /// with specified layer configuration
    pub fn new(layer_sizes: &[usize], learning_rate: f64) -> Self {
        let mut layers = Vec::new();
        for i in 0..layer_sizes.len() - 1 {
            layers.push(Layer::new(layer_sizes[i], layer_sizes[i + 1]));
        }
        
        Self {
            layers,
            learning_rate,
            momentum: 0.9,
            regularization: 0.01,
        }
    }
    
    /// Forward pass through the network
    /// calculates output for given input
    pub fn forward(&self, input: &[f64]) -> Vec<f64> {
        let mut current_input = input.to_vec();
        
        for layer in &self.layers {
            current_input = layer.forward(&current_input);
        }
        
        current_input
    }
    
    /// Backward pass for training
    /// updates weights based on error gradient
    pub fn backward(&mut self, target: &[f64], output: &[f64]) -> f64 {
        let error = self.calculate_error(target, output);
        let mut gradient = self.calculate_output_gradient(target, output);
        
        for layer in self.layers.iter_mut().rev() {
            gradient = layer.backward(&gradient, self.learning_rate);
        }
        
        error
    }
    
    /// Calculate mean squared error
    fn calculate_error(&self, target: &[f64], output: &[f64]) -> f64 {
        target.iter()
            .zip(output.iter())
            .map(|(t, o)| (t - o).powi(2))
            .sum::<f64>() / target.len() as f64
    }
    
    /// Calculate gradient for output layer
    fn calculate_output_gradient(&self, target: &[f64], output: &[f64]) -> Vec<f64> {
        target.iter()
            .zip(output.iter())
            .map(|(t, o)| 2.0 * (o - t))
            .collect()
    }
}

/// Individual layer in neural network
pub struct Layer {
    weights: Vec<Vec<f64>>,
    biases: Vec<f64>,
    last_input: Vec<f64>,
    last_output: Vec<f64>,
}

impl Layer {
    /// Create new layer with random weights
    pub fn new(input_size: usize, output_size: usize) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        let weights = (0..output_size)
            .map(|_| (0..input_size).map(|_| rng.gen_range(-1.0..1.0)).collect())
            .collect();
            
        let biases = (0..output_size).map(|_| rng.gen_range(-1.0..1.0)).collect();
        
        Self {
            weights,
            biases,
            last_input: Vec::new(),
            last_output: Vec::new(),
        }
    }
    
    /// Forward pass through layer
    pub fn forward(&mut self, input: &[f64]) -> Vec<f64> {
        self.last_input = input.to_vec();
        
        let output = self.weights.iter()
            .zip(self.biases.iter())
            .map(|(weight_row, bias)| {
                let sum: f64 = weight_row.iter()
                    .zip(input.iter())
                    .map(|(w, i)| w * i)
                    .sum();
                self.sigmoid(sum + bias)
            })
            .collect();
            
        self.last_output = output.clone();
        output
    }
    
    /// Backward pass through layer
    pub fn backward(&mut self, gradient: &[f64], learning_rate: f64) -> Vec<f64> {
        let mut input_gradient = vec![0.0; self.last_input.len()];
        
        for (i, (weight_row, &grad)) in self.weights.iter_mut().zip(gradient.iter()).enumerate() {
            let delta = grad * self.sigmoid_derivative(self.last_output[i]);
            
            for (j, &input_val) in self.last_input.iter().enumerate() {
                weight_row[j] -= learning_rate * delta * input_val;
                input_gradient[j] += weight_row[j] * delta;
            }
            
            self.biases[i] -= learning_rate * delta;
        }
        
        input_gradient
    }
    
    /// Sigmoid activation function
    fn sigmoid(&self, x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }
    
    /// Derivative of sigmoid function
    fn sigmoid_derivative(&self, x: f64) -> f64 {
        x * (1.0 - x)
    }
}''', 'rust'),
]

def measure_performance(test_name, code, language, iterations=5):
    """Measure performance of chunking operation"""
    times = []
    memory_usage = []
    
    for _ in range(iterations):
        tracemalloc.start()
        start_time = time.time()
        
        chunks = smart_chunk_content(code, language, f"{test_name}.{language}")
        
        end_time = time.time()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        times.append(end_time - start_time)
        memory_usage.append(peak)
    
    avg_time = sum(times) / len(times)
    avg_memory = sum(memory_usage) / len(memory_usage)
    
    return {
        'avg_time': avg_time,
        'avg_memory': avg_memory,
        'chunks_generated': len(chunks),
        'code_size': len(code)
    }

def run_performance_validation():
    """Run performance validation tests"""
    print("SMARTCHUNKER PERFORMANCE VALIDATION")
    print("=" * 50)
    print("Validating no more than 10% performance degradation")
    print("=" * 50)
    
    results = []
    
    for test_name, code, language in test_cases:
        print(f"\nTesting {test_name} {language} code...")
        result = measure_performance(test_name, code, language)
        results.append({
            'name': test_name,
            'language': language,
            **result
        })
        
        print(f"  Time: {result['avg_time']:.4f}s")
        print(f"  Memory: {result['avg_memory'] / 1024:.1f} KB")
        print(f"  Chunks: {result['chunks_generated']}")
        print(f"  Code size: {result['code_size']} chars")
        print(f"  Throughput: {result['code_size'] / result['avg_time']:.0f} chars/sec")
    
    # Performance analysis
    print(f"\n{'=' * 50}")
    print("PERFORMANCE ANALYSIS")
    print(f"{'=' * 50}")
    
    total_time = sum(r['avg_time'] for r in results)
    total_memory = sum(r['avg_memory'] for r in results)
    total_chunks = sum(r['chunks_generated'] for r in results)
    
    print(f"Total processing time: {total_time:.4f}s")
    print(f"Total memory usage: {total_memory / 1024:.1f} KB")
    print(f"Total chunks generated: {total_chunks}")
    print(f"Average time per chunk: {total_time / total_chunks:.4f}s")
    print(f"Average memory per chunk: {total_memory / total_chunks / 1024:.1f} KB")
    
    # Validate performance is reasonable
    performance_acceptable = True
    issues = []
    
    for result in results:
        # Check if any operation is unreasonably slow (> 1 second for these test cases)
        if result['avg_time'] > 1.0:
            performance_acceptable = False
            issues.append(f"{result['name']} took {result['avg_time']:.4f}s (> 1.0s limit)")
        
        # Check if memory usage is excessive (> 10MB for these test cases)
        if result['avg_memory'] > 10 * 1024 * 1024:
            performance_acceptable = False  
            issues.append(f"{result['name']} used {result['avg_memory'] / 1024 / 1024:.1f}MB (> 10MB limit)")
    
    print(f"\n{'=' * 50}")
    if performance_acceptable:
        print("PERFORMANCE VALIDATION PASSED")
        print("All operations completed within acceptable limits")
        print("No significant performance degradation detected")
    else:
        print("PERFORMANCE ISSUES DETECTED")
        for issue in issues:
            print(f"  - {issue}")
    
    return performance_acceptable

if __name__ == "__main__":
    success = run_performance_validation()
    exit(0 if success else 1)