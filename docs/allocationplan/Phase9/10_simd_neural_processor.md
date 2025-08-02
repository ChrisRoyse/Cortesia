# Micro-Phase 9.10: Implement SIMD Neural Processing

## Objective
Implement SIMD-accelerated neural processing for WebAssembly, enabling efficient parallel computation of activations, inhibition, and spike timing.

## Prerequisites
- Completed micro-phase 9.09 (SIMD setup)
- Understanding of WASM SIMD intrinsics
- Neural processing algorithms from Phase 1

## Task Description
Create high-performance SIMD implementations of neural processing operations including activation updates, lateral inhibition, and TTFS encoding.

## Specific Actions

1. **Create SIMD neural processor core**:
   ```rust
   // src/simd/neural_processor.rs
   use std::arch::wasm32::*;
   use wasm_bindgen::prelude::*;
   
   #[wasm_bindgen]
   pub struct SIMDNeuralProcessor {
       // Aligned buffers for SIMD operations
       weights: Vec<f32>,
       activations: Vec<f32>,
       spike_times: Vec<f32>,
       inhibition_matrix: Vec<f32>,
       
       // Configuration
       neuron_count: usize,
       use_simd: bool,
   }
   
   #[wasm_bindgen]
   impl SIMDNeuralProcessor {
       #[wasm_bindgen(constructor)]
       pub fn new(neuron_count: usize, use_simd: bool) -> Self {
           // Ensure alignment for SIMD
           let aligned_count = (neuron_count + 3) & !3; // Round up to multiple of 4
           
           Self {
               weights: vec![0.0; aligned_count],
               activations: vec![0.0; aligned_count],
               spike_times: vec![0.0; aligned_count],
               inhibition_matrix: vec![0.0; aligned_count * aligned_count],
               neuron_count,
               use_simd,
           }
       }
       
       #[wasm_bindgen]
       pub fn set_weights(&mut self, weights: &[f32]) {
           self.weights[..weights.len()].copy_from_slice(weights);
       }
       
       #[wasm_bindgen]
       pub fn get_activations(&self) -> Vec<f32> {
           self.activations[..self.neuron_count].to_vec()
       }
   }
   ```

2. **Implement SIMD activation processing**:
   ```rust
   impl SIMDNeuralProcessor {
       #[inline]
       pub unsafe fn process_activations_simd(&mut self, inputs: &[f32]) -> &[f32] {
           let len = inputs.len();
           let simd_len = len & !3; // Process in groups of 4
           
           // Process 4 neurons at a time using SIMD
           for i in (0..simd_len).step_by(4) {
               // Load 4 inputs
               let input_vec = v128_load(inputs[i..].as_ptr() as *const v128);
               
               // Load 4 weights
               let weight_vec = v128_load(self.weights[i..].as_ptr() as *const v128);
               
               // Multiply inputs by weights
               let weighted = f32x4_mul(input_vec, weight_vec);
               
               // Apply activation function (ReLU with threshold)
               let threshold = f32x4_splat(0.5);
               let zeros = f32x4_splat(0.0);
               let activated = f32x4_max(f32x4_sub(weighted, threshold), zeros);
               
               // Apply decay
               let decay = f32x4_splat(0.95);
               let current = v128_load(self.activations[i..].as_ptr() as *const v128);
               let decayed = f32x4_mul(current, decay);
               let updated = f32x4_add(activated, decayed);
               
               // Store results
               v128_store(self.activations[i..].as_mut_ptr() as *mut v128, updated);
           }
           
           // Handle remaining elements
           for i in simd_len..len {
               let weighted = inputs[i] * self.weights[i];
               let activated = (weighted - 0.5).max(0.0);
               self.activations[i] = activated + self.activations[i] * 0.95;
           }
           
           &self.activations[..self.neuron_count]
       }
       
       #[wasm_bindgen]
       pub fn process_activations(&mut self, inputs: Vec<f32>) -> Vec<f32> {
           if self.use_simd && inputs.len() >= 4 {
               unsafe { self.process_activations_simd(&inputs).to_vec() }
           } else {
               self.process_activations_scalar(&inputs).to_vec()
           }
       }
       
       fn process_activations_scalar(&mut self, inputs: &[f32]) -> &[f32] {
           for i in 0..inputs.len().min(self.neuron_count) {
               let weighted = inputs[i] * self.weights[i];
               let activated = (weighted - 0.5).max(0.0);
               self.activations[i] = activated + self.activations[i] * 0.95;
           }
           &self.activations[..self.neuron_count]
       }
   }
   ```

3. **Implement SIMD lateral inhibition**:
   ```rust
   impl SIMDNeuralProcessor {
       #[inline]
       pub unsafe fn compute_lateral_inhibition_simd(&mut self) -> &[f32] {
           let n = self.neuron_count;
           let mut inhibited = vec![0.0f32; n];
           
           // Process inhibition in 4x4 blocks
           for i in (0..n).step_by(4) {
               if i + 4 <= n {
                   let act_vec = v128_load(&self.activations[i] as *const f32 as *const v128);
                   
                   for j in (0..n).step_by(4) {
                       if j + 4 <= n && i != j {
                           // Load inhibition weights for this 4x4 block
                           let mut total_inhibition = f32x4_splat(0.0);
                           
                           for k in 0..4 {
                               let row_idx = (i + k) * n + j;
                               if row_idx + 4 <= self.inhibition_matrix.len() {
                                   let inhib_weights = v128_load(
                                       &self.inhibition_matrix[row_idx] as *const f32 as *const v128
                                   );
                                   
                                   // Load target activations
                                   let target_acts = v128_load(
                                       &self.activations[j] as *const f32 as *const v128
                                   );
                                   
                                   // Compute inhibition contribution
                                   let contrib = f32x4_mul(inhib_weights, target_acts);
                                   total_inhibition = f32x4_add(total_inhibition, contrib);
                               }
                           }
                           
                           // Apply inhibition with activation scaling
                           let inhibition = f32x4_mul(act_vec, total_inhibition);
                           let current = v128_load(&inhibited[i] as *const f32 as *const v128);
                           let updated = f32x4_add(current, inhibition);
                           v128_store(&mut inhibited[i] as *mut f32 as *mut v128, updated);
                       }
                   }
               }
           }
           
           // Apply inhibition to activations
           for i in 0..n {
               self.activations[i] = (self.activations[i] - inhibited[i]).max(0.0);
           }
           
           &self.activations[..n]
       }
       
       #[wasm_bindgen]
       pub fn apply_lateral_inhibition(&mut self) -> Vec<f32> {
           if self.use_simd && self.neuron_count >= 4 {
               unsafe { self.compute_lateral_inhibition_simd().to_vec() }
           } else {
               self.compute_lateral_inhibition_scalar().to_vec()
           }
       }
   }
   ```

4. **Implement SIMD TTFS encoding**:
   ```rust
   impl SIMDNeuralProcessor {
       #[inline]
       pub unsafe fn ttfs_encoding_simd(&mut self, complexities: &[f32]) -> &[f32] {
           let len = complexities.len().min(self.neuron_count);
           let simd_len = len & !3;
           
           // SIMD constants
           let base_time = f32x4_splat(0.1); // 100μs base
           let scale_factor = f32x4_splat(0.8); // 800μs range
           let one = f32x4_splat(1.0);
           
           for i in (0..simd_len).step_by(4) {
               let complexity = v128_load(&complexities[i] as *const f32 as *const v128);
               
               // TTFS: base_time + scale_factor * (1 - complexity)
               let inverted = f32x4_sub(one, complexity);
               let scaled = f32x4_mul(scale_factor, inverted);
               let spike_time = f32x4_add(base_time, scaled);
               
               v128_store(&mut self.spike_times[i] as *mut f32 as *mut v128, spike_time);
           }
           
           // Handle remaining
           for i in simd_len..len {
               self.spike_times[i] = 0.1 + 0.8 * (1.0 - complexities[i]);
           }
           
           &self.spike_times[..len]
       }
       
       #[wasm_bindgen]
       pub fn encode_ttfs(&mut self, complexities: Vec<f32>) -> Vec<f32> {
           if self.use_simd && complexities.len() >= 4 {
               unsafe { self.ttfs_encoding_simd(&complexities).to_vec() }
           } else {
               self.encode_ttfs_scalar(&complexities).to_vec()
           }
       }
   }
   ```

5. **Create benchmarking utilities**:
   ```rust
   #[wasm_bindgen]
   impl SIMDNeuralProcessor {
       #[wasm_bindgen]
       pub fn benchmark_simd_speedup(&mut self, iterations: u32) -> f64 {
           let test_input: Vec<f32> = (0..self.neuron_count)
               .map(|i| (i as f32) / (self.neuron_count as f32))
               .collect();
           
           // Benchmark scalar version
           let start_scalar = js_sys::Date::now();
           for _ in 0..iterations {
               self.use_simd = false;
               self.process_activations(test_input.clone());
           }
           let scalar_time = js_sys::Date::now() - start_scalar;
           
           // Benchmark SIMD version
           let start_simd = js_sys::Date::now();
           for _ in 0..iterations {
               self.use_simd = true;
               self.process_activations(test_input.clone());
           }
           let simd_time = js_sys::Date::now() - start_simd;
           
           self.use_simd = true; // Re-enable SIMD
           scalar_time / simd_time // Speedup factor
       }
   }
   ```

6. **Add tests for SIMD operations**:
   ```rust
   #[cfg(test)]
   mod tests {
       use super::*;
       
       #[test]
       fn test_simd_activation_correctness() {
           let mut processor = SIMDNeuralProcessor::new(8, true);
           let inputs = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
           processor.set_weights(&vec![1.0; 8]);
           
           let simd_result = processor.process_activations(inputs.clone());
           
           processor.use_simd = false;
           let scalar_result = processor.process_activations(inputs);
           
           for (simd, scalar) in simd_result.iter().zip(scalar_result.iter()) {
               assert!((simd - scalar).abs() < 1e-6);
           }
       }
       
       #[test]
       fn test_simd_ttfs_encoding() {
           let mut processor = SIMDNeuralProcessor::new(8, true);
           let complexities = vec![0.1, 0.3, 0.5, 0.7, 0.9, 0.2, 0.4, 0.6];
           
           let spike_times = processor.encode_ttfs(complexities);
           
           // Verify TTFS property: higher complexity = earlier spike
           for i in 0..spike_times.len() - 1 {
               if complexities[i] < complexities[i + 1] {
                   assert!(spike_times[i] > spike_times[i + 1]);
               }
           }
       }
   }
   ```

## Expected Outputs
- Complete SIMD neural processor implementation
- 4x parallel processing for activations
- Optimized lateral inhibition computation
- SIMD-accelerated TTFS encoding
- Benchmarking utilities showing speedup
- Correctness tests for all operations

## Validation
1. SIMD operations produce same results as scalar versions
2. Performance improvement of 2-4x over scalar
3. No memory alignment issues
4. All tests pass with SIMD enabled
5. Works in browsers with WASM SIMD support

## Next Steps
- Create IndexedDB wrapper (micro-phase 9.11)
- Implement storage schema (micro-phase 9.12)