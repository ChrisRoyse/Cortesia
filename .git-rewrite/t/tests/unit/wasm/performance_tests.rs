//! WASM Performance Unit Tests

use crate::unit::*;
use crate::unit::test_utils::*;
use crate::wasm::fast_interface::*;

#[cfg(test)]
mod performance_tests {
    use super::*;

    #[test]
    fn test_wasm_call_overhead() {
        let mut interface = WasmInterface::new();
        let call_count = 10000;
        
        // Measure WASM call overhead
        let (_, wasm_time) = measure_execution_time(|| {
            for i in 0..call_count {
                let _ = interface.create_entity(&format!("entity_{}", i), &format!("Entity {}", i));
            }
        });
        
        println!("WASM entity creation time for {} calls: {:?}", call_count, wasm_time);
        
        let calls_per_second = call_count as f64 / wasm_time.as_secs_f64();
        assert!(calls_per_second > 1000.0, "WASM calls too slow: {:.0} ops/sec", calls_per_second);
    }
}