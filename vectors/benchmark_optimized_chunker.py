#!/usr/bin/env python3
"""
Comprehensive Benchmarking Suite for SmartChunker Optimized
Tests performance on real LLMKG codebase and validates accuracy improvements

Benchmark Categories:
1. Performance Benchmarks - Throughput, memory usage, scalability
2. Accuracy Validation - Documentation detection, relationship preservation  
3. Real-World Testing - LLMKG codebase processing
4. Comparative Analysis - Optimized vs Base implementation
5. Production Readiness - Error handling, edge cases

Target Validation:
- 1M+ chars/sec throughput (10x improvement)
- <1GB memory usage for large batches
- 99%+ accuracy maintained
- Production-ready error handling

Author: Claude (Sonnet 4)
"""

import os
import sys
import time
import json
import logging
import traceback
import gc
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict
import concurrent.futures
import psutil
import matplotlib.pyplot as plt
import numpy as np

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from smart_chunker_optimized import SmartChunkerOptimized, PerformanceMetrics, smart_chunk_content_optimized
from smart_chunker import SmartChunker, smart_chunk_content
from ultra_reliable_core import UniversalDocumentationDetector

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Comprehensive benchmark result"""
    test_name: str
    optimized_metrics: Dict[str, Any]
    baseline_metrics: Dict[str, Any]
    improvement_factors: Dict[str, float]
    accuracy_maintained: bool
    error_count: int
    test_duration: float
    memory_efficiency: float
    success: bool
    details: Dict[str, Any]


class SmartChunkerBenchmark:
    """Comprehensive benchmarking suite for SmartChunker optimizations"""
    
    def __init__(self, llmkg_root: str):
        self.llmkg_root = Path(llmkg_root)
        self.results_dir = Path(__file__).parent / "benchmark_results"
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize chunkers
        self.optimized_chunker = SmartChunkerOptimized(
            max_chunk_size=4000,
            min_chunk_size=200,
            enable_parallel=True,
            max_workers=8,
            memory_limit_mb=1024
        )
        
        self.baseline_chunker = SmartChunker(
            max_chunk_size=4000,
            min_chunk_size=200
        )
        
        self.benchmark_results: List[BenchmarkResult] = []
        
        logger.info(f"Benchmark suite initialized with LLMKG root: {llmkg_root}")
    
    def discover_test_files(self, max_files: int = 50) -> Dict[str, List[str]]:
        """Discover test files from LLMKG codebase"""
        test_files = {
            'rust': [],
            'python': [],
            'javascript': [],
            'typescript': []
        }
        
        # Rust files
        rust_pattern = "**/*.rs"
        for rust_file in self.llmkg_root.glob(rust_pattern):
            if len(test_files['rust']) < max_files // 4:
                test_files['rust'].append(str(rust_file))
        
        # Python files  
        python_pattern = "**/*.py"
        for py_file in self.llmkg_root.glob(python_pattern):
            if len(test_files['python']) < max_files // 4:
                test_files['python'].append(str(py_file))
        
        # JavaScript/TypeScript files
        for pattern, lang in [("**/*.js", 'javascript'), ("**/*.ts", 'typescript')]:
            for js_file in self.llmkg_root.glob(pattern):
                if len(test_files[lang]) < max_files // 4:
                    test_files[lang].append(str(js_file))
        
        total_found = sum(len(files) for files in test_files.values())
        logger.info(f"Discovered {total_found} test files: "
                   f"Rust={len(test_files['rust'])}, "
                   f"Python={len(test_files['python'])}, "
                   f"JS={len(test_files['javascript'])}, "
                   f"TS={len(test_files['typescript'])}")
        
        return test_files
    
    def run_performance_benchmark(self, test_files: Dict[str, List[str]]) -> BenchmarkResult:
        """Run comprehensive performance benchmark"""
        logger.info("Starting performance benchmark...")
        
        start_time = time.time()
        all_files = []
        for file_list in test_files.values():
            all_files.extend(file_list)
        
        if not all_files:
            logger.warning("No test files found for performance benchmark")
            return BenchmarkResult(
                test_name="performance_benchmark",
                optimized_metrics={},
                baseline_metrics={},
                improvement_factors={},
                accuracy_maintained=False,
                error_count=1,
                test_duration=0.0,
                memory_efficiency=0.0,
                success=False,
                details={"error": "No test files found"}
            )
        
        # Benchmark optimized version
        logger.info("Benchmarking optimized chunker...")
        optimized_start = time.time()
        optimized_results = self.optimized_chunker.benchmark_performance(all_files, iterations=3)
        optimized_duration = time.time() - optimized_start
        
        # Benchmark baseline version  
        logger.info("Benchmarking baseline chunker...")
        baseline_start = time.time()
        baseline_results = self._benchmark_baseline_chunker(all_files)
        baseline_duration = time.time() - baseline_start
        
        # Calculate improvements
        improvement_factors = {}
        if baseline_results.get('avg_throughput_chars_per_sec', 0) > 0:
            improvement_factors['throughput'] = (
                optimized_results['benchmark_summary']['avg_throughput_chars_per_sec'] /
                baseline_results['avg_throughput_chars_per_sec']
            )
        
        if baseline_results.get('avg_memory_peak_mb', float('inf')) > 0:
            improvement_factors['memory_efficiency'] = (
                baseline_results['avg_memory_peak_mb'] /
                optimized_results['benchmark_summary']['avg_memory_peak_mb']
            )
        
        # Check if targets are met
        avg_throughput = optimized_results['benchmark_summary']['avg_throughput_chars_per_sec']
        target_achieved = avg_throughput >= 1_000_000  # 1M chars/sec
        
        test_duration = time.time() - start_time
        
        result = BenchmarkResult(
            test_name="performance_benchmark",
            optimized_metrics=optimized_results['benchmark_summary'],
            baseline_metrics=baseline_results,
            improvement_factors=improvement_factors,
            accuracy_maintained=True,  # Will be validated separately
            error_count=optimized_results['benchmark_summary'].get('errors_encountered', 0),
            test_duration=test_duration,
            memory_efficiency=improvement_factors.get('memory_efficiency', 1.0),
            success=target_achieved,
            details={
                'target_throughput_achieved': target_achieved,
                'target_throughput': 1_000_000,
                'actual_throughput': avg_throughput,
                'optimized_duration': optimized_duration,
                'baseline_duration': baseline_duration,
                'detailed_optimized': optimized_results,
                'detailed_baseline': baseline_results
            }
        )
        
        self.benchmark_results.append(result)
        
        logger.info(f"Performance benchmark completed: "
                   f"{avg_throughput:.0f} chars/sec, "
                   f"{improvement_factors.get('throughput', 0):.1f}x improvement")
        
        return result
    
    def run_accuracy_validation(self, test_files: Dict[str, List[str]]) -> BenchmarkResult:
        """Run accuracy validation benchmark"""
        logger.info("Starting accuracy validation...")
        
        start_time = time.time()
        
        # Create test cases from real files
        test_cases = self._create_accuracy_test_cases(test_files)
        
        if not test_cases:
            logger.warning("No test cases created for accuracy validation")
            return BenchmarkResult(
                test_name="accuracy_validation",
                optimized_metrics={},
                baseline_metrics={},
                improvement_factors={},
                accuracy_maintained=False,
                error_count=1,
                test_duration=0.0,
                memory_efficiency=0.0,
                success=False,
                details={"error": "No test cases created"}
            )
        
        # Test optimized version
        optimized_accuracy = self.optimized_chunker.validate_accuracy(test_cases)
        
        # Test baseline version
        baseline_accuracy = self._validate_baseline_accuracy(test_cases)
        
        # Calculate accuracy metrics
        optimized_acc = optimized_accuracy['accuracy_summary']['accuracy_percentage']
        baseline_acc = baseline_accuracy['accuracy_percentage']
        
        accuracy_maintained = optimized_acc >= 99.0 and optimized_acc >= baseline_acc * 0.95
        
        improvement_factors = {
            'accuracy_ratio': optimized_acc / baseline_acc if baseline_acc > 0 else 1.0
        }
        
        test_duration = time.time() - start_time
        
        result = BenchmarkResult(
            test_name="accuracy_validation",
            optimized_metrics=optimized_accuracy['accuracy_summary'],
            baseline_metrics=baseline_accuracy,
            improvement_factors=improvement_factors,
            accuracy_maintained=accuracy_maintained,
            error_count=0,
            test_duration=test_duration,
            memory_efficiency=1.0,
            success=accuracy_maintained,
            details={
                'target_accuracy': 99.0,
                'optimized_accuracy': optimized_acc,
                'baseline_accuracy': baseline_acc,
                'detailed_optimized': optimized_accuracy,
                'detailed_baseline': baseline_accuracy,
                'test_cases_count': len(test_cases)
            }
        )
        
        self.benchmark_results.append(result)
        
        logger.info(f"Accuracy validation completed: "
                   f"{optimized_acc:.1f}% accuracy (target: 99%+)")
        
        return result
    
    def run_scalability_test(self, test_files: Dict[str, List[str]]) -> BenchmarkResult:
        """Test scalability with different file counts"""
        logger.info("Starting scalability test...")
        
        start_time = time.time()
        
        # Test with different file counts
        file_counts = [1, 5, 10, 25, 50]
        scalability_results = {}
        
        all_files = []
        for file_list in test_files.values():
            all_files.extend(file_list)
        
        if not all_files:
            logger.warning("No files available for scalability test")
            return BenchmarkResult(
                test_name="scalability_test",
                optimized_metrics={},
                baseline_metrics={},
                improvement_factors={},
                accuracy_maintained=False,
                error_count=1,
                test_duration=0.0,
                memory_efficiency=0.0,
                success=False,
                details={"error": "No files available"}
            )
        
        for count in file_counts:
            if count > len(all_files):
                continue
                
            test_subset = all_files[:count]
            
            # Test optimized chunker
            opt_start = time.time()
            opt_results = self.optimized_chunker.chunk_files_batch(test_subset)
            opt_duration = time.time() - opt_start
            
            # Test baseline chunker
            base_start = time.time()
            base_results = self._chunk_files_baseline(test_subset)
            base_duration = time.time() - base_start
            
            scalability_results[count] = {
                'optimized_duration': opt_duration,
                'baseline_duration': base_duration,
                'optimized_chunks': sum(len(chunks) for chunks in opt_results.values()),
                'baseline_chunks': sum(len(chunks) for chunks in base_results.values()),
                'speedup': base_duration / opt_duration if opt_duration > 0 else float('inf')
            }
        
        # Calculate scalability metrics
        avg_speedup = np.mean([r['speedup'] for r in scalability_results.values() if r['speedup'] != float('inf')])
        linear_scalability = self._check_linear_scalability(scalability_results)
        
        test_duration = time.time() - start_time
        
        result = BenchmarkResult(
            test_name="scalability_test",
            optimized_metrics={'avg_speedup': avg_speedup, 'linear_scalability': linear_scalability},
            baseline_metrics={'avg_speedup': 1.0, 'linear_scalability': True},
            improvement_factors={'scalability_factor': avg_speedup},
            accuracy_maintained=True,
            error_count=0,
            test_duration=test_duration, 
            memory_efficiency=1.0,
            success=avg_speedup > 2.0 and linear_scalability,
            details={
                'file_count_results': scalability_results,
                'avg_speedup': avg_speedup,
                'linear_scalability': linear_scalability
            }
        )
        
        self.benchmark_results.append(result)
        
        logger.info(f"Scalability test completed: {avg_speedup:.1f}x average speedup")
        
        return result
    
    def run_memory_efficiency_test(self, test_files: Dict[str, List[str]]) -> BenchmarkResult:
        """Test memory efficiency and garbage collection"""
        logger.info("Starting memory efficiency test...")
        
        start_time = time.time()
        
        all_files = []
        for file_list in test_files.values():
            all_files.extend(file_list)
        
        if not all_files:
            logger.warning("No files available for memory test")
            return BenchmarkResult(
                test_name="memory_efficiency_test",
                optimized_metrics={},
                baseline_metrics={},
                improvement_factors={},
                accuracy_maintained=False,
                error_count=1,
                test_duration=0.0,
                memory_efficiency=0.0,
                success=False,
                details={"error": "No files available"}
            )
        
        # Test memory usage patterns
        process = psutil.Process()
        
        # Baseline memory measurement
        gc.collect()
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Test optimized chunker memory usage
        opt_start_memory = process.memory_info().rss / 1024 / 1024
        opt_results = self.optimized_chunker.chunk_files_batch(all_files)
        opt_peak_memory = max(opt_start_memory, process.memory_info().rss / 1024 / 1024)
        gc.collect()
        opt_end_memory = process.memory_info().rss / 1024 / 1024
        
        # Test baseline chunker memory usage
        base_start_memory = process.memory_info().rss / 1024 / 1024
        base_results = self._chunk_files_baseline(all_files)
        base_peak_memory = max(base_start_memory, process.memory_info().rss / 1024 / 1024)
        gc.collect()
        base_end_memory = process.memory_info().rss / 1024 / 1024
        
        # Calculate memory metrics
        opt_memory_used = opt_peak_memory - opt_start_memory
        base_memory_used = base_peak_memory - base_start_memory
        
        memory_efficiency = base_memory_used / opt_memory_used if opt_memory_used > 0 else 1.0
        memory_target_met = opt_peak_memory < 1024  # <1GB target
        
        test_duration = time.time() - start_time
        
        result = BenchmarkResult(
            test_name="memory_efficiency_test",
            optimized_metrics={
                'peak_memory_mb': opt_peak_memory,
                'memory_used_mb': opt_memory_used,
                'memory_leaked_mb': opt_end_memory - opt_start_memory
            },
            baseline_metrics={
                'peak_memory_mb': base_peak_memory,
                'memory_used_mb': base_memory_used,
                'memory_leaked_mb': base_end_memory - base_start_memory
            },
            improvement_factors={'memory_efficiency': memory_efficiency},
            accuracy_maintained=True,
            error_count=0,
            test_duration=test_duration,
            memory_efficiency=memory_efficiency,
            success=memory_target_met and memory_efficiency >= 1.0,
            details={
                'memory_target_met': memory_target_met,
                'memory_target_mb': 1024,
                'optimized_peak_mb': opt_peak_memory,
                'baseline_peak_mb': base_peak_memory,
                'efficiency_ratio': memory_efficiency
            }
        )
        
        self.benchmark_results.append(result)
        
        logger.info(f"Memory efficiency test completed: "
                   f"{opt_peak_memory:.1f}MB peak, "
                   f"{memory_efficiency:.1f}x efficiency")
        
        return result
    
    def run_error_handling_test(self) -> BenchmarkResult:
        """Test error handling and recovery"""
        logger.info("Starting error handling test...")
        
        start_time = time.time()
        
        # Create problematic test cases
        error_test_cases = [
            ("", "rust", "empty_file"),
            ("invalid syntax here!", "python", "syntax_error"),  
            ("a" * 10000000, "javascript", "huge_file"),  # 10MB file
            ("//\n" * 100000, "rust", "many_empty_lines"),
            ("invalid unicode: \x00\xff", "python", "encoding_issue"),
        ]
        
        optimized_errors = 0
        baseline_errors = 0
        
        # Test optimized error handling
        for content, language, test_name in error_test_cases:
            try:
                chunks = smart_chunk_content_optimized(content, language, f"test_{test_name}")
                # Should handle gracefully without crashing
            except Exception as e:
                optimized_errors += 1
                logger.debug(f"Optimized chunker error on {test_name}: {str(e)}")
        
        # Test baseline error handling
        for content, language, test_name in error_test_cases:
            try:
                chunks = smart_chunk_content(content, language, f"test_{test_name}")
            except Exception as e:
                baseline_errors += 1
                logger.debug(f"Baseline chunker error on {test_name}: {str(e)}")
        
        # Error handling should be better (fewer errors) in optimized version
        error_handling_improved = optimized_errors <= baseline_errors
        
        test_duration = time.time() - start_time
        
        result = BenchmarkResult(
            test_name="error_handling_test",
            optimized_metrics={'errors': optimized_errors},
            baseline_metrics={'errors': baseline_errors},
            improvement_factors={'error_reduction': (baseline_errors - optimized_errors + 1) / (optimized_errors + 1)},
            accuracy_maintained=True,
            error_count=optimized_errors,
            test_duration=test_duration,
            memory_efficiency=1.0,
            success=error_handling_improved,
            details={
                'test_cases_count': len(error_test_cases),
                'optimized_errors': optimized_errors,
                'baseline_errors': baseline_errors,
                'error_handling_improved': error_handling_improved
            }
        )
        
        self.benchmark_results.append(result)
        
        logger.info(f"Error handling test completed: "
                   f"{optimized_errors} vs {baseline_errors} errors")
        
        return result
    
    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run complete benchmark suite"""
        logger.info("Starting comprehensive benchmark suite...")
        
        start_time = time.time()
        
        # Discover test files
        test_files = self.discover_test_files(max_files=50)
        
        # Run all benchmarks
        benchmarks = [
            ("Performance Benchmark", lambda: self.run_performance_benchmark(test_files)),
            ("Accuracy Validation", lambda: self.run_accuracy_validation(test_files)),
            ("Scalability Test", lambda: self.run_scalability_test(test_files)),
            ("Memory Efficiency Test", lambda: self.run_memory_efficiency_test(test_files)),
            ("Error Handling Test", lambda: self.run_error_handling_test()),
        ]
        
        successful_tests = 0
        total_tests = len(benchmarks)
        
        for test_name, test_func in benchmarks:
            try:
                logger.info(f"Running {test_name}...")
                result = test_func()
                if result.success:
                    successful_tests += 1
                logger.info(f"{test_name} completed: {'PASS' if result.success else 'FAIL'}")
            except Exception as e:
                logger.error(f"Error in {test_name}: {str(e)}")
                logger.debug(traceback.format_exc())
        
        total_duration = time.time() - start_time
        
        # Calculate overall metrics
        overall_success_rate = successful_tests / total_tests if total_tests > 0 else 0
        overall_improvement = self._calculate_overall_improvement()
        
        # Generate summary report
        summary = {
            'benchmark_summary': {
                'total_tests': total_tests,
                'successful_tests': successful_tests,
                'success_rate': overall_success_rate,
                'total_duration': total_duration,
                'overall_improvement_factor': overall_improvement,
                'production_ready': overall_success_rate >= 0.8 and overall_improvement >= 5.0
            },
            'individual_results': [asdict(result) for result in self.benchmark_results],
            'recommendations': self._generate_recommendations()
        }
        
        # Save results
        self._save_benchmark_results(summary)
        
        logger.info(f"Comprehensive benchmark completed: "
                   f"{successful_tests}/{total_tests} tests passed, "
                   f"{overall_improvement:.1f}x overall improvement")
        
        return summary
    
    def _benchmark_baseline_chunker(self, files: List[str]) -> Dict[str, Any]:
        """Benchmark the baseline chunker"""
        start_time = time.time()
        total_chars = 0
        total_chunks = 0
        peak_memory = 0
        
        process = psutil.Process()
        
        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                language = self._detect_language(file_path)
                chunks = smart_chunk_content(content, language, file_path)
                
                total_chars += len(content)
                total_chunks += len(chunks)
                
                current_memory = process.memory_info().rss / 1024 / 1024
                peak_memory = max(peak_memory, current_memory)
                
            except Exception as e:
                logger.debug(f"Error processing {file_path}: {str(e)}")
                continue
        
        duration = time.time() - start_time
        
        return {
            'avg_throughput_chars_per_sec': total_chars / duration if duration > 0 else 0,
            'avg_memory_peak_mb': peak_memory,
            'total_chunks': total_chunks,
            'duration': duration
        }
    
    def _validate_baseline_accuracy(self, test_cases: List[Tuple[str, str, str]]) -> Dict[str, Any]:
        """Validate baseline chunker accuracy"""
        correct_detections = 0
        total_cases = len(test_cases)
        
        for content, language, expected in test_cases:
            try:
                chunks = smart_chunk_content(content, language, "test_case")
                has_doc = any(chunk.has_documentation for chunk in chunks)
                expected_doc = "documentation" in expected.lower()
                
                if (has_doc and expected_doc) or (not has_doc and not expected_doc):
                    correct_detections += 1
            except Exception:
                continue
        
        return {
            'accuracy_percentage': (correct_detections / total_cases) * 100 if total_cases > 0 else 0,
            'total_cases': total_cases,
            'correct_detections': correct_detections
        }
    
    def _chunk_files_baseline(self, files: List[str]) -> Dict[str, List]:
        """Chunk files using baseline chunker"""
        results = {}
        
        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                language = self._detect_language(file_path)
                chunks = smart_chunk_content(content, language, file_path)
                results[file_path] = chunks
                
            except Exception as e:
                logger.debug(f"Error processing {file_path}: {str(e)}")
                results[file_path] = []
        
        return results
    
    def _create_accuracy_test_cases(self, test_files: Dict[str, List[str]]) -> List[Tuple[str, str, str]]:
        """Create accuracy test cases from real files"""
        test_cases = []
        
        for language, files in test_files.items():
            for file_path in files[:5]:  # Use first 5 files per language
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    # Determine if file has documentation
                    detector = UniversalDocumentationDetector()
                    doc_result = detector.detect_documentation_multi_pass(content, language)
                    
                    expected = "has documentation" if doc_result['has_documentation'] else "no documentation"
                    test_cases.append((content, language, expected))
                    
                except Exception as e:
                    logger.debug(f"Error creating test case from {file_path}: {str(e)}")
                    continue
        
        return test_cases[:20]  # Limit to 20 test cases for performance
    
    def _check_linear_scalability(self, results: Dict[int, Dict]) -> bool:
        """Check if performance scales linearly with file count"""
        if len(results) < 2:
            return True
        
        file_counts = sorted(results.keys())
        durations = [results[count]['optimized_duration'] for count in file_counts]
        
        # Calculate if duration grows roughly linearly
        # Simple check: does duration increase reasonably with file count?
        for i in range(1, len(durations)):
            if durations[i] < durations[i-1] * 0.5:  # Duration decreased significantly
                return False
        
        return True
    
    def _detect_language(self, file_path: str) -> str:
        """Detect programming language from file extension"""
        ext = Path(file_path).suffix.lower()
        
        extension_map = {
            '.rs': 'rust',
            '.py': 'python', 
            '.js': 'javascript',
            '.jsx': 'javascript',
            '.ts': 'typescript',
            '.tsx': 'typescript'
        }
        
        return extension_map.get(ext, 'unknown')
    
    def _calculate_overall_improvement(self) -> float:
        """Calculate overall improvement factor across all benchmarks"""
        improvements = []
        
        for result in self.benchmark_results:
            if 'throughput' in result.improvement_factors:
                improvements.append(result.improvement_factors['throughput'])
            elif 'scalability_factor' in result.improvement_factors:
                improvements.append(result.improvement_factors['scalability_factor'])
            elif 'memory_efficiency' in result.improvement_factors:
                improvements.append(result.improvement_factors['memory_efficiency'])
        
        return np.mean(improvements) if improvements else 1.0
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on benchmark results"""
        recommendations = []
        
        # Performance recommendations
        perf_results = [r for r in self.benchmark_results if r.test_name == "performance_benchmark"]
        if perf_results and perf_results[0].success:
            recommendations.append("✅ Performance targets met - ready for production deployment")
        else:
            recommendations.append("⚠️  Performance targets not met - consider additional optimizations")
        
        # Accuracy recommendations
        acc_results = [r for r in self.benchmark_results if r.test_name == "accuracy_validation"]
        if acc_results and acc_results[0].accuracy_maintained:
            recommendations.append("✅ Accuracy maintained at 99%+ - documentation detection reliable")
        else:
            recommendations.append("⚠️  Accuracy below 99% - review documentation detection logic")
        
        # Memory recommendations
        mem_results = [r for r in self.benchmark_results if r.test_name == "memory_efficiency_test"]
        if mem_results and mem_results[0].success:
            recommendations.append("✅ Memory usage optimized - suitable for large codebases")
        else:
            recommendations.append("⚠️  High memory usage - implement additional memory optimizations")
        
        # Error handling recommendations
        err_results = [r for r in self.benchmark_results if r.test_name == "error_handling_test"]
        if err_results and err_results[0].success:
            recommendations.append("✅ Error handling robust - production ready")
        else:
            recommendations.append("⚠️  Error handling needs improvement - add more edge case handling")
        
        return recommendations
    
    def _save_benchmark_results(self, summary: Dict[str, Any]):
        """Save benchmark results to JSON file"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_file = self.results_dir / f"benchmark_results_{timestamp}.json"
        
        try:
            with open(results_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            logger.info(f"Benchmark results saved to: {results_file}")
        except Exception as e:
            logger.error(f"Failed to save benchmark results: {str(e)}")
    
    def generate_performance_plots(self, summary: Dict[str, Any]):
        """Generate performance visualization plots"""
        try:
            # Create performance comparison plots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('SmartChunker Optimization Performance Analysis', fontsize=16)
            
            # Extract data for plotting
            results = summary['individual_results']
            
            # Performance metrics plot
            perf_result = next((r for r in results if r['test_name'] == 'performance_benchmark'), None)
            if perf_result:
                metrics = ['throughput', 'memory_efficiency']
                improvements = [perf_result['improvement_factors'].get(m, 1.0) for m in metrics]
                
                axes[0, 0].bar(metrics, improvements, color=['skyblue', 'lightgreen'])
                axes[0, 0].set_title('Performance Improvements')
                axes[0, 0].set_ylabel('Improvement Factor (x)')
                axes[0, 0].axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Baseline')
                axes[0, 0].legend()
            
            # Accuracy comparison
            acc_result = next((r for r in results if r['test_name'] == 'accuracy_validation'), None)
            if acc_result:
                categories = ['Optimized', 'Baseline']
                accuracies = [
                    acc_result['optimized_metrics'].get('accuracy_percentage', 0),
                    acc_result['baseline_metrics'].get('accuracy_percentage', 0)
                ]
                
                axes[0, 1].bar(categories, accuracies, color=['orange', 'gray'])
                axes[0, 1].set_title('Accuracy Comparison')
                axes[0, 1].set_ylabel('Accuracy (%)')
                axes[0, 1].axhline(y=99, color='red', linestyle='--', alpha=0.7, label='Target 99%')
                axes[0, 1].legend()
            
            # Memory usage comparison
            mem_result = next((r for r in results if r['test_name'] == 'memory_efficiency_test'), None)
            if mem_result:
                categories = ['Optimized', 'Baseline']
                memory_usage = [
                    mem_result['optimized_metrics'].get('peak_memory_mb', 0),
                    mem_result['baseline_metrics'].get('peak_memory_mb', 0)
                ]
                
                axes[1, 0].bar(categories, memory_usage, color=['lightcoral', 'gray'])
                axes[1, 0].set_title('Memory Usage Comparison')
                axes[1, 0].set_ylabel('Peak Memory (MB)')
                axes[1, 0].axhline(y=1024, color='red', linestyle='--', alpha=0.7, label='Target <1GB')
                axes[1, 0].legend()
            
            # Overall success rate
            success_rates = [r['success'] for r in results]
            test_names = [r['test_name'].replace('_', ' ').title() for r in results]
            
            axes[1, 1].bar(test_names, [1 if s else 0 for s in success_rates], 
                          color=['green' if s else 'red' for s in success_rates])
            axes[1, 1].set_title('Test Success Rate')
            axes[1, 1].set_ylabel('Success (1) / Failure (0)')
            axes[1, 1].set_xticklabels(test_names, rotation=45, ha='right')
            
            plt.tight_layout()
            
            # Save plot
            plot_file = self.results_dir / f"performance_analysis_{time.strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            logger.info(f"Performance plots saved to: {plot_file}")
            
            plt.close()
            
        except Exception as e:
            logger.error(f"Failed to generate performance plots: {str(e)}")


def main():
    """Main benchmark execution"""
    
    # Check if LLMKG root is accessible
    llmkg_root = Path(__file__).parent.parent
    if not llmkg_root.exists():
        logger.error(f"LLMKG root directory not found: {llmkg_root}")
        return
    
    # Initialize benchmark suite
    benchmark = SmartChunkerBenchmark(str(llmkg_root))
    
    # Run comprehensive benchmark
    logger.info("=" * 60)
    logger.info("SmartChunker Optimization Benchmark Suite")
    logger.info("=" * 60)
    
    try:
        results = benchmark.run_comprehensive_benchmark()
        
        # Print summary
        summary = results['benchmark_summary']
        print("\n" + "=" * 60)
        print("BENCHMARK RESULTS SUMMARY")
        print("=" * 60)
        print(f"Tests Passed: {summary['successful_tests']}/{summary['total_tests']}")
        print(f"Success Rate: {summary['success_rate']:.1%}")
        print(f"Overall Improvement: {summary['overall_improvement_factor']:.1f}x")
        print(f"Production Ready: {'YES' if summary['production_ready'] else 'NO'}")
        print(f"Total Duration: {summary['total_duration']:.1f} seconds")
        
        print("\nRECOMMENDATIONS:")
        for rec in results['recommendations']:
            print(f"  {rec}")
        
        # Generate visualization
        benchmark.generate_performance_plots(results)
        
        print(f"\nDetailed results saved to: {benchmark.results_dir}")
        
    except Exception as e:
        logger.error(f"Benchmark failed: {str(e)}")
        logger.debug(traceback.format_exc())
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())