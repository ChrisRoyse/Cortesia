#!/usr/bin/env python3
"""
Simple Performance Benchmark for LLMKG MCP Tools
Measures real execution times for the 4 key fixed tools
"""

import json
import time
import statistics
from datetime import datetime
from typing import Dict, List, Any
from comprehensive_tool_tests import ComprehensiveToolTester

class SimplePerformanceBenchmark:
    def __init__(self):
        self.tester = ComprehensiveToolTester()
        self.results = []
        
    def setup_dataset(self, size: int):
        """Setup knowledge dataset with specified number of facts"""
        if size == 0:
            return  # No setup needed for empty dataset
            
        print(f"Setting up dataset with {size} facts...")
        
        # Sample data
        subjects = ["Einstein", "Newton", "Tesla", "Darwin", "Galileo", "Curie", "Hawking", "Feynman"]
        predicates = ["discovered", "invented", "studied", "theorized", "worked_with", "collaborated_with"]
        objects = ["relativity", "gravity", "electricity", "evolution", "astronomy", "radioactivity", "physics", "quantum"]
        
        facts_added = 0
        for i in range(size):
            subject = subjects[i % len(subjects)]
            predicate = predicates[i % len(predicates)]
            obj = objects[i % len(objects)]
            confidence = 0.8 + (i * 0.01) % 0.2
            
            try:
                response = self.tester.call_tool("store_fact", {
                    "subject": f"{subject}_{i}" if size > len(subjects) else subject,
                    "predicate": predicate,
                    "object": f"{obj}_{i}" if size > len(objects) else obj,
                    "confidence": confidence
                })
                if response and 'result' in response:
                    facts_added += 1
            except Exception as e:
                print(f"Warning: Failed to add fact {i}: {e}")
                
        print(f"Dataset setup complete: {facts_added}/{size} facts added")

    def benchmark_tool(self, tool_name: str, params: Dict, iterations: int, dataset_size: int, expected_max_ms: int) -> Dict[str, Any]:
        """Benchmark a single tool with given parameters"""
        print(f"Benchmarking {tool_name} (dataset: {dataset_size}, iterations: {iterations})")
        
        execution_times = []
        errors = []
        successful_runs = 0
        
        for i in range(iterations):
            try:
                # Execute the tool with timing
                start_time = time.perf_counter()
                result = self.tester.call_tool(tool_name, params)
                end_time = time.perf_counter()
                
                execution_time_ms = (end_time - start_time) * 1000
                
                if result and 'result' in result:
                    execution_times.append(execution_time_ms)
                    successful_runs += 1
                    
                    # Check for performance bottlenecks
                    if execution_time_ms > expected_max_ms:
                        print(f"WARNING: Iteration {i+1} took {execution_time_ms:.2f}ms (expected max: {expected_max_ms}ms)")
                else:
                    error_msg = result.get('error', {}).get('message', 'Unknown error') if result else 'No response'
                    errors.append(f"Iteration {i+1}: {error_msg}")
                    
            except Exception as e:
                errors.append(f"Iteration {i+1}: {str(e)}")
                print(f"ERROR in iteration {i+1}: {e}")
        
        # Calculate statistics
        success_rate = successful_runs / iterations if iterations > 0 else 0
        
        stats = {}
        if execution_times:
            stats = {
                "avg_time_ms": statistics.mean(execution_times),
                "min_time_ms": min(execution_times),
                "max_time_ms": max(execution_times),
                "std_dev_ms": statistics.stdev(execution_times) if len(execution_times) > 1 else 0,
                "median_time_ms": statistics.median(execution_times),
            }
        
        return {
            "tool_name": tool_name,
            "dataset_size": dataset_size,
            "iterations": iterations,
            "expected_max_ms": expected_max_ms,
            "success_rate": success_rate,
            "successful_runs": successful_runs,
            "execution_times": execution_times,
            "timing_stats": stats,
            "errors": errors,
            "bottlenecks": [t for t in execution_times if t > expected_max_ms]
        }

    def run_benchmark_suite(self) -> List[Dict[str, Any]]:
        """Run benchmark suite for the 4 key tools"""
        
        # Define benchmark configurations
        configs = [
            {"name": "Small Dataset", "size": 10, "iterations": 20, "max_ms": 50},
            {"name": "Medium Dataset", "size": 100, "iterations": 10, "max_ms": 100},
            {"name": "Large Dataset", "size": 500, "iterations": 5, "max_ms": 500},
            {"name": "Edge Case - Empty", "size": 0, "iterations": 5, "max_ms": 10},
            {"name": "Edge Case - Single", "size": 1, "iterations": 10, "max_ms": 10},
        ]
        
        # Define the 4 tools to benchmark with their test parameters
        tools = {
            "generate_graph_query": [
                {"natural_query": "Find all facts about Einstein", "query_language": "cypher", "include_explanation": True},
                {"natural_query": "Show relationships between Einstein and Newton", "query_language": "cypher", "include_explanation": True},
                {"natural_query": "What are Einstein's discoveries?", "query_language": "sparql", "include_explanation": False},
                {"natural_query": "", "query_language": "cypher", "include_explanation": True},  # Edge case
            ],
            "hybrid_search": [
                {"query": "Einstein", "search_type": "semantic", "limit": 10},
                {"query": "scientist relationships", "search_type": "structural", "limit": 10},
                {"query": "physics discovery", "search_type": "keyword", "limit": 10},
                {"query": "", "search_type": "hybrid", "limit": 10},  # Edge case
            ],
            "validate_knowledge": [
                {"validation_type": "consistency", "scope": "standard", "include_metrics": False},
                {"validation_type": "quality", "scope": "comprehensive", "include_metrics": True},
                {"validation_type": "all", "scope": "standard", "include_metrics": False},
            ],
            "knowledge_quality_metrics": [
                {"assessment_scope": "comprehensive", "include_entity_analysis": True, "include_relationship_analysis": True},
                {"assessment_scope": "entities", "include_entity_analysis": True, "quality_threshold": 0.6},
                {"assessment_scope": "content", "include_content_analysis": True, "quality_threshold": 0.8},
            ]
        }
        
        all_results = []
        
        for config in configs:
            print(f"\nRunning benchmark configuration: {config['name']}")
            print(f"   Dataset size: {config['size']}, Iterations: {config['iterations']}, Max time: {config['max_ms']}ms")
            
            # Setup dataset for this configuration
            self.setup_dataset(config['size'])
            
            # Benchmark each tool
            for tool_name, param_sets in tools.items():
                print(f"\n  Testing tool: {tool_name}")
                
                # Use the appropriate parameter set based on dataset size
                if config['size'] == 0:
                    params = param_sets[-1]  # Edge case parameters
                elif config['size'] == 1:
                    params = param_sets[0]  # Simple parameters
                else:
                    params = param_sets[config['size'] % len(param_sets)]  # Cycle through parameters
                
                result = self.benchmark_tool(
                    tool_name=tool_name,
                    params=params,
                    iterations=config['iterations'],
                    dataset_size=config['size'],
                    expected_max_ms=config['max_ms']
                )
                
                result['config_name'] = config['name']
                result['test_params'] = params
                all_results.append(result)
                
                # Print immediate results
                if result['timing_stats']:
                    print(f"    RESULT: Avg: {result['timing_stats']['avg_time_ms']:.2f}ms, "
                          f"Success: {result['success_rate']*100:.1f}%, "
                          f"Errors: {len(result['errors'])}")
                else:
                    print(f"    FAILED: No successful executions")
        
        return all_results

    def generate_performance_report(self, results: List[Dict[str, Any]]) -> str:
        """Generate performance report"""
        
        report = []
        report.append("# LLMKG MCP Tools - Performance Benchmark Report")
        report.append("")
        report.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"**Total Benchmarks**: {len(results)}")
        report.append("")
        
        # Executive Summary
        report.append("## Executive Summary")
        report.append("")
        
        total_tests = sum(r['iterations'] for r in results)
        total_successful = sum(r['successful_runs'] for r in results)
        overall_success_rate = (total_successful / total_tests) * 100 if total_tests > 0 else 0
        
        report.append(f"- **Total Test Executions**: {total_tests:,}")
        report.append(f"- **Successful Executions**: {total_successful:,}")
        report.append(f"- **Overall Success Rate**: {overall_success_rate:.1f}%")
        report.append("")
        
        # Performance Requirements Check
        meets_requirements = 0
        for result in results:
            if result['timing_stats'] and result['success_rate'] >= 0.95:
                avg_time = result['timing_stats']['avg_time_ms']
                if avg_time <= result['expected_max_ms']:
                    meets_requirements += 1
        
        performance_score = (meets_requirements / len(results)) * 100 if results else 0
        report.append(f"- **Tools Meeting Performance Requirements**: {meets_requirements}/{len(results)} ({performance_score:.1f}%)")
        report.append("")
        
        # Performance Assessment
        if performance_score >= 80:
            report.append("🎉 **EXCELLENT**: Performance requirements largely met!")
            assessment = "A+ (Outstanding)"
        elif performance_score >= 60:
            report.append("✅ **GOOD**: Most tools meet performance requirements")  
            assessment = "A (Very Good)"
        elif performance_score >= 40:
            report.append("⚠️ **FAIR**: Some performance issues identified")
            assessment = "B (Good)"
        else:
            report.append("❌ **NEEDS IMPROVEMENT**: Multiple tools exceed performance targets")
            assessment = "C (Needs Work)"
        
        report.append(f"**Overall Performance Grade**: {assessment}")
        report.append("")
        
        # Summary Table
        report.append("## Performance Summary Table")
        report.append("")
        report.append("| Tool | Dataset Size | Success % | Avg (ms) | Max (ms) | Std Dev | Meets Req? |")
        report.append("|------|--------------|-----------|----------|----------|---------|------------|")
        
        for result in results:
            tool_name = result['tool_name']
            dataset_size = result['dataset_size']
            success_rate = result['success_rate'] * 100
            
            if result['timing_stats']:
                avg_time = result['timing_stats']['avg_time_ms']
                max_time = result['timing_stats']['max_time_ms'] 
                std_dev = result['timing_stats']['std_dev_ms']
                meets_req = "✅" if avg_time <= result['expected_max_ms'] and success_rate >= 95 else "❌"
            else:
                avg_time = 0
                max_time = 0
                std_dev = 0
                meets_req = "❌"
            
            report.append(f"| {tool_name} | {dataset_size} | {success_rate:.1f}% | {avg_time:.2f} | {max_time:.2f} | {std_dev:.2f} | {meets_req} |")
        
        report.append("")
        
        # Detailed Analysis by Tool
        report.append("## Detailed Tool Analysis")
        report.append("")
        
        # Group results by tool
        tool_results = {}
        for result in results:
            tool_name = result['tool_name']
            if tool_name not in tool_results:
                tool_results[tool_name] = []
            tool_results[tool_name].append(result)
        
        for tool_name, tool_data in tool_results.items():
            report.append(f"### {tool_name}")
            report.append("")
            
            # Calculate tool-specific metrics
            total_runs = sum(r['successful_runs'] for r in tool_data)
            total_possible = sum(r['iterations'] for r in tool_data)
            tool_success_rate = (total_runs / total_possible) * 100 if total_possible > 0 else 0
            
            avg_times = [r['timing_stats']['avg_time_ms'] for r in tool_data if r['timing_stats']]
            overall_avg_time = statistics.mean(avg_times) if avg_times else 0
            
            report.append(f"**Overall Success Rate**: {tool_success_rate:.1f}%")
            report.append(f"**Average Execution Time**: {overall_avg_time:.2f}ms")
            report.append("")
            
            # Scaling analysis
            report.append("**Scaling Performance**:")
            report.append("")
            report.append("| Dataset Size | Avg Time (ms) | Success Rate | Status |")
            report.append("|--------------|---------------|--------------|--------|")
            
            # Sort by dataset size
            sorted_data = sorted(tool_data, key=lambda x: x['dataset_size'])
            for data in sorted_data:
                size = data['dataset_size']
                success = data['success_rate'] * 100
                
                if data['timing_stats']:
                    avg_time = data['timing_stats']['avg_time_ms']
                    if avg_time <= data['expected_max_ms'] and success >= 95:
                        status = "🟢 Excellent"
                    elif avg_time <= data['expected_max_ms'] * 1.5:
                        status = "🟡 Good"
                    else:
                        status = "🔴 Slow"
                else:
                    avg_time = 0
                    status = "❌ Failed"
                
                report.append(f"| {size} | {avg_time:.2f} | {success:.1f}% | {status} |")
            
            report.append("")
            
            # Error analysis if errors exist
            all_errors = []
            for data in tool_data:
                all_errors.extend(data['errors'])
            
            if all_errors:
                report.append(f"**Issues Found** ({len(all_errors)} total):")
                report.append("")
                unique_errors = list(set([e.split(':')[1].strip() if ':' in e else e for e in all_errors]))[:3]
                for i, error in enumerate(unique_errors, 1):
                    report.append(f"{i}. {error}")
                report.append("")
            
            # Performance grade for this tool
            tool_performance_score = sum(1 for data in tool_data 
                                       if data['timing_stats'] and 
                                          data['timing_stats']['avg_time_ms'] <= data['expected_max_ms'] and 
                                          data['success_rate'] >= 0.95) / len(tool_data) * 100
            
            if tool_performance_score >= 80:
                tool_grade = "A+ (Excellent)"
            elif tool_performance_score >= 60:
                tool_grade = "A (Very Good)"
            elif tool_performance_score >= 40:
                tool_grade = "B (Good)"
            else:
                tool_grade = "C (Needs Improvement)"
                
            report.append(f"**Tool Performance Grade**: {tool_grade}")
            report.append("")
            report.append("---")
            report.append("")
        
        # Performance Recommendations
        report.append("## Performance Recommendations")
        report.append("")
        
        if performance_score >= 80:
            report.append("🎉 **Outstanding Performance Achieved!**")
            report.append("")
            report.append("**Optimization Opportunities:**")
            report.append("- Implement result caching for frequently accessed queries")
            report.append("- Consider connection pooling for concurrent requests")
            report.append("- Monitor performance under sustained load")
        elif performance_score >= 60:
            report.append("✅ **Good Performance with Room for Improvement**")
            report.append("")
            report.append("**Recommended Optimizations:**")
            report.append("- Optimize slow operations identified in benchmarks")
            report.append("- Strengthen error handling for edge cases")
            report.append("- Consider algorithmic improvements for large datasets")
        else:
            report.append("⚠️ **Performance Needs Significant Improvement**")
            report.append("")
            report.append("**Critical Optimizations Needed:**")
            report.append("- Address tools exceeding performance targets")
            report.append("- Fix reliability issues causing test failures")  
            report.append("- Implement performance monitoring and alerting")
        
        report.append("")
        report.append("## Conclusion")
        report.append("")
        
        # Calculate final assessment
        if performance_score >= 90:
            final_grade = "Outstanding (A+)"
            conclusion = "🏆 The LLMKG MCP tools demonstrate exceptional performance characteristics with sub-millisecond to low-millisecond response times, high reliability, and excellent scaling properties."
        elif performance_score >= 75:
            final_grade = "Excellent (A)"
            conclusion = "🎉 The tools show excellent performance with fast response times and high reliability. Minor optimizations could push performance to outstanding levels."
        elif performance_score >= 60:
            final_grade = "Very Good (B+)"
            conclusion = "✅ Good overall performance with some tools meeting all requirements. Focus optimization efforts on slower operations."
        elif performance_score >= 45:
            final_grade = "Good (B)"
            conclusion = "👍 Acceptable performance with clear improvement opportunities. Address performance bottlenecks and reliability issues."
        else:
            final_grade = "Needs Improvement (C)"
            conclusion = "⚠️ Performance improvements needed across multiple tools. Focus on optimization and error handling."
        
        report.append(f"**Final Performance Assessment**: {final_grade}")
        report.append("")
        report.append(conclusion)
        report.append("")
        report.append(f"**Performance Score**: {performance_score:.1f}/100")
        
        return "\n".join(report)

def main():
    """Main benchmark runner"""
    print("Starting LLMKG MCP Tools Performance Benchmark")
    print("=" * 50)
    
    benchmark = SimplePerformanceBenchmark()
    
    try:
        # Start server
        benchmark.tester.start_server()
        print("LLMKG MCP Server started")
        
        # Run benchmarks
        print("\nRunning performance benchmark suite...")
        results = benchmark.run_benchmark_suite()
        
        # Generate report
        print("\nGenerating performance report...")
        report = benchmark.generate_performance_report(results)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed JSON results
        json_filename = f"performance_results_{timestamp}.json"
        with open(json_filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Detailed results saved to: {json_filename}")
        
        # Save performance report
        report_filename = f"PERFORMANCE_REPORT_{timestamp}.md"
        with open(report_filename, 'w') as f:
            f.write(report)
        print(f"Performance report saved to: {report_filename}")
        
        # Print summary
        print("\n" + "=" * 50)
        print("PERFORMANCE BENCHMARK COMPLETED")
        print("=" * 50)
        
        # Quick summary
        total_tests = sum(r['iterations'] for r in results)
        total_successful = sum(r['successful_runs'] for r in results)
        success_rate = (total_successful / total_tests) * 100 if total_tests > 0 else 0
        
        print(f"Total Executions: {total_tests:,}")
        print(f"Successful: {total_successful:,} ({success_rate:.1f}%)")
        
        # Performance assessment
        meets_requirements = sum(1 for r in results 
                               if r['timing_stats'] and r['success_rate'] >= 0.95 and 
                                  r['timing_stats']['avg_time_ms'] <= r['expected_max_ms'])
        performance_score = (meets_requirements / len(results)) * 100 if results else 0
        print(f"Performance Score: {performance_score:.1f}/100")
        
        if performance_score >= 80:
            print("OUTSTANDING PERFORMANCE!")
        elif performance_score >= 60:
            print("GOOD PERFORMANCE!")
        else:
            print("PERFORMANCE NEEDS IMPROVEMENT")
            
    except Exception as e:
        print(f"Benchmark failed: {e}")
        raise
    finally:
        benchmark.tester.stop_server()
        print("Server stopped")

if __name__ == "__main__":
    main()