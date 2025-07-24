#!/usr/bin/env python3
"""
Performance Benchmark Runner for LLMKG MCP Tools
Measures real execution times, memory usage, and scaling characteristics
"""

import asyncio
import time
import json
import statistics
import psutil
import os
from datetime import datetime
from typing import Dict, List, Any, Tuple
import subprocess
import sys

# MCP client imports
sys.path.append('.')
from comprehensive_tool_tests import ComprehensiveToolTester

class PerformanceBenchmark:
    def __init__(self):
        self.client = None
        self.results = []
        
    async def setup(self):
        """Initialize MCP client connection"""
        self.client = ComprehensiveToolTester()
        self.client.start_server()
        print("✅ Connected to LLMKG MCP server")
        
    async def cleanup(self):
        """Clean up MCP client connection"""
        if self.client:
            self.client.stop_server()
        print("✅ Disconnected from MCP server")

    def measure_memory(self) -> int:
        """Get current memory usage in bytes"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss

    async def benchmark_tool(self, tool_name: str, params: Dict, iterations: int, 
                           dataset_size: int, expected_max_ms: int) -> Dict[str, Any]:
        """Benchmark a single tool with given parameters"""
        print(f"🚀 Benchmarking {tool_name} (dataset: {dataset_size}, iterations: {iterations})")
        
        execution_times = []
        memory_usages = []
        errors = []
        successful_runs = 0
        
        # Pre-benchmark memory baseline
        baseline_memory = self.measure_memory()
        
        for i in range(iterations):
            try:
                # Measure memory before
                memory_before = self.measure_memory()
                
                # Execute the tool with timing
                start_time = time.perf_counter()
                result = self.client.call_tool(tool_name, params)
                end_time = time.perf_counter()
                
                # Measure memory after
                memory_after = self.measure_memory()
                
                execution_time_ms = (end_time - start_time) * 1000
                memory_delta = memory_after - memory_before
                
                execution_times.append(execution_time_ms)
                memory_usages.append(max(0, memory_delta))
                successful_runs += 1
                
                # Check for performance bottlenecks
                if execution_time_ms > expected_max_ms:
                    print(f"⚠️  Iteration {i+1} took {execution_time_ms:.2f}ms (expected max: {expected_max_ms}ms)")
                
            except Exception as e:
                errors.append(f"Iteration {i+1}: {str(e)}")
                print(f"❌ Error in iteration {i+1}: {e}")
        
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
                "p95_time_ms": statistics.quantiles(execution_times, n=20)[18] if len(execution_times) >= 20 else max(execution_times),
            }
        
        memory_stats = {}
        if memory_usages:
            memory_stats = {
                "avg_memory_bytes": statistics.mean(memory_usages),
                "max_memory_bytes": max(memory_usages),
                "total_memory_kb": (self.measure_memory() - baseline_memory) / 1024
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
            "memory_stats": memory_stats,
            "errors": errors,
            "bottlenecks": [t for t in execution_times if t > expected_max_ms]
        }

    async def setup_dataset(self, size: int) -> None:
        """Setup knowledge dataset with specified number of facts"""
        if size == 0:
            return  # No setup needed for empty dataset
            
        print(f"📝 Setting up dataset with {size} facts...")
        
        # Sample data for different sizes
        subjects = ["Einstein", "Newton", "Tesla", "Darwin", "Galileo", "Curie", "Hawking", "Feynman", "Bohr", "Schrödinger"]
        predicates = ["discovered", "invented", "studied", "theorized", "worked_with", "collaborated_with", "influenced", "developed"]
        objects = ["relativity", "gravity", "electricity", "evolution", "astronomy", "radioactivity", "physics", "quantum", "mechanics", "theory"]
        
        facts_added = 0
        for i in range(size):
            subject = subjects[i % len(subjects)]
            predicate = predicates[i % len(predicates)]
            obj = objects[i % len(objects)]
            confidence = 0.8 + (i * 0.01) % 0.2
            
            try:
                self.client.call_tool("store_fact", {
                    "subject": f"{subject}_{i}" if size > len(subjects) else subject,
                    "predicate": predicate,
                    "object": f"{obj}_{i}" if size > len(objects) else obj,
                    "confidence": confidence
                })
                facts_added += 1
            except Exception as e:
                print(f"Warning: Failed to add fact {i}: {e}")
                
        print(f"✅ Dataset setup complete: {facts_added}/{size} facts added")

    async def run_benchmark_suite(self) -> List[Dict[str, Any]]:
        """Run comprehensive benchmark suite for all 4 tools"""
        
        # Define benchmark configurations
        configs = [
            {"name": "Small Dataset", "size": 10, "iterations": 100, "max_ms": 10},
            {"name": "Medium Dataset", "size": 1000, "iterations": 50, "max_ms": 100},
            {"name": "Large Dataset", "size": 10000, "iterations": 10, "max_ms": 1000},
            {"name": "Edge Case - Empty", "size": 0, "iterations": 10, "max_ms": 5},
            {"name": "Edge Case - Single", "size": 1, "iterations": 20, "max_ms": 5},
        ]
        
        # Define the 4 tools to benchmark with their test parameters
        tools = {
            "generate_graph_query": [
                {"natural_query": "Find all facts about Einstein", "query_language": "cypher", "include_explanation": True},
                {"natural_query": "Show relationships between Einstein and Newton", "query_language": "cypher", "include_explanation": True},
                {"natural_query": "What are Einstein's discoveries?", "query_language": "sparql", "include_explanation": False},
                {"natural_query": "Find connections between scientists", "query_language": "gremlin", "include_explanation": True},
                {"natural_query": "", "query_language": "cypher", "include_explanation": True},  # Edge case
            ],
            "hybrid_search": [
                {"query": "Einstein", "search_type": "semantic", "limit": 10},
                {"query": "scientist relationships", "search_type": "structural", "limit": 10},
                {"query": "physics discovery", "search_type": "keyword", "limit": 10},
                {"query": "Einstein Newton connection", "search_type": "hybrid", "limit": 10},
                {"query": "", "search_type": "hybrid", "limit": 10},  # Edge case
            ],
            "validate_knowledge": [
                {"validation_type": "consistency", "scope": "standard", "include_metrics": False},
                {"validation_type": "conflicts", "scope": "standard", "include_metrics": False},
                {"validation_type": "quality", "scope": "comprehensive", "include_metrics": True},
                {"validation_type": "completeness", "entity": "Einstein", "scope": "standard"},
                {"validation_type": "all", "scope": "comprehensive", "include_metrics": True},
            ],
            "knowledge_quality_metrics": [
                {"assessment_scope": "comprehensive", "include_entity_analysis": True, "include_relationship_analysis": True},
                {"assessment_scope": "entities", "include_entity_analysis": True, "quality_threshold": 0.6},
                {"assessment_scope": "relationships", "include_relationship_analysis": True, "quality_threshold": 0.7},
                {"assessment_scope": "content", "include_content_analysis": True, "quality_threshold": 0.8},
                {"assessment_scope": "comprehensive", "quality_threshold": 0.5},  # Lower threshold test
            ]
        }
        
        all_results = []
        
        for config in configs:
            print(f"\n🎯 Running benchmark configuration: {config['name']}")
            print(f"   Dataset size: {config['size']}, Iterations: {config['iterations']}, Max time: {config['max_ms']}ms")
            
            # Setup dataset for this configuration
            await self.setup_dataset(config['size'])
            
            # Benchmark each tool
            for tool_name, param_sets in tools.items():
                print(f"\n  🔧 Testing tool: {tool_name}")
                
                # Use the appropriate parameter set based on dataset size
                if config['size'] == 0:
                    params = param_sets[4]  # Edge case parameters
                elif config['size'] == 1:
                    params = param_sets[0]  # Simple parameters
                else:
                    params = param_sets[config['size'] % len(param_sets)]  # Cycle through parameters
                
                result = await self.benchmark_tool(
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
                    print(f"    ✅ Avg: {result['timing_stats']['avg_time_ms']:.2f}ms, "
                          f"Success: {result['success_rate']*100:.1f}%, "
                          f"Errors: {len(result['errors'])}")
                else:
                    print(f"    ❌ No successful executions")
        
        return all_results

    def generate_performance_report(self, results: List[Dict[str, Any]]) -> str:
        """Generate comprehensive performance report"""
        
        report = []
        report.append("# LLMKG MCP Tools - Real Performance Benchmark Report")
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
        report.append("### Performance Requirements Validation")
        report.append("")
        meets_requirements = 0
        for result in results:
            if result['timing_stats'] and result['success_rate'] >= 0.95:
                avg_time = result['timing_stats']['avg_time_ms']
                if avg_time <= result['expected_max_ms']:
                    meets_requirements += 1
        
        performance_score = (meets_requirements / len(results)) * 100 if results else 0
        report.append(f"- **Tools Meeting Performance Requirements**: {meets_requirements}/{len(results)} ({performance_score:.1f}%)")
        report.append("")
        
        if performance_score >= 80:
            report.append("🎉 **EXCELLENT**: Performance requirements largely met!")
        elif performance_score >= 60:
            report.append("✅ **GOOD**: Most tools meet performance requirements")
        else:
            report.append("⚠️ **NEEDS IMPROVEMENT**: Multiple tools exceed performance targets")
        report.append("")
        
        # Summary Table
        report.append("## Performance Summary Table")
        report.append("")
        report.append("| Tool | Dataset | Config | Success % | Avg (ms) | Max (ms) | Memory (KB) | Meets Req? |")
        report.append("|------|---------|---------|-----------|----------|----------|-------------|------------|")
        
        for result in results:
            tool_name = result['tool_name']
            dataset_size = result['dataset_size']
            config_name = result['config_name']
            success_rate = result['success_rate'] * 100
            
            if result['timing_stats']:
                avg_time = result['timing_stats']['avg_time_ms']
                max_time = result['timing_stats']['max_time_ms']
                meets_req = "✅" if avg_time <= result['expected_max_ms'] and success_rate >= 95 else "❌"
            else:
                avg_time = 0
                max_time = 0
                meets_req = "❌"
            
            memory_kb = result['memory_stats'].get('total_memory_kb', 0) if result['memory_stats'] else 0
            
            report.append(f"| {tool_name} | {dataset_size} | {config_name} | {success_rate:.1f}% | {avg_time:.2f} | {max_time:.2f} | {memory_kb:.1f} | {meets_req} |")
        
        report.append("")
        
        # Detailed Analysis by Tool
        report.append("## Detailed Performance Analysis")
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
            report.append("**Scaling Analysis**:")
            report.append("")
            report.append("| Dataset Size | Avg Time (ms) | Success Rate | Memory (KB) | Status |")
            report.append("|--------------|---------------|--------------|-------------|--------|")
            
            # Sort by dataset size
            sorted_data = sorted(tool_data, key=lambda x: x['dataset_size'])
            for data in sorted_data:
                size = data['dataset_size']
                success = data['success_rate'] * 100
                memory_kb = data['memory_stats'].get('total_memory_kb', 0) if data['memory_stats'] else 0
                
                if data['timing_stats']:
                    avg_time = data['timing_stats']['avg_time_ms']
                    status = "✅ Good" if avg_time <= data['expected_max_ms'] and success >= 95 else "⚠️ Slow" if avg_time > data['expected_max_ms'] else "❌ Errors"
                else:
                    avg_time = 0
                    status = "❌ Failed"
                
                report.append(f"| {size} | {avg_time:.2f} | {success:.1f}% | {memory_kb:.1f} | {status} |")
            
            report.append("")
            
            # Error analysis
            all_errors = []
            for data in tool_data:
                all_errors.extend(data['errors'])
            
            if all_errors:
                report.append(f"**Error Analysis** ({len(all_errors)} total errors):")
                report.append("")
                unique_errors = list(set(all_errors))[:5]  # Show up to 5 unique errors
                for i, error in enumerate(unique_errors, 1):
                    report.append(f"{i}. {error}")
                if len(all_errors) > 5:
                    report.append(f"... and {len(all_errors) - 5} more errors")
                report.append("")
            
            # Performance bottlenecks
            bottleneck_count = sum(len(data['bottlenecks']) for data in tool_data)
            if bottleneck_count > 0:
                report.append(f"**Performance Bottlenecks**: {bottleneck_count} instances of slow execution")
                report.append("")
            
            report.append("---")
            report.append("")
        
        # Scaling Characteristics Analysis
        report.append("## Scaling Characteristics Analysis")
        report.append("")
        
        for tool_name, tool_data in tool_results.items():
            sorted_data = sorted(tool_data, key=lambda x: x['dataset_size'])
            if len(sorted_data) >= 3:  # Need at least 3 data points for scaling analysis
                
                # Calculate time complexity
                scaling_behavior = self.analyze_time_complexity(sorted_data)
                
                report.append(f"### {tool_name} Scaling")
                report.append(f"**Estimated Time Complexity**: {scaling_behavior}")
                report.append("")
        
        # Recommendations
        report.append("## Performance Recommendations")
        report.append("")
        
        recommendations = self.generate_recommendations(results)
        for rec in recommendations:
            report.append(f"- {rec}")
        
        report.append("")
        report.append("## Conclusion")
        report.append("")
        
        if performance_score >= 80:
            report.append("🎉 **EXCELLENT PERFORMANCE**: The LLMKG MCP tools demonstrate outstanding performance characteristics:")
            report.append("- Sub-millisecond to millisecond response times")
            report.append("- High reliability with 95%+ success rates")
            report.append("- Efficient memory usage")
            report.append("- Good scaling properties")
        elif performance_score >= 60:
            report.append("✅ **GOOD PERFORMANCE**: The tools perform well with some optimization opportunities:")
            report.append("- Generally fast response times")
            report.append("- Reliable operation in most scenarios") 
            report.append("- Room for improvement in edge cases")
        else:
            report.append("⚠️ **PERFORMANCE NEEDS IMPROVEMENT**: Several issues identified:")
            report.append("- Some tools exceed performance targets")
            report.append("- Error handling needs strengthening")
            report.append("- Optimization opportunities exist")
        
        report.append("")
        report.append(f"**Overall Assessment**: {performance_score:.1f}/100")
        
        return "\n".join(report)

    def analyze_time_complexity(self, sorted_data: List[Dict[str, Any]]) -> str:
        """Analyze time complexity based on scaling behavior"""
        sizes = []
        times = []
        
        for data in sorted_data:
            if data['dataset_size'] > 0 and data['timing_stats']:
                sizes.append(data['dataset_size'])
                times.append(data['timing_stats']['avg_time_ms'])
        
        if len(sizes) < 3:
            return "Insufficient data"
        
        # Calculate ratios
        time_ratios = []
        size_ratios = []
        
        for i in range(1, len(sizes)):
            if times[i-1] > 0:
                time_ratio = times[i] / times[i-1]
                size_ratio = sizes[i] / sizes[i-1]
                time_ratios.append(time_ratio)
                size_ratios.append(size_ratio)
        
        if not time_ratios:
            return "Unable to determine"
        
        avg_time_ratio = statistics.mean(time_ratios)
        avg_size_ratio = statistics.mean(size_ratios)
        
        # Classify complexity
        if avg_time_ratio <= 1.1:
            return "O(1) - Constant time ⭐⭐⭐⭐⭐"
        elif avg_time_ratio <= avg_size_ratio ** 0.5:
            return "O(√n) - Sub-linear ⭐⭐⭐⭐"
        elif avg_time_ratio <= avg_size_ratio * 1.2:
            return "O(n) - Linear ⭐⭐⭐"
        elif avg_time_ratio <= avg_size_ratio * (avg_size_ratio ** 0.5):
            return "O(n√n) - Super-linear ⭐⭐"
        elif avg_time_ratio <= avg_size_ratio ** 2:
            return "O(n²) - Quadratic ⭐"
        else:
            return "O(n²+) - Polynomial or worse ❌"

    def generate_recommendations(self, results: List[Dict[str, Any]]) -> List[str]:
        """Generate performance recommendations based on results"""
        recommendations = []
        
        # Analyze overall performance
        slow_tools = []
        error_prone_tools = []
        memory_heavy_tools = []
        
        for result in results:
            tool_name = result['tool_name']
            
            # Check for slow performance
            if result['timing_stats']:
                avg_time = result['timing_stats']['avg_time_ms']
                if avg_time > result['expected_max_ms']:
                    slow_tools.append((tool_name, avg_time, result['expected_max_ms']))
            
            # Check for errors
            if result['success_rate'] < 0.95:
                error_prone_tools.append((tool_name, result['success_rate'] * 100))
            
            # Check for memory usage
            if result['memory_stats'] and result['memory_stats'].get('total_memory_kb', 0) > 1000:
                memory_heavy_tools.append((tool_name, result['memory_stats']['total_memory_kb']))
        
        # Generate specific recommendations
        if slow_tools:
            recommendations.append("**Performance Optimization needed for slow tools:**")
            for tool, actual, expected in slow_tools[:3]:
                recommendations.append(f"  - {tool}: {actual:.1f}ms avg (target: {expected}ms) - Consider caching or algorithmic improvements")
        
        if error_prone_tools:
            recommendations.append("**Reliability improvements needed:**")
            for tool, success_rate in error_prone_tools[:3]:
                recommendations.append(f"  - {tool}: {success_rate:.1f}% success rate - Strengthen error handling and input validation")
        
        if memory_heavy_tools:
            recommendations.append("**Memory optimization opportunities:**")
            for tool, memory_kb in memory_heavy_tools[:3]:
                recommendations.append(f"  - {tool}: {memory_kb:.1f}KB usage - Consider streaming or pagination for large datasets")
        
        # General recommendations
        recommendations.extend([
            "**General Optimization Strategies:**",
            "  - Implement result caching for frequently accessed data",
            "  - Add connection pooling for database operations", 
            "  - Consider async batching for bulk operations",
            "  - Monitor performance under concurrent load",
            "  - Implement performance regression testing in CI/CD"
        ])
        
        return recommendations

async def main():
    """Main benchmark runner"""
    print("🚀 Starting LLMKG MCP Tools Performance Benchmark Suite")
    print("=" * 60)
    
    benchmark = PerformanceBenchmark()
    
    try:
        # Setup
        await benchmark.setup()
        
        # Run benchmarks
        print("\n📊 Running comprehensive benchmark suite...")
        results = await benchmark.run_benchmark_suite()
        
        # Generate report
        print("\n📝 Generating performance report...")
        report = benchmark.generate_performance_report(results)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed JSON results
        json_filename = f"performance_benchmark_results_{timestamp}.json"
        with open(json_filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"✅ Detailed results saved to: {json_filename}")
        
        # Save performance report
        report_filename = f"PERFORMANCE_BENCHMARK_REPORT_{timestamp}.md"
        with open(report_filename, 'w') as f:
            f.write(report)
        print(f"✅ Performance report saved to: {report_filename}")
        
        # Print summary
        print("\n" + "=" * 60)
        print("🎉 BENCHMARK SUITE COMPLETED SUCCESSFULLY")
        print("=" * 60)
        
        # Quick summary stats
        total_tests = sum(r['iterations'] for r in results)
        total_successful = sum(r['successful_runs'] for r in results)
        success_rate = (total_successful / total_tests) * 100 if total_tests > 0 else 0
        
        print(f"📊 Total Test Executions: {total_tests:,}")
        print(f"✅ Successful Executions: {total_successful:,}")
        print(f"📈 Overall Success Rate: {success_rate:.1f}%")
        
        # Performance summary
        meets_requirements = 0
        for result in results:
            if result['timing_stats'] and result['success_rate'] >= 0.95:
                avg_time = result['timing_stats']['avg_time_ms']
                if avg_time <= result['expected_max_ms']:
                    meets_requirements += 1
        
        performance_score = (meets_requirements / len(results)) * 100 if results else 0
        print(f"🎯 Performance Requirements Met: {meets_requirements}/{len(results)} ({performance_score:.1f}%)")
        
        if performance_score >= 80:
            print("🏆 EXCELLENT: Outstanding performance across all tools!")
        elif performance_score >= 60:
            print("👍 GOOD: Strong performance with optimization opportunities")
        else:
            print("⚠️  NEEDS WORK: Performance improvements needed")
            
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        raise
    finally:
        await benchmark.cleanup()

if __name__ == "__main__":
    asyncio.run(main())