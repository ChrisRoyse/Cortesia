#!/usr/bin/env python3
"""
Cross-Platform Test Results Comparison Script

This script compares integration test results across different platforms
(Ubuntu, Windows, macOS) to identify platform-specific issues and ensure
consistent behavior across all supported platforms.
"""

import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any, Tuple
import glob


def parse_test_xml(xml_file: Path) -> Dict[str, Any]:
    """Parse JUnit XML test results."""
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        results = {
            'file': str(xml_file),
            'total_tests': int(root.get('tests', 0)),
            'failures': int(root.get('failures', 0)),
            'errors': int(root.get('errors', 0)),
            'skipped': int(root.get('skipped', 0)),
            'time': float(root.get('time', 0.0)),
            'test_cases': []
        }
        
        for testcase in root.findall('.//testcase'):
            test_case = {
                'name': testcase.get('name'),
                'classname': testcase.get('classname'),
                'time': float(testcase.get('time', 0.0)),
                'status': 'passed'
            }
            
            # Check for failures, errors, or skips
            if testcase.find('failure') is not None:
                test_case['status'] = 'failed'
                test_case['failure'] = testcase.find('failure').text
            elif testcase.find('error') is not None:
                test_case['status'] = 'error'
                test_case['error'] = testcase.find('error').text
            elif testcase.find('skipped') is not None:
                test_case['status'] = 'skipped'
                test_case['skip_reason'] = testcase.find('skipped').get('message', '')
            
            results['test_cases'].append(test_case)
        
        return results
    except Exception as e:
        print(f"Error parsing {xml_file}: {e}")
        return None


def parse_performance_json(json_file: Path) -> Dict[str, Any]:
    """Parse performance test results from JSON."""
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error parsing {json_file}: {e}")
        return None


def extract_platform_info(artifact_name: str) -> Tuple[str, str, str]:
    """Extract platform, rust version, and features from artifact name."""
    # Expected format: integration-test-results-{os}-{rust}-{features}
    parts = artifact_name.replace('integration-test-results-', '').split('-')
    
    if len(parts) >= 3:
        os_name = parts[0]
        rust_version = parts[1]
        features = parts[2]
        return os_name, rust_version, features
    
    return 'unknown', 'unknown', 'unknown'


def collect_test_results(input_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Collect all test results organized by platform."""
    results_by_platform = defaultdict(list)
    
    # Find all result directories
    for artifact_dir in input_dir.glob('integration-test-results-*'):
        if artifact_dir.is_dir():
            platform, rust_version, features = extract_platform_info(artifact_dir.name)
            
            platform_results = {
                'platform': platform,
                'rust_version': rust_version,
                'features': features,
                'xml_results': [],
                'performance_results': [],
                'artifact_path': str(artifact_dir)
            }
            
            # Collect XML test results
            for xml_file in artifact_dir.glob('**/*.xml'):
                xml_data = parse_test_xml(xml_file)
                if xml_data:
                    platform_results['xml_results'].append(xml_data)
            
            # Collect JSON performance results
            for json_file in artifact_dir.glob('**/*.json'):
                if 'performance' in json_file.name.lower():
                    json_data = parse_performance_json(json_file)
                    if json_data:
                        platform_results['performance_results'].append(json_data)
            
            results_by_platform[platform].append(platform_results)
    
    return dict(results_by_platform)


def compare_test_coverage(platform_results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Compare test coverage across platforms."""
    coverage_comparison = {
        'platforms': list(platform_results.keys()),
        'test_coverage': {},
        'missing_tests': {},
        'inconsistent_tests': []
    }
    
    # Collect all unique test names across platforms
    all_tests = set()
    tests_by_platform = {}
    
    for platform, results_list in platform_results.items():
        platform_tests = set()
        for result in results_list:
            for xml_result in result['xml_results']:
                for test_case in xml_result['test_cases']:
                    test_name = f"{test_case['classname']}::{test_case['name']}"
                    platform_tests.add(test_name)
                    all_tests.add(test_name)
        tests_by_platform[platform] = platform_tests
    
    # Find missing tests per platform
    for platform, platform_tests in tests_by_platform.items():
        missing = all_tests - platform_tests
        if missing:
            coverage_comparison['missing_tests'][platform] = list(missing)
    
    # Calculate coverage percentages
    for platform, platform_tests in tests_by_platform.items():
        coverage_percentage = len(platform_tests) / len(all_tests) * 100 if all_tests else 0
        coverage_comparison['test_coverage'][platform] = {
            'total_tests': len(platform_tests),
            'coverage_percentage': coverage_percentage
        }
    
    return coverage_comparison


def compare_performance_metrics(platform_results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Compare performance metrics across platforms."""
    performance_comparison = {
        'platforms': list(platform_results.keys()),
        'metrics_comparison': {},
        'performance_outliers': [],
        'scaling_behavior': {}
    }
    
    # Collect all performance metrics
    metrics_by_platform = defaultdict(lambda: defaultdict(list))
    
    for platform, results_list in platform_results.items():
        for result in results_list:
            for perf_result in result['performance_results']:
                if 'metrics' in perf_result:
                    for metric_name, metric_value in perf_result['metrics'].items():
                        if isinstance(metric_value, (int, float)):
                            metrics_by_platform[platform][metric_name].append(metric_value)
    
    # Calculate statistics for each metric
    for platform, metrics in metrics_by_platform.items():
        platform_stats = {}
        for metric_name, values in metrics.items():
            if values:
                platform_stats[metric_name] = {
                    'count': len(values),
                    'min': min(values),
                    'max': max(values),
                    'avg': sum(values) / len(values),
                    'values': values
                }
        performance_comparison['metrics_comparison'][platform] = platform_stats
    
    # Identify performance outliers
    for metric_name in set().union(*[metrics.keys() for metrics in metrics_by_platform.values()]):
        metric_values = {}
        for platform, metrics in metrics_by_platform.items():
            if metric_name in metrics and metrics[metric_name]:
                avg_value = sum(metrics[metric_name]) / len(metrics[metric_name])
                metric_values[platform] = avg_value
        
        if len(metric_values) > 1:
            min_val = min(metric_values.values())
            max_val = max(metric_values.values())
            
            # Consider outlier if difference > 50%
            if max_val > min_val * 1.5:
                outlier_info = {
                    'metric': metric_name,
                    'values': metric_values,
                    'ratio': max_val / min_val,
                    'slowest_platform': max(metric_values, key=metric_values.get),
                    'fastest_platform': min(metric_values, key=metric_values.get)
                }
                performance_comparison['performance_outliers'].append(outlier_info)
    
    return performance_comparison


def compare_error_patterns(platform_results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Compare error patterns across platforms."""
    error_comparison = {
        'platforms': list(platform_results.keys()),
        'platform_specific_failures': {},
        'common_failures': [],
        'failure_rates': {}
    }
    
    # Collect failures by platform
    failures_by_platform = defaultdict(list)
    total_tests_by_platform = defaultdict(int)
    
    for platform, results_list in platform_results.items():
        for result in results_list:
            for xml_result in result['xml_results']:
                total_tests_by_platform[platform] += xml_result['total_tests']
                
                for test_case in xml_result['test_cases']:
                    if test_case['status'] in ['failed', 'error']:
                        failure_info = {
                            'test_name': f"{test_case['classname']}::{test_case['name']}",
                            'status': test_case['status'],
                            'message': test_case.get('failure', test_case.get('error', ''))
                        }
                        failures_by_platform[platform].append(failure_info)
    
    # Calculate failure rates
    for platform, failures in failures_by_platform.items():
        total_tests = total_tests_by_platform[platform]
        failure_rate = len(failures) / total_tests * 100 if total_tests > 0 else 0
        error_comparison['failure_rates'][platform] = {
            'total_tests': total_tests,
            'failed_tests': len(failures),
            'failure_rate_percent': failure_rate
        }
    
    # Find platform-specific failures
    all_failed_tests = set()
    failed_tests_by_platform = {}
    
    for platform, failures in failures_by_platform.items():
        platform_failures = set(f['test_name'] for f in failures)
        failed_tests_by_platform[platform] = platform_failures
        all_failed_tests.update(platform_failures)
    
    for platform, platform_failures in failed_tests_by_platform.items():
        other_platforms = set(platform_results.keys()) - {platform}
        other_failures = set().union(*[failed_tests_by_platform.get(p, set()) for p in other_platforms])
        
        platform_specific = platform_failures - other_failures
        if platform_specific:
            error_comparison['platform_specific_failures'][platform] = list(platform_specific)
    
    # Find common failures (failing on multiple platforms)
    for failed_test in all_failed_tests:
        platforms_with_failure = [p for p, failures in failed_tests_by_platform.items() 
                                 if failed_test in failures]
        if len(platforms_with_failure) > 1:
            error_comparison['common_failures'].append({
                'test_name': failed_test,
                'failing_platforms': platforms_with_failure
            })
    
    return error_comparison


def generate_summary(comparison_results: Dict[str, Any]) -> Dict[str, Any]:
    """Generate overall comparison summary."""
    summary = {
        'timestamp': None,
        'platforms_analyzed': comparison_results.get('test_coverage', {}).get('platforms', []),
        'overall_status': 'UNKNOWN',
        'key_findings': [],
        'recommendations': []
    }
    
    # Analyze test coverage
    test_coverage = comparison_results.get('test_coverage', {})
    if 'test_coverage' in test_coverage:
        min_coverage = min((data['coverage_percentage'] for data in test_coverage['test_coverage'].values()), default=0)
        if min_coverage < 95:
            summary['key_findings'].append(f"Test coverage inconsistency detected (minimum: {min_coverage:.1f}%)")
            summary['recommendations'].append("Review platform-specific test exclusions")
    
    # Analyze performance differences
    performance = comparison_results.get('performance_comparison', {})
    outliers = performance.get('performance_outliers', [])
    if outliers:
        max_ratio = max(o['ratio'] for o in outliers)
        summary['key_findings'].append(f"Performance outliers detected (max ratio: {max_ratio:.1f}x)")
        summary['recommendations'].append("Investigate platform-specific performance bottlenecks")
    
    # Analyze error patterns
    errors = comparison_results.get('error_comparison', {})
    platform_specific = errors.get('platform_specific_failures', {})
    if platform_specific:
        total_specific_failures = sum(len(failures) for failures in platform_specific.values())
        summary['key_findings'].append(f"Platform-specific failures detected ({total_specific_failures} tests)")
        summary['recommendations'].append("Address platform-specific compatibility issues")
    
    # Determine overall status
    if not summary['key_findings']:
        summary['overall_status'] = 'PASS'
    elif len(summary['key_findings']) <= 2:
        summary['overall_status'] = 'WARNING'
    else:
        summary['overall_status'] = 'FAIL'
    
    return summary


def main():
    parser = argparse.ArgumentParser(description='Compare integration test results across platforms')
    parser.add_argument('--input-dir', required=True, type=Path,
                        help='Directory containing platform-specific test result artifacts')
    parser.add_argument('--output', required=True, type=Path,
                        help='Output file for comparison results (JSON)')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    
    args = parser.parse_args()
    
    if not args.input_dir.exists():
        print(f"Error: Input directory {args.input_dir} does not exist")
        sys.exit(1)
    
    if args.verbose:
        print(f"Collecting test results from {args.input_dir}")
    
    # Collect all test results
    platform_results = collect_test_results(args.input_dir)
    
    if not platform_results:
        print("No test results found in input directory")
        sys.exit(1)
    
    if args.verbose:
        print(f"Found results for platforms: {list(platform_results.keys())}")
    
    # Perform comparisons
    comparison_results = {
        'test_coverage': compare_test_coverage(platform_results),
        'performance_comparison': compare_performance_metrics(platform_results),
        'error_comparison': compare_error_patterns(platform_results)
    }
    
    # Generate summary
    comparison_results['summary'] = generate_summary(comparison_results)
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(comparison_results, f, indent=2)
    
    if args.verbose:
        print(f"Comparison results saved to {args.output}")
    
    # Print summary to console
    summary = comparison_results['summary']
    print(f"\nCross-Platform Comparison Summary:")
    print(f"Status: {summary['overall_status']}")
    print(f"Platforms: {', '.join(summary['platforms_analyzed'])}")
    
    if summary['key_findings']:
        print("\nKey Findings:")
        for finding in summary['key_findings']:
            print(f"  • {finding}")
    
    if summary['recommendations']:
        print("\nRecommendations:")
        for recommendation in summary['recommendations']:
            print(f"  • {recommendation}")
    
    # Exit with appropriate code
    if summary['overall_status'] == 'FAIL':
        sys.exit(1)
    elif summary['overall_status'] == 'WARNING':
        sys.exit(0)  # Warnings don't fail CI
    else:
        sys.exit(0)


if __name__ == '__main__':
    main()