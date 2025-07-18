#!/usr/bin/env python3
"""
Cross-Platform Compatibility Verification Script

This script verifies cross-platform compatibility by analyzing comparison results
and validating that performance and behavior differences fall within acceptable
tolerance levels.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple


class CompatibilityVerifier:
    """Verifies cross-platform compatibility based on test results."""
    
    def __init__(self, tolerance: float = 0.1):
        """
        Initialize verifier with tolerance settings.
        
        Args:
            tolerance: Maximum acceptable relative difference (0.1 = 10%)
        """
        self.tolerance = tolerance
        self.issues = []
        self.warnings = []
        
    def verify_test_coverage(self, coverage_data: Dict[str, Any]) -> bool:
        """Verify test coverage consistency across platforms."""
        print("Verifying test coverage consistency...")
        
        if 'test_coverage' not in coverage_data:
            self.issues.append("No test coverage data found")
            return False
        
        coverage_percentages = [
            data['coverage_percentage'] 
            for data in coverage_data['test_coverage'].values()
        ]
        
        if not coverage_percentages:
            self.issues.append("No coverage percentages found")
            return False
        
        min_coverage = min(coverage_percentages)
        max_coverage = max(coverage_percentages)
        
        # Check if all platforms have 100% coverage
        if min_coverage < 100.0:
            platforms_with_missing = [
                platform for platform, data in coverage_data['test_coverage'].items()
                if data['coverage_percentage'] < 100.0
            ]
            self.warnings.append(
                f"Incomplete test coverage on platforms: {', '.join(platforms_with_missing)} "
                f"(minimum: {min_coverage:.1f}%)"
            )
        
        # Check coverage difference
        if max_coverage > 0:
            coverage_diff = (max_coverage - min_coverage) / max_coverage
            if coverage_diff > self.tolerance:
                self.issues.append(
                    f"Test coverage variance ({coverage_diff:.1%}) exceeds tolerance ({self.tolerance:.1%})"
                )
                return False
        
        print(f"✓ Test coverage verification passed (range: {min_coverage:.1f}% - {max_coverage:.1f}%)")
        return True
    
    def verify_performance_compatibility(self, performance_data: Dict[str, Any]) -> bool:
        """Verify performance compatibility across platforms."""
        print("Verifying performance compatibility...")
        
        if 'performance_outliers' not in performance_data:
            self.warnings.append("No performance outlier data found")
            return True
        
        outliers = performance_data['performance_outliers']
        compatibility_issues = []
        
        for outlier in outliers:
            metric = outlier['metric']
            ratio = outlier['ratio']
            slowest = outlier['slowest_platform']
            fastest = outlier['fastest_platform']
            
            # Convert ratio to relative difference
            relative_diff = (ratio - 1.0)
            
            if relative_diff > self.tolerance:
                compatibility_issues.append(
                    f"Metric '{metric}': {slowest} is {ratio:.1f}x slower than {fastest} "
                    f"(difference: {relative_diff:.1%}, tolerance: {self.tolerance:.1%})"
                )
            else:
                print(f"✓ Metric '{metric}' within tolerance ({relative_diff:.1%})")
        
        if compatibility_issues:
            self.issues.extend(compatibility_issues)
            return False
        
        if not outliers:
            print("✓ No performance outliers detected")
        else:
            print(f"✓ All {len(outliers)} performance outliers within tolerance")
        
        return True
    
    def verify_error_compatibility(self, error_data: Dict[str, Any]) -> bool:
        """Verify error pattern compatibility across platforms."""
        print("Verifying error pattern compatibility...")
        
        platform_specific_failures = error_data.get('platform_specific_failures', {})
        failure_rates = error_data.get('failure_rates', {})
        
        # Check for platform-specific failures
        total_specific_failures = sum(len(failures) for failures in platform_specific_failures.values())
        if total_specific_failures > 0:
            for platform, failures in platform_specific_failures.items():
                self.issues.append(
                    f"Platform-specific failures on {platform}: {len(failures)} tests"
                )
                if len(failures) <= 3:  # Show details for small numbers
                    for failure in failures:
                        self.issues.append(f"  - {failure}")
        
        # Check failure rate variance
        if failure_rates:
            rates = [data['failure_rate_percent'] for data in failure_rates.values()]
            min_rate = min(rates)
            max_rate = max(rates)
            
            # Allow some variance in failure rates, but flag major differences
            if max_rate > 0 and (max_rate - min_rate) / max_rate > self.tolerance:
                rate_details = {platform: f"{data['failure_rate_percent']:.1f}%" 
                              for platform, data in failure_rates.items()}
                self.warnings.append(
                    f"Failure rate variance detected: {rate_details}"
                )
        
        if total_specific_failures == 0:
            print("✓ No platform-specific failures detected")
            return True
        else:
            print(f"✗ {total_specific_failures} platform-specific failures detected")
            return False
    
    def verify_resource_usage(self, comparison_data: Dict[str, Any]) -> bool:
        """Verify resource usage compatibility across platforms."""
        print("Verifying resource usage compatibility...")
        
        # Look for memory usage patterns in performance data
        performance_data = comparison_data.get('performance_comparison', {})
        metrics_comparison = performance_data.get('metrics_comparison', {})
        
        memory_metrics = {}
        cpu_metrics = {}
        
        for platform, metrics in metrics_comparison.items():
            for metric_name, metric_data in metrics.items():
                if 'memory' in metric_name.lower():
                    memory_metrics[platform] = memory_metrics.get(platform, {})
                    memory_metrics[platform][metric_name] = metric_data['avg']
                elif 'cpu' in metric_name.lower() or 'time' in metric_name.lower():
                    cpu_metrics[platform] = cpu_metrics.get(platform, {})
                    cpu_metrics[platform][metric_name] = metric_data['avg']
        
        # Verify memory usage consistency
        memory_compatible = self._verify_resource_metric_compatibility(
            memory_metrics, "memory", max_variance=0.5  # Allow 50% variance for memory
        )
        
        # Verify CPU usage consistency
        cpu_compatible = self._verify_resource_metric_compatibility(
            cpu_metrics, "CPU", max_variance=1.0  # Allow 100% variance for CPU timing
        )
        
        return memory_compatible and cpu_compatible
    
    def _verify_resource_metric_compatibility(self, metrics: Dict[str, Dict[str, float]], 
                                           resource_type: str, max_variance: float) -> bool:
        """Helper to verify resource metric compatibility."""
        if not metrics:
            print(f"✓ No {resource_type} metrics to verify")
            return True
        
        compatible = True
        
        # Get all unique metric names
        all_metric_names = set()
        for platform_metrics in metrics.values():
            all_metric_names.update(platform_metrics.keys())
        
        for metric_name in all_metric_names:
            values = []
            platforms = []
            
            for platform, platform_metrics in metrics.items():
                if metric_name in platform_metrics:
                    values.append(platform_metrics[metric_name])
                    platforms.append(platform)
            
            if len(values) > 1:
                min_val = min(values)
                max_val = max(values)
                
                if min_val > 0:
                    variance = (max_val - min_val) / min_val
                    if variance > max_variance:
                        min_platform = platforms[values.index(min_val)]
                        max_platform = platforms[values.index(max_val)]
                        
                        self.warnings.append(
                            f"{resource_type} metric '{metric_name}' variance ({variance:.1%}) "
                            f"exceeds threshold ({max_variance:.1%}): "
                            f"{min_platform}={min_val:.2f}, {max_platform}={max_val:.2f}"
                        )
                        compatible = False
                    else:
                        print(f"✓ {resource_type} metric '{metric_name}' within variance threshold")
        
        return compatible
    
    def verify_compatibility(self, comparison_data: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """
        Perform comprehensive compatibility verification.
        
        Returns:
            Tuple of (is_compatible, verification_report)
        """
        print(f"Starting compatibility verification with tolerance {self.tolerance:.1%}\n")
        
        # Reset issues and warnings
        self.issues = []
        self.warnings = []
        
        # Perform all verification checks
        coverage_ok = self.verify_test_coverage(comparison_data.get('test_coverage', {}))
        performance_ok = self.verify_performance_compatibility(comparison_data.get('performance_comparison', {}))
        errors_ok = self.verify_error_compatibility(comparison_data.get('error_comparison', {}))
        resources_ok = self.verify_resource_usage(comparison_data)
        
        overall_compatible = coverage_ok and performance_ok and errors_ok and resources_ok
        
        # Generate verification report
        report = {
            'compatible': overall_compatible,
            'tolerance_used': self.tolerance,
            'verification_results': {
                'test_coverage': coverage_ok,
                'performance': performance_ok,
                'error_patterns': errors_ok,
                'resource_usage': resources_ok
            },
            'issues': self.issues,
            'warnings': self.warnings,
            'summary': self._generate_verification_summary(overall_compatible)
        }
        
        return overall_compatible, report
    
    def _generate_verification_summary(self, compatible: bool) -> Dict[str, Any]:
        """Generate verification summary."""
        return {
            'status': 'COMPATIBLE' if compatible else 'INCOMPATIBLE',
            'total_issues': len(self.issues),
            'total_warnings': len(self.warnings),
            'recommendation': (
                "Cross-platform compatibility verified within tolerance levels" if compatible
                else "Cross-platform compatibility issues detected - review and address before deployment"
            )
        }


def main():
    parser = argparse.ArgumentParser(description='Verify cross-platform compatibility')
    parser.add_argument('--report', required=True, type=Path,
                        help='Comparison report JSON file')
    parser.add_argument('--tolerance', type=float, default=0.1,
                        help='Maximum acceptable relative difference (default: 0.1 = 10%%)')
    parser.add_argument('--output', type=Path,
                        help='Output file for verification report (JSON)')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    parser.add_argument('--strict', action='store_true',
                        help='Treat warnings as failures')
    
    args = parser.parse_args()
    
    if not args.report.exists():
        print(f"Error: Report file {args.report} does not exist")
        sys.exit(1)
    
    # Load comparison report
    try:
        with open(args.report, 'r') as f:
            comparison_data = json.load(f)
    except Exception as e:
        print(f"Error loading report: {e}")
        sys.exit(1)
    
    # Perform compatibility verification
    verifier = CompatibilityVerifier(tolerance=args.tolerance)
    compatible, verification_report = verifier.verify_compatibility(comparison_data)
    
    # Save verification report if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(verification_report, f, indent=2)
        if args.verbose:
            print(f"\nVerification report saved to {args.output}")
    
    # Print results
    print(f"\n{'='*60}")
    print(f"COMPATIBILITY VERIFICATION RESULTS")
    print(f"{'='*60}")
    
    summary = verification_report['summary']
    print(f"Status: {summary['status']}")
    print(f"Tolerance: {args.tolerance:.1%}")
    print(f"Issues: {summary['total_issues']}")
    print(f"Warnings: {summary['total_warnings']}")
    
    if verification_report['issues']:
        print(f"\nISSUES:")
        for issue in verification_report['issues']:
            print(f"  ✗ {issue}")
    
    if verification_report['warnings']:
        print(f"\nWARNINGS:")
        for warning in verification_report['warnings']:
            print(f"  ⚠ {warning}")
    
    print(f"\nRecommendation:")
    print(f"  {summary['recommendation']}")
    
    # Determine exit code
    if not compatible:
        print(f"\n❌ Compatibility verification FAILED")
        sys.exit(1)
    elif args.strict and verification_report['warnings']:
        print(f"\n⚠️  Compatibility verification passed with WARNINGS (strict mode)")
        sys.exit(1)
    else:
        print(f"\n✅ Compatibility verification PASSED")
        sys.exit(0)


if __name__ == '__main__':
    main()