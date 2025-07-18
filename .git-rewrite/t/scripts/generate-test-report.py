#!/usr/bin/env python3

"""
Integration Test Report Generator
Processes test results and generates comprehensive HTML reports
"""

import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd


class TestReportGenerator:
    def __init__(self, input_dir: str, output_file: str, scenario: str):
        self.input_dir = Path(input_dir)
        self.output_file = Path(output_file)
        self.scenario = scenario
        self.test_results = {}
        self.performance_data = {}
        self.resource_usage = {}
        
    def collect_test_results(self) -> None:
        """Collect test results from XML files"""
        for xml_file in self.input_dir.glob("*-results.xml"):
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()
                
                suite_name = xml_file.stem.replace("-results", "")
                
                # Extract test suite information
                test_suite = {
                    'name': suite_name,
                    'tests': int(root.get('tests', 0)),
                    'failures': int(root.get('failures', 0)),
                    'errors': int(root.get('errors', 0)),
                    'time': float(root.get('time', 0)),
                    'test_cases': []
                }
                
                # Extract individual test cases
                for testcase in root.findall('.//testcase'):
                    test_case = {
                        'name': testcase.get('name'),
                        'time': float(testcase.get('time', 0)),
                        'status': 'passed'
                    }
                    
                    if testcase.find('failure') is not None:
                        test_case['status'] = 'failed'
                        test_case['failure'] = testcase.find('failure').text
                    elif testcase.find('error') is not None:
                        test_case['status'] = 'error'
                        test_case['error'] = testcase.find('error').text
                    
                    test_suite['test_cases'].append(test_case)
                
                self.test_results[suite_name] = test_suite
                
            except ET.ParseError as e:
                print(f"Error parsing {xml_file}: {e}")
                
    def collect_performance_data(self) -> None:
        """Collect performance metrics from JSON files"""
        for json_file in self.input_dir.glob("*-performance.json"):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                suite_name = json_file.stem.replace("-performance", "")
                self.performance_data[suite_name] = data
                
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error reading {json_file}: {e}")
                
    def collect_resource_usage(self) -> None:
        """Collect resource usage data from CSV files"""
        for csv_file in self.input_dir.glob("*-resource-usage.csv"):
            try:
                df = pd.read_csv(csv_file)
                suite_name = csv_file.stem.replace("-resource-usage", "")
                self.resource_usage[suite_name] = df
                
            except Exception as e:
                print(f"Error reading {csv_file}: {e}")
                
    def generate_summary_stats(self) -> Dict:
        """Generate summary statistics"""
        total_tests = sum(suite['tests'] for suite in self.test_results.values())
        total_failures = sum(suite['failures'] for suite in self.test_results.values())
        total_errors = sum(suite['errors'] for suite in self.test_results.values())
        total_time = sum(suite['time'] for suite in self.test_results.values())
        
        return {
            'total_suites': len(self.test_results),
            'total_tests': total_tests,
            'total_failures': total_failures,
            'total_errors': total_errors,
            'total_time': total_time,
            'success_rate': ((total_tests - total_failures - total_errors) / total_tests * 100) if total_tests > 0 else 0
        }
        
    def generate_performance_charts(self) -> Dict[str, str]:
        """Generate performance charts and return file paths"""
        charts = {}
        
        # Test execution times
        if self.test_results:
            suite_names = list(self.test_results.keys())
            suite_times = [self.test_results[name]['time'] for name in suite_names]
            
            plt.figure(figsize=(10, 6))
            plt.bar(suite_names, suite_times)
            plt.title('Test Suite Execution Times')
            plt.xlabel('Test Suite')
            plt.ylabel('Time (seconds)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            chart_path = self.input_dir / 'execution_times.png'
            plt.savefig(chart_path)
            plt.close()
            charts['execution_times'] = str(chart_path)
            
        # Resource usage over time
        if self.resource_usage:
            for suite_name, df in self.resource_usage.items():
                plt.figure(figsize=(12, 8))
                
                # CPU and Memory usage
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
                
                # CPU usage
                ax1.plot(df.index, df['cpu_percent'], label='CPU %', color='blue')
                ax1.set_title(f'{suite_name} - CPU Usage')
                ax1.set_ylabel('CPU %')
                ax1.legend()
                
                # Memory usage
                ax2.plot(df.index, df['memory_mb'], label='Memory MB', color='red')
                ax2.set_title(f'{suite_name} - Memory Usage')
                ax2.set_ylabel('Memory (MB)')
                ax2.set_xlabel('Time')
                ax2.legend()
                
                plt.tight_layout()
                
                chart_path = self.input_dir / f'{suite_name}_resource_usage.png'
                plt.savefig(chart_path)
                plt.close()
                charts[f'{suite_name}_resources'] = str(chart_path)
                
        return charts
        
    def generate_html_report(self) -> None:
        """Generate comprehensive HTML report"""
        summary = self.generate_summary_stats()
        charts = self.generate_performance_charts()
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LLMKG Integration Test Report - {self.scenario}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .header {{
            border-bottom: 3px solid #007acc;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        .header h1 {{
            color: #007acc;
            margin: 0;
        }}
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .summary-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}
        .summary-card h3 {{
            margin: 0 0 10px 0;
            font-size: 1.2em;
        }}
        .summary-card .value {{
            font-size: 2em;
            font-weight: bold;
        }}
        .test-suite {{
            margin-bottom: 30px;
            border: 1px solid #ddd;
            border-radius: 8px;
            overflow: hidden;
        }}
        .test-suite-header {{
            background-color: #f8f9fa;
            padding: 15px;
            border-bottom: 1px solid #ddd;
        }}
        .test-suite-header h3 {{
            margin: 0;
            color: #333;
        }}
        .test-case {{
            padding: 10px 15px;
            border-bottom: 1px solid #eee;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .test-case:last-child {{
            border-bottom: none;
        }}
        .status {{
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.8em;
            font-weight: bold;
        }}
        .status.passed {{
            background-color: #d4edda;
            color: #155724;
        }}
        .status.failed {{
            background-color: #f8d7da;
            color: #721c24;
        }}
        .status.error {{
            background-color: #fff3cd;
            color: #856404;
        }}
        .charts {{
            margin-top: 30px;
        }}
        .chart {{
            margin-bottom: 30px;
            text-align: center;
        }}
        .chart img {{
            max-width: 100%;
            border: 1px solid #ddd;
            border-radius: 8px;
        }}
        .metadata {{
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            margin-top: 30px;
        }}
        .error-details {{
            background-color: #f8f9fa;
            padding: 10px;
            margin-top: 10px;
            border-radius: 4px;
            font-family: monospace;
            font-size: 0.9em;
            white-space: pre-wrap;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>LLMKG Integration Test Report</h1>
            <p><strong>Scenario:</strong> {self.scenario}</p>
            <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="summary">
            <div class="summary-card">
                <h3>Test Suites</h3>
                <div class="value">{summary['total_suites']}</div>
            </div>
            <div class="summary-card">
                <h3>Total Tests</h3>
                <div class="value">{summary['total_tests']}</div>
            </div>
            <div class="summary-card">
                <h3>Success Rate</h3>
                <div class="value">{summary['success_rate']:.1f}%</div>
            </div>
            <div class="summary-card">
                <h3>Total Time</h3>
                <div class="value">{summary['total_time']:.1f}s</div>
            </div>
        </div>
        """
        
        # Add test suite details
        for suite_name, suite_data in self.test_results.items():
            success_rate = ((suite_data['tests'] - suite_data['failures'] - suite_data['errors']) / suite_data['tests'] * 100) if suite_data['tests'] > 0 else 0
            
            html_content += f"""
        <div class="test-suite">
            <div class="test-suite-header">
                <h3>{suite_name}</h3>
                <p>Tests: {suite_data['tests']} | Failures: {suite_data['failures']} | Errors: {suite_data['errors']} | Success Rate: {success_rate:.1f}% | Time: {suite_data['time']:.2f}s</p>
            </div>
            """
            
            for test_case in suite_data['test_cases']:
                html_content += f"""
            <div class="test-case">
                <div>
                    <strong>{test_case['name']}</strong>
                    <small>({test_case['time']:.3f}s)</small>
                </div>
                <span class="status {test_case['status']}">{test_case['status'].upper()}</span>
            </div>
                """
                
                if test_case['status'] in ['failed', 'error']:
                    error_msg = test_case.get('failure', test_case.get('error', ''))
                    html_content += f"""
            <div class="error-details">{error_msg}</div>
                    """
            
            html_content += "</div>"
        
        # Add performance charts
        if charts:
            html_content += """
        <div class="charts">
            <h2>Performance Charts</h2>
            """
            
            for chart_name, chart_path in charts.items():
                chart_filename = Path(chart_path).name
                html_content += f"""
            <div class="chart">
                <h3>{chart_name.replace('_', ' ').title()}</h3>
                <img src="{chart_filename}" alt="{chart_name}">
            </div>
                """
            
            html_content += "</div>"
        
        # Add metadata
        html_content += f"""
        <div class="metadata">
            <h2>Test Environment</h2>
            <p><strong>Scenario:</strong> {self.scenario}</p>
            <p><strong>Input Directory:</strong> {self.input_dir}</p>
            <p><strong>Report Generated:</strong> {datetime.now().isoformat()}</p>
        </div>
        
    </div>
</body>
</html>
        """
        
        # Write HTML report
        with open(self.output_file, 'w') as f:
            f.write(html_content)
            
        print(f"HTML report generated: {self.output_file}")
        
    def run(self) -> None:
        """Run the report generation process"""
        print(f"Generating test report for scenario: {self.scenario}")
        print(f"Input directory: {self.input_dir}")
        print(f"Output file: {self.output_file}")
        
        self.collect_test_results()
        self.collect_performance_data()
        self.collect_resource_usage()
        self.generate_html_report()
        
        summary = self.generate_summary_stats()
        print(f"Report completed - {summary['total_tests']} tests, {summary['success_rate']:.1f}% success rate")


def main():
    parser = argparse.ArgumentParser(description='Generate integration test report')
    parser.add_argument('--input-dir', required=True, help='Directory containing test results')
    parser.add_argument('--output', required=True, help='Output HTML file path')
    parser.add_argument('--scenario', required=True, help='Test scenario name')
    
    args = parser.parse_args()
    
    generator = TestReportGenerator(args.input_dir, args.output, args.scenario)
    generator.run()


if __name__ == '__main__':
    main()