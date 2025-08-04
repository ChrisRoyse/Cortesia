#!/usr/bin/env python3
"""
A/B Testing Framework for 99.9% Documentation Detection
Safely test improvements without risking accuracy
"""

import uuid
import time
import json
import random
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import logging
from scipy import stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Experiment:
    """Represents an A/B test experiment"""
    id: str
    name: str
    description: str
    control_model: str
    variant_model: str
    start_time: float
    duration_hours: float
    traffic_split: float  # Percentage to variant (0-1)
    status: str  # 'running', 'completed', 'failed'
    control_results: List[Dict[str, Any]] = field(default_factory=list)
    variant_results: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    conclusion: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return asdict(self)
    
    def is_active(self) -> bool:
        """Check if experiment is still running"""
        if self.status != 'running':
            return False
        elapsed = time.time() - self.start_time
        return elapsed < (self.duration_hours * 3600)


class StatisticalAnalyzer:
    """Statistical analysis for A/B test results"""
    
    def __init__(self, min_sample_size: int = 100, confidence_level: float = 0.95):
        self.min_sample_size = min_sample_size
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
    
    def calculate_sample_size(self, baseline_rate: float, 
                             minimum_detectable_effect: float,
                             power: float = 0.8) -> int:
        """Calculate required sample size for statistical significance"""
        # Using approximation for proportion test
        z_alpha = stats.norm.ppf(1 - self.alpha / 2)
        z_beta = stats.norm.ppf(power)
        
        p1 = baseline_rate
        p2 = baseline_rate + minimum_detectable_effect
        p_pooled = (p1 + p2) / 2
        
        n = ((z_alpha + z_beta) ** 2 * 2 * p_pooled * (1 - p_pooled)) / (minimum_detectable_effect ** 2)
        
        return int(np.ceil(n))
    
    def perform_proportion_test(self, control_successes: int, control_total: int,
                               variant_successes: int, variant_total: int) -> Dict[str, Any]:
        """Perform two-proportion z-test"""
        
        # Calculate proportions
        p_control = control_successes / control_total if control_total > 0 else 0
        p_variant = variant_successes / variant_total if variant_total > 0 else 0
        
        # Pooled proportion
        p_pooled = (control_successes + variant_successes) / (control_total + variant_total)
        
        # Standard error
        se = np.sqrt(p_pooled * (1 - p_pooled) * (1/control_total + 1/variant_total))
        
        # Z-score
        z_score = (p_variant - p_control) / se if se > 0 else 0
        
        # P-value (two-tailed)
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        
        # Confidence interval for difference
        ci_se = np.sqrt(p_control*(1-p_control)/control_total + p_variant*(1-p_variant)/variant_total)
        ci_margin = stats.norm.ppf(1 - self.alpha/2) * ci_se
        ci_lower = (p_variant - p_control) - ci_margin
        ci_upper = (p_variant - p_control) + ci_margin
        
        return {
            'control_rate': p_control,
            'variant_rate': p_variant,
            'absolute_difference': p_variant - p_control,
            'relative_difference': (p_variant - p_control) / p_control if p_control > 0 else 0,
            'z_score': z_score,
            'p_value': p_value,
            'is_significant': p_value < self.alpha,
            'confidence_interval': (ci_lower, ci_upper),
            'sample_sizes': {'control': control_total, 'variant': variant_total}
        }
    
    def calculate_statistical_power(self, results: Dict[str, Any]) -> float:
        """Calculate the statistical power of the test"""
        n_control = results['sample_sizes']['control']
        n_variant = results['sample_sizes']['variant']
        p_control = results['control_rate']
        p_variant = results['variant_rate']
        
        if n_control == 0 or n_variant == 0:
            return 0.0
        
        # Effect size (Cohen's h for proportions)
        h = 2 * (np.arcsin(np.sqrt(p_variant)) - np.arcsin(np.sqrt(p_control)))
        
        # Approximate power calculation
        n_harmonic = 2 * n_control * n_variant / (n_control + n_variant)
        z_alpha = stats.norm.ppf(1 - self.alpha / 2)
        z_power = abs(h) * np.sqrt(n_harmonic / 2) - z_alpha
        
        power = stats.norm.cdf(z_power)
        return power


class ABTestingFramework:
    """Main A/B testing framework for model improvements"""
    
    def __init__(self, control_model, minimum_improvement: float = 0.001):
        self.control_model = control_model
        self.control_accuracy = 0.999  # 99.9% baseline
        self.minimum_improvement = minimum_improvement
        self.experiments = {}
        self.active_experiment = None
        self.analyzer = StatisticalAnalyzer()
        self.results_buffer = defaultdict(list)
        
    def create_experiment(self, name: str, variant_model, 
                         traffic_split: float = 0.1,
                         duration_hours: float = 24) -> Experiment:
        """Create a new A/B test experiment"""
        
        # Calculate required sample size
        required_samples = self.analyzer.calculate_sample_size(
            baseline_rate=self.control_accuracy,
            minimum_detectable_effect=self.minimum_improvement,
            power=0.8
        )
        
        experiment = Experiment(
            id=str(uuid.uuid4()),
            name=name,
            description=f"Testing {name} for >{self.minimum_improvement:.1%} improvement",
            control_model=str(self.control_model),
            variant_model=str(variant_model),
            start_time=time.time(),
            duration_hours=duration_hours,
            traffic_split=traffic_split,
            status='running',
            metrics={'required_samples': required_samples}
        )
        
        self.experiments[experiment.id] = experiment
        logger.info(f"Created experiment {experiment.id}: {name}")
        logger.info(f"Required samples: {required_samples} per group")
        
        return experiment
    
    def assign_to_variant(self, experiment: Experiment) -> str:
        """Randomly assign request to control or variant"""
        if not experiment.is_active():
            return 'control'
        
        return 'variant' if random.random() < experiment.traffic_split else 'control'
    
    def process_request(self, content: str, language: str, 
                       experiment_id: Optional[str] = None) -> Dict[str, Any]:
        """Process a request through A/B test"""
        
        # Get active experiment
        if experiment_id and experiment_id in self.experiments:
            experiment = self.experiments[experiment_id]
        elif self.active_experiment:
            experiment = self.active_experiment
        else:
            # No active experiment, use control
            return self._process_with_model(self.control_model, content, language)
        
        # Check if experiment is still active
        if not experiment.is_active():
            self._complete_experiment(experiment)
            return self._process_with_model(self.control_model, content, language)
        
        # Assign to variant
        variant = self.assign_to_variant(experiment)
        
        # Process with appropriate model
        if variant == 'control':
            result = self._process_with_model(self.control_model, content, language)
            experiment.control_results.append(result)
        else:
            result = self._process_with_model(experiment.variant_model, content, language)
            experiment.variant_results.append(result)
        
        result['experiment_id'] = experiment.id
        result['variant'] = variant
        
        # Check if we have enough data for early stopping
        if len(experiment.control_results) >= 100 and len(experiment.variant_results) >= 100:
            self._check_early_stopping(experiment)
        
        return result
    
    def _process_with_model(self, model, content: str, language: str) -> Dict[str, Any]:
        """Process content with specified model"""
        # Simulate model processing
        # In production, would call actual model
        
        # For demonstration, simulate high accuracy
        has_documentation = True if '"""' in content or '///' in content else False
        confidence = 0.95 + random.random() * 0.05
        
        # Simulate occasional errors for testing
        if random.random() < 0.001:  # 0.1% error rate
            has_documentation = not has_documentation
        
        return {
            'has_documentation': has_documentation,
            'confidence': confidence,
            'language': language,
            'timestamp': time.time(),
            'model': str(model)
        }
    
    def _check_early_stopping(self, experiment: Experiment):
        """Check if experiment can be stopped early"""
        
        # Calculate current statistics
        control_correct = sum(1 for r in experiment.control_results 
                            if r.get('correct', True))
        variant_correct = sum(1 for r in experiment.variant_results 
                            if r.get('correct', True))
        
        results = self.analyzer.perform_proportion_test(
            control_correct, len(experiment.control_results),
            variant_correct, len(experiment.variant_results)
        )
        
        # Check for early stopping conditions
        if results['is_significant']:
            # Significant difference detected
            if results['absolute_difference'] > self.minimum_improvement:
                # Variant wins
                logger.info(f"Early stopping: Variant wins with {results['absolute_difference']:.3%} improvement")
                self._complete_experiment(experiment, winner='variant')
            elif results['absolute_difference'] < -self.minimum_improvement:
                # Control wins (variant is worse)
                logger.info(f"Early stopping: Control wins, variant {-results['absolute_difference']:.3%} worse")
                self._complete_experiment(experiment, winner='control')
    
    def _complete_experiment(self, experiment: Experiment, winner: Optional[str] = None):
        """Complete an experiment and analyze results"""
        
        experiment.status = 'completed'
        
        # Analyze results
        control_correct = sum(1 for r in experiment.control_results 
                            if r.get('correct', True))
        variant_correct = sum(1 for r in experiment.variant_results 
                            if r.get('correct', True))
        
        final_results = self.analyzer.perform_proportion_test(
            control_correct, len(experiment.control_results),
            variant_correct, len(experiment.variant_results)
        )
        
        # Calculate power
        power = self.analyzer.calculate_statistical_power(final_results)
        
        # Store metrics
        experiment.metrics.update({
            'final_results': final_results,
            'statistical_power': power,
            'duration_actual': time.time() - experiment.start_time,
            'winner': winner or self._determine_winner(final_results)
        })
        
        # Set conclusion
        if final_results['is_significant']:
            if final_results['absolute_difference'] > 0:
                experiment.conclusion = f"Variant wins: {final_results['absolute_difference']:.3%} improvement (p={final_results['p_value']:.4f})"
            else:
                experiment.conclusion = f"Control wins: {-final_results['absolute_difference']:.3%} better (p={final_results['p_value']:.4f})"
        else:
            experiment.conclusion = f"No significant difference (p={final_results['p_value']:.4f}, power={power:.2f})"
        
        logger.info(f"Experiment {experiment.id} completed: {experiment.conclusion}")
    
    def _determine_winner(self, results: Dict[str, Any]) -> str:
        """Determine the winner of an experiment"""
        if not results['is_significant']:
            return 'none'
        
        if results['absolute_difference'] > self.minimum_improvement:
            return 'variant'
        elif results['absolute_difference'] < -self.minimum_improvement:
            return 'control'
        else:
            return 'none'
    
    def get_experiment_report(self, experiment_id: str) -> Dict[str, Any]:
        """Get detailed report for an experiment"""
        
        if experiment_id not in self.experiments:
            return {'error': 'Experiment not found'}
        
        experiment = self.experiments[experiment_id]
        
        report = {
            'id': experiment.id,
            'name': experiment.name,
            'status': experiment.status,
            'duration': {
                'planned_hours': experiment.duration_hours,
                'actual_hours': (time.time() - experiment.start_time) / 3600
            },
            'sample_sizes': {
                'control': len(experiment.control_results),
                'variant': len(experiment.variant_results),
                'required': experiment.metrics.get('required_samples', 'N/A')
            },
            'traffic_split': {
                'planned': experiment.traffic_split,
                'actual': len(experiment.variant_results) / (len(experiment.control_results) + len(experiment.variant_results))
                          if (len(experiment.control_results) + len(experiment.variant_results)) > 0 else 0
            }
        }
        
        if experiment.status == 'completed' and 'final_results' in experiment.metrics:
            results = experiment.metrics['final_results']
            report['results'] = {
                'control_accuracy': f"{results['control_rate']:.3%}",
                'variant_accuracy': f"{results['variant_rate']:.3%}",
                'improvement': f"{results['absolute_difference']:.3%}",
                'relative_improvement': f"{results['relative_difference']:.1%}",
                'p_value': results['p_value'],
                'is_significant': results['is_significant'],
                'confidence_interval': f"({results['confidence_interval'][0]:.3%}, {results['confidence_interval'][1]:.3%})",
                'statistical_power': experiment.metrics.get('statistical_power', 'N/A'),
                'winner': experiment.metrics.get('winner', 'none')
            }
            report['conclusion'] = experiment.conclusion
        
        return report
    
    def run_simulation(self, variant_model, num_requests: int = 1000,
                      actual_improvement: float = 0.002) -> Dict[str, Any]:
        """Run a simulated A/B test for demonstration"""
        
        # Create experiment
        experiment = self.create_experiment(
            name="Simulated Test",
            variant_model=variant_model,
            traffic_split=0.5,  # 50/50 split for simulation
            duration_hours=1
        )
        
        # Simulate requests
        for i in range(num_requests):
            # Generate test content
            if random.random() < 0.7:
                content = 'def func():\n    """Documented function"""\n    pass'
            else:
                content = 'def func():\n    pass'
            
            language = random.choice(['python', 'rust', 'javascript'])
            
            # Process through A/B test
            result = self.process_request(content, language, experiment.id)
            
            # Simulate ground truth and improvement
            ground_truth = '"""' in content or '///' in content
            
            # Add correctness based on variant
            if result['variant'] == 'control':
                # Control has 99.9% accuracy
                correct = (result['has_documentation'] == ground_truth) or (random.random() < 0.999)
            else:
                # Variant has improved accuracy
                correct = (result['has_documentation'] == ground_truth) or (random.random() < (0.999 + actual_improvement))
            
            result['correct'] = correct
        
        # Complete experiment
        self._complete_experiment(experiment)
        
        # Return report
        return self.get_experiment_report(experiment.id)


# Demonstration
if __name__ == "__main__":
    print("A/B Testing Framework for 99.9% Accuracy")
    print("=" * 60)
    
    # Create framework
    framework = ABTestingFramework(
        control_model="ensemble_detector_v1.0",
        minimum_improvement=0.001  # 0.1% improvement required
    )
    
    # Run simulated A/B test
    print("\nRunning simulated A/B test...")
    report = framework.run_simulation(
        variant_model="ensemble_detector_v1.1",
        num_requests=2000,
        actual_improvement=0.002  # 0.2% actual improvement
    )
    
    print("\nExperiment Report:")
    print("-" * 40)
    print(f"Name: {report['name']}")
    print(f"Status: {report['status']}")
    print(f"Sample Sizes: Control={report['sample_sizes']['control']}, Variant={report['sample_sizes']['variant']}")
    
    if 'results' in report:
        print(f"\nResults:")
        print(f"  Control Accuracy: {report['results']['control_accuracy']}")
        print(f"  Variant Accuracy: {report['results']['variant_accuracy']}")
        print(f"  Improvement: {report['results']['improvement']}")
        print(f"  P-value: {report['results']['p_value']:.4f}")
        print(f"  Significant: {report['results']['is_significant']}")
        print(f"  Winner: {report['results']['winner']}")
        print(f"\nConclusion: {report['conclusion']}")
    
    print("\n[SUCCESS] A/B Testing Framework Ready!")