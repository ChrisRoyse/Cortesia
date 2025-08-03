#!/usr/bin/env python3
"""
Web Interface for Continuous Learning System

This module provides a Flask-based web interface for interacting with the
continuous learning system. Users can provide feedback, view system status,
and monitor learning progress through an intuitive web interface.

Features:
1. Feedback Collection Interface - Easy-to-use forms for documentation feedback
2. System Dashboard - Real-time status and performance metrics
3. Pattern Visualization - View discovered patterns and their effectiveness
4. Model Update History - Track deployments and rollbacks
5. Performance Analytics - Charts and trends for system performance
6. Admin Controls - Manual learning cycles and system configuration

Author: Claude (Sonnet 4)
"""

import json
import os
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging

# Flask and web dependencies
try:
    from flask import Flask, render_template, request, jsonify, flash, redirect, url_for, session
    from flask_wtf import FlaskForm
    from wtforms import TextAreaField, SelectField, BooleanField, FloatField, StringField, SubmitField
    from wtforms.validators import DataRequired, NumberRange, Length
    from werkzeug.security import check_password_hash, generate_password_hash
    import plotly.graph_objs as go
    import plotly.utils
except ImportError as e:
    print(f"Web interface dependencies not available: {e}")
    print("Install with: pip install flask flask-wtf wtforms plotly")

# Import our learning system
try:
    from continuous_learning_system import ContinuousLearningSystem, FeedbackRecord
    from ultra_reliable_core import UniversalDocumentationDetector
except ImportError as e:  
    print(f"Learning system components not available: {e}")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeedbackForm(FlaskForm):
    """Form for collecting user feedback"""
    content = TextAreaField('Source Code', 
                           validators=[DataRequired(), Length(min=10, max=10000)],
                           render_kw={'rows': 15, 'placeholder': 'Paste your source code here...'})
    
    language = SelectField('Programming Language', 
                          choices=[
                              ('python', 'Python'),
                              ('rust', 'Rust'),
                              ('javascript', 'JavaScript'), 
                              ('typescript', 'TypeScript'),
                              ('java', 'Java'),
                              ('cpp', 'C++'),
                              ('c', 'C'),
                              ('go', 'Go'),
                              ('other', 'Other')
                          ],
                          validators=[DataRequired()])
    
    file_path = StringField('File Path (optional)', 
                           render_kw={'placeholder': 'e.g., src/main.py'})
    
    has_documentation = BooleanField('This code has documentation')
    
    documentation_lines = StringField('Documentation Line Numbers',
                                    render_kw={'placeholder': 'e.g., 1,2,3 or 1-5'})
    
    confidence = FloatField('Your Confidence (0.0-1.0)',
                           validators=[NumberRange(min=0.0, max=1.0)],
                           default=0.9)
    
    notes = TextAreaField('Additional Notes (optional)',
                         render_kw={'rows': 3, 'placeholder': 'Any additional comments...'})
    
    submit = SubmitField('Submit Feedback')


class AdminControlForm(FlaskForm):
    """Form for admin controls"""
    action = SelectField('Action',
                        choices=[
                            ('learning_cycle', 'Run Learning Cycle'),
                            ('enable_auto_deploy', 'Enable Auto-Deployment'),
                            ('disable_auto_deploy', 'Disable Auto-Deployment'),
                            ('system_status', 'Get System Status')
                        ],
                        validators=[DataRequired()])
    
    submit = SubmitField('Execute')


class LearningWebInterface:
    """Web interface for the continuous learning system"""
    
    def __init__(self, learning_system: ContinuousLearningSystem, debug: bool = False):
        self.learning_system = learning_system
        self.app = Flask(__name__, template_folder='templates', static_folder='static')
        self.app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
        self.app.config['WTF_CSRF_ENABLED'] = True
        
        # Initialize documentation detector for real-time predictions
        self.doc_detector = UniversalDocumentationDetector()
        
        # Create templates directory
        self.templates_dir = Path('templates')
        self.templates_dir.mkdir(exist_ok=True)
        
        self.static_dir = Path('static')
        self.static_dir.mkdir(exist_ok=True)
        
        # Create templates
        self._create_templates()
        self._create_static_files()
        
        # Setup routes
        self._setup_routes()
        
        logger.info("Learning web interface initialized")
    
    def _setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def index():
            """Main dashboard"""
            status = self.learning_system.get_learning_status()
            
            # Get recent feedback statistics
            feedback_stats = self.learning_system.feedback_system.get_feedback_statistics()
            
            # Get recent patterns
            recent_patterns = self.learning_system.pattern_discovery.get_top_patterns(limit=5)
            
            # Get deployment history
            deployment_history = self.learning_system.model_updater.get_deployment_history(limit=5)
            
            return render_template('dashboard.html',
                                 status=status,
                                 feedback_stats=feedback_stats,
                                 recent_patterns=recent_patterns,
                                 deployment_history=deployment_history)
        
        @self.app.route('/feedback', methods=['GET', 'POST'])
        def feedback():
            """Feedback collection page"""
            form = FeedbackForm()
            
            if form.validate_on_submit():
                # Parse documentation lines
                doc_lines = self._parse_line_numbers(form.documentation_lines.data)
                
                # Get system prediction for comparison
                system_prediction = self.doc_detector.detect_documentation_multi_pass(
                    form.content.data, form.language.data
                )
                
                # Collect feedback
                feedback_record = self.learning_system.collect_user_feedback(
                    content=form.content.data,
                    language=form.language.data,
                    file_path=form.file_path.data or 'web_submission',
                    user_has_documentation=form.has_documentation.data,
                    user_documentation_lines=doc_lines,
                    user_confidence=form.confidence.data,
                    system_prediction=system_prediction,
                    user_id=session.get('user_id', 'web_user')
                )
                
                flash(f'Feedback submitted successfully! ID: {feedback_record.feedback_id}', 'success')
                return redirect(url_for('feedback'))
            
            return render_template('feedback.html', form=form)
        
        @self.app.route('/api/predict', methods=['POST'])
        def api_predict():
            """API endpoint for real-time documentation prediction"""
            data = request.get_json()
            
            if not data or 'content' not in data or 'language' not in data:
                return jsonify({'error': 'Missing content or language'}), 400
            
            try:
                prediction = self.doc_detector.detect_documentation_multi_pass(
                    data['content'], data['language']
                )
                
                return jsonify({
                    'has_documentation': prediction.get('has_documentation', False),
                    'confidence': prediction.get('confidence', 0.0),
                    'documentation_lines': prediction.get('documentation_lines', []),
                    'patterns_found': prediction.get('patterns_found', [])
                })
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/patterns')
        def patterns():
            """View discovered patterns"""
            all_patterns = self.learning_system.pattern_discovery.get_top_patterns(limit=50)
            
            # Group by language
            patterns_by_language = {}
            for pattern in all_patterns:
                if pattern.language not in patterns_by_language:
                    patterns_by_language[pattern.language] = []
                patterns_by_language[pattern.language].append(pattern)
            
            return render_template('patterns.html', 
                                 patterns_by_language=patterns_by_language,
                                 total_patterns=len(all_patterns))
        
        @self.app.route('/analytics')
        def analytics():
            """Analytics and performance charts"""
            # Get learning metrics history
            metrics_history = self.learning_system.learning_metrics_history
            
            if not metrics_history:
                return render_template('analytics.html', 
                                     charts_json=json.dumps([]),
                                     no_data=True)
            
            # Create accuracy trend chart
            accuracy_chart = self._create_accuracy_chart(metrics_history)
            
            # Create feedback volume chart
            feedback_chart = self._create_feedback_chart(metrics_history)
            
            # Create pattern discovery chart
            pattern_chart = self._create_pattern_chart(metrics_history)
            
            charts_json = json.dumps([accuracy_chart, feedback_chart, pattern_chart], 
                                   cls=plotly.utils.PlotlyJSONEncoder)
            
            return render_template('analytics.html', 
                                 charts_json=charts_json,
                                 metrics=metrics_history[-1] if metrics_history else None)
        
        @self.app.route('/admin', methods=['GET', 'POST'])
        def admin():
            """Admin controls"""
            form = AdminControlForm()
            result = None
            
            if form.validate_on_submit():
                action = form.action.data
                
                try:
                    if action == 'learning_cycle':
                        result = self.learning_system.force_learning_cycle()
                        flash(f'Learning cycle completed: {result}', 'success')
                    
                    elif action == 'enable_auto_deploy':
                        self.learning_system.enable_auto_deployment = True
                        flash('Auto-deployment enabled', 'success')
                    
                    elif action == 'disable_auto_deploy':
                        self.learning_system.enable_auto_deployment = False
                        flash('Auto-deployment disabled', 'warning')
                    
                    elif action == 'system_status':
                        result = self.learning_system.get_learning_status()
                        flash('System status retrieved', 'info')
                
                except Exception as e:
                    flash(f'Error executing action: {str(e)}', 'error')
            
            return render_template('admin.html', form=form, result=result)
        
        @self.app.route('/api/status')
        def api_status():
            """API endpoint for system status"""
            try:
                status = self.learning_system.get_learning_status()
                return jsonify(status)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/feedback_stats')
        def api_feedback_stats():
            """API endpoint for feedback statistics"""
            try:
                stats = self.learning_system.feedback_system.get_feedback_statistics()
                return jsonify(stats)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
    
    def _parse_line_numbers(self, line_str: str) -> List[int]:
        """Parse line numbers from string (e.g., '1,2,3' or '1-5')"""
        if not line_str:
            return []
        
        lines = []
        parts = line_str.split(',')
        
        for part in parts:
            part = part.strip()
            if '-' in part:
                # Handle ranges like '1-5'
                try:
                    start, end = part.split('-')
                    lines.extend(range(int(start), int(end) + 1))
                except ValueError:
                    continue
            else:
                # Handle single numbers
                try:
                    lines.append(int(part))
                except ValueError:
                    continue
        
        return sorted(list(set(lines)))  # Remove duplicates and sort
    
    def _create_accuracy_chart(self, metrics_history: List) -> Dict:
        """Create accuracy trend chart"""
        timestamps = [m.timestamp for m in metrics_history]
        accuracies = [m.current_accuracy * 100 for m in metrics_history]  # Convert to percentage
        
        return {
            'data': [{
                'x': [t.isoformat() for t in timestamps],
                'y': accuracies,
                'type': 'scatter',
                'mode': 'lines+markers',
                'name': 'Accuracy %',
                'line': {'color': '#2E86AB'},
                'marker': {'size': 6}
            }],
            'layout': {
                'title': 'System Accuracy Over Time',
                'xaxis': {'title': 'Time'},
                'yaxis': {'title': 'Accuracy (%)', 'range': [95, 100]},
                'showlegend': False,
                'height': 300
            }
        }
    
    def _create_feedback_chart(self, metrics_history: List) -> Dict:
        """Create feedback volume chart"""
        timestamps = [m.timestamp for m in metrics_history]
        total_feedback = [m.total_feedback_records for m in metrics_history]
        validated_feedback = [m.validated_feedback_records for m in metrics_history]
        
        return {
            'data': [
                {
                    'x': [t.isoformat() for t in timestamps],
                    'y': total_feedback,
                    'type': 'scatter',
                    'mode': 'lines',
                    'name': 'Total Feedback',
                    'line': {'color': '#A23B72'}
                },
                {
                    'x': [t.isoformat() for t in timestamps],
                    'y': validated_feedback,
                    'type': 'scatter',
                    'mode': 'lines',
                    'name': 'Validated Feedback',
                    'line': {'color': '#F18F01'}
                }
            ],
            'layout': {
                'title': 'Feedback Volume Over Time',
                'xaxis': {'title': 'Time'},
                'yaxis': {'title': 'Number of Records'},
                'height': 300
            }
        }
    
    def _create_pattern_chart(self, metrics_history: List) -> Dict:
        """Create pattern discovery chart"""
        timestamps = [m.timestamp for m in metrics_history]
        patterns_discovered = [m.patterns_discovered for m in metrics_history]
        patterns_deployed = [m.patterns_deployed for m in metrics_history]
        
        return {
            'data': [
                {
                    'x': [t.isoformat() for t in timestamps],
                    'y': patterns_discovered,
                    'type': 'bar',
                    'name': 'Patterns Discovered',
                    'marker': {'color': '#C73E1D'}
                },
                {
                    'x': [t.isoformat() for t in timestamps],
                    'y': patterns_deployed,
                    'type': 'bar',
                    'name': 'Patterns Deployed',
                    'marker': {'color': '#92B5A7'}
                }
            ],
            'layout': {
                'title': 'Pattern Discovery and Deployment',
                'xaxis': {'title': 'Time'},
                'yaxis': {'title': 'Number of Patterns'},
                'height': 300,
                'barmode': 'group'
            }
        }
    
    def _create_templates(self):
        """Create HTML templates"""
        
        # Base template
        base_template = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}Continuous Learning System{% endblock %}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        .sidebar { min-height: 100vh; background-color: #f8f9fa; }
        .accuracy-badge { font-size: 1.2rem; }
        .pattern-card { transition: transform 0.2s; }
        .pattern-card:hover { transform: translateY(-2px); }
        .code-preview { font-family: 'Courier New', monospace; font-size: 0.9rem; }
    </style>
</head>
<body>
    <div class="container-fluid">
        <div class="row">
            <!-- Sidebar -->
            <nav class="col-md-2 d-md-block bg-light sidebar">
                <div class="position-sticky pt-3">
                    <h5 class="px-3 text-primary"><i class="fas fa-brain"></i> Learning System</h5>
                    <ul class="nav flex-column">
                        <li class="nav-item">
                            <a class="nav-link" href="{{ url_for('index') }}">
                                <i class="fas fa-tachometer-alt"></i> Dashboard
                            </a>
                        </li>
                        <li class="nav-item">
                            <a class="nav-link" href="{{ url_for('feedback') }}">
                                <i class="fas fa-comment-alt"></i> Provide Feedback
                            </a>
                        </li>
                        <li class="nav-item">
                            <a class="nav-link" href="{{ url_for('patterns') }}">
                                <i class="fas fa-search"></i> Discovered Patterns
                            </a>
                        </li>
                        <li class="nav-item">
                            <a class="nav-link" href="{{ url_for('analytics') }}">
                                <i class="fas fa-chart-line"></i> Analytics
                            </a>
                        </li>
                        <li class="nav-item">
                            <a class="nav-link" href="{{ url_for('admin') }}">
                                <i class="fas fa-cog"></i> Admin Controls
                            </a>
                        </li>
                    </ul>
                </div>
            </nav>
            
            <!-- Main content -->
            <main class="col-md-10 ms-sm-auto px-md-4">
                <div class="d-flex justify-content-between flex-wrap flex-md-nowrap align-items-center pt-3 pb-2 mb-3 border-bottom">
                    <h1 class="h2">{% block header %}{% endblock %}</h1>
                </div>
                
                <!-- Flash messages -->
                {% with messages = get_flashed_messages(with_categories=true) %}
                    {% if messages %}
                        {% for category, message in messages %}
                            <div class="alert alert-{{ 'success' if category == 'success' else 'warning' if category == 'warning' else 'danger' if category == 'error' else 'info' }} alert-dismissible fade show" role="alert">
                                {{ message }}
                                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
                            </div>
                        {% endfor %}
                    {% endif %}
                {% endwith %}
                
                {% block content %}{% endblock %}
            </main>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    {% block scripts %}{% endblock %}
</body>
</html>'''
        
        # Dashboard template
        dashboard_template = '''{% extends "base.html" %}

{% block title %}Dashboard - Continuous Learning System{% endblock %}
{% block header %}System Dashboard{% endblock %}

{% block content %}
<div class="row">
    <!-- System Status Cards -->
    <div class="col-md-3">
        <div class="card text-white bg-primary mb-3">
            <div class="card-body">
                <div class="d-flex justify-content-between">
                    <div>
                        <h4 class="card-title">{{ "%.1f"|format(status.current_accuracy * 100) }}%</h4>
                        <p class="card-text">Current Accuracy</p>
                    </div>
                    <div class="align-self-center">
                        <i class="fas fa-bullseye fa-2x"></i>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <div class="col-md-3">
        <div class="card text-white bg-success mb-3">
            <div class="card-body">
                <div class="d-flex justify-content-between">
                    <div>
                        <h4 class="card-title">{{ status.total_feedback_records }}</h4>
                        <p class="card-text">Total Feedback</p>
                    </div>
                    <div class="align-self-center">
                        <i class="fas fa-comments fa-2x"></i>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <div class="col-md-3">
        <div class="card text-white bg-info mb-3">
            <div class="card-body">
                <div class="d-flex justify-content-between">
                    <div>
                        <h4 class="card-title">{{ status.patterns_discovered }}</h4>
                        <p class="card-text">Patterns Found</p>
                    </div>
                    <div class="align-self-center">
                        <i class="fas fa-search fa-2x"></i>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <div class="col-md-3">
        <div class="card text-white bg-warning mb-3">
            <div class="card-body">
                <div class="d-flex justify-content-between">
                    <div>
                        <h4 class="card-title">{{ "%.1f"|format(status.accuracy_improvement * 100) }}%</h4>
                        <p class="card-text">Improvement</p>
                    </div>
                    <div class="align-self-center">
                        <i class="fas fa-arrow-up fa-2x"></i>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>

<div class="row">
    <!-- Learning Status -->
    <div class="col-md-6">
        <div class="card">
            <div class="card-header">
                <h5><i class="fas fa-robot"></i> Learning Status</h5>
            </div>
            <div class="card-body">
                <div class="row">
                    <div class="col-6">
                        <strong>Learning Active:</strong><br>
                        <span class="badge bg-{{ 'success' if status.learning_active else 'secondary' }}">
                            {{ 'Yes' if status.learning_active else 'No' }}
                        </span>
                    </div>
                    <div class="col-6">
                        <strong>Auto-Deploy:</strong><br>
                        <span class="badge bg-{{ 'success' if status.auto_deployment_enabled else 'warning' }}">
                            {{ 'Enabled' if status.auto_deployment_enabled else 'Disabled' }}
                        </span>
                    </div>
                </div>
                <hr>
                <div class="row">
                    <div class="col-12">
                        <strong>Baseline Accuracy:</strong> {{ "%.1f"|format(status.baseline_accuracy * 100) }}%<br>
                        <strong>Learning Overhead:</strong> {{ "%.1f"|format(status.learning_overhead_percentage) }}%
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <!-- Feedback Statistics -->
    <div class="col-md-6">
        <div class="card">
            <div class="card-header">
                <h5><i class="fas fa-chart-pie"></i> Feedback Statistics</h5>
            </div>
            <div class="card-body">
                <div class="row">
                    <div class="col-6">
                        <strong>Validated:</strong> {{ feedback_stats.validated_feedback }}<br>
                        <strong>Quality Score:</strong> {{ "%.2f"|format(feedback_stats.quality_score) }}
                    </div>
                    <div class="col-6">
                        <strong>Agreement Rate:</strong> {{ "%.1f"|format(feedback_stats.system_user_agreement_rate * 100) }}%<br>
                        <strong>Avg Confidence:</strong> {{ "%.2f"|format(feedback_stats.average_user_confidence) }}
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>

<div class="row mt-3">
    <!-- Recent Patterns -->
    <div class="col-md-6">
        <div class="card">
            <div class="card-header">
                <h5><i class="fas fa-lightbulb"></i> Recent Patterns</h5>
            </div>
            <div class="card-body">
                {% if recent_patterns %}
                    {% for pattern in recent_patterns %}
                    <div class="mb-2 p-2 border rounded">
                        <small class="text-muted">{{ pattern.language }}</small><br>
                        <code>{{ pattern.pattern_regex[:50] }}{% if pattern.pattern_regex|length > 50 %}...{% endif %}</code><br>
                        <small>Accuracy: {{ "%.1f"|format(pattern.validation_accuracy * 100) }}% | Type: {{ pattern.pattern_type }}</small>
                    </div>
                    {% endfor %}
                {% else %}
                    <p class="text-muted">No patterns discovered yet.</p>
                {% endif %}
            </div>
        </div>
    </div>
    
    <!-- Deployment History -->
    <div class="col-md-6">
        <div class="card">
            <div class="card-header">
                <h5><i class="fas fa-history"></i> Recent Deployments</h5>
            </div>
            <div class="card-body">
                {% if deployment_history %}
                    {% for update in deployment_history %}
                    <div class="mb-2 p-2 border rounded">
                        <strong>{{ update.update_type }}</strong>
                        <span class="badge bg-{{ 'success' if update.deployed else 'secondary' }} float-end">
                            {{ 'Deployed' if update.deployed else 'Pending' }}
                        </span><br>
                        <small class="text-muted">{{ update.timestamp.strftime('%Y-%m-%d %H:%M') }}</small><br>
                        <small>Accuracy: {{ "%.1f"|format(update.validation_accuracy * 100) }}%</small>
                    </div>
                    {% endfor %}
                {% else %}
                    <p class="text-muted">No deployments yet.</p>
                {% endif %}
            </div>
        </div>
    </div>
</div>
{% endblock %}'''
        
        # Feedback template
        feedback_template = '''{% extends "base.html" %}

{% block title %}Provide Feedback - Continuous Learning System{% endblock %}
{% block header %}Provide Feedback{% endblock %}

{% block content %}
<div class="row">
    <div class="col-md-8">
        <div class="card">
            <div class="card-header">
                <h5><i class="fas fa-comment-alt"></i> Submit Documentation Feedback</h5>
            </div>
            <div class="card-body">
                <form method="POST">
                    {{ form.hidden_tag() }}
                    
                    <div class="mb-3">
                        {{ form.content.label(class="form-label") }}
                        {{ form.content(class="form-control") }}
                        {% if form.content.errors %}
                            <div class="text-danger">
                                {% for error in form.content.errors %}
                                    <small>{{ error }}</small>
                                {% endfor %}
                            </div>
                        {% endif %}
                    </div>
                    
                    <div class="row">
                        <div class="col-md-6">
                            <div class="mb-3">
                                {{ form.language.label(class="form-label") }}
                                {{ form.language(class="form-select") }}
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="mb-3">
                                {{ form.file_path.label(class="form-label") }}
                                {{ form.file_path(class="form-control") }}
                            </div>
                        </div>
                    </div>
                    
                    <div class="row">
                        <div class="col-md-4">
                            <div class="mb-3 form-check">
                                {{ form.has_documentation(class="form-check-input") }}
                                {{ form.has_documentation.label(class="form-check-label") }}
                            </div>
                        </div>
                        <div class="col-md-4">
                            <div class="mb-3">
                                {{ form.documentation_lines.label(class="form-label") }}
                                {{ form.documentation_lines(class="form-control") }}
                            </div>
                        </div>
                        <div class="col-md-4">
                            <div class="mb-3">
                                {{ form.confidence.label(class="form-label") }}
                                {{ form.confidence(class="form-control", step="0.1") }}
                            </div>
                        </div>
                    </div>
                    
                    <div class="mb-3">
                        {{ form.notes.label(class="form-label") }}
                        {{ form.notes(class="form-control") }}
                    </div>
                    
                    <div class="mb-3">
                        {{ form.submit(class="btn btn-primary") }}
                        <button type="button" class="btn btn-secondary" onclick="getPrediction()">
                            <i class="fas fa-robot"></i> Get System Prediction
                        </button>
                    </div>
                </form>
            </div>
        </div>
    </div>
    
    <div class="col-md-4">
        <div class="card">
            <div class="card-header">
                <h5><i class="fas fa-info-circle"></i> How to Provide Feedback</h5>
            </div>
            <div class="card-body">
                <ol>
                    <li><strong>Paste your code</strong> in the text area</li>
                    <li><strong>Select the language</strong> from the dropdown</li>
                    <li><strong>Check the box</strong> if the code has documentation</li>
                    <li><strong>Specify line numbers</strong> that contain documentation (e.g., "1,2,3" or "1-5")</li>
                    <li><strong>Rate your confidence</strong> in the assessment (0.0-1.0)</li>
                    <li><strong>Add notes</strong> if needed</li>
                    <li><strong>Submit</strong> to help improve the system!</li>
                </ol>
                
                <div class="alert alert-info mt-3">
                    <i class="fas fa-lightbulb"></i>
                    <strong>Tip:</strong> You can click "Get System Prediction" to see what the current system thinks about your code.
                </div>
            </div>
        </div>
        
        <div class="card mt-3" id="prediction-card" style="display: none;">
            <div class="card-header">
                <h5><i class="fas fa-robot"></i> System Prediction</h5>
            </div>
            <div class="card-body" id="prediction-content">
                <!-- Prediction results will be inserted here -->
            </div>
        </div>
    </div>
</div>

<script>
function getPrediction() {
    const content = document.getElementById('content').value;
    const language = document.getElementById('language').value;
    
    if (!content.trim()) {
        alert('Please enter some code first.');
        return;
    }
    
    fetch('/api/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            content: content,
            language: language
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            alert('Error: ' + data.error);
            return;
        }
        
        const predictionCard = document.getElementById('prediction-card');
        const predictionContent = document.getElementById('prediction-content');
        
        predictionContent.innerHTML = `
            <div class="mb-2">
                <strong>Has Documentation:</strong> 
                <span class="badge bg-${data.has_documentation ? 'success' : 'secondary'}">
                    ${data.has_documentation ? 'Yes' : 'No'}
                </span>
            </div>
            <div class="mb-2">
                <strong>Confidence:</strong> ${(data.confidence * 100).toFixed(1)}%
            </div>
            <div class="mb-2">
                <strong>Documentation Lines:</strong> ${data.documentation_lines.join(', ') || 'None'}
            </div>
            <div class="mb-2">
                <strong>Patterns Found:</strong> ${data.patterns_found.length || 0}
            </div>
        `;
        
        predictionCard.style.display = 'block';
    })
    .catch(error => {
        console.error('Error:', error);
        alert('Error getting prediction: ' + error);
    });
}
</script>
{% endblock %}'''
        
        # Write templates
        with open(self.templates_dir / 'base.html', 'w') as f:
            f.write(base_template)
        
        with open(self.templates_dir / 'dashboard.html', 'w') as f:
            f.write(dashboard_template)
        
        with open(self.templates_dir / 'feedback.html', 'w') as f:
            f.write(feedback_template)
        
        # Create additional templates (simplified versions)
        patterns_template = '''{% extends "base.html" %}
{% block title %}Patterns - Continuous Learning System{% endblock %}
{% block header %}Discovered Patterns{% endblock %}
{% block content %}
<div class="alert alert-info">
    <i class="fas fa-info-circle"></i>
    Found {{ total_patterns }} patterns across {{ patterns_by_language|length }} languages.
</div>

{% for language, patterns in patterns_by_language.items() %}
<div class="card mb-3">
    <div class="card-header">
        <h5><i class="fas fa-code"></i> {{ language|title }} ({{ patterns|length }} patterns)</h5>
    </div>
    <div class="card-body">
        {% for pattern in patterns %}
        <div class="pattern-card border rounded p-3 mb-2">
            <div class="row">
                <div class="col-md-8">
                    <code>{{ pattern.pattern_regex }}</code>
                    <br><small class="text-muted">Type: {{ pattern.pattern_type }}</small>
                </div>
                <div class="col-md-4 text-end">
                    <span class="badge bg-primary">{{ "%.1f"|format(pattern.validation_accuracy * 100) }}% accurate</span>
                    <br><small>{{ pattern.occurrences_found }} occurrences</small>
                </div>
            </div>
        </div>
        {% endfor %}
    </div>
</div>
{% endfor %}
{% endblock %}'''
        
        analytics_template = '''{% extends "base.html" %}
{% block title %}Analytics - Continuous Learning System{% endblock %}
{% block header %}Performance Analytics{% endblock %}
{% block content %}
{% if no_data %}
<div class="alert alert-warning">
    <i class="fas fa-exclamation-triangle"></i>
    No analytics data available yet. The system needs to run for some time to collect metrics.
</div>
{% else %}
<div class="row">
    <div class="col-12">
        <div id="charts"></div>
    </div>
</div>
{% endif %}

<script>
{% if not no_data %}
const charts = {{ charts_json|safe }};
charts.forEach((chart, index) => {
    const div = document.createElement('div');
    div.id = 'chart-' + index;
    div.style.marginBottom = '30px';
    document.getElementById('charts').appendChild(div);
    Plotly.newPlot('chart-' + index, chart.data, chart.layout);
});
{% endif %}
</script>
{% endblock %}'''
        
        admin_template = '''{% extends "base.html" %}
{% block title %}Admin - Continuous Learning System{% endblock %}
{% block header %}Admin Controls{% endblock %}
{% block content %}
<div class="row">
    <div class="col-md-6">
        <div class="card">
            <div class="card-header">
                <h5><i class="fas fa-cog"></i> System Controls</h5>
            </div>
            <div class="card-body">
                <form method="POST">
                    {{ form.hidden_tag() }}
                    <div class="mb-3">
                        {{ form.action.label(class="form-label") }}
                        {{ form.action(class="form-select") }}
                    </div>
                    <div class="mb-3">
                        {{ form.submit(class="btn btn-primary") }}
                    </div>
                </form>
            </div>
        </div>
    </div>
    
    <div class="col-md-6">
        {% if result %}
        <div class="card">
            <div class="card-header">
                <h5><i class="fas fa-clipboard-list"></i> Result</h5>
            </div>
            <div class="card-body">
                <pre>{{ result|pprint }}</pre>
            </div>
        </div>
        {% else %}
        <div class="card">
            <div class="card-header">
                <h5><i class="fas fa-info-circle"></i> Admin Information</h5>
            </div>
            <div class="card-body">
                <p>Use the controls on the left to manage the learning system:</p>
                <ul>
                    <li><strong>Run Learning Cycle:</strong> Force an immediate learning update</li>
                    <li><strong>Enable/Disable Auto-Deploy:</strong> Control automatic model deployment</li>
                    <li><strong>Get System Status:</strong> View detailed system status</li>
                </ul>
                <div class="alert alert-warning mt-3">
                    <i class="fas fa-exclamation-triangle"></i>
                    <strong>Warning:</strong> Admin actions can affect system performance. Use carefully in production.
                </div>
            </div>
        </div>
        {% endif %}
    </div>
</div>
{% endblock %}'''
        
        # Write additional templates
        with open(self.templates_dir / 'patterns.html', 'w') as f:
            f.write(patterns_template)
        
        with open(self.templates_dir / 'analytics.html', 'w') as f:
            f.write(analytics_template)
        
        with open(self.templates_dir / 'admin.html', 'w') as f:
            f.write(admin_template)
    
    def _create_static_files(self):
        """Create static CSS and JS files"""
        # This would contain custom CSS and JavaScript files
        # For now, we're using CDN resources
        pass
    
    def run(self, host='127.0.0.1', port=5000, debug=False):
        """Run the web interface"""
        logger.info(f"Starting web interface on http://{host}:{port}")
        self.app.run(host=host, port=port, debug=debug)


def create_web_interface(learning_system: ContinuousLearningSystem) -> LearningWebInterface:
    """Create web interface for the learning system"""
    return LearningWebInterface(learning_system)


def run_web_demo():
    """Run a demo of the web interface"""
    print("🚀 Starting Learning System Web Interface Demo")
    print("=" * 50)
    
    # Create learning system
    from continuous_learning_system import create_learning_system
    learning_system = create_learning_system(enable_auto_deployment=False)
    
    # Create web interface
    web_interface = create_web_interface(learning_system)
    
    print("📱 Web interface created successfully!")
    print("🌐 Starting Flask development server...")
    print("📊 Access the dashboard at: http://127.0.0.1:5000")
    print("💬 Provide feedback at: http://127.0.0.1:5000/feedback")
    print("📈 View analytics at: http://127.0.0.1:5000/analytics")
    print("⚙️  Admin controls at: http://127.0.0.1:5000/admin")
    print("\n🛑 Press Ctrl+C to stop the server")
    
    try:
        web_interface.run(debug=True)
    except KeyboardInterrupt:
        print("\n👋 Web interface stopped by user")


if __name__ == "__main__":
    # Check if required dependencies are available
    try:
        import flask
        import plotly
        run_web_demo()
    except ImportError as e:
        print(f"❌ Required dependencies not available: {e}")
        print("📦 Install with: pip install flask flask-wtf wtforms plotly")
        print("🔧 Or run without web interface using continuous_learning_system.py directly")