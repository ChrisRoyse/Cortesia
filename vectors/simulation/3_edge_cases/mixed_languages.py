"""
File mixing multiple languages and embedded code to test language detection.
This simulates real-world files that might contain SQL, HTML, CSS, JavaScript, etc.
"""

import sqlite3
import json
from typing import Dict, List, Any

class WebApplicationManager:
    """Manager that handles web application with embedded code in multiple languages."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.connection = None
        
    def initialize_database(self):
        """Initialize SQLite database with embedded SQL."""
        
        # Embedded SQL DDL statements
        create_users_table = """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username VARCHAR(50) UNIQUE NOT NULL,
            email VARCHAR(255) UNIQUE NOT NULL,
            password_hash VARCHAR(255) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP,
            is_active BOOLEAN DEFAULT TRUE,
            profile_data JSON,
            preferences TEXT
        );
        """
        
        create_posts_table = """
        CREATE TABLE IF NOT EXISTS posts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            title VARCHAR(200) NOT NULL,
            content TEXT NOT NULL,
            slug VARCHAR(250) UNIQUE NOT NULL,
            status ENUM('draft', 'published', 'archived') DEFAULT 'draft',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            published_at TIMESTAMP NULL,
            view_count INTEGER DEFAULT 0,
            metadata JSON,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
            INDEX idx_posts_user_id (user_id),
            INDEX idx_posts_status (status),
            INDEX idx_posts_published_at (published_at),
            FULLTEXT INDEX ft_posts_content (title, content)
        );
        """
        
        create_comments_table = """
        CREATE TABLE IF NOT EXISTS comments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            post_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            parent_id INTEGER NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_approved BOOLEAN DEFAULT FALSE,
            ip_address VARCHAR(45),
            user_agent TEXT,
            FOREIGN KEY (post_id) REFERENCES posts(id) ON DELETE CASCADE,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
            FOREIGN KEY (parent_id) REFERENCES comments(id) ON DELETE CASCADE
        );
        """
        
        # Execute SQL statements
        self.connection = sqlite3.connect(self.db_path)
        cursor = self.connection.cursor()
        
        cursor.execute(create_users_table)
        cursor.execute(create_posts_table)
        cursor.execute(create_comments_table)
        
        # Insert sample data with complex SQL
        sample_data_sql = """
        INSERT OR IGNORE INTO users (username, email, password_hash, profile_data) VALUES
        ('admin', 'admin@example.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewaBIgNZxPe0u3lO', 
         '{"bio": "Administrator", "avatar": "/static/avatars/admin.jpg", "social_links": {"twitter": "@admin"}}'),
        ('john_doe', 'john@example.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewaBIgNZxPe0u3lO',
         '{"bio": "Software Developer", "location": "San Francisco", "website": "https://johndoe.dev"}'),
        ('jane_smith', 'jane@example.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewaBIgNZxPe0u3lO',
         '{"bio": "UX Designer", "location": "New York", "skills": ["Design", "Prototyping", "User Research"]}');
        
        INSERT OR IGNORE INTO posts (user_id, title, content, slug, status, published_at) VALUES
        (1, 'Welcome to Our Blog', 'This is the first post on our new blog platform!', 'welcome-to-our-blog', 'published', CURRENT_TIMESTAMP),
        (2, 'Getting Started with Python', 'Python is a great language for beginners...', 'getting-started-python', 'published', CURRENT_TIMESTAMP),
        (3, 'Design Principles for Modern Web Apps', 'Good design is essential for user experience...', 'design-principles-web-apps', 'draft', NULL);
        """
        
        cursor.execute(sample_data_sql)
        self.connection.commit()
    
    def get_user_analytics(self, user_id: int) -> Dict[str, Any]:
        """Get user analytics with complex SQL query."""
        
        analytics_query = """
        WITH user_stats AS (
            SELECT 
                u.id,
                u.username,
                COUNT(DISTINCT p.id) as total_posts,
                COUNT(DISTINCT c.id) as total_comments,
                AVG(p.view_count) as avg_post_views,
                SUM(p.view_count) as total_views,
                MAX(p.created_at) as last_post_date,
                MAX(c.created_at) as last_comment_date
            FROM users u
            LEFT JOIN posts p ON u.id = p.user_id AND p.status = 'published'
            LEFT JOIN comments c ON u.id = c.user_id AND c.is_approved = TRUE
            WHERE u.id = ?
            GROUP BY u.id, u.username
        ),
        engagement_metrics AS (
            SELECT 
                p.user_id,
                COUNT(c.id) as comments_received,
                AVG(
                    CASE 
                        WHEN p.created_at > datetime('now', '-30 days') THEN p.view_count 
                        ELSE 0 
                    END
                ) as recent_avg_views
            FROM posts p
            LEFT JOIN comments c ON p.id = c.post_id AND c.is_approved = TRUE
            WHERE p.user_id = ? AND p.status = 'published'
            GROUP BY p.user_id
        )
        SELECT 
            us.*,
            COALESCE(em.comments_received, 0) as comments_received,
            COALESCE(em.recent_avg_views, 0) as recent_avg_views,
            CASE 
                WHEN us.total_posts = 0 THEN 'Newcomer'
                WHEN us.total_posts < 5 THEN 'Beginner'
                WHEN us.total_posts < 20 THEN 'Regular'
                ELSE 'Power User'
            END as user_level
        FROM user_stats us
        LEFT JOIN engagement_metrics em ON us.id = em.user_id;
        """
        
        cursor = self.connection.cursor()
        cursor.execute(analytics_query, (user_id, user_id))
        result = cursor.fetchone()
        
        if result:
            columns = [description[0] for description in cursor.description]
            return dict(zip(columns, result))
        return {}
    
    def generate_dashboard_html(self, user_data: Dict[str, Any]) -> str:
        """Generate HTML dashboard with embedded CSS and JavaScript."""
        
        # Embedded CSS styles
        css_styles = """
        <style>
            /* Modern dashboard styles */
            .dashboard-container {
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }
            
            .dashboard-header {
                background: rgba(255, 255, 255, 0.95);
                border-radius: 15px;
                padding: 30px;
                margin-bottom: 30px;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
                backdrop-filter: blur(10px);
                border: 1px solid rgba(255, 255, 255, 0.2);
            }
            
            .dashboard-header h1 {
                color: #2c3e50;
                margin: 0 0 10px 0;
                font-size: 2.5em;
                font-weight: 300;
                text-align: center;
            }
            
            .user-welcome {
                text-align: center;
                color: #7f8c8d;
                font-size: 1.2em;
                margin-bottom: 20px;
            }
            
            .stats-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }
            
            .stat-card {
                background: rgba(255, 255, 255, 0.9);
                border-radius: 12px;
                padding: 25px;
                text-align: center;
                box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
                transition: transform 0.3s ease, box-shadow 0.3s ease;
                border: 1px solid rgba(255, 255, 255, 0.3);
            }
            
            .stat-card:hover {
                transform: translateY(-5px);
                box-shadow: 0 8px 30px rgba(0, 0, 0, 0.15);
            }
            
            .stat-number {
                font-size: 2.5em;
                font-weight: bold;
                margin-bottom: 10px;
            }
            
            .stat-number.posts { color: #3498db; }
            .stat-number.views { color: #e74c3c; }
            .stat-number.comments { color: #2ecc71; }
            .stat-number.level { color: #f39c12; }
            
            .stat-label {
                color: #7f8c8d;
                font-size: 1.1em;
                text-transform: uppercase;
                letter-spacing: 1px;
            }
            
            .activity-section {
                background: rgba(255, 255, 255, 0.95);
                border-radius: 15px;
                padding: 30px;
                margin-bottom: 30px;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            }
            
            .activity-section h3 {
                color: #2c3e50;
                margin-bottom: 20px;
                font-size: 1.8em;
                border-bottom: 2px solid #ecf0f1;
                padding-bottom: 10px;
            }
            
            .progress-bar {
                width: 100%;
                height: 20px;
                background: #ecf0f1;
                border-radius: 10px;
                overflow: hidden;
                margin: 15px 0;
            }
            
            .progress-fill {
                height: 100%;
                background: linear-gradient(90deg, #3498db, #2ecc71);
                border-radius: 10px;
                transition: width 0.5s ease;
            }
            
            .btn-primary {
                background: linear-gradient(45deg, #3498db, #2ecc71);
                color: white;
                border: none;
                padding: 12px 25px;
                border-radius: 25px;
                font-size: 1.1em;
                cursor: pointer;
                transition: all 0.3s ease;
                text-decoration: none;
                display: inline-block;
                margin: 10px 5px;
            }
            
            .btn-primary:hover {
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(52, 152, 219, 0.4);
            }
            
            .chart-container {
                position: relative;
                height: 300px;
                margin: 20px 0;
            }
            
            @media (max-width: 768px) {
                .dashboard-container {
                    padding: 10px;
                }
                
                .stats-grid {
                    grid-template-columns: 1fr;
                }
                
                .dashboard-header h1 {
                    font-size: 2em;
                }
            }
            
            /* Animation keyframes */
            @keyframes fadeInUp {
                from {
                    opacity: 0;
                    transform: translateY(30px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }
            
            .animate-in {
                animation: fadeInUp 0.6s ease forwards;
            }
        </style>
        """
        
        # Embedded JavaScript
        javascript_code = """
        <script>
            // Dashboard JavaScript functionality
            class DashboardManager {
                constructor() {
                    this.chartData = null;
                    this.initializeEventListeners();
                    this.loadUserData();
                    this.startRealTimeUpdates();
                }
                
                initializeEventListeners() {
                    // Add click handlers for interactive elements
                    document.addEventListener('DOMContentLoaded', () => {
                        this.animateCards();
                        this.initializeCharts();
                        this.setupProgressBars();
                    });
                    
                    // Handle window resize for responsive charts
                    window.addEventListener('resize', () => {
                        this.resizeCharts();
                    });
                    
                    // Add keyboard shortcuts
                    document.addEventListener('keydown', (e) => {
                        if (e.ctrlKey || e.metaKey) {
                            switch(e.key) {
                                case 'r':
                                    e.preventDefault();
                                    this.refreshDashboard();
                                    break;
                                case 'd':
                                    e.preventDefault();
                                    this.downloadReport();
                                    break;
                            }
                        }
                    });
                }
                
                animateCards() {
                    const cards = document.querySelectorAll('.stat-card');
                    cards.forEach((card, index) => {
                        setTimeout(() => {
                            card.classList.add('animate-in');
                        }, index * 100);
                    });
                }
                
                async loadUserData() {
                    try {
                        const response = await fetch('/api/user/dashboard-data', {
                            method: 'GET',
                            headers: {
                                'Authorization': `Bearer ${localStorage.getItem('authToken')}`,
                                'Content-Type': 'application/json'
                            }
                        });
                        
                        if (response.ok) {
                            const data = await response.json();
                            this.updateDashboardWithData(data);
                        } else {
                            console.error('Failed to load user data:', response.statusText);
                            this.showErrorMessage('Failed to load dashboard data');
                        }
                    } catch (error) {
                        console.error('Error loading user data:', error);
                        this.showErrorMessage('Network error while loading data');
                    }
                }
                
                updateDashboardWithData(data) {
                    // Update stat cards with real data
                    const statElements = {
                        posts: document.querySelector('.stat-number.posts'),
                        views: document.querySelector('.stat-number.views'),
                        comments: document.querySelector('.stat-number.comments'),
                        level: document.querySelector('.stat-number.level')
                    };
                    
                    // Animate number counting
                    this.animateNumber(statElements.posts, 0, data.total_posts || 0, 1000);
                    this.animateNumber(statElements.views, 0, data.total_views || 0, 1500);
                    this.animateNumber(statElements.comments, 0, data.comments_received || 0, 1200);
                    
                    if (statElements.level) {
                        statElements.level.textContent = data.user_level || 'Newcomer';
                    }
                    
                    // Update progress bars
                    this.updateProgressBars(data);
                    
                    // Update charts
                    this.updateCharts(data);
                }
                
                animateNumber(element, start, end, duration) {
                    if (!element) return;
                    
                    const startTime = performance.now();
                    const difference = end - start;
                    
                    const step = (currentTime) => {
                        const elapsed = currentTime - startTime;
                        const progress = Math.min(elapsed / duration, 1);
                        
                        // Easing function for smooth animation
                        const easeOutCubic = 1 - Math.pow(1 - progress, 3);
                        const current = Math.floor(start + difference * easeOutCubic);
                        
                        element.textContent = current.toLocaleString();
                        
                        if (progress < 1) {
                            requestAnimationFrame(step);
                        }
                    };
                    
                    requestAnimationFrame(step);
                }
                
                setupProgressBars() {
                    const progressBars = document.querySelectorAll('.progress-fill');
                    progressBars.forEach(bar => {
                        const targetWidth = bar.dataset.width || '0%';
                        setTimeout(() => {
                            bar.style.width = targetWidth;
                        }, 500);
                    });
                }
                
                updateProgressBars(data) {
                    // Calculate progress percentages based on user level
                    const levelProgressMap = {
                        'Newcomer': 15,
                        'Beginner': 35,
                        'Regular': 65,
                        'Power User': 90
                    };
                    
                    const progressPercent = levelProgressMap[data.user_level] || 0;
                    const progressBar = document.querySelector('.progress-fill');
                    if (progressBar) {
                        progressBar.style.width = `${progressPercent}%`;
                        progressBar.dataset.width = `${progressPercent}%`;
                    }
                }
                
                initializeCharts() {
                    // Create a simple chart using Canvas API
                    const chartCanvas = document.getElementById('activityChart');
                    if (!chartCanvas) return;
                    
                    const ctx = chartCanvas.getContext('2d');
                    const chartData = {
                        labels: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
                        datasets: [{
                            label: 'Posts',
                            data: [2, 3, 1, 4, 2, 5, 3],
                            borderColor: '#3498db',
                            backgroundColor: 'rgba(52, 152, 219, 0.1)',
                            fill: true
                        }, {
                            label: 'Comments',
                            data: [5, 8, 3, 12, 7, 15, 9],
                            borderColor: '#2ecc71',
                            backgroundColor: 'rgba(46, 204, 113, 0.1)',
                            fill: true
                        }]
                    };
                    
                    this.drawChart(ctx, chartData);
                }
                
                drawChart(ctx, data) {
                    const canvas = ctx.canvas;
                    const width = canvas.width;
                    const height = canvas.height;
                    
                    // Clear canvas
                    ctx.clearRect(0, 0, width, height);
                    
                    // Simple line chart implementation
                    const padding = 40;
                    const chartWidth = width - 2 * padding;
                    const chartHeight = height - 2 * padding;
                    
                    // Draw axes
                    ctx.strokeStyle = '#bdc3c7';
                    ctx.lineWidth = 1;
                    ctx.beginPath();
                    ctx.moveTo(padding, padding);
                    ctx.lineTo(padding, height - padding);
                    ctx.lineTo(width - padding, height - padding);
                    ctx.stroke();
                    
                    // Draw data lines
                    data.datasets.forEach((dataset, datasetIndex) => {
                        ctx.strokeStyle = dataset.borderColor;
                        ctx.lineWidth = 3;
                        ctx.beginPath();
                        
                        dataset.data.forEach((value, index) => {
                            const x = padding + (index / (dataset.data.length - 1)) * chartWidth;
                            const y = height - padding - (value / Math.max(...dataset.data)) * chartHeight;
                            
                            if (index === 0) {
                                ctx.moveTo(x, y);
                            } else {
                                ctx.lineTo(x, y);
                            }
                        });
                        
                        ctx.stroke();
                        
                        // Draw points
                        ctx.fillStyle = dataset.borderColor;
                        dataset.data.forEach((value, index) => {
                            const x = padding + (index / (dataset.data.length - 1)) * chartWidth;
                            const y = height - padding - (value / Math.max(...dataset.data)) * chartHeight;
                            
                            ctx.beginPath();
                            ctx.arc(x, y, 4, 0, 2 * Math.PI);
                            ctx.fill();
                        });
                    });
                }
                
                async refreshDashboard() {
                    // Show loading indicator
                    this.showLoadingIndicator();
                    
                    try {
                        await this.loadUserData();
                        this.showSuccessMessage('Dashboard refreshed successfully');
                    } catch (error) {
                        this.showErrorMessage('Failed to refresh dashboard');
                    } finally {
                        this.hideLoadingIndicator();
                    }
                }
                
                async downloadReport() {
                    try {
                        const response = await fetch('/api/user/export-report', {
                            method: 'GET',
                            headers: {
                                'Authorization': `Bearer ${localStorage.getItem('authToken')}`
                            }
                        });
                        
                        if (response.ok) {
                            const blob = await response.blob();
                            const url = window.URL.createObjectURL(blob);
                            const a = document.createElement('a');
                            a.href = url;
                            a.download = `dashboard-report-${new Date().toISOString().split('T')[0]}.pdf`;
                            document.body.appendChild(a);
                            a.click();
                            window.URL.revokeObjectURL(url);
                            document.body.removeChild(a);
                            
                            this.showSuccessMessage('Report downloaded successfully');
                        } else {
                            throw new Error('Failed to download report');
                        }
                    } catch (error) {
                        console.error('Error downloading report:', error);
                        this.showErrorMessage('Failed to download report');
                    }
                }
                
                startRealTimeUpdates() {
                    // WebSocket connection for real-time updates
                    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                    const wsUrl = `${wsProtocol}//${window.location.host}/ws/dashboard`;
                    
                    try {
                        this.websocket = new WebSocket(wsUrl);
                        
                        this.websocket.onopen = () => {
                            console.log('WebSocket connected for real-time updates');
                        };
                        
                        this.websocket.onmessage = (event) => {
                            const data = JSON.parse(event.data);
                            this.handleRealTimeUpdate(data);
                        };
                        
                        this.websocket.onclose = () => {
                            console.log('WebSocket connection closed');
                            // Attempt to reconnect after 5 seconds
                            setTimeout(() => this.startRealTimeUpdates(), 5000);
                        };
                        
                        this.websocket.onerror = (error) => {
                            console.error('WebSocket error:', error);
                        };
                    } catch (error) {
                        console.error('Failed to establish WebSocket connection:', error);
                    }
                }
                
                handleRealTimeUpdate(data) {
                    switch (data.type) {
                        case 'new_comment':
                            this.incrementStat('comments', 1);
                            this.showNotification(`New comment on your post: "${data.post_title}"`);
                            break;
                        case 'post_view':
                            this.incrementStat('views', 1);
                            break;
                        case 'new_follower':
                            this.showNotification(`New follower: ${data.follower_username}`);
                            break;
                        default:
                            console.log('Unknown real-time update type:', data.type);
                    }
                }
                
                incrementStat(statType, amount) {
                    const element = document.querySelector(`.stat-number.${statType}`);
                    if (element) {
                        const currentValue = parseInt(element.textContent.replace(/,/g, '')) || 0;
                        const newValue = currentValue + amount;
                        this.animateNumber(element, currentValue, newValue, 500);
                    }
                }
                
                showNotification(message) {
                    // Create and show a notification
                    const notification = document.createElement('div');
                    notification.className = 'notification';
                    notification.textContent = message;
                    notification.style.cssText = `
                        position: fixed;
                        top: 20px;
                        right: 20px;
                        background: #2ecc71;
                        color: white;
                        padding: 15px 20px;
                        border-radius: 8px;
                        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
                        z-index: 1000;
                        animation: slideInRight 0.3s ease;
                    `;
                    
                    document.body.appendChild(notification);
                    
                    setTimeout(() => {
                        notification.style.animation = 'slideOutRight 0.3s ease';
                        setTimeout(() => {
                            document.body.removeChild(notification);
                        }, 300);
                    }, 3000);
                }
                
                showLoadingIndicator() {
                    // Implementation for loading indicator
                }
                
                hideLoadingIndicator() {
                    // Implementation for hiding loading indicator
                }
                
                showSuccessMessage(message) {
                    this.showNotification(message);
                }
                
                showErrorMessage(message) {
                    const notification = document.createElement('div');
                    notification.className = 'notification error';
                    notification.textContent = message;
                    notification.style.cssText = `
                        position: fixed;
                        top: 20px;
                        right: 20px;
                        background: #e74c3c;
                        color: white;
                        padding: 15px 20px;
                        border-radius: 8px;
                        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
                        z-index: 1000;
                        animation: slideInRight 0.3s ease;
                    `;
                    
                    document.body.appendChild(notification);
                    
                    setTimeout(() => {
                        notification.style.animation = 'slideOutRight 0.3s ease';
                        setTimeout(() => {
                            document.body.removeChild(notification);
                        }, 300);
                    }, 5000);
                }
            }
            
            // Initialize dashboard when page loads
            const dashboard = new DashboardManager();
            
            // Additional utility functions
            function formatNumber(num) {
                if (num >= 1000000) {
                    return (num / 1000000).toFixed(1) + 'M';
                } else if (num >= 1000) {
                    return (num / 1000).toFixed(1) + 'K';
                }
                return num.toString();
            }
            
            function debounce(func, wait) {
                let timeout;
                return function executedFunction(...args) {
                    const later = () => {
                        clearTimeout(timeout);
                        func(...args);
                    };
                    clearTimeout(timeout);
                    timeout = setTimeout(later, wait);
                };
            }
            
            // Export for use in other scripts
            window.DashboardManager = DashboardManager;
        </script>
        """
        
        # Main HTML structure
        html_template = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Dashboard - {user_data.get('username', 'User')}</title>
            {css_styles}
        </head>
        <body>
            <div class="dashboard-container">
                <div class="dashboard-header">
                    <h1>Welcome to Your Dashboard</h1>
                    <div class="user-welcome">
                        Hello, <strong>{user_data.get('username', 'User')}</strong>! 
                        You're a <span class="user-level">{user_data.get('user_level', 'Newcomer')}</span>
                    </div>
                </div>
                
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-number posts">{user_data.get('total_posts', 0)}</div>
                        <div class="stat-label">Total Posts</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number views">{user_data.get('total_views', 0)}</div>
                        <div class="stat-label">Total Views</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number comments">{user_data.get('comments_received', 0)}</div>
                        <div class="stat-label">Comments Received</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number level">{user_data.get('user_level', 'Newcomer')}</div>
                        <div class="stat-label">User Level</div>
                    </div>
                </div>
                
                <div class="activity-section">
                    <h3>Activity Overview</h3>
                    <p>Your progress to the next level:</p>
                    <div class="progress-bar">
                        <div class="progress-fill" data-width="65%"></div>
                    </div>
                    
                    <div class="chart-container">
                        <canvas id="activityChart" width="800" height="300"></canvas>
                    </div>
                    
                    <div style="text-align: center; margin-top: 20px;">
                        <a href="/posts/new" class="btn-primary">Create New Post</a>
                        <a href="/profile/settings" class="btn-primary">Edit Profile</a>
                        <button onclick="dashboard.downloadReport()" class="btn-primary">Download Report</button>
                    </div>
                </div>
            </div>
            
            {javascript_code}
        </body>
        </html>
        """
        
        return html_template
    
    def generate_api_config(self) -> Dict[str, Any]:
        """Generate API configuration with embedded JSON and YAML-like structures."""
        
        # Configuration that might come from a YAML file
        yaml_like_config = """
        api:
          version: "v1"
          base_url: "https://api.example.com"
          authentication:
            type: "oauth2"
            scopes:
              - "read:posts"
              - "write:posts" 
              - "read:profile"
              - "write:profile"
          rate_limiting:
            requests_per_minute: 1000
            burst_limit: 50
          endpoints:
            users:
              list: "GET /users"
              create: "POST /users"
              get: "GET /users/{id}"
              update: "PUT /users/{id}"
              delete: "DELETE /users/{id}"
            posts:
              list: "GET /posts"
              create: "POST /posts"
              get: "GET /posts/{id}"
              update: "PUT /posts/{id}"
              delete: "DELETE /posts/{id}"
              search: "GET /posts/search?q={query}"
        """
        
        # Convert to Python dict (simulating YAML parsing)
        config = {
            "api": {
                "version": "v1",
                "base_url": "https://api.example.com",
                "authentication": {
                    "type": "oauth2",
                    "scopes": ["read:posts", "write:posts", "read:profile", "write:profile"]
                },
                "rate_limiting": {
                    "requests_per_minute": 1000,
                    "burst_limit": 50
                },
                "endpoints": {
                    "users": {
                        "list": "GET /users",
                        "create": "POST /users",
                        "get": "GET /users/{id}",
                        "update": "PUT /users/{id}",
                        "delete": "DELETE /users/{id}"
                    },
                    "posts": {
                        "list": "GET /posts",
                        "create": "POST /posts", 
                        "get": "GET /posts/{id}",
                        "update": "PUT /posts/{id}",
                        "delete": "DELETE /posts/{id}",
                        "search": "GET /posts/search?q={query}"
                    }
                }
            },
            # Additional configuration with embedded regex patterns
            "validation": {
                "patterns": {
                    "email": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
                    "username": r"^[a-zA-Z0-9_]{3,20}$",
                    "password": r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$",
                    "slug": r"^[a-z0-9]+(?:-[a-z0-9]+)*$"
                }
            },
            # Database connection strings (various formats)
            "database": {
                "primary": "postgresql://user:password@localhost:5432/app_db",
                "replica": "postgresql://user:password@replica.example.com:5432/app_db",
                "redis": "redis://localhost:6379/0",
                "mongodb": "mongodb://user:password@localhost:27017/app_db"
            }
        }
        
        return config
    
    def close_connection(self):
        """Close database connection."""
        if self.connection:
            self.connection.close()

# Example usage with mixed language content
if __name__ == "__main__":
    manager = WebApplicationManager("example_app.db")
    manager.initialize_database()
    
    # Get user analytics
    user_analytics = manager.get_user_analytics(1)
    print("User Analytics:", json.dumps(user_analytics, indent=2))
    
    # Generate HTML dashboard
    html_content = manager.generate_dashboard_html(user_analytics)
    with open("dashboard.html", "w", encoding="utf-8") as f:
        f.write(html_content)
    
    # Generate API config
    api_config = manager.generate_api_config()
    print("API Config:", json.dumps(api_config, indent=2))
    
    manager.close_connection()