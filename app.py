"""
Risk Score Analysis System
A web application for analyzing and visualizing risk score distributions.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for web
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here-change-in-production'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create necessary directories
Path('uploads').mkdir(exist_ok=True)
Path('static').mkdir(exist_ok=True)
Path('static/images').mkdir(exist_ok=True)

# Global variable to store dataset
df_global = None


class RiskScoreAnalyzer:
    """Class for analyzing risk score distributions."""
    
    def __init__(self, dataframe):
        """
        Initialize the analyzer with a dataframe.
        
        Args:
            dataframe: Pandas DataFrame containing Risk_Score column
        """
        self.df = dataframe
        self.risk_scores = dataframe['Risk_Score'] if 'Risk_Score' in dataframe.columns else None
        
    def validate_data(self):
        """Validate if Risk_Score column exists and has valid data."""
        if self.risk_scores is None:
            return False, "Risk_Score column not found in dataset"
        if self.risk_scores.empty:
            return False, "Risk_Score column is empty"
        if not pd.api.types.is_numeric_dtype(self.risk_scores):
            return False, "Risk_Score column must contain numeric values"
        return True, "Data validation successful"
    
    def get_basic_statistics(self):
        """Calculate basic statistics for Risk_Score."""
        stats = {
            'count': len(self.risk_scores),
            'mean': self.risk_scores.mean(),
            'median': self.risk_scores.median(),
            'std': self.risk_scores.std(),
            'min': self.risk_scores.min(),
            'max': self.risk_scores.max(),
            'q25': self.risk_scores.quantile(0.25),
            'q75': self.risk_scores.quantile(0.75),
            'skewness': self.risk_scores.skew(),
            'kurtosis': self.risk_scores.kurtosis()
        }
        return stats
    
    def create_histogram(self):
        """
        Create a histogram of the Risk_Score column.
        
        Returns:
            Base64 encoded image string
        """
        try:
            # Create figure with specific size
            plt.figure(figsize=(12, 6))
            
            # Create histogram with KDE
            sns.histplot(data=self.df, x='Risk_Score', kde=True, bins=30, 
                        color='skyblue', edgecolor='black', alpha=0.7)
            
            # Add mean and median lines
            plt.axvline(self.risk_scores.mean(), color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: {self.risk_scores.mean():.2f}')
            plt.axvline(self.risk_scores.median(), color='green', linestyle='--', 
                       linewidth=2, label=f'Median: {self.risk_scores.median():.2f}')
            
            # Add titles and labels
            plt.title('Distribution of Risk Score', fontsize=16, fontweight='bold', pad=20)
            plt.xlabel('Risk Score', fontsize=12)
            plt.ylabel('Frequency', fontsize=12)
            plt.legend(loc='upper right')
            plt.grid(True, alpha=0.3)
            
            # Adjust layout
            plt.tight_layout()
            
            # Convert plot to base64 string
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight')
            img_buffer.seek(0)
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
            plt.close()
            
            return img_base64
            
        except Exception as e:
            logger.error(f"Error creating histogram: {e}")
            plt.close()
            raise
    
    def create_boxplot(self):
        """
        Create a box plot of the Risk_Score column.
        
        Returns:
            Base64 encoded image string
        """
        try:
            # Create figure with specific size
            plt.figure(figsize=(10, 6))
            
            # Create box plot
            box_plot = sns.boxplot(data=self.df, y='Risk_Score', 
                                  color='lightblue', width=0.3)
            
            # Add individual points for better visualization
            sns.stripplot(data=self.df, y='Risk_Score', color='red', 
                         alpha=0.3, size=4, jitter=True)
            
            # Add titles and labels
            plt.title('Box Plot of Risk Score', fontsize=16, fontweight='bold', pad=20)
            plt.ylabel('Risk Score', fontsize=12)
            plt.grid(True, alpha=0.3, axis='y')
            
            # Add statistical annotations
            stats_text = f"Min: {self.risk_scores.min():.2f}\n"
            stats_text += f"Q1: {self.risk_scores.quantile(0.25):.2f}\n"
            stats_text += f"Median: {self.risk_scores.median():.2f}\n"
            stats_text += f"Q3: {self.risk_scores.quantile(0.75):.2f}\n"
            stats_text += f"Max: {self.risk_scores.max():.2f}"
            
            plt.text(1.15, self.risk_scores.median(), stats_text, 
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                    fontsize=10, verticalalignment='center')
            
            # Adjust layout
            plt.tight_layout()
            
            # Convert plot to base64 string
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight')
            img_buffer.seek(0)
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
            plt.close()
            
            return img_base64
            
        except Exception as e:
            logger.error(f"Error creating box plot: {e}")
            plt.close()
            raise
    
    def create_combined_visualization(self):
        """
        Create a combined visualization with both histogram and box plot.
        
        Returns:
            Base64 encoded image string
        """
        try:
            # Create figure with subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Histogram
            sns.histplot(data=self.df, x='Risk_Score', kde=True, bins=30,
                        color='skyblue', edgecolor='black', alpha=0.7, ax=ax1)
            ax1.axvline(self.risk_scores.mean(), color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: {self.risk_scores.mean():.2f}')
            ax1.axvline(self.risk_scores.median(), color='green', linestyle='--', 
                       linewidth=2, label=f'Median: {self.risk_scores.median():.2f}')
            ax1.set_title('Distribution of Risk Score (Histogram)', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Risk Score', fontsize=11)
            ax1.set_ylabel('Frequency', fontsize=11)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Box plot
            sns.boxplot(data=self.df, y='Risk_Score', color='lightblue', 
                       width=0.3, ax=ax2)
            sns.stripplot(data=self.df, y='Risk_Score', color='red', 
                         alpha=0.3, size=3, jitter=True, ax=ax2)
            ax2.set_title('Distribution of Risk Score (Box Plot)', fontsize=14, fontweight='bold')
            ax2.set_ylabel('Risk Score', fontsize=11)
            ax2.grid(True, alpha=0.3, axis='y')
            
            # Main title
            fig.suptitle('Risk Score Distribution Analysis', fontsize=16, fontweight='bold', y=1.02)
            
            # Adjust layout
            plt.tight_layout()
            
            # Convert plot to base64 string
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight')
            img_buffer.seek(0)
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
            plt.close()
            
            return img_base64
            
        except Exception as e:
            logger.error(f"Error creating combined visualization: {e}")
            plt.close()
            raise


def create_sample_data():
    """Create sample data for demonstration."""
    np.random.seed(42)
    data = {
        'ID': range(1, 101),
        'Name': [f'Person_{i}' for i in range(1, 101)],
        'Age': np.random.randint(18, 80, 100),
        'Risk_Score': np.random.normal(70, 15, 100),
        'Category': np.random.choice(['Low', 'Medium', 'High'], 100, p=[0.3, 0.4, 0.3])
    }
    return pd.DataFrame(data)


# Routes
@app.route('/')
def home():
    """Home page route."""
    logger.info("Accessing home page")
    
    # Create sample data if no data exists
    global df_global
    if df_global is None:
        df_global = create_sample_data()
        logger.info("Created sample dataset")
    
    return render_template('index.html')


@app.route('/analyze')
def analyze_page():
    """Analysis page route."""
    logger.info("Accessing analysis page")
    return render_template('analyze.html')


@app.route('/api/analyze/risk-score', methods=['GET'])
def analyze_risk_score():
    """
    API endpoint to analyze risk score distribution.
    Returns statistics and visualizations.
    """
    try:
        global df_global
        
        # Use sample data if no data uploaded
        if df_global is None:
            df_global = create_sample_data()
            logger.info("Using sample dataset for analysis")
        
        # Create analyzer instance
        analyzer = RiskScoreAnalyzer(df_global)
        
        # Validate data
        is_valid, message = analyzer.validate_data()
        if not is_valid:
            return jsonify({'error': message}), 400
        
        # Get statistics
        statistics = analyzer.get_basic_statistics()
        
        # Generate visualizations
        histogram_img = analyzer.create_histogram()
        boxplot_img = analyzer.create_boxplot()
        combined_img = analyzer.create_combined_visualization()
        
        # Prepare response
        response = {
            'success': True,
            'statistics': statistics,
            'visualizations': {
                'histogram': histogram_img,
                'boxplot': boxplot_img,
                'combined': combined_img
            },
            'data_info': {
                'total_records': len(df_global),
                'columns': df_global.columns.tolist(),
                'risk_score_range': [float(df_global['Risk_Score'].min()), 
                                   float(df_global['Risk_Score'].max())]
            }
        }
        
        logger.info("Risk score analysis completed successfully")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in risk score analysis: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/upload', methods=['POST'])
def upload_data():
    """Upload dataset endpoint."""
    try:
        global df_global
        
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read the file based on extension
        if file.filename.endswith('.csv'):
            df_global = pd.read_csv(file)
        elif file.filename.endswith('.xlsx'):
            df_global = pd.read_excel(file)
        elif file.filename.endswith('.json'):
            df_global = pd.read_json(file)
        else:
            return jsonify({'error': 'Unsupported file format'}), 400
        
        # Check if Risk_Score column exists
        if 'Risk_Score' not in df_global.columns:
            return jsonify({'error': 'Risk_Score column not found in dataset'}), 400
        
        logger.info(f"Dataset uploaded successfully: {file.filename}")
        return jsonify({
            'success': True,
            'message': f'Dataset uploaded successfully',
            'rows': len(df_global),
            'columns': df_global.columns.tolist()
        })
        
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/data/preview', methods=['GET'])
def data_preview():
    """Get data preview endpoint."""
    try:
        global df_global
        
        if df_global is None:
            df_global = create_sample_data()
        
        preview = df_global.head(10).to_dict('records')
        return jsonify({
            'success': True,
            'preview': preview,
            'total_rows': len(df_global),
            'columns': df_global.columns.tolist()
        })
        
    except Exception as e:
        logger.error(f"Error getting data preview: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/about')
def about():
    """About page route."""
    return render_template('about.html')


@app.errorhandler(404)
def not_found_error(error):
    """Handle 404 errors."""
    return render_template('404.html'), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return render_template('500.html'), 500


# Create templates directory and HTML files
def create_templates():
    """Create necessary template files."""
    
    templates_dir = Path('templates')
    templates_dir.mkdir(exist_ok=True)
    
    # Create base template
    base_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}Risk Score Analysis System{% endblock %}</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        
        .navbar {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            padding: 1rem 2rem;
            box-shadow: 0 2px 20px rgba(0,0,0,0.1);
            display: flex;
            justify-content: space-between;
            align-items: center;
            position: sticky;
            top: 0;
            z-index: 1000;
        }
        
        .nav-brand {
            font-size: 1.5rem;
            font-weight: bold;
            color: #667eea;
            text-decoration: none;
        }
        
        .nav-menu {
            display: flex;
            gap: 2rem;
            align-items: center;
        }
        
        .nav-link {
            color: #333;
            text-decoration: none;
            padding: 0.5rem 1rem;
            border-radius: 5px;
            transition: all 0.3s;
            font-weight: 500;
        }
        
        .nav-link:hover {
            background: #667eea;
            color: white;
            transform: translateY(-2px);
        }
        
        .nav-link.active {
            background: #667eea;
            color: white;
        }
        
        .container {
            max-width: 1400px;
            margin: 2rem auto;
            padding: 0 2rem;
        }
        
        .content {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            padding: 2rem;
            box-shadow: 0 10px 40px rgba(0,0,0,0.1);
        }
        
        .btn {
            padding: 0.75rem 1.5rem;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-weight: 600;
            transition: all 0.3s;
            text-decoration: none;
            display: inline-block;
        }
        
        .btn-primary {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        
        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
        }
        
        .btn-secondary {
            background: #f0f0f0;
            color: #333;
        }
        
        .btn-secondary:hover {
            background: #e0e0e0;
            transform: translateY(-2px);
        }
        
        .alert {
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 1rem;
        }
        
        .alert-info {
            background: #e3f2fd;
            color: #1976d2;
            border-left: 4px solid #1976d2;
        }
        
        .alert-success {
            background: #e8f5e8;
            color: #388e3c;
            border-left: 4px solid #388e3c;
        }
        
        .alert-error {
            background: #ffebee;
            color: #d32f2f;
            border-left: 4px solid #d32f2f;
        }
        
        footer {
            text-align: center;
            padding: 2rem;
            color: white;
            margin-top: 2rem;
        }
    </style>
</head>
<body>
    <nav class="navbar">
        <a href="/" class="nav-brand">📊 Risk Analyzer</a>
        <div class="nav-menu">
            <a href="/" class="nav-link {% if request.path == '/' %}active{% endif %}">Home</a>
            <a href="/analyze" class="nav-link {% if request.path == '/analyze' %}active{% endif %}">Analyze</a>
            <a href="/about" class="nav-link {% if request.path == '/about' %}active{% endif %}">About</a>
        </div>
    </nav>
    
    <div class="container">
        <div class="content">
            {% block content %}{% endblock %}
        </div>
    </div>
    
    <footer>
        <p>&copy; 2024 Risk Score Analysis System. All rights reserved.</p>
    </footer>
</body>
</html>
    """
    
    # Create index.html
    index_html = """
{% extends "base.html" %}

{% block title %}Home - Risk Score Analysis System{% endblock %}

{% block content %}
<div style="text-align: center; padding: 3rem 1rem;">
    <h1 style="font-size: 2.5rem; margin-bottom: 1rem; color: #333;">
        Risk Score Analysis System
    </h1>
    <p style="font-size: 1.2rem; color: #666; margin-bottom: 2rem;">
        Analyze and visualize the distribution of risk scores with interactive charts and statistics
    </p>
    
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 2rem; margin-top: 3rem;">
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 2rem; border-radius: 15px;">
            <h3>📊 Histogram Analysis</h3>
            <p style="margin-top: 1rem;">View the distribution of risk scores with frequency analysis and KDE curve</p>
        </div>
        
        <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; padding: 2rem; border-radius: 15px;">
            <h3>📦 Box Plot</h3>
            <p style="margin-top: 1rem;">Identify outliers, quartiles, and the spread of your risk score data</p>
        </div>
        
        <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); color: white; padding: 2rem; border-radius: 15px;">
            <h3>📈 Statistics</h3>
            <p style="margin-top: 1rem;">Get detailed statistical measures including mean, median, and standard deviation</p>
        </div>
    </div>
    
    <div style="margin-top: 3rem;">
        <a href="/analyze" class="btn btn-primary" style="font-size: 1.1rem; padding: 1rem 2rem;">
            Start Analysis →
        </a>
    </div>
</div>
{% endblock %}
    """
    
    # Create analyze.html with the visualizations
    analyze_html = """
{% extends "base.html" %}

{% block title %}Analyze Risk Score{% endblock %}

{% block content %}
<h1 style="margin-bottom: 2rem; color: #333;">Risk Score Distribution Analysis</h1>

<div style="margin-bottom: 2rem;">
    <button onclick="loadAnalysis()" class="btn btn-primary" style="margin-right: 1rem;">
        🔄 Analyze Risk Score
    </button>
    <button onclick="uploadData()" class="btn btn-secondary">
        📁 Upload Dataset
    </button>
    <input type="file" id="fileInput" accept=".csv,.xlsx,.json" style="display: none;" onchange="handleFileUpload(event)">
</div>

<div id="loading" style="display: none; text-align: center; padding: 2rem;">
    <div style="font-size: 1.2rem; color: #667eea;">Loading analysis...</div>
</div>

<div id="error" class="alert alert-error" style="display: none;"></div>
<div id="success" class="alert alert-success" style="display: none;"></div>

<div id="statistics" style="display: none; margin-bottom: 2rem;">
    <h2 style="margin-bottom: 1rem; color: #333;">Statistical Summary</h2>
    <div id="statsGrid" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem;">
    </div>
</div>

<div id="visualizations" style="display: none;">
    <h2 style="margin-bottom: 1rem; color: #333;">Visualizations</h2>
    
    <div style="margin-bottom: 2rem;">
        <h3 style="color: #555; margin-bottom: 1rem;">Distribution of Risk Score (Histogram)</h3>
        <img id="histogram" style="width: 100%; border-radius: 10px; box-shadow: 0 5px 20px rgba(0,0,0,0.1);">
    </div>
    
    <div style="margin-bottom: 2rem;">
        <h3 style="color: #555; margin-bottom: 1rem;">Box Plot of Risk Score</h3>
        <img id="boxplot" style="width: 100%; border-radius: 10px; box-shadow: 0 5px 20px rgba(0,0,0,0.1);">
    </div>
    
    <div style="margin-bottom: 2rem;">
        <h3 style="color: #555; margin-bottom: 1rem;">Combined Analysis</h3>
        <img id="combined" style="width: 100%; border-radius: 10px; box-shadow: 0 5px 20px rgba(0,0,0,0.1);">
    </div>
</div>

<script>
function loadAnalysis() {
    const loading = document.getElementById('loading');
    const error = document.getElementById('error');
    const success = document.getElementById('success');
    const statistics = document.getElementById('statistics');
    const visualizations = document.getElementById('visualizations');
    
    loading.style.display = 'block';
    error.style.display = 'none';
    success.style.display = 'none';
    statistics.style.display = 'none';
    visualizations.style.display = 'none';
    
    fetch('/api/analyze/risk-score')
        .then(response => response.json())
        .then(data => {
            loading.style.display = 'none';
            
            if (data.error) {
                error.textContent = 'Error: ' + data.error;
                error.style.display = 'block';
                return;
            }
            
            // Display success message
            success.textContent = 'Analysis completed successfully!';
            success.style.display = 'block';
            
            // Display statistics
            const statsGrid = document.getElementById('statsGrid');
            statsGrid.innerHTML = '';
            const stats = data.statistics;
            const statItems = [
                {label: 'Count', value: stats.count},
                {label: 'Mean', value: stats.mean.toFixed(2)},
                {label: 'Median', value: stats.median.toFixed(2)},
                {label: 'Std Dev', value: stats.std.toFixed(2)},
                {label: 'Min', value: stats.min.toFixed(2)},
                {label: 'Max', value: stats.max.toFixed(2)},
                {label: 'Q25', value: stats.q25.toFixed(2)},
                {label: 'Q75', value: stats.q75.toFixed(2)},
                {label: 'Skewness', value: stats.skewness.toFixed(2)},
                {label: 'Kurtosis', value: stats.kurtosis.toFixed(2)}
            ];
            
            statItems.forEach(item => {
                const div = document.createElement('div');
                div.style.cssText = 'background: #f8f9fa; padding: 1.5rem; border-radius: 10px; text-align: center; border-left: 4px solid #667eea;';
                div.innerHTML = `
                    <div style="color: #666; font-size: 0.9rem; margin-bottom: 0.5rem;">${item.label}</div>
                    <div style="color: #333; font-size: 1.5rem; font-weight: bold;">${item.value}</div>
                `;
                statsGrid.appendChild(div);
            });
            statistics.style.display = 'block';
            
            // Display visualizations
            document.getElementById('histogram').src = 'data:image/png;base64,' + data.visualizations.histogram;
            document.getElementById('boxplot').src = 'data:image/png;base64,' + data.visualizations.boxplot;
            document.getElementById('combined').src = 'data:image/png;base64,' + data.visualizations.combined;
            visualizations.style.display = 'block';
        })
        .catch(err => {
            loading.style.display = 'none';
            error.textContent = 'Error: ' + err.message;
            error.style.display = 'block';
        });
}

function uploadData() {
    document.getElementById('fileInput').click();
}

function handleFileUpload(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    const formData = new FormData();
    formData.append('file', file);
    
    const loading = document.getElementById('loading');
    const error = document.getElementById('error');
    const success = document.getElementById('success');
    
    loading.style.display = 'block';
    error.style.display = 'none';
    success.style.display = 'none';
    
    fetch('/api/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        loading.style.display = 'none';
        
        if (data.error) {
            error.textContent = 'Error: ' + data.error;
            error.style.display = 'block';
        } else {
            success.textContent = data.message + ' (' + data.rows + ' rows loaded)';
            success.style.display = 'block';
            // Automatically load analysis after upload
            setTimeout(loadAnalysis, 1000);
        }
    })
    .catch(err => {
        loading.style.display = 'none';
        error.textContent = 'Error: ' + err.message;
        error.style.display = 'block';
    });
}

// Load analysis on page load
window.onload = function() {
    loadAnalysis();
};
</script>
{% endblock %}
    """
    
    # Create about.html
    about_html = """
{% extends "base.html" %}

{% block title %}About - Risk Score Analysis System{% endblock %}

{% block content %}
<h1 style="margin-bottom: 2rem; color: #333;">About This System</h1>

<div style="max-width: 800px; margin: 0 auto;">
    <h2 style="color: #667eea; margin-bottom: 1rem;">Risk Score Analysis System</h2>
    <p style="line-height: 1.8; color: #666; margin-bottom: 2rem;">
        This system provides comprehensive analysis and visualization of risk score distributions.
        It helps you understand the patterns, outliers, and statistical properties of your risk assessment data.
    </p>
    
    <h3 style="color: #333; margin-bottom: 1rem;">Features:</h3>
    <ul style="line-height: 2; color: #666; margin-bottom: 2rem;">
        <li>📊 Histogram visualization with KDE curve</li>
        <li>📦 Box plot analysis for outlier detection</li>
        <li>📈 Comprehensive statistical summary</li>
        <li>📁 Support for CSV, Excel, and JSON data formats</li>
        <li>🔄 Real-time interactive analysis</li>
    </ul>
    
    <h3 style="color: #333; margin-bottom: 1rem;">How to Use:</h3>
    <ol style="line-height: 2; color: #666; margin-bottom: 2rem;">
        <li>Navigate to the Analyze page</li>
        <li>Upload your dataset or use the sample data</li>
        <li>View the statistical summary and visualizations</li>
        <li>Interpret the results for your risk assessment</li>
    </ol>
</div>
{% endblock %}
    """
    
    # Create error templates
    error_404_html = """
{% extends "base.html" %}
{% block title %}Page Not Found{% endblock %}
{% block content %}
<div style="text-align: center; padding: 3rem;">
    <h1 style="font-size: 6rem; color: #667eea;">404</h1>
    <h2>Page Not Found</h2>
    <p style="color: #666; margin: 1rem 0;">The page you're looking for doesn't exist.</p>
    <a href="/" class="btn btn-primary">Go Home</a>
</div>
{% endblock %}
    """
    
    error_500_html = """
{% extends "base.html" %}
{% block title %}Server Error{% endblock %}
{% block content %}
<div style="text-align: center; padding: 3rem;">
    <h1 style="font-size: 6rem; color: #f5576c;">500</h1>
    <h2>Internal Server Error</h2>
    <p style="color: #666; margin: 1rem 0;">Something went wrong. Please try again later.</p>
    <a href="/" class="btn btn-primary">Go Home</a>
</div>
{% endblock %}
    """
    
    # Write all template files
    templates = {
        'base.html': base_html,
        'index.html': index_html,
        'analyze.html': analyze_html,
        'about.html': about_html,
        '404.html': error_404_html,
        '500.html': error_500_html
    }
    
    for filename, content in templates.items():
        with open(templates_dir / filename, 'w') as f:
            f.write(content)
    
    logger.info("Templates created successfully")


def main():
    """Main entry point."""
    try:
        logger.info("Starting Risk Score Analysis System...")
        
        # Create templates
        create_templates()
        
        # Create sample data
        global df_global
        df_global = create_sample_data()
        logger.info(f"Sample data created with {len(df_global)} records")
        
        # Run Flask app
        logger.info("Starting web server...")
        print("\n" + "="*60)
        print("Risk Score Analysis System")
        print("="*60)
        print("\n🚀 Server starting at: http://127.0.0.1:5000")
        print("📊 Home page: http://127.0.0.1:5000/")
        print("📈 Analysis page: http://127.0.0.1:5000/analyze")
        print("\nPress Ctrl+C to stop the server")
        print("="*60 + "\n")
        
        app.run(debug=True, host='0.0.0.0', port=5000)
        
    except Exception as e:
        logger.error(f"Error starting application: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
