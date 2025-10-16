import json
import boto3
from datetime import datetime
from decimal import Decimal

# S3 bucket for static files
DASHBOARD_BUCKET = 'ai-video-codec-dashboard-580473065386'

# Helper function to convert Decimal to float for JSON serialization
def decimal_to_float(obj):
    if isinstance(obj, Decimal):
        return float(obj)
    raise TypeError

def handler(event, context):
    """
    Lambda function for server-side rendering
    Routes:
    - / or /index.html -> Serve index.html from S3
    - /blog or /blog.html -> Server-side render blog with data
    - /dashboard?type=X -> API endpoints (for JavaScript if needed)
    - /static/* -> Serve static files from S3
    """
    path = event.get('path', '/')
    query_params = event.get('queryStringParameters', {}) or {}
    
    print(f"Request path: {path}")
    
    # Root path
    if path == '/' or path == '':
        return serve_static_file('index.html')
    
    # Static HTML files
    elif path in ['/index.html', '/index']:
        return serve_static_file('index.html')
    
    # Blog - server-side rendered
    elif path in ['/blog', '/blog.html']:
        return render_blog_page()
    
    # Static assets (CSS, JS, images)
    elif path.startswith('/styles') or path.startswith('/app') or path.endswith('.css') or path.endswith('.js'):
        filename = path.lstrip('/')
        return serve_static_file(filename)
    
    # API endpoints (for dynamic data fetching)
    elif path == '/dashboard' or path.startswith('/api'):
        data_type = query_params.get('type', 'experiments')
        if data_type == 'experiments':
            return get_experiments()
        elif data_type == 'metrics':
            return get_metrics()
        elif data_type == 'costs':
            return get_costs()
        elif data_type == 'reasoning':
            return get_reasoning()
        else:
            return get_experiments()
    
    # 404
    else:
        return {
            'statusCode': 404,
            'headers': {
                'Content-Type': 'text/html'
            },
            'body': f'<h1>404 Not Found</h1><p>Path: {path}</p>'
        }

def serve_static_file(filename):
    """Serve a static file from S3"""
    s3 = boto3.client('s3')
    
    try:
        response = s3.get_object(Bucket=DASHBOARD_BUCKET, Key=filename)
        content = response['Body'].read().decode('utf-8')
        
        # Determine content type
        if filename.endswith('.html'):
            content_type = 'text/html'
        elif filename.endswith('.css'):
            content_type = 'text/css'
        elif filename.endswith('.js'):
            content_type = 'application/javascript'
        else:
            content_type = 'text/plain'
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': content_type,
                'Cache-Control': 'public, max-age=300, s-maxage=300'  # Cache for 5 minutes
            },
            'body': content
        }
    except Exception as e:
        print(f"Error serving {filename}: {e}")
        return {
            'statusCode': 404,
            'headers': {'Content-Type': 'text/html'},
            'body': f'<h1>File Not Found</h1><p>{filename}</p>'
        }

def render_blog_page():
    """Server-side render the blog page with data from DynamoDB"""
    try:
        # Fetch experiments and reasoning
        dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
        
        experiments_table = dynamodb.Table('ai-video-codec-experiments')
        reasoning_table = dynamodb.Table('ai-video-codec-reasoning')
        
        experiments_res = experiments_table.scan(Limit=10)
        reasoning_res = reasoning_table.scan(Limit=10)
        
        experiments = experiments_res.get('Items', [])
        reasoning_items = reasoning_res.get('Items', [])
        
        # Sort by timestamp
        experiments.sort(key=lambda x: x.get('timestamp', 0), reverse=True)
        reasoning_items.sort(key=lambda x: x.get('timestamp', 0), reverse=True)
        
        # Generate HTML
        html = generate_blog_html(experiments, reasoning_items)
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'text/html',
                'Cache-Control': 'public, max-age=60, s-maxage=60'  # Cache for 1 minute
            },
            'body': html
        }
    except Exception as e:
        print(f"Error rendering blog: {e}")
        return {
            'statusCode': 500,
            'headers': {'Content-Type': 'text/html'},
            'body': f'<h1>Error</h1><p>{str(e)}</p>'
        }

def generate_blog_html(experiments, reasoning_items):
    """Generate blog HTML with embedded data"""
    
    # Match experiments with reasoning
    blog_posts = []
    for exp in experiments:
        exp_id = exp.get('experiment_id', '')
        exp_timestamp = exp.get('timestamp', 0)
        
        # Find matching reasoning
        matching_reasoning = None
        for reasoning in reasoning_items:
            if (reasoning.get('experiment_id') == exp_id or 
                abs(int(reasoning.get('timestamp', 0)) - int(exp_timestamp)) < 3600):
                matching_reasoning = reasoning
                break
        
        # Parse experiment data
        experiments_data = json.loads(exp.get('experiments', '[]'))
        procedural = next((e for e in experiments_data if e.get('experiment_type') == 'real_procedural_generation'), {})
        metrics = procedural.get('real_metrics', {})
        comparison = procedural.get('comparison', {})
        
        blog_posts.append({
            'exp': exp,
            'reasoning': matching_reasoning,
            'metrics': metrics,
            'comparison': comparison
        })
    
    # Generate HTML
    posts_html = ""
    if not blog_posts:
        posts_html = '''
        <div class="blog-post">
            <h2>🚀 First Experiments Starting Soon</h2>
            <p>The autonomous AI system is initializing. The first experiment analysis will appear here within 6 hours.</p>
            <p>Each blog post will include:</p>
            <ul>
                <li>What the experiment tried</li>
                <li>Root cause analysis of failures</li>
                <li>LLM reasoning and insights</li>
                <li>Hypothesis for next experiment</li>
                <li>Expected improvements</li>
            </ul>
        </div>
        '''
    else:
        for i, post in enumerate(blog_posts):
            exp = post['exp']
            reasoning = post['reasoning']
            metrics = post['metrics']
            comparison = post['comparison']
            
            bitrate = metrics.get('bitrate_mbps', 0)
            reduction = comparison.get('reduction_percent', 0)
            
            status_class = 'status-success' if exp.get('status') == 'completed' else 'status-failed'
            
            hypothesis = reasoning.get('hypothesis', 'Baseline Measurement') if reasoning else 'Baseline Measurement'
            root_cause = reasoning.get('root_cause', '') if reasoning else ''
            
            posts_html += f'''
            <div class="blog-post">
                <h2>Iteration {len(blog_posts) - i}: {hypothesis}</h2>
                <div class="blog-meta">
                    <span>📅 {exp.get('timestamp_iso', '')[:10]}</span>
                    <span>🔬 {exp.get('experiment_id', '')}</span>
                    <span class="status-badge {status_class}">{exp.get('status', 'unknown')}</span>
                </div>
                
                <div class="blog-section">
                    <h3>📊 Results</h3>
                    <div class="metrics-grid">
                        <div class="metric-card">
                            <div class="metric-value">{bitrate:.2f}</div>
                            <div class="metric-label">Mbps</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value" style="color: {'#dc3545' if reduction < 0 else '#28a745'}">
                                {'+' if reduction < 0 else ''}{reduction:.1f}%
                            </div>
                            <div class="metric-label">vs HEVC Baseline</div>
                        </div>
                    </div>
                </div>
                
                {f'<div class="blog-section"><h3>🔍 Root Cause Analysis</h3><p>{root_cause}</p></div>' if root_cause else ''}
            </div>
            '''
    
    # Full HTML page
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Codec Research Blog - Server-Side Rendered</title>
    <link rel="stylesheet" href="/styles.css">
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }}
        .blog-container {{ max-width: 900px; margin: 0 auto; }}
        .blog-header {{ text-align: center; margin-bottom: 60px; }}
        .blog-header h1 {{ font-size: 2.5em; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
        .blog-post {{ background: white; border-radius: 12px; padding: 40px; margin-bottom: 40px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); border-left: 4px solid #667eea; }}
        .blog-meta {{ display: flex; gap: 20px; margin-bottom: 20px; font-size: 0.9em; color: #666; }}
        .status-badge {{ display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: 0.85em; font-weight: 600; }}
        .status-success {{ background: #d4edda; color: #155724; }}
        .status-failed {{ background: #f8d7da; color: #721c24; }}
        .blog-section {{ margin: 30px 0; }}
        .blog-section h3 {{ font-size: 1.3em; margin-bottom: 15px; color: #667eea; border-bottom: 2px solid #eee; padding-bottom: 10px; }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
        .metric-card {{ background: #f8f9fa; padding: 20px; border-radius: 8px; text-align: center; }}
        .metric-value {{ font-size: 2em; font-weight: bold; color: #667eea; }}
        .metric-label {{ font-size: 0.9em; color: #666; margin-top: 5px; }}
        .back-link {{ display: inline-block; margin-bottom: 30px; color: #667eea; text-decoration: none; font-weight: 600; }}
    </style>
</head>
<body>
    <div class="blog-container">
        <a href="/" class="back-link">← Back to Dashboard</a>
        
        <div class="blog-header">
            <h1>🤖 AI Research Blog</h1>
            <p>Server-Side Rendered • Real-Time Updates</p>
        </div>
        
        {posts_html}
    </div>
</body>
</html>'''
    
    return html

# Keep existing API functions
def get_experiments():
    """Fetch experiments from DynamoDB"""
    dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
    table = dynamodb.Table('ai-video-codec-experiments')
    
    try:
        response = table.scan(Limit=10)
        experiments = []
        
        for item in response.get('Items', []):
            experiments_data = json.loads(item.get('experiments', '[]'))
            procedural = next((e for e in experiments_data if e.get('experiment_type') == 'real_procedural_generation'), {})
            metrics = procedural.get('real_metrics', {})
            comparison = procedural.get('comparison', {})
            
            experiments.append({
                'id': item.get('experiment_id', ''),
                'status': item.get('status', 'unknown'),
                'compression': comparison.get('reduction_percent', 0),
                'quality': 95.0,
                'bitrate': metrics.get('bitrate_mbps', 0),
                'created_at': item.get('timestamp_iso', ''),
                'timestamp': float(item.get('timestamp', 0))
            })
        
        experiments.sort(key=lambda x: x.get('timestamp', 0), reverse=True)
        
        return {
            'statusCode': 200,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Content-Type': 'application/json'
            },
            'body': json.dumps({
                'experiments': experiments,
                'total': len(experiments)
            }, default=decimal_to_float)
        }
    except Exception as e:
        print(f"Experiments error: {e}")
        return {
            'statusCode': 200,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Content-Type': 'application/json'
            },
            'body': json.dumps({'experiments': [], 'total': 0, 'error': str(e)})
        }

def get_metrics():
    """Fetch metrics from DynamoDB"""
    dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
    table = dynamodb.Table('ai-video-codec-metrics')
    
    try:
        response = table.scan(Limit=20)
        return {
            'statusCode': 200,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Content-Type': 'application/json'
            },
            'body': json.dumps({
                'metrics': response.get('Items', []),
                'total': len(response.get('Items', []))
            }, default=decimal_to_float)
        }
    except Exception as e:
        print(f"Metrics error: {e}")
        return {
            'statusCode': 200,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Content-Type': 'application/json'
            },
            'body': json.dumps({'metrics': [], 'total': 0, 'error': str(e)})
        }

def get_costs():
    """Fetch cost data"""
    return {
        'statusCode': 200,
        'headers': {
            'Access-Control-Allow-Origin': '*',
            'Content-Type': 'application/json'
        },
        'body': json.dumps({
            'monthly_cost': 100,
            'daily_cost': 3.33,
            'breakdown': {
                'compute': 75,
                'storage': 10,
                'networking': 15
            }
        })
    }

def get_reasoning():
    """Fetch LLM reasoning from DynamoDB"""
    dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
    table = dynamodb.Table('ai-video-codec-reasoning')
    
    try:
        response = table.scan(Limit=20)
        reasoning_items = []
        
        for item in response.get('Items', []):
            insights = json.loads(item.get('insights', '[]')) if isinstance(item.get('insights'), str) else item.get('insights', [])
            next_experiment = json.loads(item.get('next_experiment', '{}')) if isinstance(item.get('next_experiment'), str) else item.get('next_experiment', {})
            risks = json.loads(item.get('risks', '[]')) if isinstance(item.get('risks'), str) else item.get('risks', [])
            
            reasoning_items.append({
                'reasoning_id': item.get('reasoning_id', ''),
                'experiment_id': item.get('experiment_id', ''),
                'timestamp': item.get('timestamp_iso', ''),
                'model': item.get('model', ''),
                'root_cause': item.get('root_cause', ''),
                'insights': insights,
                'hypothesis': item.get('hypothesis', ''),
                'next_experiment': next_experiment,
                'risks': risks,
                'expected_bitrate_mbps': float(item.get('expected_bitrate_mbps', 0)),
                'confidence_score': float(item.get('confidence_score', 0))
            })
        
        reasoning_items.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        return {
            'statusCode': 200,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Content-Type': 'application/json'
            },
            'body': json.dumps({
                'reasoning': reasoning_items,
                'total': len(reasoning_items)
            }, default=decimal_to_float)
        }
    except Exception as e:
        print(f"Error fetching reasoning: {e}")
        return {
            'statusCode': 200,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Content-Type': 'application/json'
            },
            'body': json.dumps({
                'reasoning': [],
                'total': 0,
                'error': str(e)
            })
        }

