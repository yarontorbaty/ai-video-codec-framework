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
    
    # Root path (CloudFront routing issues: / comes in as /dashboard)
    # BUT: /dashboard?type=X is an API call, not the root page
    if (path == '/' or path == '' or path == '/dashboard') and not query_params.get('type'):
        return render_dashboard_page()
    
    # API endpoint via /dashboard?type=X (for backward compatibility)
    elif path == '/dashboard' and query_params.get('type'):
        data_type = query_params.get('type')
        if data_type == 'experiments':
            return get_experiments()
        elif data_type == 'experiment':
            # Get single experiment details
            exp_id = query_params.get('id')
            if exp_id:
                return get_experiment_details(exp_id)
            else:
                return {
                    'statusCode': 400,
                    'headers': {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*'},
                    'body': json.dumps({'error': 'Missing experiment id parameter'})
                }
        elif data_type == 'metrics':
            return get_metrics()
        elif data_type == 'costs':
            return get_costs()
        elif data_type == 'reasoning':
            return get_reasoning()
        elif data_type == 'infrastructure':
            return get_infrastructure()
        else:
            return get_experiments()
    
    # Static HTML files
    elif path in ['/index.html', '/index']:
        return serve_static_file('index.html')
    
    # Admin interface
    elif path in ['/admin', '/admin.html']:
        return serve_static_file('admin.html')
    
    # Admin API proxy
    elif path.startswith('/admin/'):
        # Proxy to admin API Gateway
        import urllib.request
        import urllib.parse
        
        admin_api_endpoint = 'https://mrjjwxaxma.execute-api.us-east-1.amazonaws.com/production'
        # Keep the full path including /admin
        
        try:
            # Get Authorization header from request
            headers_from_event = event.get('headers', {})
            request_headers = {'Content-Type': 'application/json'}
            
            # Forward Authorization header if present (case-insensitive lookup)
            for header_name, header_value in headers_from_event.items():
                if header_name.lower() == 'authorization':
                    request_headers['Authorization'] = header_value
                    break
            
            # Forward the request
            if event.get('httpMethod') == 'POST':
                body = event.get('body', '{}')
                req = urllib.request.Request(
                    f"{admin_api_endpoint}{path}",
                    data=body.encode('utf-8'),
                    headers=request_headers
                )
                with urllib.request.urlopen(req) as response:
                    return {
                        'statusCode': response.status,
                        'headers': {
                            'Content-Type': 'application/json',
                            'Access-Control-Allow-Origin': '*'
                        },
                        'body': response.read().decode('utf-8')
                    }
            else:
                req = urllib.request.Request(
                    f"{admin_api_endpoint}{path}",
                    headers=request_headers
                )
                with urllib.request.urlopen(req) as response:
                    return {
                        'statusCode': response.status,
                        'headers': {
                            'Content-Type': 'application/json',
                            'Access-Control-Allow-Origin': '*'
                        },
                        'body': response.read().decode('utf-8')
                    }
        except Exception as e:
            return {
                'statusCode': 500,
                'headers': {'Content-Type': 'application/json'},
                'body': json.dumps({'error': str(e)})
            }
    
    # Blog - server-side rendered
    elif path in ['/blog', '/blog.html']:
        return render_blog_page()
    
    # Static assets (CSS, JS, images)
    elif path.startswith('/styles') or path.startswith('/app') or path.endswith('.css') or path.endswith('.js'):
        filename = path.lstrip('/')
        return serve_static_file(filename)
    
    # API endpoints (for dynamic data fetching) - requires ?type= parameter
    elif path.startswith('/api'):
        data_type = query_params.get('type')
        if not data_type:
            return {'statusCode': 400, 'body': 'Missing type parameter'}
        if data_type == 'experiments':
            return get_experiments()
        elif data_type == 'metrics':
            return get_metrics()
        elif data_type == 'costs':
            return get_costs()
        elif data_type == 'reasoning':
            return get_reasoning()
        elif data_type == 'infrastructure':
            return get_infrastructure()
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

def render_dashboard_page():
    """Server-side render the dashboard with real data"""
    dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
    
    try:
        # Fetch ALL experiments with pagination
        experiments_table = dynamodb.Table('ai-video-codec-experiments')
        
        # Get ALL experiments using pagination
        all_items = []
        scan_kwargs = {}
        
        while True:
            experiments_response = experiments_table.scan(**scan_kwargs)
            all_items.extend(experiments_response.get('Items', []))
            
            # Check if there are more items to fetch
            if 'LastEvaluatedKey' not in experiments_response:
                break
            scan_kwargs['ExclusiveStartKey'] = experiments_response['LastEvaluatedKey']
        
        # Process experiments with more details
        experiments = []
        for item in all_items:
            experiments_data = json.loads(item.get('experiments', '[]'))
            procedural = next((e for e in experiments_data if e.get('experiment_type') == 'real_procedural_generation'), {})
            ai_neural = next((e for e in experiments_data if e.get('experiment_type') == 'real_ai_neural'), {})
            llm_code = next((e for e in experiments_data if e.get('experiment_type') in ['llm_generated_code', 'llm_generated_code_evolution']), {})
            
            metrics = procedural.get('real_metrics', {})
            comparison = procedural.get('comparison', {})
            
            # Determine methods used
            methods = []
            if procedural:
                methods.append('Procedural')
            if ai_neural:
                methods.append('Neural Network')
            methods_str = ' + '.join(methods) if methods else 'Hybrid'
            
            # Extract LLM evolution info
            evolution_info = None
            if llm_code:
                evolution = llm_code.get('evolution', {})
                if evolution:
                    evolution_info = {
                        'status': evolution.get('status', 'unknown'),
                        'adopted': evolution.get('adopted', False),
                        'version': evolution.get('version', 0),
                        'metrics': evolution.get('metrics', {}),
                        'reason': evolution.get('reason', '')
                    }
            
            # Parse timestamp for time of day
            timestamp_str = item.get('timestamp_iso', '')
            time_of_day = ''
            if timestamp_str:
                try:
                    from datetime import datetime
                    dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    time_of_day = dt.strftime('%I:%M %p')
                except:
                    time_of_day = timestamp_str[11:16] if len(timestamp_str) > 16 else ''
            
            experiments.append({
                'id': item.get('experiment_id', ''),
                'status': item.get('status', 'unknown'),
                'compression': comparison.get('reduction_percent', 0),
                'reduction_percent': comparison.get('reduction_percent', 0),
                'bitrate': metrics.get('bitrate_mbps', 0),
                'psnr_db': float(metrics.get('psnr_db')) if metrics.get('psnr_db') else None,
                'ssim': float(metrics.get('ssim')) if metrics.get('ssim') else None,
                'quality': metrics.get('quality'),
                'quality_verified': metrics.get('quality_verified', False),
                'achievement_tier': comparison.get('achievement_tier'),
                'target_achieved': comparison.get('target_achieved', False),
                'current_phase': item.get('current_phase', 'unknown'),
                'phase_completed': item.get('phase_completed', 'unknown'),
                'timestamp': item.get('timestamp_iso', ''),
                'time_of_day': time_of_day,
                'methods': methods_str,
                'evolution': evolution_info,
                'video_url': procedural.get('video_url'),
                'decoder_s3_key': procedural.get('decoder_s3_key'),
                'full_data': experiments_data  # Keep full data for blog linking
            })
        
        experiments.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        # Fetch infrastructure
        ec2 = boto3.client('ec2', region_name='us-east-1')
        ec2_response = ec2.describe_instances(
            Filters=[
                {'Name': 'tag:Project', 'Values': ['ai-video-codec']},
                {'Name': 'instance-state-name', 'Values': ['running', 'stopped', 'pending']}
            ]
        )
        
        running_instances = 0
        total_instances = 0
        instance_types = []
        for reservation in ec2_response.get('Reservations', []):
            for instance in reservation.get('Instances', []):
                total_instances += 1
                if instance['State']['Name'] == 'running':
                    running_instances += 1
                    instance_types.append(instance['InstanceType'])
        
        # Fetch costs from Cost Explorer
        ce = boto3.client('ce', region_name='us-east-1')
        monthly_cost = 0
        cost_breakdown = {'EC2': 0, 'S3': 0, 'Lambda': 0, 'Other': 0}
        
        try:
            from datetime import timedelta
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=7)
            
            response = ce.get_cost_and_usage(
                TimePeriod={
                    'Start': start_date.strftime('%Y-%m-%d'),
                    'End': end_date.strftime('%Y-%m-%d')
                },
                Granularity='DAILY',
                Metrics=['UnblendedCost'],
                GroupBy=[{'Type': 'SERVICE', 'Key': 'SERVICE'}],
                Filter={
                    'Tags': {
                        'Key': 'Project',
                        'Values': ['ai-video-codec']
                    }
                }
            )
            
            # Calculate total and breakdown
            total_cost = 0
            for result in response.get('ResultsByTime', []):
                for group in result.get('Groups', []):
                    service = group['Keys'][0]
                    cost = float(group['Metrics']['UnblendedCost']['Amount'])
                    total_cost += cost
                    
                    if 'EC2' in service or 'Elastic Compute' in service:
                        cost_breakdown['EC2'] += cost
                    elif 'S3' in service or 'Simple Storage' in service:
                        cost_breakdown['S3'] += cost
                    elif 'Lambda' in service:
                        cost_breakdown['Lambda'] += cost
                    else:
                        cost_breakdown['Other'] += cost
            
            monthly_cost = (total_cost / 7) * 30 if total_cost > 0 else 0
            
            # Scale breakdown to monthly
            for key in cost_breakdown:
                cost_breakdown[key] = round((cost_breakdown[key] / 7) * 30, 2)
                
        except Exception as e:
            print(f"Cost fetch error: {e}")
            # Use estimated costs based on instance types
            ec2_hourly_rates = {
                'c6i.xlarge': 0.17,
                'g4dn.xlarge': 0.526,
                't3.medium': 0.0416
            }
            for itype in instance_types:
                monthly_cost += ec2_hourly_rates.get(itype, 0.1) * 730  # hours per month
            cost_breakdown = {
                'EC2': round(monthly_cost * 0.85, 2),
                'S3': round(monthly_cost * 0.08, 2),
                'Lambda': round(monthly_cost * 0.05, 2),
                'Other': round(monthly_cost * 0.02, 2)
            }
        
        # Generate experiments table rows with blog links (show latest 50)
        experiments_html = ""
        for i, exp in enumerate(experiments[:50]):
            status_class = 'completed' if exp['status'] == 'completed' else 'running'
            
            # Positive reduction = good (smaller file), negative = bad (larger file)
            compression = exp['compression']
            if compression > 0:
                compression_display = f'<span style="color: green;">↓ {compression:.1f}%</span>'
            elif compression < 0:
                compression_display = f'<span style="color: red;">↑ {abs(compression):.1f}%</span>'
            else:
                compression_display = f'{compression:.1f}%'
            
            # PSNR display
            psnr = exp.get('psnr_db')
            if psnr and psnr > 0:
                psnr_color = '#10b981' if psnr >= 30 else ('#f59e0b' if psnr >= 25 else '#ef4444')
                psnr_display = f'<span style="color: {psnr_color}; font-weight: 600;">{psnr:.1f} dB</span>'
            else:
                psnr_display = '<span style="color: #666;">—</span>'
            
            # Quality display
            quality = exp.get('quality')
            if quality and quality != 'unknown':
                quality_colors = {'excellent': '#10b981', 'good': '#20c997', 'acceptable': '#f59e0b', 'poor': '#ef4444'}
                quality_color = quality_colors.get(quality, '#666')
                quality_display = f'<span style="color: {quality_color}; font-weight: 600;">{quality.title()}</span>'
            else:
                quality_display = '<span style="color: #666;">—</span>'
            
            # Achievement tier
            tier = exp.get('achievement_tier', '🎯 In Progress')
            tier_colors = {'🏆 90% Reduction': '#fbbf24', '🥇 70% Reduction': '#10b981', '🥈 50% Reduction': '#3b82f6'}
            tier_color = tier_colors.get(tier, '#94a3b8')
            tier_display = f'<span style="color: {tier_color}; font-size: 0.9em;">{tier}</span>'
            
            # Phase badge
            phase = exp.get('current_phase', 'unknown')
            phase_data = {
                'design': ('fa-lightbulb', '#3b82f6'),
                'deploy': ('fa-upload', '#8b5cf6'),
                'validation': ('fa-check-circle', '#f59e0b'),
                'execution': ('fa-play-circle', '#10b981'),
                'quality_verification': ('fa-eye', '#ec4899'),
                'analysis': ('fa-chart-line', '#06b6d4')
            }
            phase_icon, phase_color = phase_data.get(phase, ('fa-question', '#94a3b8'))
            phase_display = f'<i class="fas {phase_icon}" style="color: {phase_color};"></i>'
            
            # Evolution badge
            evolution_badge = ''
            if exp.get('evolution'):
                evo = exp['evolution']
                if evo.get('adopted'):
                    version = evo.get('version', 0)
                    evolution_badge = f'<span style="background: #28a745; color: white; padding: 2px 8px; border-radius: 10px; font-size: 0.75em; margin-left: 5px;" title="Code v{version} adopted">🎉 v{version}</span>'
                elif evo.get('status') == 'rejected':
                    reason = evo.get('reason', 'Not better')
                    evolution_badge = f'<span style="background: #ffc107; color: #333; padding: 2px 8px; border-radius: 10px; font-size: 0.75em; margin-left: 5px;" title="{reason}">⏭️</span>'
            
            experiments_html += f'''
                <div class="table-row" style="cursor: pointer; grid-template-columns: 1fr 0.6fr 0.5fr 0.9fr 0.7fr 0.6fr 0.7fr 0.7fr 0.7fr 0.5fr 0.7fr 0.4fr;" onclick="window.location.href='/blog.html#exp-{i+1}'">
                    <div class="col">{exp['id'][:18]}...</div>
                    <div class="col"><span class="status-badge {status_class}">{exp['status']}</span></div>
                    <div class="col">{exp.get('time_of_day', 'N/A')}</div>
                    <div class="col">{exp['methods']}{evolution_badge}</div>
                    <div class="col">{compression_display}</div>
                    <div class="col">{tier_display}</div>
                    <div class="col">{psnr_display}</div>
                    <div class="col">{quality_display}</div>
                    <div class="col">{exp['bitrate']:.2f} Mbps</div>
                    <div class="col">{phase_display}</div>
                    <div class="col">{exp['timestamp'][:10]}</div>
                    <div class="col"><a href="/blog.html#exp-{i+1}" style="color: #667eea; text-decoration: none;"><i class="fas fa-arrow-right"></i></a></div>
                </div>
            '''
        
        # Generate full HTML
        html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AiV1 - AI Video Codec Dashboard</title>
    <link rel="stylesheet" href="/styles.css">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <meta http-equiv="refresh" content="60">
</head>
<body>
    <div class="dashboard">
        <header class="header">
            <div class="header-content">
                <div class="logo">
                    <i class="fas fa-video"></i>
                    <h1>AiV1</h1>
                </div>
                <nav style="display: flex; gap: 20px; align-items: center;">
                    <a href="/blog.html" style="color: #667eea; text-decoration: none; font-weight: 500; display: flex; align-items: center; gap: 0.5rem;">
                        <i class="fas fa-book-open"></i> Research Blog
                    </a>
                    <div class="status-indicator">
                        <div class="status-dot connected"></div>
                        <span>Live • {running_instances} Running</span>
                    </div>
                </nav>
            </div>
        </header>

        <main class="main-content">
            <section class="overview-section">
                <div class="overview-grid">
                    <div class="card">
                        <div class="card-header">
                            <i class="fas fa-flask"></i>
                            <h3>Total Experiments</h3>
                        </div>
                        <div class="card-content">
                            <div class="metric-value">{len(experiments)}</div>
                            <div class="metric-label">AI Codec Iterations</div>
                        </div>
                    </div>
                    <div class="card">
                        <div class="card-header">
                            <i class="fas fa-compress"></i>
                            <h3>Best Compression</h3>
                        </div>
                        <div class="card-content">
                            <div class="metric-value">{max([e['compression'] for e in experiments], default=0):.1f}%</div>
                            <div class="metric-label">vs HEVC Baseline</div>
                        </div>
                    </div>
                    <div class="card">
                        <div class="card-header">
                            <i class="fas fa-server"></i>
                            <h3>Infrastructure</h3>
                        </div>
                        <div class="card-content">
                            <div class="metric-value">{running_instances}/{total_instances}</div>
                            <div class="metric-label">Active Instances</div>
                        </div>
                    </div>
                    <div class="card">
                        <div class="card-header">
                            <i class="fas fa-dollar-sign"></i>
                            <h3>Monthly Cost</h3>
                        </div>
                        <div class="card-content">
                            <div class="metric-value">${monthly_cost:.2f}</div>
                            <div class="metric-label">Estimated Monthly</div>
                            <div style="margin-top: 1rem; font-size: 0.85rem; color: #718096;">
                                <div style="display: flex; justify-content: space-between; margin: 0.25rem 0;">
                                    <span><i class="fas fa-server"></i> EC2:</span>
                                    <strong>${cost_breakdown['EC2']:.2f}</strong>
                                </div>
                                <div style="display: flex; justify-content: space-between; margin: 0.25rem 0;">
                                    <span><i class="fas fa-database"></i> S3:</span>
                                    <strong>${cost_breakdown['S3']:.2f}</strong>
                                </div>
                                <div style="display: flex; justify-content: space-between; margin: 0.25rem 0;">
                                    <span><i class="fas fa-bolt"></i> Lambda:</span>
                                    <strong>${cost_breakdown['Lambda']:.2f}</strong>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </section>

            <section class="experiments-section">
                <h2><i class="fas fa-flask"></i> Recent Experiments</h2>
                <div class="experiments-table">
                    <div class="table-header" style="grid-template-columns: 1fr 0.6fr 0.5fr 0.9fr 0.7fr 0.6fr 0.7fr 0.7fr 0.7fr 0.5fr 0.7fr 0.4fr;">
                        <div>Experiment ID</div>
                        <div>Status</div>
                        <div>Time</div>
                        <div>Methods</div>
                        <div>Reduction</div>
                        <div><i class="fas fa-trophy"></i> Tier</div>
                        <div><i class="fas fa-chart-line"></i> PSNR</div>
                        <div><i class="fas fa-eye"></i> Quality</div>
                        <div>Bitrate</div>
                        <div><i class="fas fa-tasks"></i> Phase</div>
                        <div>Date</div>
                        <div>Details</div>
                    </div>
                    <div class="table-body">
                        {experiments_html if experiments_html else '<div class="table-row loading">No experiments yet</div>'}
                    </div>
                </div>
            </section>

            <footer style="margin-top: 3rem; text-align: center; color: rgba(255,255,255,0.7); font-size: 0.875rem;">
                <p><i class="fas fa-bolt"></i> Server-Side Rendered • Auto-refresh every 60s • <a href="/blog.html" style="color: #667eea;"><i class="fas fa-book-open"></i> View Research Blog</a></p>
            </footer>
        </main>
    </div>
</body>
</html>
'''
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'text/html',
                'Cache-Control': 'public, max-age=30, s-maxage=30'  # Cache for 30 seconds only
            },
            'body': html
        }
    except Exception as e:
        print(f"Dashboard render error: {e}")
        import traceback
        traceback.print_exc()
        return {
            'statusCode': 500,
            'headers': {'Content-Type': 'text/html'},
            'body': f'<h1>Error rendering dashboard</h1><pre>{str(e)}</pre>'
        }

def generate_next_steps_html(next_experiment):
    """Generate HTML for next steps section."""
    if not next_experiment or not isinstance(next_experiment, dict):
        return ""
    
    approach = next_experiment.get('approach', '')
    changes = next_experiment.get('changes', [])
    expected_improvement = next_experiment.get('expected_improvement', '')
    
    if not approach and not changes:
        return ""
    
    # Parse changes if it's a string
    if isinstance(changes, str):
        try:
            changes = json.loads(changes)
        except:
            changes = [changes] if changes else []
    
    changes_html = ""
    for change in changes:
        changes_html += f"<li><i class='fas fa-arrow-right'></i> {change}</li>"
    
    html = '<div class="blog-section next-steps">'
    html += '<h3><i class="fas fa-arrow-circle-right"></i> Next Steps</h3>'
    
    if approach:
        html += f'<p><strong>Approach:</strong> {approach}</p>'
    
    if changes_html:
        html += f'<p><strong>Planned Changes:</strong></p><ul>{changes_html}</ul>'
    
    if expected_improvement:
        html += f'<p><strong>Expected Improvement:</strong> {expected_improvement}</p>'
    
    html += '</div>'
    return html

def render_blog_page():
    """Server-side render the blog page with data from DynamoDB"""
    try:
        # Fetch experiments and reasoning
        dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
        
        experiments_table = dynamodb.Table('ai-video-codec-experiments')
        reasoning_table = dynamodb.Table('ai-video-codec-reasoning')
        
        # Fetch ALL experiments with pagination
        experiments = []
        scan_kwargs = {}
        while True:
            experiments_res = experiments_table.scan(**scan_kwargs)
            experiments.extend(experiments_res.get('Items', []))
            if 'LastEvaluatedKey' not in experiments_res:
                break
            scan_kwargs['ExclusiveStartKey'] = experiments_res['LastEvaluatedKey']
        
        # Fetch ALL reasoning with pagination
        reasoning_items = []
        scan_kwargs = {}
        while True:
            reasoning_res = reasoning_table.scan(**scan_kwargs)
            reasoning_items.extend(reasoning_res.get('Items', []))
            if 'LastEvaluatedKey' not in reasoning_res:
                break
            scan_kwargs['ExclusiveStartKey'] = reasoning_res['LastEvaluatedKey']
        
        # Sort by timestamp
        experiments.sort(key=lambda x: x.get('timestamp', 0), reverse=True)
        reasoning_items.sort(key=lambda x: x.get('timestamp', 0), reverse=True)
        
        # Generate HTML
        html = generate_blog_html(experiments, reasoning_items)
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'text/html',
                'Cache-Control': 'no-cache, no-store, must-revalidate',
                'Pragma': 'no-cache',
                'Expires': '0'
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
        
        # Find matching reasoning by experiment_id (most reliable)
        matching_reasoning = None
        for reasoning in reasoning_items:
            if reasoning.get('experiment_id') == exp_id:
                matching_reasoning = reasoning
                break
        
        # Fallback: find by timestamp if experiment_id doesn't match (within 60 seconds)
        if not matching_reasoning:
            for reasoning in reasoning_items:
                if abs(int(reasoning.get('timestamp', 0)) - int(exp_timestamp)) < 60:
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
    
    # Calculate summary statistics
    completed_experiments = [p for p in blog_posts if p['exp'].get('status') == 'completed']
    total_experiments = len(blog_posts)
    best_bitrate = min([p['metrics'].get('bitrate_mbps', 999) for p in completed_experiments], default=10.0)
    
    tier90_count = sum(1 for p in completed_experiments if p['metrics'].get('bitrate_mbps', 999) <= 1.0)
    tier70_count = sum(1 for p in completed_experiments if 1.0 < p['metrics'].get('bitrate_mbps', 999) <= 3.0)
    tier50_count = sum(1 for p in completed_experiments if 3.0 < p['metrics'].get('bitrate_mbps', 999) <= 5.0)
    
    # Generate summary section
    summary_html = f'''
    <div class="summary-section">
        <h2>🎯 Research Progress Summary</h2>
        <p>Our autonomous AI system has conducted {total_experiments} compression experiments, systematically exploring different approaches to achieve significant bitrate reductions while maintaining video quality.</p>
        <div class="summary-stats">
            <div class="summary-stat">
                <div class="value">{total_experiments}</div>
                <div class="label">Total Experiments</div>
            </div>
            <div class="summary-stat">
                <div class="value">{best_bitrate:.2f} Mbps</div>
                <div class="label">Best Bitrate</div>
            </div>
            <div class="summary-stat">
                <div class="value">🏆 {tier90_count} | 🥇 {tier70_count} | 🥈 {tier50_count}</div>
                <div class="label">Tier Achievements</div>
            </div>
        </div>
    </div>
    '''
    
    # Generate blog posts
    posts_html = ""
    if not blog_posts:
        posts_html = '''
        <div class="blog-post">
            <h2>🚀 First Experiments Starting Soon</h2>
            <p>The autonomous AI system is initializing. The first experiment analysis will appear here within 6 hours.</p>
            <p>Each blog post will include:</p>
            <ul>
                <li><strong>Methods:</strong> Detailed description of what this experiment is trying</li>
                <li><strong>Results:</strong> Performance metrics and achievement tier</li>
                <li><strong>Analysis:</strong> What worked, what didn't, and why</li>
                <li><strong>Recommendations:</strong> What the AI suggests trying next</li>
            </ul>
        </div>
        '''
    else:
        for i, post in enumerate(blog_posts):
            exp = post['exp']
            reasoning = post['reasoning']
            metrics = post['metrics']
            comparison = post['comparison']
            
            # Get previous post for context
            previous_post = blog_posts[i+1] if i+1 < len(blog_posts) else None
            
            # Parse experiment data
            experiments_data = json.loads(exp.get('experiments', '[]'))
            procedural = next((e for e in experiments_data if e.get('experiment_type') == 'real_procedural_generation'), {})
            
            bitrate = metrics.get('bitrate_mbps', 0)
            psnr = metrics.get('psnr_db', 0)
            ssim = metrics.get('ssim', 0)
            quality = metrics.get('quality', 'unknown')
            reduction = comparison.get('reduction_percent', 0)
            achievement_tier = comparison.get('achievement_tier', '🎯 In Progress')
            
            status_class = 'status-success' if exp.get('status') == 'completed' else 'status-failed'
            
            # Generate short title from approach
            approach = procedural.get('approach', '') or (reasoning.get('hypothesis', '') if reasoning else '')
            short_title = 'Compression Experiment'
            if 'JPEG' in approach:
                short_title = 'JPEG-based Compression'
            elif 'Downsample' in approach or 'Spatial' in approach:
                short_title = 'Spatial Downsampling'
            elif 'Quantization' in approach:
                short_title = 'Advanced Quantization'
            elif 'DCT' in approach:
                short_title = 'DCT Transform'
            elif 'Neural' in approach or 'PyTorch' in approach:
                short_title = 'Neural Codec'
            elif 'Wavelet' in approach:
                short_title = 'Wavelet Transform'
            elif 'Hybrid' in approach:
                short_title = 'Hybrid Approach'
            elif approach:
                # Use first few words of approach
                short_title = ' '.join(approach.split()[:4])
            
            # Get detailed methods description
            methods_description = ''
            if reasoning:
                generated_code = reasoning.get('generated_code', {})
                if isinstance(generated_code, str):
                    try:
                        generated_code = json.loads(generated_code)
                    except:
                        generated_code = {}
                methods_description = generated_code.get('description', approach)
            if not methods_description:
                methods_description = approach or 'Hybrid compression approach combining multiple techniques.'
            
            # Build on previous results section
            previous_context_html = ''
            if previous_post:
                prev_bitrate = previous_post['metrics'].get('bitrate_mbps', 10)
                prev_approach = previous_post['reasoning'].get('hypothesis', 'previous experiment') if previous_post['reasoning'] else 'previous experiment'
                improvement_text = 'improved' if bitrate < prev_bitrate else 'did not improve'
                previous_context_html = f'''
                <div class="blog-section">
                    <h3>🔄 Building on Previous Results</h3>
                    <p>The previous experiment ({prev_approach}) achieved {prev_bitrate:.2f} Mbps. This experiment {improvement_text} upon that result, achieving {bitrate:.2f} Mbps.</p>
                </div>
                '''
            
            # Recommendations section
            recommendations_html = ''
            next_experiment = reasoning.get('next_experiment', {}) if reasoning else {}
            if isinstance(next_experiment, str):
                try:
                    next_experiment = json.loads(next_experiment)
                except:
                    next_experiment = {}
            
            if next_experiment and isinstance(next_experiment, dict):
                suggested_approach = next_experiment.get('approach', '')
                changes = next_experiment.get('changes', [])
                if isinstance(changes, str):
                    try:
                        changes = json.loads(changes)
                    except:
                        changes = []
                risks = next_experiment.get('risks', [])
                if isinstance(risks, str):
                    try:
                        risks = json.loads(risks)
                    except:
                        risks = []
                expected_bitrate = next_experiment.get('expected_bitrate_mbps', 0)
                expected_psnr = next_experiment.get('expected_psnr_db', 0)
                
                changes_html = ''.join([f"<li>{change}</li>" for change in changes]) if changes else ''
                risks_html = ''.join([f"<li>{risk}</li>" for risk in risks]) if risks else ''
                
                recommendations_html = f'''
                <div class="blog-section">
                    <h3>💡 Recommendations for Next Iteration</h3>
                    {f'<p><strong>Suggested Approach:</strong> {suggested_approach}</p>' if suggested_approach else ''}
                    {f'<p><strong>Proposed Changes:</strong></p><ul>{changes_html}</ul>' if changes_html else ''}
                    {f'<p><strong>Risks:</strong></p><ul>{risks_html}</ul>' if risks_html else ''}
                    {f'<p><strong>Expected Metrics:</strong> {expected_bitrate:.2f} Mbps bitrate, {expected_psnr:.1f} dB PSNR</p>' if expected_bitrate > 0 else ''}
                </div>
                '''
            
            posts_html += f'''
            <div class="blog-post" id="exp-{i+1}">
                <h2>Experiment #{len(blog_posts) - i}: {short_title}</h2>
                <div class="blog-meta">
                    <span><i class="fas fa-calendar"></i> {exp.get('timestamp_iso', '')[:10]}</span>
                    <span><i class="fas fa-microscope"></i> {exp.get('experiment_id', '')}</span>
                    <span class="status-badge {status_class}">{exp.get('status', 'unknown')}</span>
                    <span class="achievement-badge">{achievement_tier}</span>
                </div>
                
                {previous_context_html}
                
                <div class="blog-section">
                    <h3>🔬 Methods</h3>
                    <p>{methods_description}</p>
                </div>
                
                <div class="blog-section">
                    <h3>📊 Results</h3>
                    <div class="metrics-grid">
                        <div class="metric-card">
                            <div class="metric-value">{bitrate:.2f}</div>
                            <div class="metric-label">Bitrate (Mbps)</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value" style="color: {'#dc3545' if reduction < 0 else '#28a745'}">
                                {reduction:.1f}%
                            </div>
                            <div class="metric-label">Reduction vs HEVC</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{achievement_tier[:2]}</div>
                            <div class="metric-label">{achievement_tier[3:]}</div>
                        </div>
                        {f'<div class="metric-card"><div class="metric-value">{psnr:.1f} dB</div><div class="metric-label">PSNR</div></div>' if psnr > 0 else ''}
                        {f'<div class="metric-card"><div class="metric-value">{quality.title()}</div><div class="metric-label">Quality</div></div>' if quality and quality != 'unknown' else ''}
                    </div>
                </div>
                
                {recommendations_html}
            </div>
            '''
    
    # Full HTML page
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="refresh" content="60">
    <title>AI Codec Research Blog - Server-Side Rendered</title>
    <link rel="stylesheet" href="/styles.css">
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }}
        .blog-container {{ max-width: 900px; margin: 0 auto; }}
        .blog-header {{ text-align: center; margin-bottom: 60px; }}
        .blog-header h1 {{ font-size: 2.5em; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
        .summary-section {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 12px; padding: 40px; margin-bottom: 50px; box-shadow: 0 4px 20px rgba(102, 126, 234, 0.3); }}
        .summary-section h2 {{ font-size: 2em; margin-bottom: 20px; color: white; }}
        .summary-section p {{ font-size: 1.1em; line-height: 1.8; margin-bottom: 15px; opacity: 0.95; }}
        .summary-stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-top: 30px; }}
        .summary-stat {{ background: rgba(255, 255, 255, 0.2); border-radius: 8px; padding: 20px; text-align: center; }}
        .summary-stat .value {{ font-size: 2.5em; font-weight: bold; margin-bottom: 5px; }}
        .summary-stat .label {{ font-size: 0.9em; opacity: 0.9; }}
        .blog-post {{ background: white; border-radius: 12px; padding: 40px; margin-bottom: 40px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); border-left: 4px solid #667eea; }}
        .blog-meta {{ display: flex; gap: 20px; margin-bottom: 20px; font-size: 0.9em; color: #666; flex-wrap: wrap; }}
        .status-badge {{ display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: 0.85em; font-weight: 600; }}
        .status-success {{ background: #d4edda; color: #155724; }}
        .status-failed {{ background: #f8d7da; color: #721c24; }}
        .achievement-badge {{ display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: 0.85em; font-weight: 600; background: #e3f2fd; color: #1976d2; }}
        .blog-section {{ margin: 30px 0; }}
        .blog-section h3 {{ font-size: 1.3em; margin-bottom: 15px; color: #667eea; border-bottom: 2px solid #eee; padding-bottom: 10px; }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 20px; margin: 20px 0; }}
        .metric-card {{ background: #f8f9fa; padding: 20px; border-radius: 8px; text-align: center; }}
        .metric-value {{ font-size: 2em; font-weight: bold; color: #667eea; }}
        .metric-label {{ font-size: 0.9em; color: #666; margin-top: 5px; }}
        .back-link {{ display: inline-block; margin-bottom: 30px; color: #667eea; text-decoration: none; font-weight: 600; }}
        .next-steps {{ background: #f0f7ff; border-left: 4px solid #4CAF50; padding: 20px; border-radius: 8px; }}
        .next-steps h3 {{ color: #4CAF50; }}
        .next-steps ul {{ list-style: none; padding-left: 0; }}
        .next-steps li {{ padding: 8px 0; }}
        .next-steps .fa-arrow-right {{ color: #4CAF50; margin-right: 10px; }}
    </style>
</head>
<body>
    <div class="blog-container">
        <a href="/" class="back-link">← Back to Dashboard</a>
        
        <div class="blog-header">
            <h1>🤖 AI Research Blog</h1>
            <p>Autonomous Learning Journey • Real-Time Updates</p>
        </div>
        
        {summary_html if summary_html else ''}
        
        {posts_html}
    </div>
</body>
</html>'''
    
    return html

# Keep existing API functions
def get_experiments():
    """Fetch experiments from DynamoDB - now returns ALL experiments with full details"""
    dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
    table = dynamodb.Table('ai-video-codec-experiments')
    
    try:
        # Get ALL experiments with pagination
        all_items = []
        scan_kwargs = {}
        
        while True:
            response = table.scan(**scan_kwargs)
            all_items.extend(response.get('Items', []))
            
            # Check if there are more items to fetch
            if 'LastEvaluatedKey' not in response:
                break
            scan_kwargs['ExclusiveStartKey'] = response['LastEvaluatedKey']
        
        experiments = []
        
        for item in all_items:
            experiments_data = json.loads(item.get('experiments', '[]'))
            procedural = next((e for e in experiments_data if e.get('experiment_type') == 'real_procedural_generation'), {})
            metrics = procedural.get('real_metrics', {})
            comparison = procedural.get('comparison', {})
            
            experiments.append({
                'id': item.get('experiment_id', ''),
                'status': item.get('status', 'unknown'),
                'compression': comparison.get('reduction_percent', 0),
                'reduction_percent': comparison.get('reduction_percent', 0),
                'bitrate': metrics.get('bitrate_mbps', 0),
                'psnr_db': float(metrics.get('psnr_db')) if metrics.get('psnr_db') else None,
                'ssim': float(metrics.get('ssim')) if metrics.get('ssim') else None,
                'quality': metrics.get('quality'),
                'quality_verified': metrics.get('quality_verified', False),
                'achievement_tier': comparison.get('achievement_tier'),
                'target_achieved': comparison.get('target_achieved', False),
                'current_phase': item.get('current_phase', 'unknown'),
                'phase_completed': item.get('phase_completed', 'unknown'),
                'created_at': item.get('timestamp_iso', ''),
                'timestamp': float(item.get('timestamp', 0)),
                'video_url': procedural.get('video_url'),
                'decoder_s3_key': procedural.get('decoder_s3_key')
            })
        
        # Sort by timestamp descending (newest first)
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
        import traceback
        traceback.print_exc()
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
    """Fetch cost data from AWS Cost Explorer"""
    ce = boto3.client('ce', region_name='us-east-1')
    
    try:
        # Get costs for the last 7 days
        from datetime import datetime, timedelta
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=7)
        
        response = ce.get_cost_and_usage(
            TimePeriod={
                'Start': start_date.strftime('%Y-%m-%d'),
                'End': end_date.strftime('%Y-%m-%d')
            },
            Granularity='DAILY',
            Metrics=['UnblendedCost'],
            Filter={
                'Tags': {
                    'Key': 'Project',
                    'Values': ['ai-video-codec']
                }
            }
        )
        
        # Calculate total and daily average
        total_cost = 0
        for result in response.get('ResultsByTime', []):
            total_cost += float(result['Total']['UnblendedCost']['Amount'])
        
        daily_cost = total_cost / 7 if total_cost > 0 else 0
        monthly_cost = daily_cost * 30
        
        return {
            'statusCode': 200,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Content-Type': 'application/json'
            },
            'body': json.dumps({
                'monthly_cost': round(monthly_cost, 2),
                'daily_cost': round(daily_cost, 2),
                'weekly_cost': round(total_cost, 2),
                'breakdown': {
                    'compute': round(monthly_cost * 0.75, 2),
                    'storage': round(monthly_cost * 0.10, 2),
                    'networking': round(monthly_cost * 0.15, 2)
                }
            })
        }
    except Exception as e:
        print(f"Cost error: {e}")
        # Return zeros instead of mock data on error
        return {
            'statusCode': 200,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Content-Type': 'application/json'
            },
            'body': json.dumps({
                'monthly_cost': 0,
                'daily_cost': 0,
                'weekly_cost': 0,
                'breakdown': {
                    'compute': 0,
                    'storage': 0,
                    'networking': 0
                },
                'error': str(e)
            })
        }

def get_infrastructure():
    """Fetch infrastructure status from EC2"""
    ec2 = boto3.client('ec2', region_name='us-east-1')
    
    try:
        response = ec2.describe_instances(
            Filters=[
                {'Name': 'tag:Project', 'Values': ['ai-video-codec']},
                {'Name': 'instance-state-name', 'Values': ['running', 'stopped', 'pending']}
            ]
        )
        
        instances = []
        for reservation in response.get('Reservations', []):
            for instance in reservation.get('Instances', []):
                name_tag = next((tag['Value'] for tag in instance.get('Tags', []) if tag['Key'] == 'Name'), 'Unknown')
                purpose_tag = next((tag['Value'] for tag in instance.get('Tags', []) if tag['Key'] == 'Purpose'), 'Unknown')
                
                instances.append({
                    'id': instance['InstanceId'],
                    'name': name_tag,
                    'type': instance['InstanceType'],
                    'state': instance['State']['Name'],
                    'purpose': purpose_tag,
                    'launch_time': instance.get('LaunchTime', datetime.now()).isoformat() if 'LaunchTime' in instance else ''
                })
        
        return {
            'statusCode': 200,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Content-Type': 'application/json'
            },
            'body': json.dumps({
                'instances': instances,
                'total': len(instances),
                'running': len([i for i in instances if i['state'] == 'running'])
            }, default=str)
        }
    except Exception as e:
        print(f"Infrastructure error: {e}")
        return {
            'statusCode': 200,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Content-Type': 'application/json'
            },
            'body': json.dumps({
                'instances': [],
                'total': 0,
                'running': 0,
                'error': str(e)
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

def get_experiment_details(experiment_id):
    """Fetch detailed experiment data including LLM analysis"""
    dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
    experiments_table = dynamodb.Table('ai-video-codec-experiments')
    reasoning_table = dynamodb.Table('ai-video-codec-reasoning')
    
    try:
        # Query experiments table
        exp_response = experiments_table.query(
            KeyConditionExpression='experiment_id = :id',
            ExpressionAttributeValues={':id': experiment_id}
        )
        
        if not exp_response.get('Items'):
            return {
                'statusCode': 404,
                'headers': {
                    'Access-Control-Allow-Origin': '*',
                    'Content-Type': 'application/json'
                },
                'body': json.dumps({'error': 'Experiment not found'})
            }
        
        exp_item = exp_response['Items'][0]
        
        # Parse experiments JSON
        experiments_data = json.loads(exp_item.get('experiments', '[]'))
        procedural = next((e for e in experiments_data if e.get('experiment_type') == 'real_procedural_generation'), {})
        
        # Try to get reasoning data
        try:
            reasoning_response = reasoning_table.query(
                KeyConditionExpression='reasoning_id = :id',
                ExpressionAttributeValues={':id': experiment_id}
            )
            reasoning_item = reasoning_response['Items'][0] if reasoning_response.get('Items') else {}
        except:
            reasoning_item = {}
        
        # Parse JSON fields
        def safe_json_parse(value, default):
            if isinstance(value, str):
                try:
                    return json.loads(value)
                except:
                    return default
            return value if value else default
        
        insights = safe_json_parse(reasoning_item.get('insights'), [])
        next_experiment = safe_json_parse(reasoning_item.get('next_experiment'), {})
        risks = safe_json_parse(reasoning_item.get('risks'), [])
        generated_code = safe_json_parse(reasoning_item.get('generated_code'), {})
        
        # Build detailed response
        details = {
            'experiment_id': experiment_id,
            'status': exp_item.get('status', 'unknown'),
            'approach': procedural.get('approach', reasoning_item.get('hypothesis', 'Compression experiment')),
            'hypothesis': reasoning_item.get('hypothesis', ''),
            'root_cause': reasoning_item.get('root_cause', ''),
            'insights': insights,
            'next_experiment': next_experiment,
            'risks': risks,
            'expected_bitrate_mbps': float(reasoning_item.get('expected_bitrate_mbps', 0)) if reasoning_item.get('expected_bitrate_mbps') else 0,
            'expected_psnr_db': float(reasoning_item.get('expected_psnr_db', 0)) if reasoning_item.get('expected_psnr_db') else 0,
            'confidence_score': float(reasoning_item.get('confidence_score', 0)) if reasoning_item.get('confidence_score') else 0,
            'generated_code': generated_code,
            'real_metrics': procedural.get('real_metrics', {}),
            'comparison': procedural.get('comparison', {})
        }
        
        return {
            'statusCode': 200,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Content-Type': 'application/json'
            },
            'body': json.dumps(details, default=decimal_to_float)
        }
    except Exception as e:
        print(f"Error fetching experiment details: {e}")
        import traceback
        traceback.print_exc()
        return {
            'statusCode': 500,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Content-Type': 'application/json'
            },
            'body': json.dumps({'error': str(e)})
        }

