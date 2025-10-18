"""
V3.0 Public Dashboard Lambda

Server-side rendered dashboard showing AI Video Codec experiments
with beautiful blog posts and comparison videos
"""

import json
import boto3
import os
from datetime import datetime
from decimal import Decimal

# Configuration
DYNAMODB_TABLE = os.environ.get('DYNAMODB_TABLE', 'ai-codec-v3-experiments')
S3_BUCKET = os.environ.get('S3_BUCKET', 'ai-codec-v3-artifacts-580473065386')

# Initialize AWS clients
dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
s3 = boto3.client('s3', region_name='us-east-1')


def lambda_handler(event, context):
    """Handle dashboard requests"""
    
    path = event.get('rawPath', event.get('path', '/'))
    
    if path == '/' or path == '/dashboard':
        return render_dashboard()
    elif path.startswith('/blog/'):
        experiment_id = path.split('/')[-1]
        return render_blog_post(experiment_id)
    else:
        return {
            'statusCode': 404,
            'headers': {'Content-Type': 'text/html'},
            'body': '<h1>404 Not Found</h1>'
        }


def render_dashboard():
    """Render main dashboard page"""
    
    # Get all experiments
    table = dynamodb.Table(DYNAMODB_TABLE)
    response = table.scan()
    experiments = response.get('Items', [])
    
    # Sort by iteration
    experiments.sort(key=lambda x: int(x.get('iteration', 0)))
    
    # Generate HTML
    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Video Codec v3.0 - Research Dashboard</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        
        .header {{
            background: white;
            border-radius: 20px;
            padding: 40px;
            margin-bottom: 30px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }}
        
        .header h1 {{
            font-size: 3em;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }}
        
        .header .subtitle {{
            font-size: 1.2em;
            color: #666;
            margin-bottom: 30px;
        }}
        
        .reference-videos {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-top: 30px;
            padding-top: 30px;
            border-top: 2px solid #eee;
        }}
        
        .reference-card {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 12px;
            border: 2px solid #e0e0e0;
        }}
        
        .reference-card h3 {{
            color: #333;
            margin-bottom: 10px;
            font-size: 1.1em;
        }}
        
        .reference-card p {{
            color: #666;
            font-size: 0.9em;
            margin-bottom: 15px;
        }}
        
        .reference-card a {{
            display: inline-block;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 10px 20px;
            border-radius: 8px;
            text-decoration: none;
            font-weight: 600;
            transition: transform 0.2s;
        }}
        
        .reference-card a:hover {{
            transform: translateY(-2px);
        }}
        
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .stat-card {{
            background: white;
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            text-align: center;
        }}
        
        .stat-value {{
            font-size: 2.5em;
            font-weight: bold;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        
        .stat-label {{
            color: #666;
            font-size: 0.9em;
            margin-top: 5px;
        }}
        
        .experiments {{
            display: grid;
            gap: 20px;
        }}
        
        .experiment-card {{
            background: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            transition: transform 0.2s;
        }}
        
        .experiment-card:hover {{
            transform: translateY(-5px);
        }}
        
        .experiment-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding-bottom: 20px;
            border-bottom: 2px solid #eee;
        }}
        
        .iteration {{
            font-size: 1.5em;
            font-weight: bold;
            color: #333;
        }}
        
        .status {{
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: 600;
            font-size: 0.9em;
        }}
        
        .status.success {{
            background: #d4edda;
            color: #155724;
        }}
        
        .status.failed {{
            background: #f8d7da;
            color: #721c24;
        }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        
        .metric {{
            text-align: center;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 10px;
        }}
        
        .metric-value {{
            font-size: 1.8em;
            font-weight: bold;
            color: #667eea;
        }}
        
        .metric-label {{
            color: #666;
            font-size: 0.85em;
            margin-top: 5px;
        }}
        
        .reasoning {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            border-left: 4px solid #667eea;
            margin: 20px 0;
        }}
        
        .reasoning-title {{
            font-weight: bold;
            color: #333;
            margin-bottom: 10px;
        }}
        
        .reasoning-text {{
            color: #666;
            line-height: 1.6;
        }}
        
        .actions {{
            display: flex;
            gap: 15px;
            margin-top: 20px;
        }}
        
        .btn {{
            padding: 12px 24px;
            border-radius: 10px;
            text-decoration: none;
            font-weight: 600;
            transition: transform 0.2s;
            display: inline-flex;
            align-items: center;
            gap: 8px;
        }}
        
        .btn:hover {{
            transform: translateY(-2px);
        }}
        
        .btn-primary {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }}
        
        .btn-secondary {{
            background: #6c757d;
            color: white;
        }}
        
        .btn-disabled {{
            background: #e0e0e0;
            color: #999;
            cursor: not-allowed;
        }}
        
        .footer {{
            text-align: center;
            color: white;
            margin-top: 40px;
            padding: 20px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎬 AI Video Codec Research</h1>
            <p class="subtitle">LLM-Generated Video Compression Algorithms • v3.0</p>
            
            <div class="reference-videos">
                <div class="reference-card">
                    <h3>📹 Source Video</h3>
                    <p>Original uncompressed test video (640x480, 60 frames, 30fps)</p>
                    <a href="#" onclick="alert('Source video is generated fresh for each experiment'); return false;">
                        🎥 View Source Video
                    </a>
                </div>
                <div class="reference-card">
                    <h3>🎯 HEVC Baseline</h3>
                    <p>H.265/HEVC compressed reference for comparison</p>
                    <a href="#" onclick="alert('HEVC baseline: Professional codec for quality comparison'); return false;">
                        🎥 View HEVC Baseline
                    </a>
                </div>
            </div>
        </div>
        
        <div class="stats">
            {generate_stats(experiments)}
        </div>
        
        <div class="experiments">
            {generate_experiment_cards(experiments)}
        </div>
        
        <div class="footer">
            <p>🤖 Powered by Claude AI • Built with AWS Lambda, DynamoDB, S3</p>
            <p style="margin-top: 10px; opacity: 0.8;">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
        </div>
    </div>
</body>
</html>
"""
    
    return {
        'statusCode': 200,
        'headers': {
            'Content-Type': 'text/html',
            'Cache-Control': 'public, max-age=60'
        },
        'body': html
    }


def generate_stats(experiments):
    """Generate statistics cards"""
    total = len(experiments)
    successful = len([e for e in experiments if e.get('status') == 'success'])
    failed = total - successful
    
    avg_psnr = 0
    avg_ssim = 0
    count = 0
    
    for exp in experiments:
        if exp.get('status') == 'success':
            metrics = exp.get('metrics', {})
            psnr = float(metrics.get('psnr_db', 0))
            ssim = float(metrics.get('ssim', 0))
            if psnr > 0:
                avg_psnr += psnr
                avg_ssim += ssim
                count += 1
    
    if count > 0:
        avg_psnr /= count
        avg_ssim /= count
    
    return f"""
        <div class="stat-card">
            <div class="stat-value">{total}</div>
            <div class="stat-label">Total Experiments</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{successful}</div>
            <div class="stat-label">Successful</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{avg_psnr:.1f}</div>
            <div class="stat-label">Avg PSNR (dB)</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{avg_ssim:.3f}</div>
            <div class="stat-label">Avg SSIM</div>
        </div>
    """


def generate_experiment_cards(experiments):
    """Generate experiment cards HTML"""
    cards = []
    
    for exp in experiments:
        iteration = exp.get('iteration', 0)
        status = exp.get('status', 'unknown')
        experiment_id = exp.get('experiment_id', '')
        
        metrics = exp.get('metrics', {})
        psnr = float(metrics.get('psnr_db', 0))
        ssim = float(metrics.get('ssim', 0))
        bitrate = float(metrics.get('bitrate_mbps', 0))
        compression = float(metrics.get('compression_ratio', 0))
        
        artifacts = exp.get('artifacts', {})
        video_url = artifacts.get('video_url') if isinstance(artifacts, dict) else None
        decoder_key = artifacts.get('decoder_s3_key') if isinstance(artifacts, dict) else None
        
        reasoning = exp.get('llm_reasoning', 'No reasoning provided')
        
        # Truncate reasoning for card view
        reasoning_preview = reasoning[:200] + '...' if len(reasoning) > 200 else reasoning
        
        status_class = 'success' if status == 'success' else 'failed'
        
        card = f"""
        <div class="experiment-card">
            <div class="experiment-header">
                <div class="iteration">Iteration {iteration}</div>
                <div class="status {status_class}">{status.upper()}</div>
            </div>
            
            {f'''
            <div class="metrics-grid">
                <div class="metric">
                    <div class="metric-value">{psnr:.1f}</div>
                    <div class="metric-label">PSNR (dB)</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{ssim:.3f}</div>
                    <div class="metric-label">SSIM</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{bitrate:.2f}</div>
                    <div class="metric-label">Bitrate (Mbps)</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{compression:.2f}x</div>
                    <div class="metric-label">Compression</div>
                </div>
            </div>
            ''' if status == 'success' else '<p style="color: #721c24; padding: 20px; text-align: center;">❌ Experiment failed to complete</p>'}
            
            <div class="reasoning">
                <div class="reasoning-title">🤖 LLM Reasoning:</div>
                <div class="reasoning-text">{reasoning_preview}</div>
            </div>
            
            <div class="actions">
                {'<a href="/blog/' + experiment_id + '" class="btn btn-primary">📝 Read Full Blog Post</a>' if status == 'success' else '<span class="btn btn-disabled">📝 No Blog Post</span>'}
                {'<a href="' + video_url + '" class="btn btn-secondary" target="_blank">🎥 Watch Video</a>' if video_url else '<span class="btn btn-disabled">🎥 No Video</span>'}
                {'<a href="https://' + S3_BUCKET + '.s3.amazonaws.com/' + decoder_key + '" class="btn btn-secondary" target="_blank">💾 Download Decoder</a>' if decoder_key else '<span class="btn btn-disabled">💾 No Decoder</span>'}
            </div>
        </div>
        """
        cards.append(card)
    
    return '\n'.join(cards)


def render_blog_post(experiment_id):
    """Render detailed blog post for an experiment"""
    
    # Get experiment from DynamoDB
    table = dynamodb.Table(DYNAMODB_TABLE)
    response = table.query(
        KeyConditionExpression='experiment_id = :id',
        ExpressionAttributeValues={':id': experiment_id}
    )
    
    items = response.get('Items', [])
    if not items:
        return {
            'statusCode': 404,
            'headers': {'Content-Type': 'text/html'},
            'body': '<h1>Experiment not found</h1>'
        }
    
    exp = items[0]
    
    # Generate blog post HTML
    html = generate_blog_html(exp)
    
    return {
        'statusCode': 200,
        'headers': {
            'Content-Type': 'text/html',
            'Cache-Control': 'public, max-age=300'
        },
        'body': html
    }


def generate_blog_html(exp):
    """Generate beautiful blog post HTML"""
    
    iteration = exp.get('iteration', 0)
    status = exp.get('status', 'unknown')
    experiment_id = exp.get('experiment_id', '')
    timestamp = exp.get('timestamp', 0)
    
    metrics = exp.get('metrics', {})
    psnr = float(metrics.get('psnr_db', 0))
    ssim = float(metrics.get('ssim', 0))
    bitrate = float(metrics.get('bitrate_mbps', 0))
    compression = float(metrics.get('compression_ratio', 0))
    original_size = int(metrics.get('original_size_bytes', 0))
    compressed_size = int(metrics.get('compressed_size_bytes', 0))
    
    artifacts = exp.get('artifacts', {})
    video_url = artifacts.get('video_url') if isinstance(artifacts, dict) else None
    decoder_key = artifacts.get('decoder_s3_key') if isinstance(artifacts, dict) else None
    
    reasoning = exp.get('llm_reasoning', 'No reasoning provided')
    
    # Format timestamp
    dt = datetime.fromtimestamp(int(timestamp))
    date_str = dt.strftime('%B %d, %Y at %H:%M UTC')
    
    # Determine quality assessment
    if psnr >= 30:
        quality = "Excellent"
        quality_color = "#28a745"
    elif psnr >= 25:
        quality = "Good"
        quality_color = "#17a2b8"
    elif psnr >= 20:
        quality = "Fair"
        quality_color = "#ffc107"
    else:
        quality = "Poor"
        quality_color = "#dc3545"
    
    # Generate insights
    insights = generate_insights(psnr, ssim, compression, bitrate)
    
    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Iteration {iteration} - AI Video Codec Research</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: Georgia, 'Times New Roman', serif;
            background: #f5f5f5;
            line-height: 1.8;
        }}
        
        .hero {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 80px 20px;
            text-align: center;
        }}
        
        .hero h1 {{
            font-size: 3em;
            margin-bottom: 20px;
        }}
        
        .hero .meta {{
            font-size: 1.1em;
            opacity: 0.9;
        }}
        
        .container {{
            max-width: 900px;
            margin: -50px auto 0;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.2);
            padding: 60px;
        }}
        
        .back-link {{
            display: inline-block;
            color: #667eea;
            text-decoration: none;
            margin-bottom: 30px;
            font-weight: 600;
        }}
        
        .back-link:hover {{
            text-decoration: underline;
        }}
        
        .quality-badge {{
            display: inline-block;
            padding: 10px 20px;
            border-radius: 20px;
            font-weight: bold;
            margin: 20px 0;
            background: {quality_color};
            color: white;
        }}
        
        .section {{
            margin: 40px 0;
        }}
        
        .section h2 {{
            color: #333;
            font-size: 2em;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 3px solid #667eea;
        }}
        
        .metrics-showcase {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 20px;
            margin: 30px 0;
        }}
        
        .metric-box {{
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            padding: 30px;
            border-radius: 15px;
            text-align: center;
            border: 2px solid #dee2e6;
        }}
        
        .metric-box .value {{
            font-size: 3em;
            font-weight: bold;
            color: #667eea;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }}
        
        .metric-box .label {{
            color: #666;
            margin-top: 10px;
            font-size: 1.1em;
        }}
        
        .reasoning-box {{
            background: #f8f9fa;
            padding: 30px;
            border-radius: 15px;
            border-left: 5px solid #667eea;
            font-style: italic;
            line-height: 1.8;
        }}
        
        .insights {{
            background: #fff3cd;
            border-left: 5px solid #ffc107;
            padding: 25px;
            border-radius: 10px;
            margin: 30px 0;
        }}
        
        .insights h3 {{
            color: #856404;
            margin-bottom: 15px;
        }}
        
        .insights ul {{
            list-style: none;
            padding-left: 0;
        }}
        
        .insights li {{
            padding: 8px 0;
            color: #856404;
        }}
        
        .insights li:before {{
            content: "💡 ";
            margin-right: 8px;
        }}
        
        .video-section {{
            margin: 40px 0;
            text-align: center;
        }}
        
        .video-section a {{
            display: inline-block;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px 40px;
            border-radius: 10px;
            text-decoration: none;
            font-weight: 600;
            font-size: 1.2em;
            margin: 10px;
            transition: transform 0.2s;
        }}
        
        .video-section a:hover {{
            transform: translateY(-3px);
        }}
        
        .technical-details {{
            background: #f8f9fa;
            padding: 25px;
            border-radius: 10px;
            margin: 30px 0;
        }}
        
        .technical-details table {{
            width: 100%;
            border-collapse: collapse;
        }}
        
        .technical-details td {{
            padding: 12px;
            border-bottom: 1px solid #dee2e6;
        }}
        
        .technical-details td:first-child {{
            font-weight: bold;
            color: #667eea;
            width: 40%;
        }}
        
        .footer-nav {{
            margin-top: 60px;
            padding-top: 30px;
            border-top: 2px solid #eee;
            text-align: center;
        }}
    </style>
</head>
<body>
    <div class="hero">
        <h1>Iteration {iteration}: {quality} Quality Achieved</h1>
        <p class="meta">Experiment conducted on {date_str}</p>
    </div>
    
    <div class="container">
        <a href="/dashboard" class="back-link">← Back to Dashboard</a>
        
        <div class="quality-badge">Quality Assessment: {quality}</div>
        
        <div class="section">
            <h2>📊 Performance Metrics</h2>
            <div class="metrics-showcase">
                <div class="metric-box">
                    <div class="value">{psnr:.2f}</div>
                    <div class="label">PSNR (dB)</div>
                </div>
                <div class="metric-box">
                    <div class="value">{ssim:.3f}</div>
                    <div class="label">SSIM Score</div>
                </div>
                <div class="metric-box">
                    <div class="value">{bitrate:.2f}</div>
                    <div class="label">Bitrate (Mbps)</div>
                </div>
                <div class="metric-box">
                    <div class="value">{compression:.2f}x</div>
                    <div class="label">Compression Ratio</div>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>🤖 AI's Approach</h2>
            <div class="reasoning-box">
                "{reasoning}"
            </div>
        </div>
        
        <div class="insights">
            <h3>💡 Key Insights</h3>
            <ul>
                {insights}
            </ul>
        </div>
        
        <div class="section">
            <h2>🔬 Technical Details</h2>
            <div class="technical-details">
                <table>
                    <tr>
                        <td>Experiment ID</td>
                        <td><code>{experiment_id}</code></td>
                    </tr>
                    <tr>
                        <td>Original Size</td>
                        <td>{original_size:,} bytes ({original_size/1024/1024:.2f} MB)</td>
                    </tr>
                    <tr>
                        <td>Compressed Size</td>
                        <td>{compressed_size:,} bytes ({compressed_size/1024/1024:.2f} MB)</td>
                    </tr>
                    <tr>
                        <td>Bitrate</td>
                        <td>{bitrate:.2f} Mbps</td>
                    </tr>
                    <tr>
                        <td>Status</td>
                        <td>✅ {status.upper()}</td>
                    </tr>
                </table>
            </div>
        </div>
        
        <div class="video-section">
            <h2>🎬 View Results</h2>
            {f'<a href="{video_url}" target="_blank">🎥 Watch Reconstructed Video</a>' if video_url else ''}
            {f'<a href="https://{S3_BUCKET}.s3.amazonaws.com/{decoder_key}" target="_blank">💾 Download Decoder Code</a>' if decoder_key else ''}
        </div>
        
        <div class="footer-nav">
            <a href="/dashboard" class="back-link">← Return to All Experiments</a>
        </div>
    </div>
</body>
</html>
"""


def generate_insights(psnr, ssim, compression, bitrate):
    """Generate insights based on metrics"""
    insights = []
    
    if psnr >= 30:
        insights.append("<li>Excellent PSNR indicates minimal quality loss</li>")
    elif psnr >= 25:
        insights.append("<li>Good PSNR shows acceptable quality for most applications</li>")
    elif psnr < 20:
        insights.append("<li>Low PSNR suggests significant quality degradation</li>")
    
    if ssim >= 0.9:
        insights.append("<li>Outstanding structural similarity to original</li>")
    elif ssim >= 0.8:
        insights.append("<li>Good preservation of visual structure</li>")
    elif ssim < 0.7:
        insights.append("<li>Structural differences may be noticeable</li>")
    
    if compression > 1.0:
        insights.append("<li>Successfully compressed video data</li>")
    elif compression < 1.0:
        insights.append("<li>Output file is larger than input - compression algorithm needs improvement</li>")
    
    if bitrate < 5.0:
        insights.append("<li>Low bitrate suitable for bandwidth-constrained applications</li>")
    elif bitrate > 10.0:
        insights.append("<li>High bitrate ensures quality but requires more bandwidth</li>")
    
    return '\n'.join(insights)

