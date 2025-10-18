"""
AiV1 Video Codec Research v3.0 - Dashboard Lambda

Complete dashboard with dark theme, working downloads, and full blog posts.
"""

import json
import boto3
import os
from datetime import datetime
from decimal import Decimal
from urllib.parse import quote

# Configuration
DYNAMODB_TABLE = os.environ.get('DYNAMODB_TABLE', 'ai-codec-v3-experiments')
S3_BUCKET = os.environ.get('S3_BUCKET', 'ai-codec-v3-artifacts-580473065386')
GITHUB_REPO = "https://github.com/yarontorbaty/ai-video-codec-framework"

# Reference video S3 keys
SOURCE_VIDEO_KEY = "reference/source.mp4"
HEVC_VIDEO_KEY = "reference/hevc_baseline.mp4"

# Initialize AWS clients
dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
s3 = boto3.client('s3', region_name='us-east-1')


def lambda_handler(event, context):
    """Handle dashboard requests"""
    path = event.get('rawPath', event.get('path', '/'))
    
    if path == '/' or path == '/dashboard':
        return render_dashboard()
    elif path == '/api/experiments':
        # API endpoint for real-time updates
        return get_experiments_api()
    elif path.startswith('/blog/'):
        experiment_id = path.split('/')[-1]
        return render_blog_post(experiment_id)
    else:
        return {
            'statusCode': 404,
            'headers': {'Content-Type': 'text/html'},
            'body': '<h1>404 Not Found</h1>'
        }


def get_experiments_api():
    """API endpoint for real-time experiment updates"""
    try:
        # Get all experiments
        table = dynamodb.Table(DYNAMODB_TABLE)
        response = table.scan()
        experiments = response.get('Items', [])
        
        # Sort by iteration
        experiments.sort(key=lambda x: int(x.get('iteration', 0)), reverse=True)
        
        # Separate by status
        successful = [e for e in experiments if e.get('status') == 'success']
        in_progress = [e for e in experiments if e.get('status') == 'in_progress']
        failed = [e for e in experiments if e.get('status') == 'failed']
        
        # Convert Decimal to float for JSON serialization
        def convert_decimals(obj):
            if isinstance(obj, Decimal):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_decimals(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_decimals(item) for item in obj]
            return obj
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Cache-Control': 'no-cache, no-store, must-revalidate'
            },
            'body': json.dumps({
                'successful': convert_decimals(successful),
                'in_progress': convert_decimals(in_progress),
                'failed': convert_decimals(failed),
                'total': len(experiments)
            })
        }
    except Exception as e:
        print(f"Error in API: {e}")
        return {
            'statusCode': 500,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({'error': str(e)})
        }


def generate_presigned_url(s3_key, expiration=604800, download=False, filename=None):
    """
    Generate presigned URL with max 7-day expiration
    
    Args:
        s3_key: S3 object key
        expiration: Seconds (max 604800 = 7 days)
        download: If True, add Content-Disposition header for download
        filename: Optional filename for download
    """
    try:
        params = {
            'Bucket': S3_BUCKET,
            'Key': s3_key
        }
        
        # Add response headers for downloads
        if download and filename:
            params['ResponseContentDisposition'] = f'attachment; filename="{filename}"'
        
        url = s3.generate_presigned_url(
            'get_object',
            Params=params,
            ExpiresIn=min(expiration, 604800)  # Max 7 days
        )
        return url
    except Exception as e:
        print(f"Error generating presigned URL for {s3_key}: {e}")
        return None


def get_quality_label(metric_type, value):
    """Get quality label for a metric"""
    if metric_type == 'psnr':
        if value >= 38:
            return ('Excellent', '#4ade80')
        elif value >= 32:
            return ('Good', '#60a5fa')
        elif value >= 25:
            return ('Acceptable', '#fbbf24')
        else:
            return ('Poor', '#f87171')
    elif metric_type == 'ssim':
        if value >= 0.95:
            return ('Excellent', '#4ade80')
        elif value >= 0.85:
            return ('Good', '#60a5fa')
        elif value >= 0.75:
            return ('Acceptable', '#fbbf24')
        else:
            return ('Poor', '#f87171')
    elif metric_type == 'bitrate':
        if value <= 3.0:
            return ('Excellent', '#4ade80')
        elif value <= 6.0:
            return ('Good', '#60a5fa')
        elif value <= 10.0:
            return ('Acceptable', '#fbbf24')
        else:
            return ('Poor', '#f87171')
    return ('Unknown', '#6b7280')


def get_tier(psnr, ssim, bitrate):
    """Determine achievement tier"""
    psnr_score = 0
    ssim_score = 0
    bitrate_score = 0
    
    if psnr >= 38:
        psnr_score = 3
    elif psnr >= 32:
        psnr_score = 2
    elif psnr >= 26:
        psnr_score = 1
    
    if ssim >= 0.95:
        ssim_score = 3
    elif ssim >= 0.80:
        ssim_score = 2
    elif ssim >= 0.65:
        ssim_score = 1
    
    if bitrate <= 1.0:
        bitrate_score = 3
    elif bitrate <= 3.0:
        bitrate_score = 2
    elif bitrate <= 5.0:
        bitrate_score = 1
    
    total_score = psnr_score + ssim_score + bitrate_score
    
    if total_score >= 7:
        return ('Gold', '#FFD700')
    elif total_score >= 4:
        return ('Silver', '#C0C0C0')
    elif total_score >= 2:
        return ('Bronze', '#CD7F32')
    else:
        return ('', '')


def generate_llm_summary(experiments):
    """Generate AI-powered project summary"""
    successful = [e for e in experiments if e.get('status') == 'success']
    failed = [e for e in experiments if e.get('status') == 'failed']
    
    if not successful:
        return "No successful experiments yet. The system is learning and adapting with each iteration."
    
    avg_psnr = sum(float(e.get('metrics', {}).get('psnr_db', 0)) for e in successful) / len(successful)
    avg_ssim = sum(float(e.get('metrics', {}).get('ssim', 0)) for e in successful) / len(successful)
    avg_compression = sum(float(e.get('metrics', {}).get('compression_ratio', 0)) for e in successful) / len(successful)
    
    best_exp = max(successful, key=lambda x: float(x.get('metrics', {}).get('psnr_db', 0)))
    best_psnr = float(best_exp.get('metrics', {}).get('psnr_db', 0))
    best_iteration = best_exp.get('iteration', 0)
    
    success_rate = len(successful)/len(experiments)*100 if experiments else 0
    
    return f"""
    <strong>Research Progress Update:</strong><br><br>
    
    After {len(experiments)} iterations, achieved {success_rate:.0f}% success rate ({len(successful)}/{len(experiments)} experiments). 
    LLM-powered codec evolution produces avg PSNR of {avg_psnr:.1f}dB and SSIM of {avg_ssim:.3f}.<br><br>
    
    <strong>Best Performance:</strong> Iteration {best_iteration} achieved {best_psnr:.2f}dB PSNR. 
    Current compression ratio of {avg_compression:.2f}x shows room for optimization.<br><br>
    
    <strong>Learning:</strong> {len(failed)} failed experiments provide training data. 
    Each failure refines code generation strategy toward production-grade ratios.<br><br>
    
    <strong>Next:</strong> Focus on >10x compression while maintaining quality. 
    SSIM {avg_ssim:.3f} suggests solid foundation.
    """.strip()


def render_dashboard():
    """Render main dashboard page with dark theme and real-time updates"""
    
    # Get all experiments
    table = dynamodb.Table(DYNAMODB_TABLE)
    response = table.scan()
    experiments = response.get('Items', [])
    
    # Sort by iteration
    experiments.sort(key=lambda x: int(x.get('iteration', 0)), reverse=True)
    
    # Separate by status
    successful = [e for e in experiments if e.get('status') == 'success']
    in_progress = [e for e in experiments if e.get('status') == 'in_progress']
    failed = [e for e in experiments if e.get('status') == 'failed']
    
    # Generate LLM summary
    llm_summary = generate_llm_summary(experiments)
    
    # Find best results
    best_results = []
    for exp in successful:
        metrics = exp.get('metrics', {})
        psnr = float(metrics.get('psnr_db', 0))
        ssim = float(metrics.get('ssim', 0))
        bitrate = float(metrics.get('bitrate_mbps', 0))
        
        if psnr > 0:
            tier, color = get_tier(psnr, ssim, bitrate)
            if tier:
                best_results.append({
                    'exp': exp,
                    'tier': tier,
                    'color': color,
                    'psnr': psnr,
                    'ssim': ssim,
                    'bitrate': bitrate
                })
    
    tier_order = {'Gold': 0, 'Silver': 1, 'Bronze': 2}
    best_results.sort(key=lambda x: (tier_order.get(x['tier'], 999), -x['psnr']))
    best_results = best_results[:3]
    
    # Generate reference video URLs (7-day expiration)
    source_url = generate_presigned_url(SOURCE_VIDEO_KEY, download=False)
    hevc_url = generate_presigned_url(HEVC_VIDEO_KEY, download=False)
    
    # Generate HTML
    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AiV1 Video Codec Research v3.0</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0f172a;
            color: #e2e8f0;
            height: 100vh;
            overflow: hidden;
        }}
        
        /* Header */
        .header {{
            background: linear-gradient(135deg, #1e40af 0%, #7c3aed 100%);
            color: white;
            padding: 16px 24px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
        }}
        
        .header h1 {{
            font-size: 1.5em;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 12px;
        }}
        
        .header-links {{
            display: flex;
            gap: 16px;
        }}
        
        .header-links a {{
            color: white;
            text-decoration: none;
            padding: 8px 16px;
            border-radius: 6px;
            background: rgba(255,255,255,0.2);
            transition: background 0.2s;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        
        .header-links a:hover {{
            background: rgba(255,255,255,0.3);
        }}
        
        /* Layout */
        .container {{
            display: flex;
            height: calc(100vh - 120px);
        }}
        
        /* Sidebar */
        .sidebar {{
            width: 240px;
            background: #1e293b;
            border-right: 1px solid #334155;
            padding: 24px 0;
        }}
        
        .nav-item {{
            padding: 14px 24px;
            cursor: pointer;
            transition: all 0.2s;
            border-left: 3px solid transparent;
            display: flex;
            align-items: center;
            gap: 12px;
            color: #94a3b8;
        }}
        
        .nav-item:hover {{
            background: #334155;
            color: #e2e8f0;
        }}
        
        .nav-item.active {{
            background: #334155;
            border-left-color: #3b82f6;
            color: #3b82f6;
            font-weight: 600;
        }}
        
        /* Main content */
        .main-content {{
            flex: 1;
            overflow-y: auto;
            padding: 24px;
        }}
        
        /* Reference videos */
        .reference-section {{
            background: #1e293b;
            padding: 20px;
            border-radius: 12px;
            margin-bottom: 24px;
            display: flex;
            gap: 20px;
            border: 1px solid #334155;
        }}
        
        .ref-video {{
            flex: 1;
            text-align: center;
        }}
        
        .ref-video h3 {{
            font-size: 1em;
            color: #94a3b8;
            margin-bottom: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
        }}
        
        .ref-video a {{
            display: inline-block;
            padding: 10px 20px;
            background: #3b82f6;
            color: white;
            text-decoration: none;
            border-radius: 6px;
            font-size: 0.9em;
            transition: background 0.2s;
        }}
        
        .ref-video a:hover {{
            background: #2563eb;
        }}
        
        /* LLM Summary */
        .llm-summary {{
            background: #422006;
            border-left: 4px solid #f59e0b;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 24px;
            font-size: 0.9em;
            line-height: 1.7;
        }}
        
        /* HEVC Baseline Card */
        .hevc-baseline-card {{
            background: linear-gradient(135deg, #164e63 0%, #155e75 100%);
            border: 2px solid #06b6d4;
            padding: 24px;
            border-radius: 12px;
            margin-bottom: 24px;
            box-shadow: 0 4px 12px rgba(6, 182, 212, 0.2);
        }}
        
        .hevc-baseline-card h3 {{
            color: #22d3ee;
            font-size: 1.3em;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        
        .baseline-metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }}
        
        .baseline-metric {{
            background: rgba(255, 255, 255, 0.05);
            padding: 16px;
            border-radius: 8px;
            border: 1px solid rgba(34, 211, 238, 0.3);
            text-align: center;
        }}
        
        .baseline-label {{
            font-size: 0.85em;
            color: #94a3b8;
            margin-bottom: 8px;
            font-weight: 500;
        }}
        
        .baseline-value {{
            font-size: 1.8em;
            font-weight: bold;
            color: #22d3ee;
            margin-bottom: 8px;
        }}
        
        .baseline-target {{
            font-size: 0.8em;
            color: #e0f2fe;
            font-weight: 600;
            padding: 4px 8px;
            background: rgba(34, 211, 238, 0.1);
            border-radius: 4px;
            display: inline-block;
        }}
        
        .baseline-note {{
            background: rgba(255, 255, 255, 0.05);
            padding: 12px 16px;
            border-radius: 8px;
            font-size: 0.9em;
            line-height: 1.6;
            color: #cbd5e1;
            border-left: 3px solid #22d3ee;
        }}
        
        .baseline-note i {{
            color: #22d3ee;
            margin-right: 8px;
        }}
        
        /* Best Results */
        .best-results {{
            background: #1e293b;
            padding: 20px;
            border-radius: 12px;
            margin-bottom: 24px;
            border: 1px solid #334155;
        }}
        
        .best-results h2 {{
            font-size: 1.2em;
            margin-bottom: 20px;
            color: #e2e8f0;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        
        .tier-cards {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 20px;
        }}
        
        .tier-card {{
            padding: 20px;
            border-radius: 12px;
            border: 2px solid;
            background: #0f172a;
        }}
        
        .tier-badge {{
            font-size: 1.3em;
            font-weight: bold;
            margin-bottom: 12px;
        }}
        
        .tier-metrics {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 12px;
            margin-top: 16px;
        }}
        
        .tier-metric {{
            text-align: center;
        }}
        
        .tier-metric-value {{
            font-size: 1.3em;
            font-weight: bold;
            color: #e2e8f0;
        }}
        
        .tier-metric-label {{
            font-size: 0.8em;
            color: #94a3b8;
            margin-top: 4px;
        }}
        
        /* Tabs */
        .tabs {{
            display: flex;
            gap: 12px;
            margin-bottom: 24px;
            border-bottom: 2px solid #334155;
        }}
        
        .tab {{
            padding: 12px 24px;
            cursor: pointer;
            border-bottom: 3px solid transparent;
            transition: all 0.2s;
            font-weight: 500;
            color: #94a3b8;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        
        .tab:hover {{
            background: #1e293b;
            color: #e2e8f0;
        }}
        
        .tab.active {{
            color: #3b82f6;
            border-bottom-color: #3b82f6;
        }}
        
        .tab-content {{
            display: none;
        }}
        
        .tab-content.active {{
            display: block;
        }}
        
        /* Table */
        .table-container {{
            background: #1e293b;
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid #334155;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        
        th {{
            background: #0f172a;
            padding: 16px;
            text-align: left;
            font-weight: 600;
            font-size: 0.85em;
            color: #94a3b8;
            border-bottom: 2px solid #334155;
        }}
        
        td {{
            padding: 16px;
            border-bottom: 1px solid #334155;
            font-size: 0.9em;
        }}
        
        tr:hover {{
            background: #0f172a;
        }}
        
        .quality-badge {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 14px;
            font-size: 0.75em;
            font-weight: 600;
            color: #0f172a;
            margin-left: 8px;
        }}
        
        .blog-link {{
            color: #3b82f6;
            text-decoration: none;
            font-weight: 500;
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }}
        
        .blog-link:hover {{
            text-decoration: underline;
        }}
        
        /* Pagination */
        .pagination {{
            display: flex;
            justify-content: center;
            gap: 8px;
            padding: 20px;
            background: #0f172a;
            border-top: 1px solid #334155;
        }}
        
        .page-btn {{
            padding: 8px 14px;
            border: 1px solid #334155;
            background: #1e293b;
            cursor: pointer;
            border-radius: 6px;
            color: #e2e8f0;
            transition: all 0.2s;
        }}
        
        .page-btn:hover {{
            background: #334155;
        }}
        
        .page-btn.active {{
            background: #3b82f6;
            color: white;
            border-color: #3b82f6;
        }}
        
        /* Footer */
        .footer {{
            background: #1e293b;
            padding: 16px 24px;
            text-align: center;
            border-top: 1px solid #334155;
            font-size: 0.85em;
            color: #94a3b8;
        }}
        
        .footer a {{
            color: #3b82f6;
            text-decoration: none;
        }}
        
        .footer a:hover {{
            text-decoration: underline;
        }}
        
        /* Failed experiments */
        .error-log {{
            background: #450a0a;
            border-left: 4px solid #ef4444;
            padding: 12px;
            border-radius: 6px;
            font-family: 'Courier New', monospace;
            font-size: 0.8em;
            margin-top: 8px;
            max-height: 120px;
            overflow-y: auto;
            color: #fca5a5;
        }}
        
        .llm-reasoning {{
            background: #164e63;
            border-left: 4px solid #06b6d4;
            padding: 12px;
            border-radius: 6px;
            font-size: 0.85em;
            margin-top: 8px;
            font-style: italic;
            color: #a5f3fc;
        }}
        
        /* Scrollbar */
        ::-webkit-scrollbar {{
            width: 10px;
        }}
        
        ::-webkit-scrollbar-track {{
            background: #0f172a;
        }}
        
        ::-webkit-scrollbar-thumb {{
            background: #334155;
            border-radius: 5px;
        }}
        
        ::-webkit-scrollbar-thumb:hover {{
            background: #475569;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1><i class="fas fa-video"></i> AiV1 Video Codec Research v3.0</h1>
        <div class="header-links">
            <a href="{GITHUB_REPO}" target="_blank"><i class="fab fa-github"></i> GitHub</a>
            <a href="{GITHUB_REPO}/tree/v3.0" target="_blank"><i class="fas fa-code-branch"></i> v3.0</a>
        </div>
    </div>
    
    <div class="container">
        <div class="sidebar">
            <div class="nav-item active" data-section="overview"><i class="fas fa-chart-line"></i> Overview</div>
            <div class="nav-item" data-section="best"><i class="fas fa-trophy"></i> Best Results</div>
            <div class="nav-item" data-section="experiments"><i class="fas fa-flask"></i> Experiments</div>
            <div class="nav-item" data-section="references"><i class="fas fa-film"></i> References</div>
        </div>
        
        <div class="main-content">
            <section id="overview">
                <div class="llm-summary">
                    <i class="fas fa-robot"></i> <strong>AI Analysis:</strong><br>
                    {llm_summary}
                </div>
                
                <!-- HEVC Baseline Threshold -->
                <div class="hevc-baseline-card">
                    <h3><i class="fas fa-bullseye"></i> HEVC Baseline - Our Threshold to Beat</h3>
                    <div class="baseline-metrics">
                        <div class="baseline-metric">
                            <div class="baseline-label">PSNR (vs SOURCE)</div>
                            <div class="baseline-value">27.82 dB</div>
                            <div class="baseline-target">Target: ≥ 27.82 dB</div>
                        </div>
                        <div class="baseline-metric">
                            <div class="baseline-label">SSIM (vs SOURCE)</div>
                            <div class="baseline-value">0.6826</div>
                            <div class="baseline-target">Target: ≥ 0.6826</div>
                        </div>
                        <div class="baseline-metric">
                            <div class="baseline-label">Bitrate</div>
                            <div class="baseline-value">10.18 Mbps</div>
                            <div class="baseline-target">Target: &lt; 10.18 Mbps</div>
                        </div>
                        <div class="baseline-metric">
                            <div class="baseline-label">File Size</div>
                            <div class="baseline-value">12.14 MB</div>
                            <div class="baseline-target">Target: &lt; 12.14 MB</div>
                        </div>
                    </div>
                    <div class="baseline-note">
                        <i class="fas fa-info-circle"></i> 
                        <strong>Goal:</strong> Match or beat HEVC quality (PSNR/SSIM) at lower bitrate/size.
                        Measured on actual 10Mbps HEVC encoding vs uncompressed source.
                    </div>
                </div>
            </section>
            
            <section id="references" class="reference-section">
                <div class="ref-video">
                    <h3><i class="fas fa-file-video"></i> Source Video (HD Raw)</h3>
                    <a href="{source_url if source_url else '#'}" target="_blank"><i class="fas fa-play"></i> Watch Source</a>
                </div>
                <div class="ref-video">
                    <h3><i class="fas fa-compress"></i> HEVC Baseline (10Mbps)</h3>
                    <a href="{hevc_url if hevc_url else '#'}" target="_blank"><i class="fas fa-play"></i> Watch HEVC</a>
                </div>
            </section>
            
            <section id="best">
                {generate_best_results_html(best_results)}
            </section>
            
            <section id="experiments">
                <div class="tabs">
                    <div class="tab active" data-tab="successful">
                        <i class="fas fa-check-circle"></i> Successful (<span id="successful-count">{len(successful)}</span>)
                    </div>
                    <div class="tab" data-tab="in-progress">
                        <i class="fas fa-spinner fa-spin"></i> In Progress (<span id="in-progress-count">{len(in_progress)}</span>)
                    </div>
                    <div class="tab" data-tab="failed">
                        <i class="fas fa-times-circle"></i> Failed (<span id="failed-count">{len(failed)}</span>)
                    </div>
                </div>
                
                <div id="successful" class="tab-content active">
                    {generate_successful_table(successful)}
                </div>
                
                <div id="in-progress" class="tab-content">
                    {generate_in_progress_table(in_progress)}
                </div>
                
                <div id="failed" class="tab-content">
                    {generate_failed_table(failed)}
                </div>
            </section>
        </div>
    </div>
    
    <div class="footer">
        Created by <a href="https://www.linkedin.com/in/yaron-torbaty/" target="_blank"><i class="fab fa-linkedin"></i> Yaron Torbaty</a> | 
        Powered by Claude AI & AWS | 
        <a href="{GITHUB_REPO}" target="_blank"><i class="fab fa-github"></i> GitHub</a>
    </div>
    
    <script>
        // Real-time updates
        let lastRefreshTime = Date.now();
        
        async function refreshExperiments() {{
            try {{
                const response = await fetch('/api/experiments');
                const data = await response.json();
                
                // Update counts
                document.getElementById('successful-count').textContent = data.successful.length;
                document.getElementById('in-progress-count').textContent = data.in_progress.length;
                document.getElementById('failed-count').textContent = data.failed.length;
                
                // Check if we need to reload entire page (new successful or failed experiments)
                const currentSuccessful = document.querySelectorAll('#successful tbody tr').length;
                const currentFailed = document.querySelectorAll('#failed tbody tr').length;
                
                if (data.successful.length !== currentSuccessful || data.failed.length !== currentFailed) {{
                    // Reload page to update all sections (LLM summary, best results, failed list, etc.)
                    location.reload();
                    return;
                }}
                
                // Update in-progress table
                updateInProgressTable(data.in_progress);
                
                lastRefreshTime = Date.now();
            }} catch (err) {{
                console.error('Failed to refresh experiments:', err);
            }}
        }}
        
        function updateInProgressTable(experiments) {{
            const tableBody = document.querySelector('#in-progress tbody');
            if (!tableBody) return;
            
            if (experiments.length === 0) {{
                tableBody.innerHTML = '<tr><td colspan="5" style="text-align: center; padding: 20px; color: #94a3b8;">No experiments in progress</td></tr>';
                return;
            }}
            
            tableBody.innerHTML = experiments.map(exp => {{
                const iteration = exp.iteration || 'N/A';
                const experimentId = exp.experiment_id || '';
                const startedAt = exp.started_at ? new Date(exp.started_at).toLocaleString() : 'Just now';
                const phase = exp.phase || 'Starting...';
                
                return `
                    <tr>
                        <td>${{iteration}}</td>
                        <td>${{experimentId}}</td>
                        <td><span class="badge" style="background: #fbbf24; color: #0f172a;"><i class="fas fa-spinner fa-spin"></i> ${{phase}}</span></td>
                        <td>${{startedAt}}</td>
                        <td><i class="fas fa-circle-notch fa-spin" style="color: #fbbf24;"></i> Running</td>
                    </tr>
                `;
            }}).join('');
        }}
        
        // Auto-refresh every 5 seconds
        setInterval(refreshExperiments, 5000);
        
        // Tab switching
        document.querySelectorAll('.tab').forEach(tab => {{
            tab.addEventListener('click', () => {{
                const tabName = tab.getAttribute('data-tab');
                document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
                document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
                tab.classList.add('active');
                document.getElementById(tabName).classList.add('active');
            }});
        }});
        
        // Sidebar navigation
        document.querySelectorAll('.nav-item').forEach(item => {{
            item.addEventListener('click', () => {{
                const sectionId = item.getAttribute('data-section');
                document.querySelectorAll('.nav-item').forEach(i => i.classList.remove('active'));
                item.classList.add('active');
                document.getElementById(sectionId).scrollIntoView({{ behavior: 'smooth' }});
            }});
        }});
        
        // Pagination
        window.showPage = function(page, tableId) {{
            const table = document.getElementById(tableId);
            const rows = table.querySelectorAll('tbody tr');
            const rowsPerPage = 10;
            const totalPages = Math.ceil(rows.length / rowsPerPage);
            
            const currentPage = Math.max(1, Math.min(page, totalPages));
            
            rows.forEach((row, index) => {{
                const startIndex = (currentPage - 1) * rowsPerPage;
                const endIndex = startIndex + rowsPerPage;
                row.style.display = (index >= startIndex && index < endIndex) ? '' : 'none';
            }});
            
            const pagination = table.nextElementSibling;
            pagination.querySelectorAll('.page-btn').forEach((btn, index) => {{
                btn.classList.toggle('active', index + 1 === currentPage);
            }});
        }};
        
        // Initialize pagination
        window.addEventListener('load', () => {{
            if (document.getElementById('successTable')) {{
                showPage(1, 'successTable');
            }}
            if (document.getElementById('failedTable')) {{
                showPage(1, 'failedTable');
            }}
        }});
    </script>
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


def generate_best_results_html(best_results):
    """Generate best results section"""
    if not best_results:
        return '<div class="best-results"><h2><i class="fas fa-trophy"></i> Best Results</h2><p>No experiments have achieved tier status yet.</p></div>'
    
    cards_html = []
    for result in best_results:
        exp = result['exp']
        experiment_id = exp.get('experiment_id', '')
        iteration = exp.get('iteration', 0)
        
        icon = '🥇' if result['tier'] == 'Gold' else '🥈' if result['tier'] == 'Silver' else '🥉'
        
        card = f"""
        <div class="tier-card" style="border-color: {result['color']};">
            <div class="tier-badge" style="color: {result['color']};">{icon} {result['tier']}</div>
            <div style="font-size: 1.1em; margin-bottom: 8px;"><strong>Iteration {iteration}</strong></div>
            <div class="tier-metrics">
                <div class="tier-metric">
                    <div class="tier-metric-value">{result['psnr']:.1f}</div>
                    <div class="tier-metric-label">PSNR (dB)</div>
                </div>
                <div class="tier-metric">
                    <div class="tier-metric-value">{result['ssim']:.3f}</div>
                    <div class="tier-metric-label">SSIM</div>
                </div>
                <div class="tier-metric">
                    <div class="tier-metric-value">{result['bitrate']:.1f}</div>
                    <div class="tier-metric-label">Bitrate</div>
                </div>
            </div>
            <div style="margin-top: 16px;">
                <a href="/blog/{experiment_id}" class="blog-link" target="_blank"><i class="fas fa-newspaper"></i> Read Blog Post</a>
            </div>
        </div>
        """
        cards_html.append(card)
    
    return f"""
    <div class="best-results">
        <h2><i class="fas fa-trophy"></i> Top Achievements</h2>
        <div class="tier-cards">
            {''.join(cards_html)}
        </div>
    </div>
    """


def generate_successful_table(experiments):
    """Generate successful experiments table"""
    if not experiments:
        return '<p>No successful experiments yet.</p>'
    
    rows = []
    for exp in experiments:
        experiment_id = exp.get('experiment_id', '')
        iteration = exp.get('iteration', 0)
        
        metrics = exp.get('metrics', {})
        psnr = float(metrics.get('psnr_db', 0))
        ssim = float(metrics.get('ssim', 0))
        bitrate = float(metrics.get('bitrate_mbps', 0))
        compression = float(metrics.get('compression_ratio', 0))
        
        psnr_label, psnr_color = get_quality_label('psnr', psnr)
        ssim_label, ssim_color = get_quality_label('ssim', ssim)
        bitrate_label, bitrate_color = get_quality_label('bitrate', bitrate)
        
        tier, tier_color = get_tier(psnr, ssim, bitrate)
        
        # Generate download URLs with Content-Disposition
        artifacts = exp.get('artifacts', {})
        video_url = None
        decoder_url = None
        
        if isinstance(artifacts, dict):
            decoder_s3_key = artifacts.get('decoder_s3_key')
            video_s3_key = f"videos/{experiment_id}/reconstructed.mp4"
            
            if decoder_s3_key:
                decoder_url = generate_presigned_url(
                    decoder_s3_key, 
                    download=True, 
                    filename=f"{experiment_id}_decoder.py"
                )
                video_url = generate_presigned_url(
                    video_s3_key, 
                    download=True, 
                    filename=f"{experiment_id}_video.mp4"
                )
        
        row = f"""
        <tr>
            <td><strong>{iteration}</strong></td>
            <td>
                {psnr:.2f} dB
                <span class="quality-badge" style="background: {psnr_color};">{psnr_label}</span>
            </td>
            <td>
                {ssim:.3f}
                <span class="quality-badge" style="background: {ssim_color};">{ssim_label}</span>
            </td>
            <td>
                {bitrate:.2f} Mbps
                <span class="quality-badge" style="background: {bitrate_color};">{bitrate_label}</span>
            </td>
            <td>{compression:.2f}x</td>
            <td>{tier if tier else '-'}</td>
            <td>
                <a href="/blog/{experiment_id}" class="blog-link" target="_blank"><i class="fas fa-newspaper"></i> Blog</a>
                {f' | <a href="{video_url}" class="blog-link"><i class="fas fa-download"></i> Video</a>' if video_url else ''}
                {f' | <a href="{decoder_url}" class="blog-link"><i class="fas fa-download"></i> Decoder</a>' if decoder_url else ''}
            </td>
        </tr>
        """
        rows.append(row)
    
    total_pages = (len(rows) + 9) // 10
    pagination_html = '<div class="pagination">'
    for i in range(1, min(total_pages + 1, 11)):
        pagination_html += f'<button class="page-btn" onclick="showPage({i}, \'successTable\')">{i}</button>'
    pagination_html += '</div>'
    
    return f"""
    <div class="table-container">
        <table id="successTable">
            <thead>
                <tr>
                    <th>Iter</th>
                    <th>PSNR</th>
                    <th>SSIM</th>
                    <th>Bitrate</th>
                    <th>Compression</th>
                    <th>Tier</th>
                    <th>Actions</th>
                </tr>
            </thead>
            <tbody>
                {''.join(rows)}
            </tbody>
        </table>
        {pagination_html}
    </div>
    """


def generate_in_progress_table(experiments):
    """Generate in-progress experiments table"""
    if not experiments:
        return '<p>No experiments currently in progress.</p>'
    
    rows = []
    for exp in experiments:
        experiment_id = exp.get('experiment_id', '')
        iteration = exp.get('iteration', 0)
        started_at = exp.get('started_at', '')
        phase = exp.get('phase', 'Starting...')
        
        # Format timestamp
        if started_at:
            try:
                if isinstance(started_at, str):
                    dt = datetime.fromisoformat(started_at.replace('Z', '+00:00'))
                    started_at = dt.strftime('%Y-%m-%d %H:%M:%S')
            except:
                started_at = 'Just now'
        else:
            started_at = 'Just now'
        
        row = f"""
        <tr>
            <td><strong>{iteration}</strong></td>
            <td>{experiment_id}</td>
            <td>
                <span class="badge" style="background: #fbbf24; color: #0f172a;">
                    <i class="fas fa-spinner fa-spin"></i> {phase}
                </span>
            </td>
            <td>{started_at}</td>
            <td>
                <i class="fas fa-circle-notch fa-spin" style="color: #fbbf24;"></i> Running
            </td>
        </tr>
        """
        rows.append(row)
    
    return f"""
    <div class="table-container">
        <table id="inProgressTable">
            <thead>
                <tr>
                    <th>Iteration</th>
                    <th>Experiment ID</th>
                    <th>Phase</th>
                    <th>Started</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
                {''.join(rows)}
            </tbody>
        </table>
    </div>
    """


def generate_failed_table(experiments):
    """Generate failed experiments table"""
    if not experiments:
        return '<p>No failed experiments.</p>'
    
    rows = []
    for exp in experiments:
        iteration = exp.get('iteration', 0)
        error = exp.get('error', 'Unknown error')
        reasoning = exp.get('llm_reasoning', 'No reasoning provided')
        
        row = f"""
        <tr>
            <td><strong>{iteration}</strong></td>
            <td>
                <div class="error-log">{error[:250]}{'...' if len(error) > 250 else ''}</div>
                <div class="llm-reasoning">
                    <strong>LLM Reasoning:</strong> {reasoning[:350]}{'...' if len(reasoning) > 350 else ''}
                </div>
            </td>
        </tr>
        """
        rows.append(row)
    
    total_pages = (len(rows) + 9) // 10
    pagination_html = '<div class="pagination">'
    for i in range(1, min(total_pages + 1, 11)):
        pagination_html += f'<button class="page-btn" onclick="showPage({i}, \'failedTable\')">{i}</button>'
    pagination_html += '</div>'
    
    return f"""
    <div class="table-container">
        <table id="failedTable">
            <thead>
                <tr>
                    <th style="width: 120px;">Iteration</th>
                    <th>Error Details & LLM Analysis</th>
                </tr>
            </thead>
            <tbody>
                {''.join(rows)}
            </tbody>
        </table>
        {pagination_html}
    </div>
    """


def render_blog_post(experiment_id):
    """Render detailed blog post for experiment"""
    
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
            'body': generate_404_page()
        }
    
    exp = items[0]
    iteration = exp.get('iteration', 0)
    status = exp.get('status', 'unknown')
    
    # Extract metrics
    metrics = exp.get('metrics', {})
    psnr = float(metrics.get('psnr_db', 0))
    ssim = float(metrics.get('ssim', 0))
    bitrate = float(metrics.get('bitrate_mbps', 0))
    compression = float(metrics.get('compression_ratio', 0))
    
    # Get quality labels
    psnr_label, psnr_color = get_quality_label('psnr', psnr)
    ssim_label, ssim_color = get_quality_label('ssim', ssim)
    bitrate_label, bitrate_color = get_quality_label('bitrate', bitrate)
    
    # Get tier
    tier, tier_color = get_tier(psnr, ssim, bitrate)
    tier_icon = '🥇' if tier == 'Gold' else '🥈' if tier == 'Silver' else '🥉' if tier == 'Bronze' else ''
    
    # Get artifacts
    artifacts = exp.get('artifacts', {})
    decoder_s3_key = artifacts.get('decoder_s3_key') if isinstance(artifacts, dict) else None
    video_s3_key = f"videos/{experiment_id}/reconstructed.mp4"
    
    video_url = generate_presigned_url(video_s3_key, download=True, filename=f"{experiment_id}_video.mp4") if decoder_s3_key else None
    decoder_url = generate_presigned_url(decoder_s3_key, download=True, filename=f"{experiment_id}_decoder.py") if decoder_s3_key else None
    
    # Get timestamp
    timestamp_iso = exp.get('timestamp_iso', '')
    try:
        dt = datetime.fromisoformat(timestamp_iso.replace('Z', '+00:00'))
        timestamp_str = dt.strftime('%B %d, %Y at %I:%M %p')
    except:
        timestamp_str = 'Unknown date'
    
    # Get LLM reasoning
    llm_reasoning = exp.get('llm_reasoning', 'No reasoning provided')
    
    # Generate blog HTML
    blog_html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Experiment {iteration} - AiV1 Research</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0f172a;
            color: #e2e8f0;
            line-height: 1.6;
        }}
        
        .header {{
            background: linear-gradient(135deg, #1e40af 0%, #7c3aed 100%);
            color: white;
            padding: 20px 0;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
        }}
        
        .container {{
            max-width: 1000px;
            margin: 0 auto;
            padding: 0 24px;
        }}
        
        .header-content {{
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        
        .back-link {{
            color: white;
            text-decoration: none;
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 8px 16px;
            background: rgba(255,255,255,0.2);
            border-radius: 6px;
            transition: background 0.2s;
        }}
        
        .back-link:hover {{
            background: rgba(255,255,255,0.3);
        }}
        
        .content {{
            padding: 40px 0;
        }}
        
        .blog-title {{
            font-size: 2.5em;
            margin-bottom: 16px;
            color: #e2e8f0;
        }}
        
        .blog-meta {{
            color: #94a3b8;
            font-size: 0.9em;
            margin-bottom: 32px;
            display: flex;
            align-items: center;
            gap: 20px;
        }}
        
        .tier-badge-large {{
            display: inline-block;
            padding: 8px 16px;
            border-radius: 8px;
            font-size: 1em;
            font-weight: bold;
            border: 2px solid;
        }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 32px 0;
        }}
        
        .metric-card {{
            background: #1e293b;
            padding: 24px;
            border-radius: 12px;
            border: 1px solid #334155;
        }}
        
        .metric-label {{
            font-size: 0.85em;
            color: #94a3b8;
            margin-bottom: 8px;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: #e2e8f0;
        }}
        
        .quality-badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 14px;
            font-size: 0.8em;
            font-weight: 600;
            color: #0f172a;
            margin-top: 8px;
        }}
        
        .section {{
            background: #1e293b;
            padding: 32px;
            border-radius: 12px;
            margin: 24px 0;
            border: 1px solid #334155;
        }}
        
        .section-title {{
            font-size: 1.5em;
            margin-bottom: 20px;
            color: #e2e8f0;
            display: flex;
            align-items: center;
            gap: 12px;
        }}
        
        .reasoning {{
            background: #164e63;
            border-left: 4px solid #06b6d4;
            padding: 20px;
            border-radius: 8px;
            font-style: italic;
            line-height: 1.8;
            color: #a5f3fc;
        }}
        
        .download-section {{
            display: flex;
            gap: 16px;
            margin: 24px 0;
        }}
        
        .download-btn {{
            flex: 1;
            padding: 16px 24px;
            background: #3b82f6;
            color: white;
            text-decoration: none;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 12px;
            font-weight: 600;
            transition: background 0.2s;
        }}
        
        .download-btn:hover {{
            background: #2563eb;
        }}
        
        .footer {{
            background: #1e293b;
            padding: 20px 0;
            text-align: center;
            border-top: 1px solid #334155;
            color: #94a3b8;
            font-size: 0.85em;
        }}
        
        .footer a {{
            color: #3b82f6;
            text-decoration: none;
        }}
        
        .footer a:hover {{
            text-decoration: underline;
        }}
    </style>
</head>
<body>
    <div class="header">
        <div class="container">
            <div class="header-content">
                <h1><i class="fas fa-video"></i> AiV1 Research</h1>
                <a href="/" class="back-link"><i class="fas fa-arrow-left"></i> Back to Dashboard</a>
            </div>
        </div>
    </div>
    
    <div class="container content">
        <h1 class="blog-title">Experiment Iteration {iteration}</h1>
        
        <div class="blog-meta">
            <span><i class="far fa-calendar"></i> {timestamp_str}</span>
            <span><i class="fas fa-flask"></i> {experiment_id}</span>
            {f'<span class="tier-badge-large" style="border-color: {tier_color}; color: {tier_color};">{tier_icon} {tier} Tier</span>' if tier else ''}
        </div>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label"><i class="fas fa-signal"></i> PSNR</div>
                <div class="metric-value">{psnr:.2f} dB</div>
                <span class="quality-badge" style="background: {psnr_color};">{psnr_label}</span>
            </div>
            
            <div class="metric-card">
                <div class="metric-label"><i class="fas fa-image"></i> SSIM</div>
                <div class="metric-value">{ssim:.3f}</div>
                <span class="quality-badge" style="background: {ssim_color};">{ssim_label}</span>
            </div>
            
            <div class="metric-card">
                <div class="metric-label"><i class="fas fa-tachometer-alt"></i> Bitrate</div>
                <div class="metric-value">{bitrate:.2f}</div>
                <div style="font-size: 0.8em; color: #94a3b8; margin-top: 4px;">Mbps</div>
                <span class="quality-badge" style="background: {bitrate_color};">{bitrate_label}</span>
            </div>
            
            <div class="metric-card">
                <div class="metric-label"><i class="fas fa-compress"></i> Compression</div>
                <div class="metric-value">{compression:.2f}x</div>
            </div>
        </div>
        
        {f'''
        <div class="download-section">
            <a href="{video_url}" class="download-btn"><i class="fas fa-download"></i> Download Reconstructed Video</a>
            <a href="{decoder_url}" class="download-btn"><i class="fas fa-download"></i> Download Decoder Code</a>
        </div>
        ''' if video_url and decoder_url else ''}
        
        <div class="section">
            <h2 class="section-title"><i class="fas fa-brain"></i> LLM Reasoning</h2>
            <div class="reasoning">
                {llm_reasoning}
            </div>
        </div>
        
        <div class="section">
            <h2 class="section-title"><i class="fas fa-chart-bar"></i> Analysis</h2>
            <p>
                This experiment achieved a PSNR of {psnr:.2f}dB and SSIM of {ssim:.3f}, which is considered <strong>{psnr_label.lower()}</strong> quality. 
                The compression ratio of {compression:.2f}x at {bitrate:.2f} Mbps bitrate shows {'excellent progress' if bitrate < 3 else 'good progress' if bitrate < 6 else 'room for improvement'}.
            </p>
            <br>
            <p>
                The structural similarity index (SSIM) of {ssim:.3f} indicates that the reconstructed video maintains {'excellent' if ssim >= 0.95 else 'good' if ssim >= 0.85 else 'acceptable'} 
                structural similarity to the original. This metric is particularly important as it correlates better with human perception than PSNR alone.
            </p>
            {f'<br><p><strong>Achievement:</strong> This experiment earned a <strong style="color: {tier_color};">{tier_icon} {tier} Tier</strong> ranking, placing it among the top-performing experiments.</p>' if tier else ''}
        </div>
    </div>
    
    <div class="footer">
        <div class="container">
            Created by <a href="https://www.linkedin.com/in/yaron-torbaty/" target="_blank"><i class="fab fa-linkedin"></i> Yaron Torbaty</a> | 
            Powered by Claude AI & AWS | 
            <a href="{GITHUB_REPO}" target="_blank"><i class="fab fa-github"></i> GitHub</a>
        </div>
    </div>
</body>
</html>
"""
    
    return {
        'statusCode': 200,
        'headers': {
            'Content-Type': 'text/html',
            'Cache-Control': 'public, max-age=300'
        },
        'body': blog_html
    }


def generate_404_page():
    """Generate 404 page"""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>404 - Experiment Not Found</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0f172a;
            color: #e2e8f0;
            display: flex;
            align-items: center;
            justify-content: center;
            height: 100vh;
            margin: 0;
        }
        .error-container {
            text-align: center;
        }
        h1 { font-size: 4em; color: #3b82f6; }
        p { font-size: 1.2em; color: #94a3b8; }
        a { color: #3b82f6; text-decoration: none; }
        a:hover { text-decoration: underline; }
    </style>
</head>
<body>
    <div class="error-container">
        <h1>404</h1>
        <p>Experiment not found</p>
        <p><a href="/">← Back to Dashboard</a></p>
    </div>
</body>
</html>
"""
