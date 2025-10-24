#!/usr/bin/env python3
"""
Universal Web Dashboard for Autoencoder Training
Works with any iteration by auto-detecting the latest log file
"""

from flask import Flask, render_template_string
import subprocess
import re
import glob
import os
from datetime import datetime

app = Flask(__name__)

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Autoencoder Training Monitor</title>
    <meta http-equiv="refresh" content="10">
    <style>
        body {
            font-family: 'Monaco', 'Courier New', monospace;
            background: #0d1117;
            color: #c9d1d9;
            margin: 0;
            padding: 20px;
        }
        .container { max-width: 1400px; margin: 0 auto; }
        h1 {
            color: #58a6ff;
            border-bottom: 2px solid #21262d;
            padding-bottom: 10px;
        }
        h2 {
            color: #8b949e;
            margin-top: 30px;
            border-left: 4px solid #58a6ff;
            padding-left: 10px;
        }
        .section {
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 6px;
            padding: 20px;
            margin: 20px 0;
        }
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        .metric-card {
            background: #0d1117;
            border: 1px solid #30363d;
            border-radius: 6px;
            padding: 15px;
        }
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            color: #58a6ff;
            margin: 10px 0;
        }
        .metric-label {
            color: #8b949e;
            font-size: 0.9em;
        }
        .status-running { color: #3fb950; }
        .status-error { color: #f85149; }
        .status-waiting { color: #d29922; }
        .progress-bar {
            background: #21262d;
            height: 20px;
            border-radius: 4px;
            overflow: hidden;
            margin: 10px 0;
        }
        .progress-fill {
            background: linear-gradient(90deg, #3fb950 0%, #58a6ff 100%);
            height: 100%;
            transition: width 0.3s ease;
        }
        .gpu-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
        }
        .gpu-card {
            background: #0d1117;
            border: 1px solid #30363d;
            border-radius: 6px;
            padding: 15px;
        }
        .gpu-name { color: #58a6ff; font-weight: bold; margin-bottom: 10px; }
        .gpu-stat { margin: 5px 0; }
        .log {
            background: #0d1117;
            border: 1px solid #30363d;
            border-radius: 6px;
            padding: 15px;
            font-size: 0.85em;
            line-height: 1.6;
            overflow-x: auto;
            white-space: pre-wrap;
            max-height: 400px;
            overflow-y: auto;
        }
        .info-box {
            background: #1c2128;
            border-left: 4px solid #58a6ff;
            padding: 12px;
            margin: 15px 0;
            border-radius: 3px;
        }
        .warning-box {
            background: #332B00;
            border-left: 4px solid #d29922;
            padding: 12px;
            margin: 15px 0;
            border-radius: 3px;
        }
        .error-box {
            background: #2d0f0f;
            border-left: 4px solid #f85149;
            padding: 12px;
            margin: 15px 0;
            border-radius: 3px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🧠 Autoencoder Training Monitor</h1>
        <div class="info-box">
            <strong>Instance:</strong> {{ instance }} | 
            <strong>Log File:</strong> {{ log_file }} | 
            <strong>Updated:</strong> {{ timestamp }}
        </div>
        
        <div class="section">
            <h2>📊 Training Progress</h2>
            <div class="metric-grid">
                <div class="metric-card">
                    <div class="metric-label">Current Epoch</div>
                    <div class="metric-value">{{ current_epoch }} / {{ total_epochs }}</div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {{ (current_epoch / total_epochs * 100) if total_epochs > 0 else 0 }}%"></div>
                    </div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Validation PSNR</div>
                    <div class="metric-value" style="color: {% if val_psnr and val_psnr > 0 %}#3fb950{% elif val_psnr and val_psnr < 0 %}#f85149{% else %}#d29922{% endif %}">
                        {{ "%.2f"|format(val_psnr) if val_psnr is not none else "N/A" }} dB
                    </div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Validation Loss</div>
                    <div class="metric-value">{{ "%.6f"|format(val_loss) if val_loss else "N/A" }}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Best PSNR</div>
                    <div class="metric-value" style="color: #3fb950">{{ "%.2f"|format(best_psnr) if best_psnr else "N/A" }} dB</div>
                </div>
            </div>
            
            {% if training_status %}
            <div class="{% if 'ERROR' in training_status or 'OOM' in training_status %}error-box{% elif 'Running' in training_status %}info-box{% else %}warning-box{% endif %}">
                <strong>Status:</strong> {{ training_status }}
            </div>
            {% endif %}
        </div>
        
        <div class="section">
            <h2>🎮 GPU Status</h2>
            <div class="gpu-grid">
                {% for gpu in gpus %}
                <div class="gpu-card">
                    <div class="gpu-name">GPU {{ gpu.index }}: {{ gpu.name }}</div>
                    <div class="gpu-stat">GPU Utilization: {{ gpu.utilization }}%</div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {{ gpu.utilization }}%"></div>
                    </div>
                    <div class="gpu-stat">Memory: {{ gpu.memory_used }} / {{ gpu.memory_total }} MiB</div>
                    <div class="gpu-stat" style="color: {% if gpu.memory_pct > 80 %}#f85149{% elif gpu.memory_pct > 60 %}#d29922{% else %}#3fb950{% endif %}">Memory Usage: {{ "%.1f"|format(gpu.memory_pct) }}%</div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {{ gpu.memory_pct }}%"></div>
                    </div>
                    <div class="gpu-stat">Temp: {{ gpu.temperature }}°C</div>
                </div>
                {% endfor %}
            </div>
        </div>
        
        <div class="section">
            <h2>📝 Recent Training Log (Last 50 lines)</h2>
            <div class="log">{{ training_log }}</div>
        </div>
        
        <div class="section">
            <h2>💾 Saved Checkpoints</h2>
            <div class="log">{{ saved_models }}</div>
        </div>
    </div>
</body>
</html>
'''

def run_command(cmd):
    """Run shell command and return output"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
        return result.stdout.strip()
    except:
        return ""

def find_latest_log():
    """Auto-detect the latest training log file"""
    log_dir = "/home/ec2-user/autoencoder_training"
    
    # Priority 1: Check for iteration-specific logs (most recent first)
    iter_logs = sorted(glob.glob(f"{log_dir}/training_iter*.log"), 
                       key=os.path.getmtime, reverse=True)
    if iter_logs:
        return iter_logs[0]  # Return most recent iteration log
    
    # Priority 2: Fall back to generic training.log
    generic_log = f"{log_dir}/training.log"
    if os.path.exists(generic_log):
        return generic_log
    
    return None

def parse_training_log():
    """Parse training log for metrics (works with any log format)"""
    log_file = find_latest_log()
    
    if not log_file or not os.path.exists(log_file):
        return {
            'log_file': 'No log file found',
            'current_epoch': 0,
            'total_epochs': 100,
            'val_psnr': None,
            'val_loss': None,
            'best_psnr': None,
            'training_status': 'No training detected',
            'training_log': 'No log file found. Training may not have started yet.'
        }
    
    log = run_command(f'tail -5000 {log_file}')
    
    current_epoch = 0
    total_epochs = 100
    val_psnr = None
    val_loss = None
    best_psnr = None
    training_status = "Initializing..."
    
    # Parse completed epoch numbers - flexible regex for various formats
    # Matches: "Epoch X/Y (time):", "Epoch X/Y:", etc.
    epoch_patterns = [
        r'Epoch\s+(\d+)/(\d+)\s*\([^)]+\):',  # "Epoch 1/100 (123.4s):"
        r'Epoch\s+(\d+)/(\d+)\s*\(',          # "Epoch 1/100 (123.4s)"
        r'Epoch\s+(\d+)/(\d+):',              # "Epoch 1/100:"
        r'Epoch\s+(\d+)\s*/\s*(\d+)',         # "Epoch 1 / 100"
    ]
    
    for pattern in epoch_patterns:
        epoch_match = re.findall(pattern, log)
        if epoch_match:
            current_epoch = int(epoch_match[-1][0])
            total_epochs = int(epoch_match[-1][1])
            break
    
    # Parse PSNR (can be negative)
    psnr_match = re.findall(r'PSNR:\s*([-\d.]+)\s*dB', log)
    if psnr_match:
        psnr_values = [float(p) for p in psnr_match]
        val_psnr = psnr_values[-1]
        # Only consider positive PSNR for "best"
        positive_psnr = [p for p in psnr_values if p > 0]
        best_psnr = max(positive_psnr) if positive_psnr else val_psnr
    
    # Parse validation loss
    loss_match = re.findall(r'Val Loss:\s*([\d.]+)', log)
    if loss_match:
        val_loss = float(loss_match[-1])
    
    # Determine training status
    if 'OutOfMemoryError' in log or 'OOM' in log:
        training_status = "❌ ERROR: Out of Memory (OOM)"
    elif 'Error' in log[-2000:] or 'Exception' in log[-2000:]:
        training_status = "❌ ERROR: Training crashed"
    elif current_epoch > 0:
        training_status = f"✅ Running - Epoch {current_epoch}/{total_epochs}"
    elif 'Ready to train' in log:
        training_status = "🔄 Starting training..."
    elif 'Loading dataset' in log:
        training_status = "📥 Loading dataset..."
    else:
        training_status = "⏳ Initializing..."
    
    # For display, show last 50 meaningful lines
    log_lines = log.split('\n')
    # Filter out noise
    filtered_log = [line for line in log_lines if line.strip() and 
                    'Processing first batch' not in line and
                    '[Rank' not in line[:10]]  # Remove most DDP spam but keep important ones
    display_log = '\n'.join(filtered_log[-50:]) if filtered_log else log[-2000:]
    
    return {
        'log_file': os.path.basename(log_file),
        'current_epoch': current_epoch,
        'total_epochs': total_epochs,
        'val_psnr': val_psnr,
        'val_loss': val_loss,
        'best_psnr': best_psnr,
        'training_status': training_status,
        'training_log': display_log if display_log else "Waiting for training output..."
    }

def parse_gpu_status():
    """Parse GPU status"""
    gpu_output = run_command('nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits')
    
    gpus = []
    for line in gpu_output.split('\n'):
        if line.strip():
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 6:
                memory_used = int(parts[3]) if parts[3].isdigit() else 0
                memory_total = int(parts[4]) if parts[4].isdigit() else 1
                memory_pct = (memory_used / memory_total * 100) if memory_total > 0 else 0
                
                gpus.append({
                    'index': parts[0],
                    'name': parts[1],
                    'utilization': int(parts[2]) if parts[2].isdigit() else 0,
                    'memory_used': memory_used,
                    'memory_total': memory_total,
                    'memory_pct': memory_pct,
                    'temperature': int(parts[5]) if parts[5].isdigit() else 0,
                })
    
    return gpus

def get_saved_models():
    """List saved model checkpoints"""
    models_dirs = [
        "/home/ec2-user/autoencoder_training/trained_models_iter*",
        "/home/ec2-user/autoencoder_training/trained_models",
    ]
    
    all_models = []
    for pattern in models_dirs:
        all_models.extend(glob.glob(pattern))
    
    if not all_models:
        return "No checkpoints found yet"
    
    output_lines = []
    for model_dir in all_models:
        if os.path.isdir(model_dir):
            models = run_command(f'ls -lh {model_dir}/*.pth 2>/dev/null | tail -10')
            if models:
                output_lines.append(f"\n📁 {os.path.basename(model_dir)}:\n{models}")
    
    return '\n'.join(output_lines) if output_lines else "No .pth files found yet"

@app.route('/')
def index():
    """Main dashboard page"""
    training_data = parse_training_log()
    gpus = parse_gpu_status()
    saved_models = get_saved_models()
    
    hostname = run_command('hostname')
    instance = hostname if hostname else "g5.12xlarge"
    
    return render_template_string(
        HTML_TEMPLATE,
        instance=instance,
        log_file=training_data['log_file'],
        current_epoch=training_data['current_epoch'],
        total_epochs=training_data['total_epochs'],
        val_psnr=training_data['val_psnr'],
        val_loss=training_data['val_loss'],
        best_psnr=training_data['best_psnr'],
        training_status=training_data['training_status'],
        gpus=gpus,
        training_log=training_data['training_log'],
        saved_models=saved_models,
        timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    )

if __name__ == '__main__':
    # Run on all interfaces so it's accessible externally
    app.run(host='0.0.0.0', port=8080, debug=False)

