#!/usr/bin/env python3
"""
Web Dashboard for True Autoencoder Training
Real-time monitoring via web browser
"""

from flask import Flask, render_template_string
import subprocess
import re
import time
import os
import sys
from datetime import datetime

app = Flask(__name__)

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>True Autoencoder Training Monitor</title>
    <meta http-equiv="refresh" content="15">
    <style>
        body {
            font-family: 'Monaco', 'Courier New', monospace;
            background: #0d1117;
            color: #c9d1d9;
            margin: 0;
            padding: 20px;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
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
        .metric {
            display: inline-block;
            margin: 10px 20px 10px 0;
            padding: 10px 15px;
            background: #0d1117;
            border-radius: 6px;
            border-left: 3px solid #58a6ff;
        }
        .metric-label {
            color: #8b949e;
            font-size: 12px;
            text-transform: uppercase;
        }
        .metric-value {
            color: #58a6ff;
            font-size: 24px;
            font-weight: bold;
        }
        .log {
            background: #0d1117;
            border: 1px solid #30363d;
            border-radius: 6px;
            padding: 15px;
            font-size: 12px;
            overflow-x: auto;
            white-space: pre;
        }
        .gpu-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 15px;
        }
        .gpu-card {
            background: #0d1117;
            border: 1px solid #30363d;
            border-radius: 6px;
            padding: 15px;
        }
        .gpu-name {
            color: #58a6ff;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .gpu-stat {
            margin: 5px 0;
            font-size: 13px;
        }
        .progress-bar {
            background: #21262d;
            height: 8px;
            border-radius: 4px;
            overflow: hidden;
            margin: 5px 0;
        }
        .progress-fill {
            background: linear-gradient(90deg, #58a6ff, #1f6feb);
            height: 100%;
            transition: width 0.3s;
        }
        .timestamp {
            color: #8b949e;
            font-size: 12px;
            text-align: right;
        }
        .status-good { color: #3fb950; }
        .status-warn { color: #d29922; }
        .status-error { color: #f85149; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 True Autoencoder Training Monitor</h1>
        <div class="timestamp">Last updated: {{ timestamp }}</div>
        
        <div class="section">
            <h2>📊 Training Progress</h2>
            <div class="metric">
                <div class="metric-label">Current Epoch</div>
                <div class="metric-value">{{ current_epoch }}/{{ total_epochs }}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Val PSNR</div>
                <div class="metric-value {{ 'status-good' if val_psnr and val_psnr > 35 else 'status-warn' if val_psnr and val_psnr > 25 else '' }}">{{ "%.2f" % val_psnr if val_psnr else "--" }} dB</div>
            </div>
            <div class="metric">
                <div class="metric-label">Val Loss</div>
                <div class="metric-value">{{ "%.6f" % val_loss if val_loss else "--" }}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Best PSNR</div>
                <div class="metric-value status-good">{{ "%.2f" % best_psnr if best_psnr else "--" }} dB</div>
            </div>
        </div>
        
        <div class="section">
            <h2>🎮 GPU Status</h2>
            <div class="gpu-grid">
                {% for gpu in gpus %}
                <div class="gpu-card">
                    <div class="gpu-name">GPU {{ gpu.index }}: {{ gpu.name }}</div>
                    <div class="gpu-stat">Utilization: {{ gpu.utilization }}%</div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {{ gpu.utilization }}%"></div>
                    </div>
                    <div class="gpu-stat">Memory: {{ gpu.memory_used }} / {{ gpu.memory_total }} MiB</div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {{ (gpu.memory_used / gpu.memory_total * 100) if gpu.memory_total > 0 else 0 }}%"></div>
                    </div>
                    <div class="gpu-stat">Temp: {{ gpu.temperature }}°C</div>
                </div>
                {% endfor %}
            </div>
        </div>
        
        <div class="section">
            <h2>📝 Recent Training Log</h2>
            <div class="log">{{ training_log }}</div>
        </div>
        
        <div class="section">
            <h2>💾 Saved Models</h2>
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

def parse_training_log():
    """Parse training log for metrics"""
    log = run_command('tail -5000 /home/ec2-user/autoencoder_training/training.log')
    
    current_epoch = 0
    total_epochs = 100
    val_psnr = None
    val_loss = None
    best_psnr = None
    
    # Parse completed epoch numbers (format: "Epoch X/Y (time):")
    epoch_match = re.findall(r'Epoch (\d+)/(\d+) \([\d.]+s\):', log)
    if epoch_match:
        current_epoch = int(epoch_match[-1][0])
        total_epochs = int(epoch_match[-1][1])
    
    # Parse PSNR (can be negative)
    psnr_match = re.findall(r'PSNR: ([-\d.]+) dB', log)
    if psnr_match:
        psnr_values = [float(p) for p in psnr_match]
        val_psnr = psnr_values[-1]
        # Only consider positive PSNR for "best"
        positive_psnr = [p for p in psnr_values if p > 0]
        best_psnr = max(positive_psnr) if positive_psnr else val_psnr
    
    # Parse validation loss
    loss_match = re.findall(r'Val Loss: ([\d.]+)', log)
    if loss_match:
        val_loss = float(loss_match[-1])
    
    # For display, show last 100 lines that aren't spam
    log_lines = log.split('\n')
    filtered_log = [line for line in log_lines if 'Processing first batch' not in line]
    display_log = '\n'.join(filtered_log[-100:]) if filtered_log else log[-2000:]
    
    return {
        'current_epoch': current_epoch,
        'total_epochs': total_epochs,
        'val_psnr': val_psnr,
        'val_loss': val_loss,
        'best_psnr': best_psnr,
        'training_log': display_log if display_log else "Waiting for training to start..."
    }

def parse_gpu_status():
    """Parse GPU status"""
    gpu_output = run_command('nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits')
    
    gpus = []
    for line in gpu_output.split('\n'):
        if line.strip():
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 6:
                gpus.append({
                    'index': parts[0],
                    'name': parts[1],
                    'utilization': int(parts[2]) if parts[2].isdigit() else 0,
                    'memory_used': int(parts[3]) if parts[3].isdigit() else 0,
                    'memory_total': int(parts[4]) if parts[4].isdigit() else 1,
                    'temperature': int(parts[5]) if parts[5].isdigit() else 0,
                })
    
    return gpus

def get_saved_models():
    """Get list of saved models"""
    models = run_command('ls -lh /home/ec2-user/autoencoder_training/trained_models/*.pth 2>/dev/null | tail -10')
    return models if models else "No models saved yet"

@app.route('/')
def index():
    """Main dashboard page"""
    # Gather all data
    training_data = parse_training_log()
    gpus = parse_gpu_status()
    saved_models = get_saved_models()
    
    return render_template_string(
        HTML_TEMPLATE,
        timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        current_epoch=training_data['current_epoch'],
        total_epochs=training_data['total_epochs'],
        val_psnr=training_data['val_psnr'],
        val_loss=training_data['val_loss'],
        best_psnr=training_data['best_psnr'],
        training_log=training_data['training_log'],
        gpus=gpus,
        saved_models=saved_models
    )

if __name__ == '__main__':
    print("="*60, flush=True)
    print("True Autoencoder Training Dashboard", flush=True)
    print("="*60, flush=True)
    print("Starting web server on port 8080...", flush=True)
    print("Access at: http://<instance-public-ip>:8080", flush=True)
    print("="*60, flush=True)
    
    # Add error handling to prevent crashes
    try:
        app.run(host='0.0.0.0', port=8080, debug=False, threaded=True)
    except Exception as e:
        print(f"Dashboard error: {e}", flush=True)
        # Restart after 5 seconds
        time.sleep(5)
        os.execv(sys.executable, ['python3'] + sys.argv)

