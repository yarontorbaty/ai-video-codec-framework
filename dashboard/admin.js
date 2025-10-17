// Admin Interface with 2FA Authentication
const API_BASE = 'https://aiv1codec.com';
let sessionToken = null;
let pendingUsername = null;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    // Check for existing session
    sessionToken = localStorage.getItem('adminToken');
    
    if (sessionToken) {
        // Verify token is still valid
        verifySession();
    } else {
        showLoginScreen();
    }
    
    // Setup login form
    const loginForm = document.getElementById('loginForm');
    if (loginForm) {
        loginForm.addEventListener('submit', handleLogin);
    }
});

function showLoginScreen() {
    document.getElementById('loginScreen').style.display = 'flex';
    if (document.getElementById('adminInterface')) {
        document.getElementById('adminInterface').style.display = 'none';
    }
}

function show2FAScreen() {
    const loginBox = document.querySelector('.login-box');
    loginBox.innerHTML = `
        <h1><i class="fas fa-shield-alt"></i> Enter 2FA Code</h1>
        <p style="text-align: center; color: #666; margin-bottom: 20px;">
            A 6-digit code has been sent to your email
        </p>
        <form id="twoFAForm">
            <div class="form-group">
                <label>Verification Code</label>
                <input type="text" id="twoFACode" required placeholder="000000" maxlength="6" 
                       pattern="[0-9]{6}" inputmode="numeric" autocomplete="one-time-code">
            </div>
            <button type="submit" class="login-btn">
                <i class="fas fa-check"></i> Verify
            </button>
            <button type="button" onclick="showLoginScreen(); location.reload();" 
                    style="width: 100%; margin-top: 10px; padding: 12px; background: #6c757d; border: none; border-radius: 8px; color: white; cursor: pointer;">
                <i class="fas fa-arrow-left"></i> Back to Login
            </button>
            <div id="twoFAError" class="error-message"></div>
        </form>
    `;
    
    document.getElementById('twoFAForm').addEventListener('submit', handle2FAVerify);
    document.getElementById('twoFACode').focus();
}

function showAdminInterface() {
    document.getElementById('loginScreen').style.display = 'none';
    if (document.getElementById('adminInterface')) {
        document.getElementById('adminInterface').style.display = 'block';
    }
    // Set dark background for admin interface
    document.body.style.background = 'linear-gradient(135deg, #0f172a 0%, #1e293b 100%)';
    document.body.style.minHeight = '100vh';
}

async function handleLogin(event) {
    event.preventDefault();
    
    const username = document.getElementById('username').value;
    const password = document.getElementById('password').value;
    const errorDiv = document.getElementById('loginError');
    
    errorDiv.textContent = '';
    
    try {
        const response = await fetch(`${API_BASE}/admin/login`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username, password })
        });
        
        const data = await response.json();
        
        if (data.success && data.requires_2fa) {
            // 2FA required - show verification screen
            pendingUsername = username;
            show2FAScreen();
        } else if (data.success && data.token) {
            // Direct login (2FA not configured)
            sessionToken = data.token;
            localStorage.setItem('adminToken', sessionToken);
            localStorage.setItem('adminUsername', data.username);
            loadAdminInterface();
        } else {
            errorDiv.textContent = data.error || 'Login failed';
        }
    } catch (error) {
        errorDiv.textContent = 'Connection error. Please try again.';
        console.error('Login error:', error);
    }
}

async function handle2FAVerify(event) {
    event.preventDefault();
    
    const code = document.getElementById('twoFACode').value;
    const errorDiv = document.getElementById('twoFAError');
    
    errorDiv.textContent = '';
    
    try {
        const response = await fetch(`${API_BASE}/admin/verify-2fa`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                username: pendingUsername, 
                code: code 
            })
        });
        
        const data = await response.json();
        
        if (data.success && data.token) {
            sessionToken = data.token;
            localStorage.setItem('adminToken', sessionToken);
            localStorage.setItem('adminUsername', data.username);
            loadAdminInterface();
        } else {
            errorDiv.textContent = data.error || '2FA verification failed';
        }
    } catch (error) {
        errorDiv.textContent = 'Connection error. Please try again.';
        console.error('2FA verification error:', error);
    }
}

async function verifySession() {
    try {
        const response = await fetch(`${API_BASE}/admin/status`, {
            headers: {
                'Authorization': `Bearer ${sessionToken}`
            }
        });
        
        if (response.status === 401) {
            // Session expired
            logout();
        } else if (response.ok) {
            // Session valid - load interface
            loadAdminInterface();
        }
    } catch (error) {
        console.error('Session verification error:', error);
        logout();
    }
}

function logout() {
    sessionToken = null;
    pendingUsername = null;
    localStorage.removeItem('adminToken');
    localStorage.removeItem('adminUsername');
    showLoginScreen();
    location.reload();
}

function loadAdminInterface() {
    // Build the admin interface dynamically
    let container = document.getElementById('adminInterface');
    if (!container) {
        container = document.createElement('div');
        container.id = 'adminInterface';
        container.style.display = 'none';
        document.body.appendChild(container);
    }
    
    container.innerHTML = `
        <nav style="background: linear-gradient(135deg, #1e293b 0%, #334155 100%); padding: 15px 30px; display: flex; justify-content: space-between; align-items: center; box-shadow: 0 4px 6px rgba(0,0,0,0.3);">
            <div style="display: flex; align-items: center; gap: 10px;">
                <i class="fas fa-robot" style="color: #60a5fa; font-size: 1.5em;"></i>
                <strong style="color: #f1f5f9; font-size: 1.2em;">AiV1 Codec - Admin Control</strong>
            </div>
            <div>
                <span style="margin-right: 20px; color: #cbd5e1;">
                    <i class="fas fa-user"></i> ${localStorage.getItem('adminUsername')}
                </span>
                <button onclick="logout()" style="padding: 10px 20px; background: #ef4444; border: none; border-radius: 8px; color: white; cursor: pointer; font-weight: 600; box-shadow: 0 2px 4px rgba(0,0,0,0.2);">
                    <i class="fas fa-sign-out-alt"></i> Logout
                </button>
            </div>
        </nav>
        
        <div style="max-width: 1400px; margin: 0 auto; padding: 20px;">
            <h1 style="text-align: center; margin: 40px 0; color: #f8fafc; font-size: 2.5em; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
                <i class="fas fa-shield-alt" style="color: #60a5fa;"></i> Admin Dashboard
            </h1>
            
            <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; margin: 30px 0;">
                <div style="background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%); padding: 25px; border-radius: 16px; border: 2px solid #60a5fa; box-shadow: 0 8px 16px rgba(0,0,0,0.3);">
                    <h3 style="color: #dbeafe; margin-bottom: 15px; font-size: 1.1em;"><i class="fas fa-flask"></i> Total Experiments</h3>
                    <div id="totalExperiments" style="font-size: 2.5em; font-weight: bold; color: #ffffff;">-</div>
                </div>
                <div style="background: linear-gradient(135deg, #047857 0%, #10b981 100%); padding: 25px; border-radius: 16px; border: 2px solid #34d399; box-shadow: 0 8px 16px rgba(0,0,0,0.3);">
                    <h3 style="color: #d1fae5; margin-bottom: 15px; font-size: 1.1em;"><i class="fas fa-play"></i> Running Now</h3>
                    <div id="runningNow" style="font-size: 2.5em; font-weight: bold; color: #ffffff;">-</div>
                </div>
                <div style="background: linear-gradient(135deg, #d97706 0%, #f59e0b 100%); padding: 25px; border-radius: 16px; border: 2px solid #fbbf24; box-shadow: 0 8px 16px rgba(0,0,0,0.3);">
                    <h3 style="color: #fef3c7; margin-bottom: 15px; font-size: 1.1em;"><i class="fas fa-chart-line"></i> Best Bitrate</h3>
                    <div id="bestBitrate" style="font-size: 2.5em; font-weight: bold; color: #ffffff;">-</div>
                </div>
            </div>
            
            <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 15px; margin: 30px 0;">
                <button onclick="executeCommand('start_experiment')" style="padding: 18px; background: linear-gradient(135deg, #059669 0%, #10b981 100%); border: 2px solid #34d399; border-radius: 12px; color: white; font-weight: 700; cursor: pointer; font-size: 1.1em; box-shadow: 0 4px 8px rgba(0,0,0,0.3); transition: transform 0.2s;">
                    <i class="fas fa-play"></i> Start Experiment
                </button>
                <button onclick="executeCommand('stop_experiments')" style="padding: 18px; background: linear-gradient(135deg, #dc2626 0%, #ef4444 100%); border: 2px solid #f87171; border-radius: 12px; color: white; font-weight: 700; cursor: pointer; font-size: 1.1em; box-shadow: 0 4px 8px rgba(0,0,0,0.3); transition: transform 0.2s;">
                    <i class="fas fa-stop"></i> Stop All
                </button>
                <button onclick="executeCommand('pause_autonomous')" style="padding: 18px; background: linear-gradient(135deg, #d97706 0%, #f59e0b 100%); border: 2px solid #fbbf24; border-radius: 12px; color: white; font-weight: 700; cursor: pointer; font-size: 1.1em; box-shadow: 0 4px 8px rgba(0,0,0,0.3); transition: transform 0.2s;">
                    <i class="fas fa-pause"></i> Pause Autonomous
                </button>
            </div>
            
            <div id="commandStatus" style="margin: 20px 0; padding: 18px; border-radius: 12px; display: none; font-size: 1.1em; font-weight: 600; border: 2px solid;"></div>
            
            <!-- Experiments Table -->
            <div style="background: linear-gradient(135deg, #1e293b 0%, #334155 100%); border-radius: 16px; padding: 25px; margin-top: 30px; border: 2px solid #475569; box-shadow: 0 8px 16px rgba(0,0,0,0.3);">
                <h2 style="margin-bottom: 20px; color: #f1f5f9; font-size: 1.8em;">
                    <i class="fas fa-flask" style="color: #60a5fa;"></i> Recent Experiments
                </h2>
                
                <div id="experimentsTable" style="overflow-x: auto;">
                    <div style="text-align: center; padding: 40px; color: #94a3b8;">
                        <i class="fas fa-spinner fa-spin" style="font-size: 2em; margin-bottom: 10px;"></i>
                        <p>Loading experiments...</p>
                    </div>
                </div>
            </div>
            
            <!-- LLM Chat Interface -->
            <div style="background: linear-gradient(135deg, #1e293b 0%, #334155 100%); border-radius: 16px; padding: 25px; margin-top: 30px; border: 2px solid #475569; box-shadow: 0 8px 16px rgba(0,0,0,0.3);">
                <h2 style="margin-bottom: 20px; color: #f1f5f9; font-size: 1.8em;">
                    <i class="fas fa-brain" style="color: #a78bfa;"></i> Governing LLM Chat
                </h2>
                
                <div id="chatMessages" style="height: 400px; overflow-y: auto; background: #0f172a; border-radius: 12px; padding: 20px; margin-bottom: 15px; border: 2px solid #1e293b; box-shadow: inset 0 2px 8px rgba(0,0,0,0.3);">
                    <div style="text-align: center; color: #94a3b8; padding: 40px;">
                        <i class="fas fa-comment-dots" style="font-size: 3em; margin-bottom: 20px; color: #64748b;"></i>
                        <p style="font-size: 1.2em; font-weight: 600; color: #cbd5e1;">Start a conversation with the Governing LLM</p>
                        <p style="font-size: 1em; margin-top: 10px;">Ask for experiment suggestions, insights, or provide guidance.</p>
                    </div>
                </div>
                
                <div style="display: flex; gap: 12px;">
                    <input 
                        type="text" 
                        id="chatInput" 
                        placeholder="Ask the LLM for experiment suggestions or provide guidance..."
                        style="flex: 1; padding: 16px; background: #0f172a; border: 2px solid #475569; border-radius: 12px; color: #f1f5f9; font-size: 1.05em; outline: none; transition: border-color 0.3s;"
                        onkeypress="if(event.key === 'Enter') sendChatMessage()"
                        onfocus="this.style.borderColor='#60a5fa'"
                        onblur="this.style.borderColor='#475569'"
                    />
                    <button 
                        onclick="sendChatMessage()" 
                        style="padding: 16px 28px; background: linear-gradient(135deg, #2563eb 0%, #3b82f6 100%); border: 2px solid #60a5fa; border-radius: 12px; color: white; font-weight: 700; cursor: pointer; font-size: 1.05em; box-shadow: 0 4px 8px rgba(0,0,0,0.3); transition: transform 0.2s;"
                        onmouseover="this.style.transform='translateY(-2px)'"
                        onmouseout="this.style.transform='translateY(0)'">
                        <i class="fas fa-paper-plane"></i> Send
                    </button>
                </div>
            </div>
        </div>
    `;
    
    showAdminInterface();
    loadSystemStatus();
    loadChatHistory();
    loadExperiments();
    setInterval(loadSystemStatus, 30000); // Update every 30 seconds
    setInterval(loadExperiments, 30000); // Update experiments every 30 seconds
}

async function loadSystemStatus() {
    try {
        const response = await fetch(`${API_BASE}/admin/status`, {
            headers: { 'Authorization': `Bearer ${sessionToken}` }
        });
        
        if (response.status === 401) {
            logout();
            return;
        }
        
        const data = await response.json();
        
        if (document.getElementById('totalExperiments')) {
            document.getElementById('totalExperiments').textContent = data.total_experiments || 0;
            document.getElementById('runningNow').textContent = data.running_now || 0;
            document.getElementById('bestBitrate').textContent = data.best_bitrate ? `${data.best_bitrate.toFixed(2)} Mbps` : 'N/A';
        }
    } catch (error) {
        console.error('Error loading status:', error);
    }
}

async function executeCommand(command) {
    const statusDiv = document.getElementById('commandStatus');
    statusDiv.style.display = 'block';
    statusDiv.style.background = 'linear-gradient(135deg, #1e40af 0%, #3b82f6 100%)';
    statusDiv.style.borderColor = '#60a5fa';
    statusDiv.style.color = '#ffffff';
    statusDiv.textContent = `Executing: ${command}...`;
    
    try {
        const response = await fetch(`${API_BASE}/admin/execute`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${sessionToken}`
            },
            body: JSON.stringify({ command })
        });
        
        if (response.status === 401) {
            logout();
            return;
        }
        
        const data = await response.json();
        
        if (data.success) {
            statusDiv.style.background = 'linear-gradient(135deg, #047857 0%, #10b981 100%)';
            statusDiv.style.borderColor = '#34d399';
            statusDiv.style.color = '#ffffff';
            statusDiv.textContent = `✓ ${data.message}`;
        } else {
            statusDiv.style.background = 'linear-gradient(135deg, #dc2626 0%, #ef4444 100%)';
            statusDiv.style.borderColor = '#f87171';
            statusDiv.style.color = '#ffffff';
            statusDiv.textContent = `✗ ${data.message}`;
        }
        
        // Refresh status
        setTimeout(loadSystemStatus, 2000);
        
    } catch (error) {
        statusDiv.style.background = 'linear-gradient(135deg, #dc2626 0%, #ef4444 100%)';
        statusDiv.style.borderColor = '#f87171';
        statusDiv.style.color = '#ffffff';
        statusDiv.textContent = `✗ Error: ${error.message}`;
    }
}

async function loadChatHistory() {
    try {
        const response = await fetch(`${API_BASE}/admin/chat`, {
            headers: { 'Authorization': `Bearer ${sessionToken}` }
        });
        
        if (response.status === 401) {
            logout();
            return;
        }
        
        const data = await response.json();
        
        const chatContainer = document.getElementById('chatMessages');
        if (data.messages && data.messages.length > 0) {
            chatContainer.innerHTML = '';
            data.messages.forEach(msg => {
                appendChatMessage(msg.role, msg.content, msg.timestamp);
            });
        }
    } catch (error) {
        console.error('Error loading chat history:', error);
    }
}

async function sendChatMessage() {
    const input = document.getElementById('chatInput');
    const message = input.value.trim();
    
    if (!message) return;
    
    // Clear input
    input.value = '';
    
    // Add user message to chat
    appendChatMessage('user', message, Date.now());
    
    // Show loading indicator
    const loadingId = 'loading-' + Date.now();
    appendChatMessage('assistant', '<i class="fas fa-spinner fa-spin"></i> Thinking...', Date.now(), loadingId);
    
    try {
        const response = await fetch(`${API_BASE}/admin/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${sessionToken}`
            },
            body: JSON.stringify({ message })
        });
        
        if (response.status === 401) {
            logout();
            return;
        }
        
        const data = await response.json();
        
        // Remove loading indicator
        const loadingEl = document.getElementById(loadingId);
        if (loadingEl) loadingEl.remove();
        
        if (data.response) {
            appendChatMessage('assistant', data.response, Date.now());
        } else {
            appendChatMessage('assistant', 'Error: ' + (data.error || 'No response'), Date.now());
        }
    } catch (error) {
        // Remove loading indicator
        const loadingEl = document.getElementById(loadingId);
        if (loadingEl) loadingEl.remove();
        
        appendChatMessage('assistant', 'Error: ' + error.message, Date.now());
    }
}

async function loadExperiments() {
    try {
        const response = await fetch(`${API_BASE}/admin/experiments`, {
            headers: { 'Authorization': `Bearer ${sessionToken}` }
        });
        
        if (response.status === 401) {
            logout();
            return;
        }
        
        const data = await response.json();
        
        if (data.success && data.experiments) {
            // Update total count if element exists
            if (data.total_count && document.getElementById('totalExperiments')) {
                document.getElementById('totalExperiments').textContent = data.total_count;
            }
            renderExperimentsTable(data.experiments);
        }
    } catch (error) {
        console.error('Error loading experiments:', error);
        document.getElementById('experimentsTable').innerHTML = `
            <div style="text-align: center; padding: 40px; color: #f87171;">
                <i class="fas fa-exclamation-triangle" style="font-size: 2em; margin-bottom: 10px;"></i>
                <p>Error loading experiments</p>
            </div>
        `;
    }
}

function renderExperimentsTable(experiments) {
    const tableContainer = document.getElementById('experimentsTable');
    
    if (!experiments || experiments.length === 0) {
        tableContainer.innerHTML = `
            <div style="text-align: center; padding: 40px; color: #94a3b8;">
                <i class="fas fa-flask" style="font-size: 2em; margin-bottom: 10px;"></i>
                <p>No experiments found</p>
                <p style="font-size: 0.9em; margin-top: 10px;">Start your first experiment using the controls above</p>
            </div>
        `;
        return;
    }
    
    let tableHTML = `
        <table style="width: 100%; border-collapse: collapse; color: #f1f5f9;">
            <thead>
                <tr style="background: rgba(0,0,0,0.3); border-bottom: 2px solid #475569;">
                    <th style="padding: 15px; text-align: left; color: #cbd5e1; font-weight: 600;">Experiment ID</th>
                    <th style="padding: 15px; text-align: left; color: #cbd5e1; font-weight: 600;">Time</th>
                    <th style="padding: 15px; text-align: center; color: #cbd5e1; font-weight: 600;">Status</th>
                    <th style="padding: 15px; text-align: center; color: #cbd5e1; font-weight: 600;">Tests Run</th>
                    <th style="padding: 15px; text-align: center; color: #cbd5e1; font-weight: 600;">Best Bitrate</th>
                    <th style="padding: 15px; text-align: center; color: #cbd5e1; font-weight: 600;"><i class="fas fa-trophy"></i> Achievement</th>
                    <th style="padding: 15px; text-align: center; color: #cbd5e1; font-weight: 600;"><i class="fas fa-chart-line"></i> PSNR</th>
                    <th style="padding: 15px; text-align: center; color: #cbd5e1; font-weight: 600;"><i class="fas fa-eye"></i> Quality</th>
                    <th style="padding: 15px; text-align: center; color: #cbd5e1; font-weight: 600;"><i class="fas fa-clock"></i> Runtime</th>
                    <th style="padding: 15px; text-align: center; color: #cbd5e1; font-weight: 600;"><i class="fas fa-list-ol"></i> Phase</th>
                    <th style="padding: 15px; text-align: center; color: #cbd5e1; font-weight: 600;"><i class="fas fa-code"></i> Code</th>
                    <th style="padding: 15px; text-align: center; color: #cbd5e1; font-weight: 600;"><i class="fas fa-code-branch"></i> Ver</th>
                    <th style="padding: 15px; text-align: center; color: #cbd5e1; font-weight: 600;"><i class="fab fa-github"></i> Git</th>
                    <th style="padding: 15px; text-align: center; color: #cbd5e1; font-weight: 600;"><i class="fas fa-video"></i> Media</th>
                    <th style="padding: 15px; text-align: center; color: #cbd5e1; font-weight: 600;"><i class="fas fa-bug"></i> Analysis</th>
                    <th style="padding: 15px; text-align: center; color: #cbd5e1; font-weight: 600;"><i class="fas fa-user-cog"></i> Human</th>
                    <th style="padding: 15px; text-align: center; color: #cbd5e1; font-weight: 600;">Actions</th>
                </tr>
            </thead>
            <tbody>
    `;
    
    experiments.forEach((exp, index) => {
        const statusColor = exp.status === 'completed' ? '#10b981' : 
                           exp.status === 'running' ? '#3b82f6' : 
                           exp.status === 'failed' ? '#ef4444' : '#94a3b8';
        
        const time = new Date(exp.timestamp * 1000).toLocaleString();
        const bitrate = exp.best_bitrate ? `${exp.best_bitrate.toFixed(2)} Mbps` : 'N/A';
        
        // Quality metrics (PSNR, SSIM, quality)
        const psnr = exp.psnr_db || null;
        const ssim = exp.ssim || null;
        const quality = exp.quality || null;
        
        let psnrDisplay = '<span style="color: #666;">—</span>';
        if (psnr !== null && psnr > 0) {
            const psnrColor = psnr >= 30 ? '#10b981' : (psnr >= 25 ? '#f59e0b' : '#ef4444');
            const psnrLabel = psnr >= 35 ? 'Excellent' : (psnr >= 30 ? 'Good' : (psnr >= 25 ? 'Acceptable' : 'Poor'));
            psnrDisplay = `<div style="display: flex; flex-direction: column; align-items: center; gap: 2px;">
                <span style="font-weight: 600; color: ${psnrColor}; font-size: 1.1em;">${psnr.toFixed(1)} dB</span>
                <span style="font-size: 0.75em; color: ${psnrColor}88;">${psnrLabel}</span>
            </div>`;
        }
        
        let qualityDisplay = '<span style="color: #666;">—</span>';
        if (quality && quality !== 'unknown') {
            const qualityColors = {
                'excellent': '#10b981',
                'good': '#20c997',
                'acceptable': '#f59e0b',
                'poor': '#ef4444'
            };
            const qualityEmoji = {
                'excellent': '🏆',
                'good': '✅',
                'acceptable': '⚠️',
                'poor': '❌'
            };
            const qColor = qualityColors[quality] || '#94a3b8';
            const qEmoji = qualityEmoji[quality] || '❓';
            qualityDisplay = `<div style="display: flex; flex-direction: column; align-items: center; gap: 2px;">
                <span style="font-size: 1.3em;">${qEmoji}</span>
                <span style="font-size: 0.75em; color: ${qColor}; font-weight: 600;">${quality.toUpperCase()}</span>
                ${ssim !== null && ssim > 0 ? `<span style="font-size: 0.7em; color: #94a3b8;">SSIM: ${ssim.toFixed(3)}</span>` : ''}
            </div>`;
        }
        
        // Code evolution fields
        const codeChanged = exp.code_changed || false;
        const version = exp.version || 0;
        const evolutionStatus = exp.evolution_status || 'N/A';
        const githubCommitted = exp.github_committed || false;
        const githubHash = exp.github_commit_hash || '';
        const improvement = exp.improvement || '';
        
        // Code badge
        const codeBadge = codeChanged ? 
            `<span style="background: #667eea; color: white; padding: 4px 8px; border-radius: 6px; font-size: 0.75em;" title="LLM generated code">✨ LLM</span>` :
            `<span style="color: #666;">—</span>`;
        
        // Version display  
        const versionDisplay = codeChanged ? 
            `<span style="font-weight: 600; color: #60a5fa; font-size: 1.1em;">v${version}</span>` :
            `<span style="color: #666;">—</span>`;
        
        // GitHub badge
        let githubBadge = '<span style="color: #666;">—</span>';
        if (githubCommitted && githubHash) {
            const shortHash = githubHash.substring(0, 7);
            githubBadge = `<a href="https://github.com/your-repo/commit/${githubHash}" target="_blank" style="color: #10b981; text-decoration: none; font-family: monospace;" title="${improvement}">
                <i class="fab fa-github"></i> ${shortHash}
            </a>`;
        } else if (exp.deployment_status === 'deployed') {
            githubBadge = `<span style="color: #f59e0b;" title="Deployed locally">📦 Local</span>`;
        }
        
        // Human intervention badge
        let humanBadge = '<span style="color: #666;">—</span>';
        if (exp.needs_human && exp.human_intervention_reasons && exp.human_intervention_reasons.length > 0) {
            const reasonsText = exp.human_intervention_reasons.map(r => 
                `${r.phase}: ${r.reason}`
            ).join('\\n');
            humanBadge = `<button onclick="showHumanIntervention('${exp.id}', ${JSON.stringify(exp.human_intervention_reasons).replace(/"/g, '&quot;')})"
                style="padding: 6px 10px; background: #dc262622; border: 1px solid #dc2626; border-radius: 6px; color: #dc2626; cursor: pointer; font-size: 0.85em; font-weight: 600; transition: all 0.2s; animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;"
                onmouseover="this.style.background='#dc262644'"
                onmouseout="this.style.background='#dc262622'"
                title="${reasonsText}">
                <i class="fas fa-user-cog"></i> HUMAN NEEDED
            </button>`;
        }
        
        // Runtime display with progress bar
        const elapsedSeconds = exp.elapsed_seconds || 0;
        const estimatedSeconds = exp.estimated_duration_seconds || 106;
        const progressPercent = Math.min(100, (elapsedSeconds / estimatedSeconds) * 100);
        
        const formatTime = (seconds) => {
            if (seconds < 60) return `${seconds}s`;
            const mins = Math.floor(seconds / 60);
            const secs = seconds % 60;
            return `${mins}m ${secs}s`;
        };
        
        const runtimeColor = exp.status === 'completed' ? '#10b981' : 
                            elapsedSeconds > estimatedSeconds * 1.5 ? '#ef4444' : 
                            elapsedSeconds > estimatedSeconds ? '#f59e0b' : '#3b82f6';
        
        let runtimeDisplay = `<div style="display: flex; flex-direction: column; align-items: center; gap: 4px;">
            <span style="font-weight: 600; color: ${runtimeColor};">${formatTime(elapsedSeconds)}</span>
            <span style="font-size: 0.75em; color: #94a3b8;">est: ${formatTime(estimatedSeconds)}</span>`;
        
        // Progress bar (only for running experiments)
        if (exp.status === 'running') {
            runtimeDisplay += `
                <div style="width: 100%; height: 4px; background: rgba(255,255,255,0.1); border-radius: 2px; overflow: hidden;">
                    <div style="width: ${progressPercent}%; height: 100%; background: ${runtimeColor}; transition: width 0.5s;"></div>
                </div>`;
        }
        runtimeDisplay += `</div>`;
        
        // Phase display with progress indicator
        const phaseData = {
            'design': { icon: 'fa-lightbulb', color: '#3b82f6', label: 'Design', order: 1 },
            'deploy': { icon: 'fa-upload', color: '#8b5cf6', label: 'Deploy', order: 2 },
            'validation': { icon: 'fa-check-circle', color: '#f59e0b', label: 'Validate', order: 3 },
            'execution': { icon: 'fa-play-circle', color: '#10b981', label: 'Execute', order: 4 },
            'quality_verification': { icon: 'fa-eye', color: '#ec4899', label: 'Quality Check', order: 5 },
            'analysis': { icon: 'fa-chart-line', color: '#06b6d4', label: 'Analyze', order: 6 },
            'complete': { icon: 'fa-check-double', color: '#10b981', label: 'Complete', order: 7 },
            'unknown': { icon: 'fa-question', color: '#94a3b8', label: 'Unknown', order: 0 }
        };
        
        const currentPhase = exp.current_phase || exp.phase_completed || 'unknown';
        const phase = phaseData[currentPhase] || phaseData['unknown'];
        
        let phaseBadge = `<div style="display: flex; flex-direction: column; align-items: center; gap: 4px;">
            <span style="padding: 6px 10px; background: ${phase.color}22; border: 1px solid ${phase.color}; border-radius: 6px; color: ${phase.color}; font-size: 0.85em; font-weight: 600; white-space: nowrap;">
                <i class="fas ${phase.icon}"></i> ${phase.label}
            </span>`;
        
        // Show retry counts if applicable
        const valRetries = exp.validation_retries || 0;
        const execRetries = exp.execution_retries || 0;
        if (valRetries > 0 || execRetries > 0) {
            phaseBadge += `<span style="font-size: 0.75em; color: #94a3b8;">`;
            if (valRetries > 0) phaseBadge += `Val: ${valRetries}x `;
            if (execRetries > 0) phaseBadge += `Exec: ${execRetries}x`;
            phaseBadge += `</span>`;
        }
        phaseBadge += `</div>`;
        
        // Parse experiments JSON to get video_url, decoder_s3_key, and achievement tier
        let videoUrl = null;
        let decoderKey = null;
        let achievementTier = null;
        let reductionPercent = 0;
        try {
            if (exp.experiments) {
                const experimentsData = typeof exp.experiments === 'string' ? JSON.parse(exp.experiments) : exp.experiments;
                if (experimentsData && experimentsData.length > 0) {
                    videoUrl = experimentsData[0].video_url;
                    decoderKey = experimentsData[0].decoder_s3_key;
                    const comparison = experimentsData[0].comparison || {};
                    achievementTier = comparison.achievement_tier;
                    reductionPercent = comparison.reduction_percent || 0;
                }
            }
        } catch (e) {
            // Ignore parsing errors
        }
        
        // Achievement tier display
        let achievementDisplay = '<span style="color: #666;">—</span>';
        if (achievementTier) {
            let tierColor, tierBg, tierIcon;
            if (achievementTier.includes('90%')) {
                tierColor = '#fbbf24';
                tierBg = '#fef3c7';
                tierIcon = '🏆';
            } else if (achievementTier.includes('70%')) {
                tierColor = '#10b981';
                tierBg = '#d1fae5';
                tierIcon = '🥇';
            } else if (achievementTier.includes('50%')) {
                tierColor = '#60a5fa';
                tierBg = '#dbeafe';
                tierIcon = '🥈';
            } else {
                tierColor = '#94a3b8';
                tierBg = '#f1f5f9';
                tierIcon = '🎯';
            }
            achievementDisplay = `<div style="display: flex; flex-direction: column; align-items: center; gap: 4px;">
                <span style="font-size: 1.5em;">${tierIcon}</span>
                <span style="padding: 4px 8px; background: ${tierBg}; border: 1px solid ${tierColor}; border-radius: 6px; color: ${tierColor}; font-size: 0.75em; font-weight: 600; white-space: nowrap;">
                    ${achievementTier}
                </span>
                <span style="font-size: 0.7em; color: #94a3b8;">${reductionPercent.toFixed(1)}% vs HEVC</span>
            </div>`;
        }

        // Media download buttons
        let mediaBadge = '<span style="color: #666;">—</span>';
        if (videoUrl || decoderKey) {
            mediaBadge = '<div style="display: flex; flex-direction: column; gap: 4px;">';
            if (videoUrl) {
                mediaBadge += `<a href="${videoUrl}" target="_blank" style="padding: 4px 8px; background: #ec489922; border: 1px solid #ec4899; border-radius: 4px; color: #ec4899; text-decoration: none; font-size: 0.75em; font-weight: 600; white-space: nowrap;">
                    <i class="fas fa-video"></i> Video
                </a>`;
            }
            if (decoderKey) {
                mediaBadge += `<a href="https://ai-video-codec-videos-580473065386.s3.amazonaws.com/${decoderKey}" target="_blank" style="padding: 4px 8px; background: #0ea5e922; border: 1px solid #0ea5e9; border-radius: 4px; color: #0ea5e9; text-decoration: none; font-size: 0.75em; font-weight: 600; white-space: nowrap;">
                    <i class="fas fa-code"></i> Decoder
                </a>`;
            }
            mediaBadge += '</div>';
        }
        
        // Failure analysis badge
        let analysisBadge = '<span style="color: #666;">—</span>';
        if (exp.failure_analysis) {
            const fa = exp.failure_analysis;
            const severityColors = {
                'critical': '#ef4444',
                'high': '#f59e0b',
                'medium': '#eab308',
                'low': '#94a3b8'
            };
            const severityColor = severityColors[fa.severity] || '#94a3b8';
            const categoryIcons = {
                'syntax_error': 'fa-code',
                'import_error': 'fa-puzzle-piece',
                'validation_error': 'fa-shield-alt',
                'runtime_error': 'fa-exclamation-triangle',
                'timeout': 'fa-clock',
                'resource_error': 'fa-memory',
                'logic_error': 'fa-brain'
            };
            const icon = categoryIcons[fa.category] || 'fa-bug';
            
            analysisBadge = `<button onclick="showFailureAnalysis('${exp.id}', ${JSON.stringify(fa).replace(/"/g, '&quot;')})" 
                style="padding: 6px 10px; background: ${severityColor}22; border: 1px solid ${severityColor}; border-radius: 6px; color: ${severityColor}; cursor: pointer; font-size: 0.85em; font-weight: 600; transition: all 0.2s;"
                onmouseover="this.style.background='${severityColor}44'"
                onmouseout="this.style.background='${severityColor}22'"
                title="${fa.root_cause}">
                <i class="fas ${icon}"></i> ${fa.severity.toUpperCase()}
            </button>`;
        }
        
        // Add status indicator color to row if code was adopted
        const rowHighlight = (evolutionStatus === 'adopted') ? 'rgba(16, 185, 129, 0.05)' : 'transparent';
        
        tableHTML += `
            <tr style="border-bottom: 1px solid rgba(255,255,255,0.1); transition: background 0.2s; background: ${rowHighlight};" 
                onmouseover="this.style.background='rgba(59, 130, 246, 0.1)'" 
                onmouseout="this.style.background='${rowHighlight}'">
                <td style="padding: 15px; font-family: monospace; font-size: 0.9em; color: #93c5fd;">${exp.id}</td>
                <td style="padding: 15px; color: #cbd5e1; font-size: 0.95em;">${time}</td>
                <td style="padding: 15px; text-align: center;">
                    <span style="padding: 6px 12px; background: ${statusColor}33; color: ${statusColor}; border-radius: 6px; font-weight: 600; font-size: 0.9em;">
                        ${exp.status.toUpperCase()}
                    </span>
                </td>
                <td style="padding: 15px; text-align: center; color: #e0e7ff; font-weight: 600;">${exp.experiments_run}</td>
                <td style="padding: 15px; text-align: center; color: #a5f3fc; font-weight: 600;">${bitrate}</td>
                <td style="padding: 15px; text-align: center;">${achievementDisplay}</td>
                <td style="padding: 15px; text-align: center;">${psnrDisplay}</td>
                <td style="padding: 15px; text-align: center;">${qualityDisplay}</td>
                <td style="padding: 15px; text-align: center;">${runtimeDisplay}</td>
                <td style="padding: 15px; text-align: center;">${phaseBadge}</td>
                <td style="padding: 15px; text-align: center;">${codeBadge}</td>
                <td style="padding: 15px; text-align: center;">${versionDisplay}</td>
                <td style="padding: 15px; text-align: center;">${githubBadge}</td>
                <td style="padding: 15px; text-align: center;">${mediaBadge}</td>
                <td style="padding: 15px; text-align: center;">${analysisBadge}</td>
                <td style="padding: 15px; text-align: center;">${humanBadge}</td>
                <td style="padding: 15px; text-align: center;">
                    <button onclick="viewExperimentDetails('${exp.id}')" 
                            title="View experiment details"
                            style="padding: 8px 12px; background: linear-gradient(135deg, #2563eb 0%, #3b82f6 100%); border: 1px solid #60a5fa; border-radius: 6px; color: white; cursor: pointer; margin: 0 3px; font-size: 1em;">
                        <i class="fas fa-eye"></i>
                    </button>
                    <button onclick="rerunExperiment('${exp.id}')" 
                            title="Rerun this experiment"
                            style="padding: 8px 12px; background: linear-gradient(135deg, #059669 0%, #10b981 100%); border: 1px solid #34d399; border-radius: 6px; color: white; cursor: pointer; margin: 0 3px; font-size: 1em;">
                        <i class="fas fa-redo"></i>
                    </button>
                </td>
            </tr>
        `;
    });
    
    tableHTML += `
            </tbody>
        </table>
    `;
    
    tableContainer.innerHTML = tableHTML;
}

function viewExperimentDetails(experimentId) {
    // Open blog with the experiment highlighted
    window.open(`/blog.html#${experimentId}`, '_blank');
}

async function rerunExperiment(experimentId) {
    if (!confirm(`Rerun experiment ${experimentId}?\n\nThis will start a new experiment with similar parameters.`)) {
        return;
    }
    
    const statusDiv = document.getElementById('commandStatus');
    statusDiv.style.display = 'block';
    statusDiv.style.background = 'linear-gradient(135deg, #1e40af 0%, #3b82f6 100%)';
    statusDiv.style.borderColor = '#60a5fa';
    statusDiv.style.color = '#ffffff';
    statusDiv.textContent = `Rerunning experiment...`;
    
    try {
        // For now, just start a new experiment
        await executeCommand('start_experiment');
    } catch (error) {
        statusDiv.style.background = 'linear-gradient(135deg, #dc2626 0%, #ef4444 100%)';
        statusDiv.style.borderColor = '#f87171';
        statusDiv.textContent = `✗ Error: ${error.message}`;
    }
}

function showHumanIntervention(experimentId, reasons) {
    // Create modal overlay
    const modalOverlay = document.createElement('div');
    modalOverlay.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(0, 0, 0, 0.8);
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 10000;
        backdrop-filter: blur(4px);
    `;
    
    // Create modal content
    const modal = document.createElement('div');
    modal.style.cssText = `
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-radius: 16px;
        padding: 32px;
        max-width: 600px;
        width: 90%;
        max-height: 80vh;
        overflow-y: auto;
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
        border: 1px solid rgba(220, 38, 38, 0.3);
    `;
    
    // Parse reasons if it's a string
    if (typeof reasons === 'string') {
        try {
            reasons = JSON.parse(reasons.replace(/&quot;/g, '"'));
        } catch (e) {
            console.error('Failed to parse reasons:', e);
            reasons = [];
        }
    }
    
    // Build reasons list
    const reasonsHTML = reasons.map(r => `
        <div style="margin-bottom: 16px; padding: 16px; background: rgba(220, 38, 38, 0.1); border-left: 4px solid #dc2626; border-radius: 8px;">
            <div style="font-weight: 600; color: #fca5a5; margin-bottom: 8px; text-transform: uppercase;">
                <i class="fas fa-layer-group"></i> Phase: ${r.phase}
            </div>
            <div style="color: #cbd5e1; line-height: 1.6;">
                <strong>Reason:</strong> ${r.reason}
            </div>
            ${r.last_failure ? `
                <div style="color: #94a3b8; font-size: 0.9em; margin-top: 8px;">
                    <strong>Details:</strong> ${r.last_failure.root_cause || r.last_error || 'No additional details'}
                </div>
            ` : ''}
        </div>
    `).join('');
    
    modal.innerHTML = `
        <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 24px;">
            <div>
                <h2 style="color: #dc2626; margin: 0; font-size: 1.5em; animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;">
                    <i class="fas fa-user-cog"></i> Human Intervention Required
                </h2>
                <p style="color: #94a3b8; margin: 8px 0 0 0; font-size: 0.9em;">
                    Experiment ID: ${experimentId}
                </p>
            </div>
            <button onclick="this.closest('[style*=fixed]').remove()" 
                    style="background: transparent; border: 1px solid #475569; border-radius: 8px; color: #94a3b8; padding: 8px 12px; cursor: pointer; font-size: 1.2em;">
                ✕
            </button>
        </div>
        
        <div style="background: rgba(220, 38, 38, 0.2); border-radius: 8px; padding: 16px; margin-bottom: 24px; border: 1px solid rgba(220, 38, 38, 0.3);">
            <div style="color: #fca5a5; font-weight: 600; margin-bottom: 8px;">
                <i class="fas fa-exclamation-triangle"></i> System Cannot Proceed Automatically
            </div>
            <div style="color: #e2e8f0; line-height: 1.6;">
                The autonomous system has encountered issues that exceed its self-healing capabilities.
                Manual intervention is required to resolve the following:
            </div>
        </div>
        
        <div style="margin-bottom: 24px;">
            <h3 style="color: #cbd5e1; margin: 0 0 16px 0; font-size: 1.1em;">
                <i class="fas fa-clipboard-list"></i> Intervention Reasons
            </h3>
            ${reasonsHTML}
        </div>
        
        <div style="border-top: 1px solid rgba(255,255,255,0.1); padding-top: 20px;">
            <button onclick="this.closest('[style*=fixed]').remove()" 
                    style="width: 100%; padding: 12px; background: linear-gradient(135deg, #dc2626 0%, #991b1b 100%); border: 1px solid #ef4444; border-radius: 8px; color: white; cursor: pointer; font-weight: 600; font-size: 1em;">
                <i class="fas fa-check"></i> Acknowledged
            </button>
        </div>
    `;
    
    modalOverlay.appendChild(modal);
    document.body.appendChild(modalOverlay);
    
    // Click outside to close
    modalOverlay.addEventListener('click', (e) => {
        if (e.target === modalOverlay) {
            modalOverlay.remove();
        }
    });
}

function showFailureAnalysis(experimentId, analysis) {
    const severityColors = {
        'critical': '#ef4444',
        'high': '#f59e0b',
        'medium': '#eab308',
        'low': '#94a3b8'
    };
    const severityColor = severityColors[analysis.severity] || '#94a3b8';
    
    const categoryLabels = {
        'syntax_error': 'Syntax Error',
        'import_error': 'Import Error',
        'validation_error': 'Validation Error',
        'runtime_error': 'Runtime Error',
        'timeout': 'Timeout',
        'resource_error': 'Resource Error',
        'logic_error': 'Logic Error'
    };
    const categoryLabel = categoryLabels[analysis.category] || analysis.category;
    
    // Create modal
    const modal = document.createElement('div');
    modal.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.8);
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 10000;
        backdrop-filter: blur(5px);
    `;
    
    modal.innerHTML = `
        <div style="
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            border: 2px solid rgba(255,255,255,0.1);
            border-radius: 16px;
            padding: 32px;
            max-width: 600px;
            width: 90%;
            box-shadow: 0 20px 60px rgba(0,0,0,0.5);
        ">
            <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 24px;">
                <div>
                    <h2 style="color: #f1f5f9; margin: 0 0 8px 0; font-size: 1.5em;">
                        <i class="fas fa-bug" style="color: ${severityColor};"></i>
                        Failure Analysis
                    </h2>
                    <p style="color: #94a3b8; margin: 0; font-family: monospace; font-size: 0.9em;">
                        ${experimentId}
                    </p>
                </div>
                <button onclick="this.parentElement.parentElement.parentElement.remove()" 
                    style="background: rgba(255,255,255,0.1); border: none; color: #cbd5e1; width: 32px; height: 32px; border-radius: 8px; cursor: pointer; font-size: 1.2em; transition: all 0.2s;">
                    ×
                </button>
            </div>
            
            <div style="margin-bottom: 20px;">
                <div style="display: inline-block; padding: 8px 16px; background: ${severityColor}22; border: 1px solid ${severityColor}; border-radius: 8px; margin-bottom: 20px;">
                    <span style="color: ${severityColor}; font-weight: 600; font-size: 0.9em;">
                        ${categoryLabel}
                    </span>
                    <span style="color: ${severityColor}; margin: 0 8px;">•</span>
                    <span style="color: ${severityColor}; font-weight: 600; font-size: 0.9em;">
                        ${analysis.severity.toUpperCase()} SEVERITY
                    </span>
                </div>
            </div>
            
            <div style="background: rgba(0,0,0,0.3); border-left: 4px solid ${severityColor}; padding: 16px; border-radius: 8px; margin-bottom: 20px;">
                <h3 style="color: #cbd5e1; margin: 0 0 12px 0; font-size: 0.9em; text-transform: uppercase; letter-spacing: 1px;">
                    <i class="fas fa-search"></i> Root Cause
                </h3>
                <p style="color: #f1f5f9; margin: 0; line-height: 1.6;">
                    ${analysis.root_cause}
                </p>
            </div>
            
            <div style="background: rgba(16, 185, 129, 0.1); border-left: 4px solid #10b981; padding: 16px; border-radius: 8px;">
                <h3 style="color: #cbd5e1; margin: 0 0 12px 0; font-size: 0.9em; text-transform: uppercase; letter-spacing: 1px;">
                    <i class="fas fa-wrench"></i> Suggested Fix
                </h3>
                <p style="color: #f1f5f9; margin: 0; line-height: 1.6;">
                    ${analysis.fix_suggestion}
                </p>
            </div>
            
            <div style="margin-top: 24px; text-align: right;">
                <button onclick="this.parentElement.parentElement.parentElement.remove()" 
                    style="
                        padding: 12px 24px;
                        background: linear-gradient(135deg, #2563eb 0%, #3b82f6 100%);
                        border: none;
                        border-radius: 8px;
                        color: white;
                        font-weight: 600;
                        cursor: pointer;
                        transition: all 0.2s;
                    "
                    onmouseover="this.style.transform='translateY(-2px)'; this.style.boxShadow='0 4px 12px rgba(59,130,246,0.4)'"
                    onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='none'">
                    Close
                </button>
            </div>
        </div>
    `;
    
    // Close on background click
    modal.addEventListener('click', (e) => {
        if (e.target === modal) {
            modal.remove();
        }
    });
    
    document.body.appendChild(modal);
}

function appendChatMessage(role, content, timestamp, id) {
    const chatContainer = document.getElementById('chatMessages');
    
    // Remove empty state if present
    const emptyState = chatContainer.querySelector('[style*="text-align: center"]');
    if (emptyState) emptyState.remove();
    
    const messageDiv = document.createElement('div');
    if (id) messageDiv.id = id;
    
    const isUser = role === 'user';
    messageDiv.style.cssText = `
        margin-bottom: 16px;
        padding: 16px 20px;
        border-radius: 12px;
        background: ${isUser ? 'linear-gradient(135deg, #1e40af 0%, #2563eb 100%)' : 'linear-gradient(135deg, #047857 0%, #059669 100%)'};
        border: 2px solid ${isUser ? '#3b82f6' : '#10b981'};
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    `;
    
    const time = new Date(timestamp).toLocaleTimeString();
    messageDiv.innerHTML = `
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 10px;">
            <i class="fas ${isUser ? 'fa-user' : 'fa-robot'}" style="color: ${isUser ? '#93c5fd' : '#6ee7b7'}; font-size: 1.2em;"></i>
            <strong style="color: #f1f5f9; font-size: 1.1em;">${isUser ? 'You' : 'LLM'}</strong>
            <span style="color: #cbd5e1; font-size: 0.9em;">${time}</span>
        </div>
        <div style="white-space: pre-wrap; line-height: 1.6; color: #f8fafc; font-size: 1.05em;">${content}</div>
    `;
    
    chatContainer.appendChild(messageDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}
