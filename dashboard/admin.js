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
        <nav style="background: rgba(0,0,0,0.3); padding: 15px 30px; display: flex; justify-content: space-between; align-items: center;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <i class="fas fa-robot"></i>
                <strong>AiV1 Codec - Admin Control</strong>
            </div>
            <div>
                <span style="margin-right: 20px;">
                    <i class="fas fa-user"></i> ${localStorage.getItem('adminUsername')}
                </span>
                <button onclick="logout()" style="padding: 8px 16px; background: #dc2626; border: none; border-radius: 6px; color: white; cursor: pointer;">
                    <i class="fas fa-sign-out-alt"></i> Logout
                </button>
            </div>
        </nav>
        
        <div style="max-width: 1400px; margin: 0 auto; padding: 20px;">
            <h1 style="text-align: center; margin: 40px 0;">
                <i class="fas fa-shield-alt"></i> Admin Dashboard
            </h1>
            
            <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; margin: 30px 0;">
                <div style="background: rgba(59, 130, 246, 0.2); padding: 20px; border-radius: 12px; border: 1px solid rgba(59, 130, 246, 0.3);">
                    <h3><i class="fas fa-flask"></i> Total Experiments</h3>
                    <div id="totalExperiments" style="font-size: 2em; font-weight: bold;">-</div>
                </div>
                <div style="background: rgba(16, 185, 129, 0.2); padding: 20px; border-radius: 12px; border: 1px solid rgba(16, 185, 129, 0.3);">
                    <h3><i class="fas fa-play"></i> Running Now</h3>
                    <div id="runningNow" style="font-size: 2em; font-weight: bold;">-</div>
                </div>
                <div style="background: rgba(251, 191, 36, 0.2); padding: 20px; border-radius: 12px; border: 1px solid rgba(251, 191, 36, 0.3);">
                    <h3><i class="fas fa-chart-line"></i> Best Bitrate</h3>
                    <div id="bestBitrate" style="font-size: 2em; font-weight: bold;">-</div>
                </div>
            </div>
            
            <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; margin: 30px 0;">
                <button onclick="executeCommand('start_experiment')" style="padding: 15px; background: #10b981; border: none; border-radius: 8px; color: white; font-weight: 600; cursor: pointer;">
                    <i class="fas fa-play"></i> Start Experiment
                </button>
                <button onclick="executeCommand('stop_experiments')" style="padding: 15px; background: #dc2626; border: none; border-radius: 8px; color: white; font-weight: 600; cursor: pointer;">
                    <i class="fas fa-stop"></i> Stop All
                </button>
                <button onclick="executeCommand('pause_autonomous')" style="padding: 15px; background: #f59e0b; border: none; border-radius: 8px; color: white; font-weight: 600; cursor: pointer;">
                    <i class="fas fa-pause"></i> Pause Autonomous
                </button>
            </div>
            
            <div id="commandStatus" style="margin: 20px 0; padding: 15px; border-radius: 8px; display: none;"></div>
        </div>
    `;
    
    showAdminInterface();
    loadSystemStatus();
    setInterval(loadSystemStatus, 30000); // Update every 30 seconds
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
    statusDiv.style.background = 'rgba(59, 130, 246, 0.2)';
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
            statusDiv.style.background = 'rgba(16, 185, 129, 0.2)';
            statusDiv.textContent = `✓ ${data.message}`;
        } else {
            statusDiv.style.background = 'rgba(220, 38, 38, 0.2)';
            statusDiv.textContent = `✗ ${data.message}`;
        }
        
        // Refresh status
        setTimeout(loadSystemStatus, 2000);
        
    } catch (error) {
        statusDiv.style.background = 'rgba(220, 38, 38, 0.2)';
        statusDiv.textContent = `✗ Error: ${error.message}`;
    }
}
