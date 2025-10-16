// Admin Chat Interface
const API_BASE = 'https://aiv1codec.com';
let conversationHistory = [];

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    loadSystemStatus();
    setInterval(loadSystemStatus, 30000); // Update every 30 seconds
    
    // Load chat history from localStorage
    const saved = localStorage.getItem('adminChatHistory');
    if (saved) {
        conversationHistory = JSON.parse(saved);
        conversationHistory.forEach(msg => {
            addMessageToUI(msg.role, msg.content, false);
        });
    }
});

// Send message to LLM
async function sendMessage() {
    const input = document.getElementById('chatInput');
    const message = input.value.trim();
    
    if (!message) return;
    
    // Add user message to UI
    addMessageToUI('user', message);
    conversationHistory.push({ role: 'user', content: message });
    
    // Clear input
    input.value = '';
    
    // Disable send button
    const sendBtn = document.getElementById('sendBtn');
    sendBtn.disabled = true;
    sendBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Thinking<span class="loading-dots"></span>';
    
    try {
        // Send to backend
        const response = await fetch(`${API_BASE}/admin/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                message: message,
                history: conversationHistory.slice(-10) // Last 10 messages for context
            })
        });
        
        if (!response.ok) {
            throw new Error('Failed to get response from LLM');
        }
        
        const data = await response.json();
        
        // Add LLM response to UI
        addMessageToUI('llm', data.response);
        conversationHistory.push({ role: 'llm', content: data.response });
        
        // Execute any commands if present
        if (data.commands && data.commands.length > 0) {
            for (const cmd of data.commands) {
                await executeCommand(cmd);
            }
        }
        
        // Save history
        localStorage.setItem('adminChatHistory', JSON.stringify(conversationHistory));
        
    } catch (error) {
        console.error('Error:', error);
        addMessageToUI('system', `Error: ${error.message}`);
    } finally {
        sendBtn.disabled = false;
        sendBtn.innerHTML = '<i class="fas fa-paper-plane"></i> Send';
    }
}

// Add message to UI
function addMessageToUI(role, content, save = true) {
    const messagesDiv = document.getElementById('chatMessages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;
    
    const icon = role === 'user' ? 'fa-user' : 
                 role === 'llm' ? 'fa-robot' : 
                 'fa-info-circle';
    
    const roleLabel = role === 'user' ? 'You' : 
                     role === 'llm' ? 'Governing LLM' : 
                     'System';
    
    messageDiv.innerHTML = `
        <div class="message-header">
            <i class="fas ${icon}"></i> ${roleLabel} - ${new Date().toLocaleTimeString()}
        </div>
        <div class="message-content">${formatMessage(content)}</div>
    `;
    
    messagesDiv.appendChild(messageDiv);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
    
    if (save && role !== 'system') {
        conversationHistory.push({ role, content });
    }
}

// Format message with markdown-like syntax
function formatMessage(text) {
    // Code blocks
    text = text.replace(/`([^`]+)`/g, '<code>$1</code>');
    // Line breaks
    text = text.replace(/\n/g, '<br>');
    // Bold
    text = text.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    return text;
}

// Quick commands
function sendQuickCommand(command) {
    document.getElementById('chatInput').value = command;
    sendMessage();
}

// Handle Enter key
function handleKeyPress(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
}

// Clear chat
function clearChat() {
    if (confirm('Clear all chat history?')) {
        conversationHistory = [];
        localStorage.removeItem('adminChatHistory');
        document.getElementById('chatMessages').innerHTML = `
            <div class="message system">
                <div class="message-content">
                    <strong>Chat Cleared</strong><br>
                    Ready for new conversation.
                </div>
            </div>
        `;
    }
}

// Load system status
async function loadSystemStatus() {
    try {
        const response = await fetch(`${API_BASE}/admin/status`);
        if (!response.ok) throw new Error('Failed to load status');
        
        const data = await response.json();
        
        // Update stats
        document.getElementById('totalExperiments').textContent = data.total_experiments || '--';
        document.getElementById('runningNow').textContent = data.running_now || '0';
        document.getElementById('bestBitrate').textContent = data.best_bitrate ? 
            `${data.best_bitrate.toFixed(2)} Mbps` : '--';
        document.getElementById('successRate').textContent = data.success_rate ? 
            `${Math.round(data.success_rate)}%` : '--';
        
        // Update status indicator
        const indicator = document.getElementById('statusIndicator');
        if (data.running_now > 0) {
            indicator.className = 'status-indicator running';
        } else if (data.autonomous_enabled) {
            indicator.className = 'status-indicator idle';
        } else {
            indicator.className = 'status-indicator stopped';
        }
        
    } catch (error) {
        console.error('Error loading status:', error);
    }
}

// Control functions
async function startExperiment() {
    if (!confirm('Start a new experiment now?')) return;
    
    addMessageToUI('system', 'Starting new experiment...');
    
    try {
        const response = await fetch(`${API_BASE}/admin/command`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ command: 'start_experiment' })
        });
        
        const data = await response.json();
        addMessageToUI('system', data.message || 'Experiment started');
        loadSystemStatus();
    } catch (error) {
        addMessageToUI('system', `Error: ${error.message}`);
    }
}

async function stopExperiments() {
    if (!confirm('Stop all running experiments?')) return;
    
    addMessageToUI('system', 'Stopping all experiments...');
    
    try {
        const response = await fetch(`${API_BASE}/admin/command`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ command: 'stop_experiments' })
        });
        
        const data = await response.json();
        addMessageToUI('system', data.message || 'Experiments stopped');
        loadSystemStatus();
    } catch (error) {
        addMessageToUI('system', `Error: ${error.message}`);
    }
}

async function pauseAutonomous() {
    addMessageToUI('system', 'Pausing autonomous mode...');
    
    try {
        const response = await fetch(`${API_BASE}/admin/command`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ command: 'pause_autonomous' })
        });
        
        const data = await response.json();
        addMessageToUI('system', data.message || 'Autonomous mode paused');
        loadSystemStatus();
    } catch (error) {
        addMessageToUI('system', `Error: ${error.message}`);
    }
}

async function resumeAutonomous() {
    addMessageToUI('system', 'Resuming autonomous mode...');
    
    try {
        const response = await fetch(`${API_BASE}/admin/command`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ command: 'resume_autonomous' })
        });
        
        const data = await response.json();
        addMessageToUI('system', data.message || 'Autonomous mode resumed');
        loadSystemStatus();
    } catch (error) {
        addMessageToUI('system', `Error: ${error.message}`);
    }
}

// Execute command from LLM
async function executeCommand(cmd) {
    addMessageToUI('system', `Executing: ${cmd.description}`);
    
    try {
        const response = await fetch(`${API_BASE}/admin/execute`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(cmd)
        });
        
        const data = await response.json();
        if (data.success) {
            addMessageToUI('system', `✅ ${data.message}`);
        } else {
            addMessageToUI('system', `❌ ${data.message}`);
        }
    } catch (error) {
        addMessageToUI('system', `Error executing command: ${error.message}`);
    }
}

