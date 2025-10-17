// AI Video Codec Framework Dashboard JavaScript

class Dashboard {
    constructor() {
        this.awsConfig = {
            region: 'us-east-1',
            accessKeyId: null, // Will be set from environment
            secretAccessKey: null, // Will be set from environment
        };
        
        this.refreshInterval = 30000; // 30 seconds
        this.isConnected = false;
        this.charts = {};
        
        this.init();
    }

    async init() {
        try {
            // Hide loading overlay
            this.hideLoading();
            
            // Initialize charts
            this.initCharts();
            
            // Start data refresh loop
            this.startRefreshLoop();
            
            // Load initial data
            await this.loadData();
            
        } catch (error) {
            console.error('Dashboard initialization failed:', error);
            this.showError('Failed to initialize dashboard');
        }
    }

    hideLoading() {
        const overlay = document.getElementById('loading-overlay');
        if (overlay) {
            overlay.classList.add('hidden');
            setTimeout(() => {
                overlay.style.display = 'none';
            }, 300);
        }
    }

    showError(message) {
        const statusText = document.getElementById('status-text');
        const statusDot = document.getElementById('status-dot');
        
        if (statusText) statusText.textContent = 'Error';
        if (statusDot) {
            statusDot.className = 'status-dot';
        }
        
        console.error(message);
    }

    initCharts() {
        // Cost breakdown pie chart
        const costCtx = document.getElementById('cost-chart');
        if (costCtx) {
            this.charts.cost = new Chart(costCtx, {
                type: 'doughnut',
                data: {
                    labels: ['Training Workers', 'Inference Workers', 'Storage', 'Orchestrator'],
                    datasets: [{
                        data: [0, 0, 0, 0],
                        backgroundColor: [
                            '#667eea',
                            '#764ba2',
                            '#f093fb',
                            '#f5576c'
                        ],
                        borderWidth: 0
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            position: 'bottom',
                            labels: {
                                padding: 20,
                                usePointStyle: true
                            }
                        }
                    }
                }
            });
        }

        // Compression progress line chart
        const compressionCtx = document.getElementById('compression-chart');
        if (compressionCtx) {
            this.charts.compression = new Chart(compressionCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Compression %',
                        data: [],
                        borderColor: '#667eea',
                        backgroundColor: 'rgba(102, 126, 234, 0.1)',
                        tension: 0.4,
                        fill: true
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 100,
                            ticks: {
                                callback: function(value) {
                                    return value + '%';
                                }
                            }
                        }
                    },
                    plugins: {
                        legend: {
                            display: false
                        }
                    }
                }
            });
        }

        // Quality progress line chart
        const qualityCtx = document.getElementById('quality-chart');
        if (qualityCtx) {
            this.charts.quality = new Chart(qualityCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'PSNR (dB)',
                        data: [],
                        borderColor: '#38a169',
                        backgroundColor: 'rgba(56, 161, 105, 0.1)',
                        tension: 0.4,
                        fill: true
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 50,
                            ticks: {
                                callback: function(value) {
                                    return value + ' dB';
                                }
                            }
                        }
                    },
                    plugins: {
                        legend: {
                            display: false
                        }
                    }
                }
            });
        }
    }

    async loadData() {
        try {
            // Simulate API calls (replace with actual AWS API calls)
            const data = await this.fetchDashboardData();
            this.updateDashboard(data);
            this.updateConnectionStatus(true);
        } catch (error) {
            console.error('Failed to load data:', error);
            this.updateConnectionStatus(false);
        }
    }

    async fetchDashboardData() {
        try {
            // Get API base URL - using the correct API Gateway endpoint
            const apiBaseUrl = 'https://pbv4wnw8zd.execute-api.us-east-1.amazonaws.com/production';

            // Fetch real data from API endpoints (dashboard route with query params)
            const [experimentsResponse, metricsResponse, costsResponse] = await Promise.allSettled([
                fetch(`${apiBaseUrl}/dashboard?type=experiments`),
                fetch(`${apiBaseUrl}/dashboard?type=metrics`),
                fetch(`${apiBaseUrl}/dashboard?type=costs`)
            ]);

            // Process metrics data
            let metrics = {};
            if (metricsResponse.status === 'fulfilled' && metricsResponse.value.ok) {
                metrics = await metricsResponse.value.json();
            }

            // Process experiments data
            let experiments = [];
            if (experimentsResponse.status === 'fulfilled' && experimentsResponse.value.ok) {
                experiments = await experimentsResponse.value.json();
            }

            // Process costs data
            let costs = {};
            if (costsResponse.status === 'fulfilled' && costsResponse.value.ok) {
                costs = await costsResponse.value.json();
            }

            // Return structured data with fallbacks for missing data
            return {
                experiments: {
                    total: experiments.length || 0,
                    recent: experiments.filter(exp => exp.status === 'running').length || 0,
                    bestCompression: experiments.length > 0 ? Math.max(...experiments.map(exp => exp.compression || 0)) : 0,
                    bestQuality: experiments.length > 0 ? Math.max(...experiments.map(exp => exp.quality || 0)) : 0
                },
                infrastructure: {
                    orchestrator: {
                        status: metrics.orchestrator_ip ? 'healthy' : 'unknown',
                        cpu: metrics.orchestrator_cpu || 'No data',
                        memory: metrics.orchestrator_memory || 'No data'
                    },
                    training: {
                        status: metrics.training_queue ? 'healthy' : 'unknown',
                        active: 'No data',
                        queue: 'No data',
                        spot: 'No data'
                    },
                    inference: {
                        status: metrics.evaluation_queue ? 'healthy' : 'unknown',
                        active: 'No data',
                        queue: 'No data',
                        fps: 'No data'
                    },
                    storage: {
                        status: 'unknown',
                        s3Objects: 'No data',
                        efsUsage: 'No data',
                        dynamodbItems: 'No data'
                    }
                },
                costs: {
                    monthly: costs.monthly_cost || 0,
                    breakdown: {
                        training: costs.breakdown?.training || 0,
                        inference: costs.breakdown?.inference || 0,
                        storage: costs.breakdown?.storage || 0,
                        orchestrator: costs.breakdown?.orchestrator || 0
                    }
                },
                experiments: experiments.length > 0 ? experiments : [
                    {
                        id: 'No data',
                        status: 'No data',
                        compression: 'No data',
                        quality: 'No data',
                        duration: 'No data',
                        cost: 'No data'
                    }
                ]
            };
        } catch (error) {
            console.error('Failed to fetch dashboard data:', error);
            // Return empty data structure with "No data" indicators
            return {
                experiments: {
                    total: 0,
                    recent: 0,
                    bestCompression: 0,
                    bestQuality: 0
                },
                infrastructure: {
                    orchestrator: {
                        status: 'unknown',
                        cpu: 'No data',
                        memory: 'No data'
                    },
                    training: {
                        status: 'unknown',
                        active: 'No data',
                        queue: 'No data',
                        spot: 'No data'
                    },
                    inference: {
                        status: 'unknown',
                        active: 'No data',
                        queue: 'No data',
                        fps: 'No data'
                    },
                    storage: {
                        status: 'unknown',
                        s3Objects: 'No data',
                        efsUsage: 'No data',
                        dynamodbItems: 'No data'
                    }
                },
                costs: {
                    monthly: 0,
                    breakdown: {
                        training: 0,
                        inference: 0,
                        storage: 0,
                        orchestrator: 0
                    }
                },
                experiments: [
                    {
                        id: 'No data',
                        status: 'No data',
                        compression: 'No data',
                        quality: 'No data',
                        duration: 'No data',
                        cost: 'No data'
                    }
                ]
            };
        }
    }

    updateDashboard(data) {
        // Update overview metrics
        this.updateOverview(data.experiments, data.costs);
        
        // Update infrastructure status
        this.updateInfrastructure(data.infrastructure);
        
        // Update experiments table
        this.updateExperimentsTable(data.experiments);
        
        // Update cost breakdown
        this.updateCostBreakdown(data.costs);
        
        // Update performance charts
        this.updatePerformanceCharts(data.experiments);
        
        // Update last updated time
        this.updateLastUpdated();
    }

    updateOverview(experiments, costs) {
        // Total experiments
        const totalExperiments = document.getElementById('total-experiments');
        if (totalExperiments) {
            totalExperiments.textContent = experiments.total || 'No data';
        }

        // Best compression
        const bestCompression = document.getElementById('best-compression');
        if (bestCompression) {
            bestCompression.textContent = experiments.bestCompression > 0 ? experiments.bestCompression + '%' : 'No data';
        }

        // Best quality
        const bestQuality = document.getElementById('best-quality');
        if (bestQuality) {
            bestQuality.textContent = experiments.bestQuality > 0 ? experiments.bestQuality : 'No data';
        }

        // Monthly cost
        const monthlyCost = document.getElementById('monthly-cost');
        if (monthlyCost) {
            monthlyCost.textContent = costs.monthly > 0 ? '$' + costs.monthly.toLocaleString() : 'No data';
        }
    }

    updateInfrastructure(infrastructure) {
        // Orchestrator status
        this.updateInfraStatus('orchestrator', infrastructure.orchestrator);
        this.updateInfraDetail('orchestrator-cpu', infrastructure.orchestrator.cpu + (typeof infrastructure.orchestrator.cpu === 'number' ? '%' : ''));
        this.updateInfraDetail('orchestrator-memory', infrastructure.orchestrator.memory + (typeof infrastructure.orchestrator.memory === 'number' ? '%' : ''));

        // Training workers status
        this.updateInfraStatus('training', infrastructure.training);
        this.updateInfraDetail('training-active', infrastructure.training.active);
        this.updateInfraDetail('training-queue', infrastructure.training.queue);
        this.updateInfraDetail('training-spot', infrastructure.training.spot);

        // Inference workers status
        this.updateInfraStatus('inference', infrastructure.inference);
        this.updateInfraDetail('inference-active', infrastructure.inference.active);
        this.updateInfraDetail('inference-queue', infrastructure.inference.queue);
        this.updateInfraDetail('inference-fps', infrastructure.inference.fps);

        // Storage status
        this.updateInfraStatus('storage', infrastructure.storage);
        this.updateInfraDetail('s3-objects', infrastructure.storage.s3Objects);
        this.updateInfraDetail('efs-usage', infrastructure.storage.efsUsage);
        this.updateInfraDetail('dynamodb-items', infrastructure.storage.dynamodbItems);
    }

    updateInfraStatus(component, data) {
        const statusElement = document.getElementById(component + '-status');
        if (statusElement) {
            const statusDot = statusElement.querySelector('.status-dot');
            const statusText = statusElement.querySelector('span');
            
            if (statusDot) {
                statusDot.className = 'status-dot ' + data.status;
            }
            if (statusText) {
                statusText.textContent = data.status.charAt(0).toUpperCase() + data.status.slice(1);
            }
        }
    }

    updateInfraDetail(id, value) {
        const element = document.getElementById(id);
        if (element) {
            element.textContent = value;
        }
    }

    updateExperimentsTable(experiments) {
        const container = document.getElementById('experiments-table-container');
        if (!container) return;

        // Hide loading
        const loading = document.getElementById('experiments-loading');
        if (loading) loading.style.display = 'none';

        if (!experiments || experiments.length === 0) {
            container.innerHTML = '<div style="padding: 20px; text-align: center; color: #94a3b8;">No experiments yet</div>';
            return;
        }

        let tableHTML = `
            <table style="width: 100%; border-collapse: collapse; color: #f1f5f9;">
                <thead>
                    <tr style="background: rgba(0,0,0,0.3); border-bottom: 2px solid #475569;">
                        <th style="padding: 15px; text-align: left; color: #cbd5e1; font-weight: 600;">Experiment ID</th>
                        <th style="padding: 15px; text-align: left; color: #cbd5e1; font-weight: 600;">Time</th>
                        <th style="padding: 15px; text-align: center; color: #cbd5e1; font-weight: 600;">Status</th>
                        <th style="padding: 15px; text-align: center; color: #cbd5e1; font-weight: 600;">Bitrate</th>
                        <th style="padding: 15px; text-align: center; color: #cbd5e1; font-weight: 600;"><i class="fas fa-chart-line"></i> PSNR</th>
                        <th style="padding: 15px; text-align: center; color: #cbd5e1; font-weight: 600;"><i class="fas fa-eye"></i> Quality</th>
                        <th style="padding: 15px; text-align: center; color: #cbd5e1; font-weight: 600;"><i class="fas fa-list-ol"></i> Phase</th>
                        <th style="padding: 15px; text-align: center; color: #cbd5e1; font-weight: 600;">Actions</th>
                    </tr>
                </thead>
                <tbody>
        `;

        experiments.forEach((exp) => {
            const statusColor = exp.status === 'completed' ? '#10b981' : 
                               exp.status === 'running' ? '#3b82f6' : 
                               exp.status === 'failed' ? '#ef4444' : '#94a3b8';
            
            const time = new Date(exp.timestamp * 1000).toLocaleString();
            const bitrate = exp.best_bitrate ? `${exp.best_bitrate.toFixed(2)} Mbps` : 'N/A';
            
            // Quality metrics
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

            // Phase display
            const phaseData = {
                'design': { icon: 'fa-lightbulb', color: '#3b82f6', label: 'Design' },
                'deploy': { icon: 'fa-upload', color: '#8b5cf6', label: 'Deploy' },
                'validation': { icon: 'fa-check-circle', color: '#f59e0b', label: 'Validate' },
                'execution': { icon: 'fa-play-circle', color: '#10b981', label: 'Execute' },
                'quality_verification': { icon: 'fa-eye', color: '#ec4899', label: 'Quality Check' },
                'analysis': { icon: 'fa-chart-line', color: '#06b6d4', label: 'Analyze' },
                'complete': { icon: 'fa-check-double', color: '#10b981', label: 'Complete' },
                'unknown': { icon: 'fa-question', color: '#94a3b8', label: 'Unknown' }
            };
            
            const currentPhase = exp.current_phase || exp.phase_completed || 'unknown';
            const phase = phaseData[currentPhase] || phaseData['unknown'];
            
            let phaseBadge = `<span style="padding: 6px 10px; background: ${phase.color}22; border: 1px solid ${phase.color}; border-radius: 6px; color: ${phase.color}; font-size: 0.85em; font-weight: 600; white-space: nowrap;">
                <i class="fas ${phase.icon}"></i> ${phase.label}
            </span>`;

            tableHTML += `
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.1); transition: background 0.2s;" 
                    onmouseover="this.style.background='rgba(59, 130, 246, 0.1)'" 
                    onmouseout="this.style.background='transparent'">
                    <td style="padding: 15px; font-family: monospace; font-size: 0.9em; color: #93c5fd;">${exp.id}</td>
                    <td style="padding: 15px; color: #cbd5e1; font-size: 0.95em;">${time}</td>
                    <td style="padding: 15px; text-align: center;">
                        <span style="padding: 6px 12px; background: ${statusColor}33; color: ${statusColor}; border-radius: 6px; font-weight: 600; font-size: 0.9em;">
                            ${exp.status.toUpperCase()}
                        </span>
                    </td>
                    <td style="padding: 15px; text-align: center; color: #a5f3fc; font-weight: 600;">${bitrate}</td>
                    <td style="padding: 15px; text-align: center;">${psnrDisplay}</td>
                    <td style="padding: 15px; text-align: center;">${qualityDisplay}</td>
                    <td style="padding: 15px; text-align: center;">${phaseBadge}</td>
                    <td style="padding: 15px; text-align: center;">
                        <button onclick="window.open('/blog.html#${exp.id}', '_blank')" 
                                style="padding: 8px 12px; background: linear-gradient(135deg, #2563eb 0%, #3b82f6 100%); border: 1px solid #60a5fa; border-radius: 6px; color: white; cursor: pointer; font-size: 0.85em; font-weight: 600;">
                            <i class="fas fa-eye"></i> View Details
                        </button>
                    </td>
                </tr>
            `;
        });

        tableHTML += `
                </tbody>
            </table>
        `;

        container.innerHTML = tableHTML;
    }

    updateCostBreakdown(costs) {
        const breakdown = costs.breakdown;
        const total = costs.monthly;
        
        // Update cost values
        this.updateCostItem('cost-training', breakdown.training);
        this.updateCostItem('cost-inference', breakdown.inference);
        this.updateCostItem('cost-storage', breakdown.storage);
        this.updateCostItem('cost-orchestrator', breakdown.orchestrator);
        
        // Update cost percentages
        if (total > 0) {
            this.updateCostPercentage('cost-training-pct', (breakdown.training / total * 100).toFixed(1));
            this.updateCostPercentage('cost-inference-pct', (breakdown.inference / total * 100).toFixed(1));
            this.updateCostPercentage('cost-storage-pct', (breakdown.storage / total * 100).toFixed(1));
            this.updateCostPercentage('cost-orchestrator-pct', (breakdown.orchestrator / total * 100).toFixed(1));
        } else {
            this.updateCostPercentage('cost-training-pct', 'No data');
            this.updateCostPercentage('cost-inference-pct', 'No data');
            this.updateCostPercentage('cost-storage-pct', 'No data');
            this.updateCostPercentage('cost-orchestrator-pct', 'No data');
        }
        
        // Update cost chart
        if (this.charts.cost) {
            this.charts.cost.data.datasets[0].data = [
                breakdown.training,
                breakdown.inference,
                breakdown.storage,
                breakdown.orchestrator
            ];
            this.charts.cost.update();
        }
    }

    updateCostItem(id, value) {
        const element = document.getElementById(id);
        if (element) {
            element.textContent = value > 0 ? '$' + value.toLocaleString() : 'No data';
        }
    }

    updateCostPercentage(id, percentage) {
        const element = document.getElementById(id);
        if (element) {
            element.textContent = percentage === 'No data' ? 'No data' : percentage + '%';
        }
    }

    updatePerformanceCharts(experiments) {
        // Generate sample data for charts
        const labels = [];
        const compressionData = [];
        const qualityData = [];
        
        for (let i = 0; i < 10; i++) {
            labels.push(`Exp ${i + 1}`);
            compressionData.push(Math.floor(Math.random() * 30) + 70);
            qualityData.push(Math.floor(Math.random() * 15) + 35);
        }
        
        // Update compression chart
        if (this.charts.compression) {
            this.charts.compression.data.labels = labels;
            this.charts.compression.data.datasets[0].data = compressionData;
            this.charts.compression.update();
        }
        
        // Update quality chart
        if (this.charts.quality) {
            this.charts.quality.data.labels = labels;
            this.charts.quality.data.datasets[0].data = qualityData;
            this.charts.quality.update();
        }
    }

    updateConnectionStatus(connected) {
        this.isConnected = connected;
        
        const statusText = document.getElementById('status-text');
        const statusDot = document.getElementById('status-dot');
        
        if (statusText && statusDot) {
            if (connected) {
                statusText.textContent = 'Connected';
                statusDot.className = 'status-dot connected';
            } else {
                statusText.textContent = 'Disconnected';
                statusDot.className = 'status-dot';
            }
        }
    }

    updateLastUpdated() {
        const lastUpdated = document.getElementById('last-updated');
        if (lastUpdated) {
            lastUpdated.textContent = new Date().toLocaleTimeString();
        }
    }

    startRefreshLoop() {
        setInterval(() => {
            this.loadData();
        }, this.refreshInterval);
    }
}

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new Dashboard();
});

// Add some utility functions for AWS API integration
class AWSIntegration {
    constructor() {
        this.awsConfig = {
            region: 'us-east-1'
        };
    }

    async getCloudFormationOutputs(stackName) {
        // This would make actual AWS API calls
        // For now, return mock data
        return {
            orchestratorIp: '54.123.45.67',
            trainingQueueUrl: 'https://sqs.us-east-1.amazonaws.com/123456789012/ai-video-codec-training-queue',
            evaluationQueueUrl: 'https://sqs.us-east-1.amazonaws.com/123456789012/ai-video-codec-evaluation-queue'
        };
    }

    async getDynamoDBMetrics(tableName) {
        // This would query DynamoDB for experiment data
        return {
            itemCount: Math.floor(Math.random() * 100),
            experiments: []
        };
    }

    async getS3Metrics(bucketName) {
        // This would get S3 object counts and sizes
        return {
            objectCount: Math.floor(Math.random() * 1000),
            totalSize: Math.floor(Math.random() * 1000000)
        };
    }

    async getCostMetrics() {
        // This would query AWS Cost Explorer API
        return {
            monthlyCost: Math.floor(Math.random() * 2000) + 1000,
            dailyCost: Math.floor(Math.random() * 100) + 50
        };
    }
}

// Export for use in other modules
window.Dashboard = Dashboard;
window.AWSIntegration = AWSIntegration;
