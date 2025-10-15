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
        // Simulate API response (replace with actual AWS API calls)
        return {
            experiments: {
                total: Math.floor(Math.random() * 50) + 10,
                recent: Math.floor(Math.random() * 5),
                bestCompression: Math.floor(Math.random() * 20) + 80,
                bestQuality: Math.floor(Math.random() * 10) + 40
            },
            infrastructure: {
                orchestrator: {
                    status: 'healthy',
                    cpu: Math.floor(Math.random() * 40) + 20,
                    memory: Math.floor(Math.random() * 30) + 40
                },
                training: {
                    status: 'healthy',
                    active: Math.floor(Math.random() * 3) + 1,
                    queue: Math.floor(Math.random() * 10),
                    spot: true
                },
                inference: {
                    status: 'healthy',
                    active: Math.floor(Math.random() * 2) + 1,
                    queue: Math.floor(Math.random() * 5),
                    fps: Math.floor(Math.random() * 20) + 30
                },
                storage: {
                    status: 'healthy',
                    s3Objects: Math.floor(Math.random() * 1000) + 100,
                    efsUsage: Math.floor(Math.random() * 50) + 10,
                    dynamodbItems: Math.floor(Math.random() * 500) + 50
                }
            },
            costs: {
                monthly: Math.floor(Math.random() * 2000) + 1000,
                breakdown: {
                    training: Math.floor(Math.random() * 1000) + 500,
                    inference: Math.floor(Math.random() * 500) + 200,
                    storage: Math.floor(Math.random() * 200) + 100,
                    orchestrator: Math.floor(Math.random() * 200) + 100
                }
            },
            experiments: [
                {
                    id: 'exp_001',
                    status: 'completed',
                    compression: 85,
                    quality: 42,
                    duration: '2h 15m',
                    cost: '$12.50'
                },
                {
                    id: 'exp_002',
                    status: 'running',
                    compression: 78,
                    quality: 38,
                    duration: '1h 30m',
                    cost: '$8.75'
                },
                {
                    id: 'exp_003',
                    status: 'failed',
                    compression: 45,
                    quality: 25,
                    duration: '45m',
                    cost: '$5.20'
                }
            ]
        };
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
            totalExperiments.textContent = experiments.total;
        }

        // Best compression
        const bestCompression = document.getElementById('best-compression');
        if (bestCompression) {
            bestCompression.textContent = experiments.bestCompression + '%';
        }

        // Best quality
        const bestQuality = document.getElementById('best-quality');
        if (bestQuality) {
            bestQuality.textContent = experiments.bestQuality;
        }

        // Monthly cost
        const monthlyCost = document.getElementById('monthly-cost');
        if (monthlyCost) {
            monthlyCost.textContent = '$' + costs.monthly.toLocaleString();
        }
    }

    updateInfrastructure(infrastructure) {
        // Orchestrator status
        this.updateInfraStatus('orchestrator', infrastructure.orchestrator);
        this.updateInfraDetail('orchestrator-cpu', infrastructure.orchestrator.cpu + '%');
        this.updateInfraDetail('orchestrator-memory', infrastructure.orchestrator.memory + '%');

        // Training workers status
        this.updateInfraStatus('training', infrastructure.training);
        this.updateInfraDetail('training-active', infrastructure.training.active);
        this.updateInfraDetail('training-queue', infrastructure.training.queue);
        this.updateInfraDetail('training-spot', infrastructure.training.spot ? 'Yes' : 'No');

        // Inference workers status
        this.updateInfraStatus('inference', infrastructure.inference);
        this.updateInfraDetail('inference-active', infrastructure.inference.active);
        this.updateInfraDetail('inference-queue', infrastructure.inference.queue);
        this.updateInfraDetail('inference-fps', infrastructure.inference.fps + ' fps');

        // Storage status
        this.updateInfraStatus('storage', infrastructure.storage);
        this.updateInfraDetail('s3-objects', infrastructure.storage.s3Objects.toLocaleString());
        this.updateInfraDetail('efs-usage', infrastructure.storage.efsUsage + ' GB');
        this.updateInfraDetail('dynamodb-items', infrastructure.storage.dynamodbItems.toLocaleString());
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
        const tableBody = document.getElementById('experiments-table-body');
        if (!tableBody) return;

        tableBody.innerHTML = '';

        experiments.forEach(exp => {
            const row = document.createElement('div');
            row.className = 'table-row';
            
            row.innerHTML = `
                <div class="col">${exp.id}</div>
                <div class="col">
                    <span class="status-badge ${exp.status}">${exp.status}</span>
                </div>
                <div class="col">${exp.compression}%</div>
                <div class="col">${exp.quality} dB</div>
                <div class="col">${exp.duration}</div>
                <div class="col">${exp.cost}</div>
            `;
            
            tableBody.appendChild(row);
        });
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
        this.updateCostPercentage('cost-training-pct', (breakdown.training / total * 100).toFixed(1));
        this.updateCostPercentage('cost-inference-pct', (breakdown.inference / total * 100).toFixed(1));
        this.updateCostPercentage('cost-storage-pct', (breakdown.storage / total * 100).toFixed(1));
        this.updateCostPercentage('cost-orchestrator-pct', (breakdown.orchestrator / total * 100).toFixed(1));
        
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
            element.textContent = '$' + value.toLocaleString();
        }
    }

    updateCostPercentage(id, percentage) {
        const element = document.getElementById(id);
        if (element) {
            element.textContent = percentage + '%';
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
