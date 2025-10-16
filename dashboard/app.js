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
