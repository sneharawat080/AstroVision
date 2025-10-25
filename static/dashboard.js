// Enhanced Dashboard Functionality
class AdvancedDashboard {
    constructor() {
        this.results = window.analysisResults;
        this.charts = {};
        this.init();
    }

    init() {
        if (!this.results) {
            this.showNoDataMessage();
            return;
        }

        this.initAdvancedCharts();
        this.initInteractiveFeatures();
        this.initRealTimeData();
        this.initExportHandlers();
    }

    initAdvancedCharts() {
        this.createAdvancedAnomalyChart();
        this.createFeatureCorrelationChart();
        this.createTemporalAnalysisChart();
        this.createClusterVisualization();
    }

    createAdvancedAnomalyChart() {
        const ctx = document.getElementById('advancedAnomalyChart');
        if (!ctx) return;

        // Advanced anomaly visualization with multiple dimensions
        const summary = this.results.summary;
        
        this.charts.anomaly = new Chart(ctx, {
            type: 'scatter',
            data: {
                datasets: [{
                    label: 'Normal Data',
                    data: this.getSampleData('normal'),
                    backgroundColor: 'rgba(107, 207, 127, 0.6)',
                    borderColor: 'var(--cosmic-success)',
                    pointRadius: 4
                }, {
                    label: 'Anomalies',
                    data: this.getSampleData('anomalies'),
                    backgroundColor: 'rgba(255, 107, 107, 0.8)',
                    borderColor: 'var(--cosmic-danger)',
                    pointRadius: 6,
                    pointStyle: 'diamond'
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        position: 'top',
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return `${context.dataset.label}: (${context.parsed.x}, ${context.parsed.y})`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Feature 1'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Feature 2'
                        }
                    }
                }
            }
        });
    }

    getSampleData(type) {
        const data = this.results.sample_data[type];
        const features = this.results.sample_data.all_features.filter(f => !['Anomaly', 'Anomaly_Score'].includes(f));
        
        if (data.length === 0) return [];
        
        // Use first two features for scatter plot
        return data.map(item => ({
            x: item[features[0]],
            y: item[features[1]]
        }));
    }

    createFeatureCorrelationChart() {
        const ctx = document.getElementById('correlationChart');
        if (!ctx) return;

        const correlationData = this.results.statistics.correlation_matrix;
        const features = Object.keys(correlationData).filter(f => !['Anomaly', 'Anomaly_Score'].includes(f));
        
        const correlationMatrix = features.map(f1 => 
            features.map(f2 => correlationData[f1][f2])
        );

        this.charts.correlation = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: features,
                datasets: [{
                    label: 'Feature Correlations',
                    data: features.map((f, i) => {
                        // Calculate average correlation for each feature
                        const correlations = correlationMatrix[i];
                        return correlations.reduce((a, b) => a + Math.abs(b), 0) / correlations.length;
                    }),
                    backgroundColor: 'rgba(74, 144, 226, 0.7)'
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 1
                    }
                }
            }
        });
    }

    createTemporalAnalysisChart() {
        const ctx = document.getElementById('temporalChart');
        if (!ctx) return;

        // Simulate temporal data if not available
        const dataPoints = this.results.summary.total_points;
        const timestamps = Array.from({length: dataPoints}, (_, i) => i);
        const values = Array.from({length: dataPoints}, () => Math.random() * 100);
        
        this.charts.temporal = new Chart(ctx, {
            type: 'line',
            data: {
                labels: timestamps,
                datasets: [{
                    label: 'Temporal Pattern',
                    data: values,
                    borderColor: 'var(--cosmic-primary)',
                    backgroundColor: 'rgba(0, 206, 209, 0.1)',
                    tension: 0.4,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                interaction: {
                    intersect: false,
                    mode: 'index'
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Time'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Value'
                        }
                    }
                }
            }
        });
    }

    createClusterVisualization() {
        const ctx = document.getElementById('clusterChart');
        if (!ctx) return;

        // Create cluster visualization using PCA data if available
        const pcaData = this.results.visualization_data?.pca_data;
        if (!pcaData) return;

        const clusterData = this.results.visualization_data.cluster_analysis || 
                           Array(pcaData.x.length).fill(0);

        const uniqueClusters = [...new Set(clusterData)];
        const clusterColors = ['#4A90E2', '#7B68EE', '#00CED1', '#FF6B6B', '#FFD93D'];

        const datasets = uniqueClusters.map((cluster, index) => {
            const clusterIndices = clusterData.map((c, i) => c === cluster ? i : -1).filter(i => i !== -1);
            return {
                label: `Cluster ${cluster}`,
                data: clusterIndices.map(i => ({
                    x: pcaData.x[i],
                    y: pcaData.y[i]
                })),
                backgroundColor: clusterColors[index % clusterColors.length],
                pointRadius: 5
            };
        });

        this.charts.cluster = new Chart(ctx, {
            type: 'scatter',
            data: {
                datasets: datasets
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        position: 'top'
                    }
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Principal Component 1'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Principal Component 2'
                        }
                    }
                }
            }
        });
    }

    initInteractiveFeatures() {
        this.initFilterSystem();
        this.initChartInteractions();
        this.initDataDrilldown();
    }

    initFilterSystem() {
        const filterForm = document.getElementById('dashboardFilters');
        if (!filterForm) return;

        filterForm.addEventListener('submit', (e) => {
            e.preventDefault();
            this.applyFilters(new FormData(filterForm));
        });
    }

    applyFilters(filters) {
        // Apply filters to charts and data
        console.log('Applying filters:', Object.fromEntries(filters));
        
        // Update charts based on filters
        Object.values(this.charts).forEach(chart => {
            chart.update();
        });
    }

    initChartInteractions() {
        // Add click handlers to charts for drill-down
        Object.values(this.charts).forEach(chart => {
            chart.canvas.onclick = (evt) => {
                const points = chart.getElementsAtEventForMode(evt, 'nearest', { intersect: true }, true);
                if (points.length) {
                    const firstPoint = points[0];
                    this.handleChartClick(chart, firstPoint);
                }
            };
        });
    }

    handleChartClick(chart, point) {
        const datasetIndex = point.datasetIndex;
        const index = point.index;
        
        console.log('Chart clicked:', { chart, datasetIndex, index });
        
        // Show detailed information about the clicked data point
        this.showDataPointDetails(datasetIndex, index);
    }

    showDataPointDetails(datasetIndex, index) {
        // Create a modal or update a details panel with specific data point information
        const detailsPanel = document.getElementById('dataPointDetails');
        if (detailsPanel) {
            detailsPanel.innerHTML = `
                <h4>Data Point Details</h4>
                <p>Dataset: ${datasetIndex}</p>
                <p>Index: ${index}</p>
                <p>More details would be shown here...</p>
            `;
            detailsPanel.style.display = 'block';
        }
    }

    initDataDrilldown() {
        // Initialize drill-down functionality for hierarchical data exploration
        const drilldownButtons = document.querySelectorAll('.drilldown-btn');
        drilldownButtons.forEach(btn => {
            btn.addEventListener('click', () => {
                const level = btn.getAttribute('data-level');
                this.drillDown(level);
            });
        });
    }

    drillDown(level) {
        console.log('Drilling down to level:', level);
        // Implement drill-down logic based on the specified level
    }

    initRealTimeData() {
        // Set up WebSocket or polling for real-time data updates
        this.setupDataStream();
    }

    setupDataStream() {
        // Simulate real-time data updates
        setInterval(() => {
            this.updateRealTimeData();
        }, 3000);
    }

    updateRealTimeData() {
        // Update charts with new data
        Object.values(this.charts).forEach(chart => {
            // Add new data point or update existing ones
            if (chart.data.datasets.length > 0) {
                const lastDataPoint = chart.data.datasets[0].data[chart.data.datasets[0].data.length - 1];
                const newDataPoint = this.generateNewDataPoint(lastDataPoint);
                
                // Add new point and remove oldest if needed
                chart.data.datasets.forEach(dataset => {
                    dataset.data.push(newDataPoint);
                    if (dataset.data.length > 50) {
                        dataset.data.shift();
                    }
                });
                
                chart.update('quiet');
            }
        });
    }

    generateNewDataPoint(previousPoint) {
        // Generate a new data point based on the previous one with some randomness
        if (typeof previousPoint === 'number') {
            return previousPoint + (Math.random() - 0.5) * 10;
        } else if (typeof previousPoint === 'object' && previousPoint.x !== undefined) {
            return {
                x: previousPoint.x + 1,
                y: previousPoint.y + (Math.random() - 0.5) * 2
            };
        }
        return Math.random() * 100;
    }

    initExportHandlers() {
        // Initialize export functionality for dashboard data
        const exportButtons = document.querySelectorAll('.export-btn');
        exportButtons.forEach(btn => {
            btn.addEventListener('click', () => {
                const format = btn.getAttribute('data-format');
                this.exportDashboardData(format);
            });
        });
    }

    exportDashboardData(format) {
        console.log(`Exporting dashboard data in ${format} format`);
        
        // Implement export logic based on format (PNG, CSV, JSON, etc.)
        switch (format) {
            case 'png':
                this.exportChartsAsPNG();
                break;
            case 'csv':
                this.exportDataAsCSV();
                break;
            case 'json':
                this.exportDataAsJSON();
                break;
        }
    }

    exportChartsAsPNG() {
        Object.entries(this.charts).forEach(([name, chart]) => {
            const link = document.createElement('a');
            link.download = `chart-${name}.png`;
            link.href = chart.toBase64Image();
            link.click();
        });
    }

    exportDataAsCSV() {
        // Export main data as CSV
        const data = this.results.sample_data;
        let csvContent = 'data:text/csv;charset=utf-8,';
        
        // Create CSV header
        const headers = Object.keys(data.normal[0] || {});
        csvContent += headers.join(',') + '\n';
        
        // Add data rows
        [...data.normal, ...data.anomalies].forEach(row => {
            csvContent += headers.map(header => row[header]).join(',') + '\n';
        });
        
        const encodedUri = encodeURI(csvContent);
        const link = document.createElement('a');
        link.setAttribute('href', encodedUri);
        link.setAttribute('download', 'astrovision_data.csv');
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }

    exportDataAsJSON() {
        const dataStr = JSON.stringify(this.results, null, 2);
        const dataBlob = new Blob([dataStr], { type: 'application/json' });
        
        const link = document.createElement('a');
        link.href = URL.createObjectURL(dataBlob);
        link.download = 'astrovision_analysis.json';
        link.click();
        URL.revokeObjectURL(link.href);
    }

    showNoDataMessage() {
        const dashboardContainer = document.querySelector('.dashboard-container');
        if (dashboardContainer) {
            dashboardContainer.innerHTML = `
                <div class="no-data-message">
                    <i class="fas fa-chart-line"></i>
                    <h3>No Analysis Data Available</h3>
                    <p>Please run an analysis first to view the dashboard.</p>
                    <a href="/upload" class="cta-button primary">
                        <i class="fas fa-rocket"></i>
                        Start Analysis
                    </a>
                </div>
            `;
        }
    }
}

// Additional dashboard utilities
const DashboardUtils = {
    createPerformanceMetrics: (results) => {
        // Calculate various performance metrics
        const metrics = {
            accuracy: Math.random() * 0.2 + 0.8, // Simulated accuracy
            precision: Math.random() * 0.2 + 0.75,
            recall: Math.random() * 0.2 + 0.7,
            f1Score: Math.random() * 0.2 + 0.72
        };
        
        return metrics;
    },
    
    generateInsights: (results) => {
        // Generate AI-powered insights based on analysis results
        const insights = [];
        
        const anomalyRate = results.summary.anomaly_percentage;
        if (anomalyRate > 10) {
            insights.push('High anomaly detection rate detected. Consider reviewing data quality.');
        } else if (anomalyRate < 1) {
            insights.push('Low anomaly rate suggests clean, well-behaved data.');
        }
        
        // Add more insight generation logic based on specific patterns
        
        return insights;
    },
    
    formatChartData: (rawData, options = {}) => {
        // Utility function to format data for charts
        const { normalize = false, logScale = false } = options;
        
        let processedData = [...rawData];
        
        if (normalize) {
            const max = Math.max(...processedData);
            processedData = processedData.map(val => val / max);
        }
        
        if (logScale) {
            processedData = processedData.map(val => Math.log10(val + 1));
        }
        
        return processedData;
    }
};

// Initialize dashboard when DOM is loaded
if (document.querySelector('.dashboard-section')) {
    document.addEventListener('DOMContentLoaded', function() {
        new AdvancedDashboard();
    });
}