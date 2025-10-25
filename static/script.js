// Enhanced AstroVision JavaScript - Main Application Logic

class CosmicUI {
    constructor() {
        this.currentStep = 1;
        this.uploadedFile = null;
        this.init();
    }

    init() {
        this.initFileUpload();
        this.initAlgorithmSelection();
        this.initFormSubmission();
        this.initSampleData();
        this.initSmoothScrolling();
        this.initAnimations();
        this.initWorkflowSteps();
        this.initRangeInputs();
        this.initAnalysisConfig();
    }

    initFileUpload() {
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('datafile');
        const filePreview = document.getElementById('filePreview');

        if (!uploadArea) return;

        // Drag and drop functionality
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.style.borderColor = 'var(--cosmic-primary)';
            uploadArea.style.background = 'rgba(0, 206, 209, 0.05)';
        });

        uploadArea.addEventListener('dragleave', (e) => {
            e.preventDefault();
            uploadArea.style.borderColor = '';
            uploadArea.style.background = '';
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.style.borderColor = '';
            uploadArea.style.background = '';
            
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                fileInput.files = files;
                this.handleFileUpload(files[0]);
            }
        });

        fileInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) {
                this.handleFileUpload(file);
            }
        });
    }

    handleFileUpload(file) {
        this.uploadedFile = file;
        this.displayFileInfo(file);
        this.showAnalysisConfig();
        this.updateWorkflowStep(2);
    }

    displayFileInfo(file) {
        const filePreview = document.getElementById('filePreview');
        if (!filePreview) return;

        const fileSize = this.formatFileSize(file.size);
        const fileType = file.type || 'CSV File';

        filePreview.innerHTML = `
            <div class="file-info-content">
                <div class="file-icon">
                    <i class="fas fa-file-csv"></i>
                </div>
                <div class="file-details">
                    <div class="file-name">${file.name}</div>
                    <div class="file-meta">
                        <span><i class="fas fa-weight-hanging"></i> ${fileSize}</span>
                        <span><i class="fas fa-file-code"></i> ${fileType}</span>
                        <span><i class="fas fa-check-circle"></i> Ready for Analysis</span>
                    </div>
                </div>
                <div class="file-status success">
                    <i class="fas fa-check"></i>
                </div>
            </div>
        `;
        filePreview.style.display = 'block';

        // Add success animation
        filePreview.classList.add('file-uploaded');
        setTimeout(() => {
            filePreview.classList.remove('file-uploaded');
        }, 2000);
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    showAnalysisConfig() {
        const analysisConfig = document.getElementById('analysisConfig');
        if (analysisConfig) {
            analysisConfig.style.display = 'block';
            analysisConfig.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    }

    initAnalysisConfig() {
        // Initialize analysis configuration section
        const analyzeNowBtn = document.getElementById('analyzeNowBtn');
        if (analyzeNowBtn) {
            analyzeNowBtn.addEventListener('click', () => {
                this.scrollToAnalysisConfig();
            });
        }
    }

    scrollToAnalysisConfig() {
        const analysisConfig = document.getElementById('analysisConfig');
        if (analysisConfig) {
            analysisConfig.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    }

    initAlgorithmSelection() {
        const algorithmOptions = document.querySelectorAll('input[name="algorithm"]');
        const parameterGroups = document.querySelectorAll('.algorithm-specific');

        algorithmOptions.forEach(option => {
            option.addEventListener('change', (e) => {
                const selectedAlgorithm = e.target.value;
                
                // Hide all parameter groups
                parameterGroups.forEach(group => {
                    group.classList.remove('active');
                });

                // Show parameters for selected algorithm
                const relevantParams = document.querySelectorAll(`[data-algorithm="${selectedAlgorithm}"]`);
                relevantParams.forEach(param => {
                    param.classList.add('active');
                });

                // Update UI state
                this.updateAlgorithmUI(selectedAlgorithm);
            });
        });

        // Initialize with default algorithm
        const defaultAlgorithm = document.querySelector('input[name="algorithm"]:checked');
        if (defaultAlgorithm) {
            defaultAlgorithm.dispatchEvent(new Event('change'));
        }
    }

    updateAlgorithmUI(algorithm) {
        // Update algorithm cards active state
        const algorithmCards = document.querySelectorAll('.algorithm-card-enhanced');
        algorithmCards.forEach(card => {
            card.classList.remove('active');
        });

        const activeCard = document.querySelector(`input[value="${algorithm}"] + label .algorithm-card-enhanced`);
        if (activeCard) {
            activeCard.classList.add('active');
        }

        // Update algorithm description
        this.updateAlgorithmDescription(algorithm);
    }

    updateAlgorithmDescription(algorithm) {
        const descriptions = {
            'isolationforest': 'Isolation Forest efficiently isolates anomalies by randomly selecting features and split values. It works well for high-dimensional datasets and has low computational complexity.',
            'oneclasssvm': 'One-Class SVM learns a decision boundary around the normal data. Points outside this boundary are considered anomalies. Effective for novelty detection.',
            'localoutlierfactor': 'LOF compares the local density of a point to its neighbors. Points with significantly lower density are flagged as anomalies.',
            'dbscan': 'DBSCAN identifies dense regions and marks points in low-density areas as noise/anomalies. Good for data with clusters.',
            'ellipticenvelope': 'Elliptic Envelope fits a robust covariance estimate to the data, assuming the inlier data is Gaussian distributed.',
            'randomforest': 'Random Forest uses ensemble learning to detect anomalies by learning the data distribution and identifying outliers.'
        };

        const descElement = document.getElementById('algorithmDescription');
        if (descElement && descriptions[algorithm]) {
            descElement.textContent = descriptions[algorithm];
        }
    }

    initFormSubmission() {
        const uploadForm = document.getElementById('uploadForm');
        const analyzeBtn = document.getElementById('analyzeBtn');

        if (!uploadForm) return;

        uploadForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            if (!this.uploadedFile) {
                this.showNotification('Please upload a dataset first.', 'error');
                return;
            }

            const formData = new FormData(uploadForm);
            
            this.setLoadingState(true);
            this.updateWorkflowStep(3);

            try {
                const response = await fetch('/analyze', {
                    method: 'POST',
                    body: formData
                });

                const result = await response.json();

                if (result.success) {
                    this.showNotification('Cosmic analysis initiated successfully! Processing your data...', 'success');
                    this.updateWorkflowStep(4);
                    
                    // Add a small delay for better UX
                    setTimeout(() => {
                        window.location.href = result.redirect;
                    }, 2000);
                    
                } else {
                    this.showNotification(result.error || 'An unexpected error occurred during analysis.', 'error');
                    this.updateWorkflowStep(2);
                }
            } catch (error) {
                this.showNotification('Network error. Please check your connection and try again.', 'error');
                console.error('Analysis error:', error);
                this.updateWorkflowStep(2);
            } finally {
                this.setLoadingState(false);
            }
        });
    }

    setLoadingState(loading) {
        const analyzeBtn = document.getElementById('analyzeBtn');
        if (!analyzeBtn) return;

        if (loading) {
            analyzeBtn.disabled = true;
            analyzeBtn.classList.add('loading');
            analyzeBtn.querySelector('.button-text').textContent = 'Analyzing Cosmic Data...';
        } else {
            analyzeBtn.disabled = false;
            analyzeBtn.classList.remove('loading');
            analyzeBtn.querySelector('.button-text').textContent = 'Launch Cosmic Analysis';
        }
    }

    initSampleData() {
        const generateSampleBtn = document.getElementById('generateSample');
        if (!generateSampleBtn) return;

        generateSampleBtn.addEventListener('click', async (e) => {
            e.preventDefault();
            
            this.showNotification('Generating sample astronomical dataset...', 'info');
            
            try {
                // Create sample data
                const sampleData = this.generateSampleDataset();
                this.downloadCSV(sampleData, 'sample_astronomical_data.csv');
                
                this.showNotification('Sample dataset downloaded successfully!', 'success');
            } catch (error) {
                this.showNotification('Error generating sample data.', 'error');
                console.error('Sample data error:', error);
            }
        });
    }

    generateSampleDataset() {
        // Generate realistic astronomical sample data
        const headers = ['timestamp', 'right_ascension', 'declination', 'brightness', 'luminosity', 'temperature', 'radiation_level', 'redshift', 'proper_motion', 'parallax'];
        
        const data = [];
        const baseDate = new Date();
        
        for (let i = 0; i < 1000; i++) {
            const row = {
                timestamp: new Date(baseDate.getTime() - i * 3600000).toISOString(),
                right_ascension: (Math.random() * 360).toFixed(6),
                declination: (Math.random() * 180 - 90).toFixed(6),
                brightness: (2000 + Math.random() * 1000).toFixed(2),
                luminosity: (1e6 + Math.random() * 5e5).toFixed(2),
                temperature: (5000 + Math.random() * 2000).toFixed(2),
                radiation_level: (80 + Math.random() * 40).toFixed(2),
                redshift: (Math.random() * 0.2).toFixed(6),
                proper_motion: (Math.random() * 20).toFixed(4),
                parallax: (5 + Math.random() * 10).toFixed(4)
            };
            
            // Add some anomalies (5% of data)
            if (Math.random() < 0.05) {
                row.brightness = (Math.random() * 10000).toFixed(2); // Extreme brightness
                row.radiation_level = (Math.random() * 500).toFixed(2); // High radiation
            }
            
            data.push(row);
        }
        
        return { headers, data };
    }

    downloadCSV(data, filename) {
        const csvContent = [
            data.headers.join(','),
            ...data.data.map(row => data.headers.map(header => row[header]).join(','))
        ].join('\n');

        const blob = new Blob([csvContent], { type: 'text/csv' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.style.display = 'none';
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
    }

    initWorkflowSteps() {
        // Initialize workflow step management
        this.updateWorkflowStep(1);
    }

    updateWorkflowStep(step) {
        this.currentStep = step;
        const steps = document.querySelectorAll('.workflow-steps .step');
        
        steps.forEach((stepElement, index) => {
            if (index + 1 <= step) {
                stepElement.classList.add('active');
            } else {
                stepElement.classList.remove('active');
            }
        });
    }

    initSmoothScrolling() {
        // Smooth scrolling for anchor links
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', function (e) {
                e.preventDefault();
                const target = document.querySelector(this.getAttribute('href'));
                if (target) {
                    target.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                }
            });
        });
    }

    initAnimations() {
        // Intersection Observer for scroll animations
        const observerOptions = {
            threshold: 0.1,
            rootMargin: '0px 0px -50px 0px'
        };

        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('animate-in');
                }
            });
        }, observerOptions);

        // Observe elements for animation
        document.querySelectorAll('.feature-card, .algorithm-card, .stat-card, .dashboard-card').forEach(el => {
            observer.observe(el);
        });
    }

    initRangeInputs() {
        const rangeInputs = document.querySelectorAll('input[type="range"]');
        rangeInputs.forEach(input => {
            const valueDisplay = input.nextElementSibling?.querySelector('span');
            if (valueDisplay) {
                // Initial display
                valueDisplay.textContent = input.value;
                
                // Update on input
                input.addEventListener('input', () => {
                    valueDisplay.textContent = input.value;
                    
                    // Add visual feedback
                    const percent = (input.value - input.min) / (input.max - input.min) * 100;
                    input.style.background = `linear-gradient(to right, var(--cosmic-primary) 0%, var(--cosmic-primary) ${percent}%, rgba(255, 255, 255, 0.1) ${percent}%, rgba(255, 255, 255, 0.1) 100%)`;
                });
                
                // Initialize gradient
                const percent = (input.value - input.min) / (input.max - input.min) * 100;
                input.style.background = `linear-gradient(to right, var(--cosmic-primary) 0%, var(--cosmic-primary) ${percent}%, rgba(255, 255, 255, 0.1) ${percent}%, rgba(255, 255, 255, 0.1) 100%)`;
            }
        });
    }

    showNotification(message, type = 'info') {
        // Remove existing notification
        const existingNotification = document.getElementById('notification');
        if (existingNotification) {
            existingNotification.remove();
        }

        // Create new notification
        const notification = document.createElement('div');
        notification.id = 'notification';
        notification.className = `notification ${type}`;
        notification.textContent = message;

        document.body.appendChild(notification);

        // Show notification
        setTimeout(() => {
            notification.classList.add('show');
        }, 100);

        // Auto hide after 5 seconds
        setTimeout(() => {
            notification.classList.remove('show');
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.parentNode.removeChild(notification);
                }
            }, 300);
        }, 5000);
    }

    // Utility methods for analysis page
    initAnalysisPage() {
        if (!document.querySelector('.analysis-content')) return;

        this.initAnalysisTabs();
        this.initDataTables();
        this.initChartInteractions();
    }

    initAnalysisTabs() {
        const tabButtons = document.querySelectorAll('.tab-button');
        const tabContents = document.querySelectorAll('.tab-content');

        tabButtons.forEach(button => {
            button.addEventListener('click', () => {
                const tabId = button.getAttribute('data-tab');
                
                // Update active tab button
                tabButtons.forEach(btn => btn.classList.remove('active'));
                button.classList.add('active');
                
                // Show corresponding content
                tabContents.forEach(content => {
                    content.classList.remove('active');
                    if (content.id === `${tabId}-tab`) {
                        content.classList.add('active');
                    }
                });

                // Trigger custom event for tab change
                this.onAnalysisTabChange(tabId);
            });
        });

        // Data explorer tabs
        const dataTabButtons = document.querySelectorAll('.data-tab-button');
        const dataContents = document.querySelectorAll('.data-content');

        dataTabButtons.forEach(button => {
            button.addEventListener('click', () => {
                const dataTabId = button.getAttribute('data-data-tab');
                
                dataTabButtons.forEach(btn => btn.classList.remove('active'));
                button.classList.add('active');
                
                dataContents.forEach(content => {
                    content.classList.remove('active');
                    if (content.id === `${dataTabId}-data`) {
                        content.classList.add('active');
                    }
                });
            });
        });
    }

    onAnalysisTabChange(tabId) {
        // Load specific content when tabs change
        switch (tabId) {
            case 'visualizations':
                this.loadVisualizations();
                break;
            case 'insights':
                this.loadAdvancedInsights();
                break;
            case 'data':
                this.initDataTableSorting();
                break;
        }
    }

    loadVisualizations() {
        // Initialize or update visualizations when tab becomes active
        console.log('Loading visualizations...');
        // This will be handled by the dashboard.js file
    }

    loadAdvancedInsights() {
        // Load additional insights or analysis
        console.log('Loading advanced insights...');
    }

    initDataTables() {
        // Add basic table enhancements
        const tables = document.querySelectorAll('.data-table');
        tables.forEach(table => {
            this.enhanceDataTable(table);
        });
    }

    enhanceDataTable(table) {
        // Add hover effects and basic styling
        const rows = table.querySelectorAll('tbody tr');
        rows.forEach(row => {
            row.addEventListener('mouseenter', () => {
                row.style.backgroundColor = 'rgba(0, 206, 209, 0.05)';
            });
            
            row.addEventListener('mouseleave', () => {
                row.style.backgroundColor = '';
            });
        });
    }

    initDataTableSorting() {
        // Initialize sorting functionality for data tables
        const tables = document.querySelectorAll('.data-table');
        tables.forEach(table => {
            this.addTableSorting(table);
        });
    }

    addTableSorting(table) {
        const headers = table.querySelectorAll('th');
        headers.forEach((header, index) => {
            header.style.cursor = 'pointer';
            header.addEventListener('click', () => {
                this.sortTableByColumn(table, index);
            });
        });
    }

    sortTableByColumn(table, columnIndex) {
        const tbody = table.querySelector('tbody');
        const rows = Array.from(tbody.querySelectorAll('tr'));
        const isNumeric = this.isColumnNumeric(rows, columnIndex);
        
        rows.sort((a, b) => {
            const aValue = a.cells[columnIndex].textContent.trim();
            const bValue = b.cells[columnIndex].textContent.trim();
            
            if (isNumeric) {
                return parseFloat(aValue) - parseFloat(bValue);
            } else {
                return aValue.localeCompare(bValue);
            }
        });

        // Remove existing rows
        rows.forEach(row => tbody.removeChild(row));
        
        // Add sorted rows
        rows.forEach(row => tbody.appendChild(row));
    }

    isColumnNumeric(rows, columnIndex) {
        if (rows.length === 0) return false;
        const sampleValue = rows[0].cells[columnIndex].textContent.trim();
        return !isNaN(sampleValue) && !isNaN(parseFloat(sampleValue));
    }

    initChartInteractions() {
        // Initialize chart interaction handlers
        console.log('Initializing chart interactions...');
    }
}

// Analysis Page Manager
class AnalysisPageManager {
    constructor() {
        this.results = window.analysisResults;
        this.init();
    }

    init() {
        if (!this.results) {
            this.showNoDataMessage();
            return;
        }

        this.initQuickStats();
        this.initExportHandlers();
        this.initRealTimeUpdates();
    }

    initQuickStats() {
        // Update any dynamic stats or metrics
        const statElements = document.querySelectorAll('.stat-number, .stat-value');
        statElements.forEach(element => {
            if (element.textContent.includes('%')) {
                this.animateValue(element, 0, parseFloat(element.textContent), 2000);
            }
        });
    }

    animateValue(element, start, end, duration) {
        let startTimestamp = null;
        const step = (timestamp) => {
            if (!startTimestamp) startTimestamp = timestamp;
            const progress = Math.min((timestamp - startTimestamp) / duration, 1);
            const value = progress * (end - start) + start;
            element.textContent = value.toFixed(2) + '%';
            if (progress < 1) {
                window.requestAnimationFrame(step);
            }
        };
        window.requestAnimationFrame(step);
    }

    initExportHandlers() {
        // Initialize export functionality
        const exportButtons = document.querySelectorAll('[data-export]');
        exportButtons.forEach(button => {
            button.addEventListener('click', (e) => {
                const format = button.getAttribute('data-export');
                this.handleExport(format);
            });
        });
    }

    handleExport(format) {
        switch (format) {
            case 'pdf':
                window.open('/generate_report', '_blank');
                break;
            case 'csv':
                window.open('/export_anomalies', '_blank');
                break;
            case 'json':
                this.exportAsJSON();
                break;
        }
    }

    exportAsJSON() {
        const dataStr = JSON.stringify(this.results, null, 2);
        const dataBlob = new Blob([dataStr], { type: 'application/json' });
        const url = URL.createObjectURL(dataBlob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `astrovision_analysis_${new Date().getTime()}.json`;
        link.click();
        URL.revokeObjectURL(url);
    }

    initRealTimeUpdates() {
        // Simulate real-time updates for dashboard
        setInterval(() => {
            this.updateLiveMetrics();
        }, 5000);
    }

    updateLiveMetrics() {
        // Update any live metrics
        const liveElements = document.querySelectorAll('.live-metric');
        liveElements.forEach(element => {
            const currentValue = parseFloat(element.textContent);
            const fluctuation = (Math.random() - 0.5) * 0.02; // ±1% fluctuation
            const newValue = currentValue * (1 + fluctuation);
            element.textContent = newValue.toFixed(2);
        });
    }

    showNoDataMessage() {
        const container = document.querySelector('.analysis-content, .dashboard-content');
        if (container) {
            container.innerHTML = `
                <div class="no-results">
                    <div class="no-results-icon">
                        <i class="fas fa-search"></i>
                    </div>
                    <h2>No Analysis Data Available</h2>
                    <p>Please upload a dataset and run anomaly detection to view results.</p>
                    <a href="/upload" class="cta-button primary">
                        <i class="fas fa-upload"></i>
                        Start New Analysis
                    </a>
                </div>
            `;
        }
    }
}

// Utility functions
const AstroUtils = {
    // Format numbers with commas and decimal places
    formatNumber: (num, decimals = 2) => {
        return parseFloat(num).toLocaleString('en-US', {
            minimumFractionDigits: decimals,
            maximumFractionDigits: decimals
        });
    },
    
    // Format percentages
    formatPercentage: (num, decimals = 2) => {
        return `${parseFloat(num).toFixed(decimals)}%`;
    },
    
    // Generate random color
    randomColor: () => {
        const colors = [
            '#00CED1', '#4A90E2', '#7B68EE', '#FF6B6B', '#FFD93D', '#6BCF7F'
        ];
        return colors[Math.floor(Math.random() * colors.length)];
    },
    
    // Debounce function for performance
    debounce: (func, wait) => {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    },
    
    // Throttle function for performance
    throttle: (func, limit) => {
        let inThrottle;
        return function(...args) {
            if (!inThrottle) {
                func.apply(this, args);
                inThrottle = true;
                setTimeout(() => inThrottle = false, limit);
            }
        };
    },
    
    // Download data as file
    downloadFile: (data, filename, type = 'text/plain') => {
        const file = new Blob([data], { type: type });
        const a = document.createElement('a');
        const url = URL.createObjectURL(file);
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        setTimeout(() => {
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);
        }, 0);
    }
};

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Initialize main UI controller
    window.cosmicUI = new CosmicUI();
    
    // Initialize analysis page if we're on analysis page
    if (document.querySelector('.analysis-content')) {
        window.analysisManager = new AnalysisPageManager();
        window.cosmicUI.initAnalysisPage();
    }
    
    // Initialize dashboard if we're on dashboard page
    if (document.querySelector('.dashboard-section')) {
        // Dashboard initialization will be handled by dashboard.js
        console.log('Dashboard page loaded');
    }

    // Add CSS for animations
    const style = document.createElement('style');
    style.textContent = `
        .feature-card, .algorithm-card, .stat-card, .dashboard-card {
            opacity: 0;
            transform: translateY(30px);
            transition: all 0.6s ease;
        }
        
        .feature-card.animate-in, 
        .algorithm-card.animate-in, 
        .stat-card.animate-in,
        .dashboard-card.animate-in {
            opacity: 1;
            transform: translateY(0);
        }
        
        .file-uploaded {
            animation: fileUploaded 0.5s ease;
        }
        
        @keyframes fileUploaded {
            0% { transform: scale(1); }
            50% { transform: scale(1.05); }
            100% { transform: scale(1); }
        }
        
        .live-metric {
            transition: all 0.3s ease;
        }
        
        .algorithm-card-enhanced.active {
            border-color: var(--cosmic-primary) !important;
            background: rgba(0, 206, 209, 0.1) !important;
            box-shadow: var(--glow-primary) !important;
        }
        
        .file-info-content {
            display: flex;
            align-items: center;
            gap: 1rem;
        }
        
        .file-icon {
            font-size: 2rem;
            color: var(--cosmic-primary);
        }
        
        .file-details {
            flex: 1;
        }
        
        .file-name {
            font-weight: 600;
            color: var(--text-primary);
            margin-bottom: 0.5rem;
        }
        
        .file-meta {
            display: flex;
            gap: 1rem;
            flex-wrap: wrap;
        }
        
        .file-meta span {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            color: var(--text-secondary);
            font-size: 0.9rem;
        }
        
        .file-status {
            padding: 0.5rem;
            border-radius: 50%;
            color: white;
        }
        
        .file-status.success {
            background: var(--cosmic-success);
        }
        
        input[type="range"] {
            -webkit-appearance: none;
            width: 100%;
            height: 6px;
            border-radius: 3px;
            outline: none;
        }
        
        input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 18px;
            height: 18px;
            border-radius: 50%;
            background: var(--cosmic-primary);
            cursor: pointer;
            border: 2px solid white;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.3);
        }
    `;
    document.head.appendChild(style);

    // Add loading state management
    document.addEventListener('astrovision:loading-start', () => {
        document.body.style.cursor = 'wait';
    });
    
    document.addEventListener('astrovision:loading-end', () => {
        document.body.style.cursor = 'default';
    });
});

// Make utilities globally available
window.AstroUtils = AstroUtils;

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { CosmicUI, AnalysisPageManager, AstroUtils };
}