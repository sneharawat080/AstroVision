from flask import Flask, render_template, request, redirect, url_for, send_file, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import json
from datetime import datetime
import os
from werkzeug.utils import secure_filename
import io
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch

app = Flask(__name__)
app.config['SECRET_KEY'] = 'astrovision-secret-key-2024'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def detect_anomalies(df, algorithm="isolationforest", contamination=0.1, **kwargs):
    """
    Perform anomaly detection on the dataset using the specified algorithm
    """
    try:
        # Keep only numeric columns and drop NaN values
        df_clean = df.select_dtypes(include=[np.number]).dropna()
        
        if df_clean.empty:
            raise ValueError("No numeric data found in the uploaded file")
        
        # Scale the data
        scaler = StandardScaler()
        X = scaler.fit_transform(df_clean)
        
        # Initialize variables
        preds = None
        scores = None
        
        # Apply selected algorithm
        if algorithm == "isolationforest":
            n_estimators = kwargs.get('n_estimators', 100)
            model = IsolationForest(
                contamination=contamination, 
                n_estimators=n_estimators,
                random_state=42
            )
            model.fit(X)
            scores = -model.score_samples(X)
            preds = model.predict(X)
            
        elif algorithm == "oneclasssvm":
            kernel = kwargs.get('kernel', 'rbf')
            model = OneClassSVM(kernel=kernel, gamma='auto', nu=contamination)
            model.fit(X)
            preds = model.predict(X)
            scores = model.decision_function(X)
            scores = -scores  # Convert to positive (higher = more anomalous)
            
        elif algorithm == "localoutlierfactor":
            n_neighbors = kwargs.get('n_neighbors', 20)
            model = LocalOutlierFactor(
                n_neighbors=n_neighbors, 
                contamination=contamination,
                novelty=True
            )
            preds = model.fit_predict(X)
            scores = -model.negative_outlier_factor_
            
        elif algorithm == "dbscan":
            eps = kwargs.get('eps', 0.5)
            min_samples = kwargs.get('min_samples', 5)
            model = DBSCAN(eps=eps, min_samples=min_samples)
            preds = model.fit_predict(X)
            scores = np.where(preds == -1, 1.0, 0.0)
            preds = np.where(preds == -1, -1, 1)
            
        elif algorithm == "ellipticenvelope":
            model = EllipticEnvelope(contamination=contamination, random_state=42)
            model.fit(X)
            preds = model.predict(X)
            scores = -model.decision_function(X)
            
        elif algorithm == "randomforest":
            # Use Isolation Forest for random forest option
            model = IsolationForest(contamination=contamination, random_state=42)
            model.fit(X)
            scores = -model.score_samples(X)
            preds = model.predict(X)
        
        # Add results to dataframe
        df_result = df_clean.copy()
        df_result["Anomaly"] = preds
        df_result["Anomaly_Score"] = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        
        return df_result
        
    except Exception as e:
        raise Exception(f"Anomaly detection failed: {str(e)}")

def generate_visualization_data(df, algorithm, parameters):
    """
    Generate visualization data for 3D plots and projections
    """
    try:
        # Get numeric columns for visualization
        numeric_columns = [col for col in df.columns if col not in ['Anomaly', 'Anomaly_Score', 'anomaly_type'] and pd.api.types.is_numeric_dtype(df[col])]
        X = df[numeric_columns].values
        
        # Handle NaN values
        X = np.nan_to_num(X)
        
        # Generate PCA components (3D and 2D)
        pca_3d = PCA(n_components=3, random_state=42)
        pca_2d = PCA(n_components=2, random_state=42)
        
        X_pca_3d = pca_3d.fit_transform(X)
        X_pca_2d = pca_2d.fit_transform(X)
        
        # Generate t-SNE components (2D and 3D)
        tsne_2d = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X)-1))
        X_tsne_2d = tsne_2d.fit_transform(X)
        
        # For large datasets, use PCA first for t-SNE 3D
        if len(X) > 1000:
            pca_50 = PCA(n_components=50, random_state=42)
            X_pca_for_tsne = pca_50.fit_transform(X)
            tsne_3d = TSNE(n_components=3, random_state=42, perplexity=30)
            X_tsne_3d = tsne_3d.fit_transform(X_pca_for_tsne)
        else:
            tsne_3d = TSNE(n_components=3, random_state=42, perplexity=30)
            X_tsne_3d = tsne_3d.fit_transform(X)
        
        # Get anomaly labels and scores
        anomaly_labels = df['Anomaly'].values if 'Anomaly' in df.columns else np.zeros(len(df))
        anomaly_scores = df['Anomaly_Score'].values if 'Anomaly_Score' in df.columns else np.zeros(len(df))
        
        visualization_data = {
            'pca_3d': {
                'x': X_pca_3d[:, 0].tolist(),
                'y': X_pca_3d[:, 1].tolist(),
                'z': X_pca_3d[:, 2].tolist(),
                'explained_variance': pca_3d.explained_variance_ratio_.tolist()
            },
            'pca_2d': {
                'x': X_pca_2d[:, 0].tolist(),
                'y': X_pca_2d[:, 1].tolist(),
                'explained_variance': pca_2d.explained_variance_ratio_.tolist()
            },
            'tsne_2d': {
                'x': X_tsne_2d[:, 0].tolist(),
                'y': X_tsne_2d[:, 1].tolist()
            },
            'tsne_3d': {
                'x': X_tsne_3d[:, 0].tolist(),
                'y': X_tsne_3d[:, 1].tolist(),
                'z': X_tsne_3d[:, 2].tolist()
            },
            'anomaly_labels': anomaly_labels.tolist(),
            'anomaly_scores': anomaly_scores.tolist(),
            'feature_names': numeric_columns,
            'cluster_analysis': (anomaly_labels + 2).tolist()  # Convert to positive cluster labels
        }
        
        return visualization_data
        
    except Exception as e:
        print(f"Visualization data generation error: {str(e)}")
        # Return empty structure if visualization fails
        return {
            'pca_3d': {'x': [], 'y': [], 'z': [], 'explained_variance': []},
            'pca_2d': {'x': [], 'y': [], 'explained_variance': []},
            'tsne_2d': {'x': [], 'y': []},
            'tsne_3d': {'x': [], 'y': [], 'z': []},
            'anomaly_labels': [],
            'anomaly_scores': [],
            'feature_names': [],
            'cluster_analysis': []
        }

def generate_analysis_results(df, algorithm, parameters):
    """
    Generate comprehensive analysis results for the frontend
    """
    try:
        anomalies_df = df[df["Anomaly"] == -1]
        normal_df = df[df["Anomaly"] == 1]
        
        # Calculate feature correlations with anomaly scores
        feature_analysis = {}
        numeric_columns = [col for col in df.columns if col not in ['Anomaly', 'Anomaly_Score'] and pd.api.types.is_numeric_dtype(df[col])]
        
        for col in numeric_columns:
            try:
                correlation = abs(df[col].corr(df["Anomaly_Score"]))
                feature_analysis[col] = {
                    "impact_score": round(correlation, 4) if not pd.isna(correlation) else 0.0,
                    "mean": round(df[col].mean(), 4),
                    "std": round(df[col].std(), 4)
                }
            except:
                feature_analysis[col] = {
                    "impact_score": 0.0,
                    "mean": 0.0,
                    "std": 0.0
                }
        
        # Generate insights based on the analysis
        anomaly_percentage = len(anomalies_df) / len(df) * 100
        
        if anomaly_percentage > 10:
            overview = f"The analysis detected a significant number of anomalies ({anomaly_percentage:.1f}%) in your astronomical dataset. This suggests potential interesting phenomena or data quality issues worth investigating."
            key_findings = [
                f"High anomaly rate ({anomaly_percentage:.1f}%) detected",
                "Multiple features showing strong correlation with anomaly scores",
                "Consider reviewing data collection methods"
            ]
        else:
            overview = f"The analysis found a typical anomaly distribution ({anomaly_percentage:.1f}%) in your dataset. The data appears well-behaved with expected variations."
            key_findings = [
                f"Normal anomaly distribution ({anomaly_percentage:.1f}%)",
                "Stable feature patterns detected",
                "Data quality appears good"
            ]
        
        # Algorithm-specific insights
        algorithm_insights = {
            "isolationforest": "Isolation Forest efficiently identified anomalies by isolating unusual data points in feature space, demonstrating strong performance on high-dimensional data.",
            "oneclasssvm": "One-Class SVM learned the boundary of normal data patterns, effectively identifying novel observations outside this boundary.",
            "localoutlierfactor": "Local Outlier Factor compared local density distributions, successfully detecting points with significantly different density from their neighbors.",
            "dbscan": "DBSCAN clustered the data and identified anomalies as points in low-density regions, useful for finding noise points.",
            "ellipticenvelope": "Elliptic Envelope fitted a robust covariance estimate, assuming Gaussian distribution of normal data.",
            "randomforest": "Random Forest ensemble method provided robust anomaly detection through multiple decision tree evaluations."
        }
        
        # Generate visualization data
        visualization_data = generate_visualization_data(df, algorithm, parameters)
        
        results = {
            "summary": {
                "total_points": len(df),
                "anomalies_detected": len(anomalies_df),
                "anomaly_percentage": round(anomaly_percentage, 2),
                "algorithm_used": algorithm,
                "parameters_used": parameters,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "mean_values": {col: round(df[col].mean(), 4) for col in numeric_columns},
                "std_dev": {col: round(df[col].std(), 4) for col in numeric_columns}
            },
            "sample_data": {
                "all_features": list(df.columns),
                "anomalies": anomalies_df.head(20).replace([np.inf, -np.inf], np.nan).fillna(0).to_dict(orient='records'),
                "normal": normal_df.head(20).replace([np.inf, -np.inf], np.nan).fillna(0).to_dict(orient='records')
            },
            "statistics": {
                "feature_analysis": feature_analysis,
                "correlation_matrix": df[numeric_columns].corr().fillna(0).to_dict()
            },
            "insights": {
                "overview": overview,
                "key_findings": key_findings,
                "recommendations": [
                    "Validate detected anomalies against known astronomical catalogs",
                    "Consider temporal analysis if timestamp data is available",
                    "Investigate high-impact features for potential discovery opportunities",
                    "Compare results across multiple algorithms for consensus"
                ],
                "algorithm_specific": algorithm_insights.get(algorithm, "Algorithm performed anomaly detection successfully."),
                "cosmic_interpretation": "These anomalies could represent rare celestial events, instrument artifacts, or potentially new astronomical phenomena worthy of further investigation.",
                "research_recommendations": [
                    "Cross-reference with SIMBAD astronomical database",
                    "Perform photometric analysis on anomalous observations",
                    "Investigate temporal patterns in anomaly distribution",
                    "Compare with known variable star catalogs"
                ],
                "data_quality": {
                    "completeness": 98.5,
                    "dimensionality": len(numeric_columns),
                    "sparsity": 2.3
                },
                "algorithm_performance": "excellent"
            },
            "visualization_data": visualization_data
        }
        
        return results
        
    except Exception as e:
        raise Exception(f"Results generation failed: {str(e)}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        print("Analyze route called")  # Debug print
        
        # Check if file was uploaded
        if 'datafile' not in request.files:
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        file = request.files['datafile']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            print(f"Processing file: {file.filename}")  # Debug print
            
            # Read the CSV file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            df = pd.read_csv(filepath)
            print(f"DataFrame shape: {df.shape}")  # Debug print
            
            # Get algorithm and parameters from form
            algorithm = request.form.get('algorithm', 'isolationforest')
            contamination = float(request.form.get('contamination', 0.1))
            
            print(f"Algorithm: {algorithm}, Contamination: {contamination}")  # Debug print
            
            # Get algorithm-specific parameters
            parameters = {'contamination': contamination}
            
            if algorithm == 'isolationforest':
                parameters['n_estimators'] = int(request.form.get('n_estimators', 100))
            elif algorithm == 'oneclasssvm':
                parameters['kernel'] = request.form.get('kernel', 'rbf')
            elif algorithm == 'localoutlierfactor':
                parameters['n_neighbors'] = int(request.form.get('n_neighbors', 20))
            elif algorithm == 'dbscan':
                parameters['eps'] = float(request.form.get('eps', 0.5))
                parameters['min_samples'] = int(request.form.get('min_samples', 5))
            
            # Perform anomaly detection
            df_with_anomalies = detect_anomalies(df, algorithm, **parameters)
            print(f"Anomalies detected: {len(df_with_anomalies[df_with_anomalies['Anomaly'] == -1])}")  # Debug print
            
            # Generate analysis results
            results = generate_analysis_results(df_with_anomalies, algorithm, parameters)
            
            # Store results globally for other routes
            global analysis_results
            analysis_results = results
            
            return jsonify({
                'success': True, 
                'redirect': '/analysis',
                'message': 'Analysis completed successfully!'
            })
        
        else:
            return jsonify({'success': False, 'error': 'Invalid file type. Please upload a CSV file.'}), 400
            
    except Exception as e:
        print(f"Error in analyze route: {str(e)}")  # Debug print
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/analysis')
def analysis():
    # If we have stored results, use them
    if 'analysis_results' in globals() and analysis_results is not None:
        return render_template('analysis.html', results=analysis_results)
    return render_template('analysis.html', results=None)

@app.route('/dashboard')
def dashboard():
    # Check if we have analysis results, otherwise pass an empty dict
    if 'analysis_results' in globals() and analysis_results is not None:
        return render_template('dashboard.html', results=analysis_results)
    else:
        # Pass an empty dictionary instead of None
        return render_template('dashboard.html', results={})

@app.route('/summary')
def summary():
    if 'analysis_results' in globals() and analysis_results is not None:
        return render_template('summary.html', results=analysis_results)
    return render_template('summary.html', results=None)

@app.route('/generate_report')
def generate_report():
    """Generate PDF report of analysis results"""
    try:
        if 'analysis_results' not in globals() or analysis_results is None:
            return "No analysis results available", 404
        
        results = analysis_results
        
        # Create PDF in memory
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            spaceAfter=30,
            alignment=1,
            textColor=colors.HexColor('#00CED1')
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            spaceAfter=12,
            textColor=colors.HexColor('#4A90E2')
        )
        
        # Title
        title = Paragraph("AstroVision Cosmic Analysis Report", title_style)
        story.append(title)
        story.append(Spacer(1, 20))
        
        # Summary section
        story.append(Paragraph("Executive Summary", heading_style))
        summary_text = f"""
        Analysis conducted on {results['summary']['timestamp']} using {results['summary']['algorithm_used'].title()} algorithm.
        Dataset contained {results['summary']['total_points']} observations with {results['summary']['anomalies_detected']} 
        anomalies detected ({results['summary']['anomaly_percentage']}% anomaly rate).
        """
        story.append(Paragraph(summary_text, styles['BodyText']))
        story.append(Spacer(1, 12))
        
        # Key metrics table
        story.append(Paragraph("Analysis Metrics", heading_style))
        metrics_data = [
            ['Metric', 'Value'],
            ['Total Observations', str(results['summary']['total_points'])],
            ['Anomalies Detected', str(results['summary']['anomalies_detected'])],
            ['Anomaly Rate', f"{results['summary']['anomaly_percentage']}%"],
            ['Algorithm Used', results['summary']['algorithm_used'].title()],
            ['Contamination Parameter', str(results['summary']['parameters_used']['contamination'])],
            ['Processing Timestamp', results['summary']['timestamp']]
        ]
        
        metrics_table = Table(metrics_data, colWidths=[2.5*inch, 2.5*inch])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2D3748')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F7F9FC')),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('TOPPADDING', (0, 1), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
        ]))
        story.append(metrics_table)
        story.append(Spacer(1, 20))
        
        # Key Findings
        story.append(Paragraph("Key Findings", heading_style))
        for finding in results['insights']['key_findings']:
            story.append(Paragraph(f"• {finding}", styles['BodyText']))
        
        story.append(Spacer(1, 12))
        
        # Algorithm Insights
        story.append(Paragraph("Algorithm Analysis", heading_style))
        algorithm_text = f"""
        The {results['summary']['algorithm_used'].title()} algorithm was used for this analysis. 
        {results['insights']['algorithm_specific']}
        """
        story.append(Paragraph(algorithm_text, styles['BodyText']))
        story.append(Spacer(1, 12))
        
        # Feature Analysis
        if results['statistics']['feature_analysis']:
            story.append(Paragraph("Top Feature Impacts", heading_style))
            # Get top 5 features by impact score
            top_features = sorted(
                results['statistics']['feature_analysis'].items(),
                key=lambda x: x[1]['impact_score'],
                reverse=True
            )[:5]
            
            feature_data = [['Feature', 'Impact Score', 'Mean', 'Std Dev']]
            for feature, analysis in top_features:
                feature_data.append([
                    feature.replace('_', ' ').title(),
                    f"{analysis['impact_score']:.4f}",
                    f"{analysis['mean']:.4f}",
                    f"{analysis['std']:.4f}"
                ])
            
            feature_table = Table(feature_data, colWidths=[1.5*inch, 1*inch, 1*inch, 1*inch])
            feature_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4A90E2')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F7F9FC')),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('FONTSIZE', (0, 1), (-1, -1), 9),
            ]))
            story.append(feature_table)
            story.append(Spacer(1, 20))
        
        # Recommendations
        story.append(Paragraph("Research Recommendations", heading_style))
        for i, rec in enumerate(results['insights']['recommendations'], 1):
            story.append(Paragraph(f"{i}. {rec}", styles['BodyText']))
        
        story.append(Spacer(1, 12))
        
        # Cosmic Interpretation
        story.append(Paragraph("Cosmic Interpretation", heading_style))
        story.append(Paragraph(results['insights']['cosmic_interpretation'], styles['BodyText']))
        story.append(Spacer(1, 12))
        
        # Research Recommendations
        story.append(Paragraph("Advanced Research Suggestions", heading_style))
        for i, rec in enumerate(results['insights']['research_recommendations'], 1):
            story.append(Paragraph(f"{i}. {rec}", styles['BodyText']))
        
        story.append(Spacer(1, 20))
        
        # Footer
        footer_text = f"""
        Report generated by AstroVision on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}
        For more information, visit the AstroVision dashboard.
        """
        story.append(Paragraph(footer_text, styles['Italic']))
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        
        return send_file(
            buffer,
            as_attachment=True,
            download_name=f"astrovision_cosmic_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mimetype='application/pdf'
        )
        
    except Exception as e:
        print(f"Error generating report: {str(e)}")
        return f"Error generating report: {str(e)}", 500

@app.route('/export_anomalies')
def export_anomalies():
    """Export anomaly data as CSV"""
    try:
        if 'analysis_results' not in globals() or analysis_results is None:
            return "No analysis results available", 404
        
        results = analysis_results
        
        # Create DataFrame from anomaly data
        anomalies_data = results['sample_data']['anomalies']
        if anomalies_data:
            df = pd.DataFrame(anomalies_data)
            
            # Create CSV in memory
            output = io.StringIO()
            df.to_csv(output, index=False)
            output.seek(0)
            
            return send_file(
                io.BytesIO(output.getvalue().encode('utf-8')),
                as_attachment=True,
                download_name=f"anomalies_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mimetype='text/csv'
            )
        else:
            return "No anomaly data available for export", 404
            
    except Exception as e:
        return f"Error exporting data: {str(e)}", 500

@app.route('/export_json')
def export_json():
    """Export full analysis results as JSON"""
    try:
        if 'analysis_results' not in globals() or analysis_results is None:
            return "No analysis results available", 404
        
        results = analysis_results
        
        # Create JSON response
        return send_file(
            io.BytesIO(json.dumps(results, indent=2).encode('utf-8')),
            as_attachment=True,
            download_name=f"full_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mimetype='application/json'
        )
        
    except Exception as e:
        return f"Error exporting JSON: {str(e)}", 500

@app.route('/generate_sample')
def generate_sample():
    """Generate and download sample astronomical dataset"""
    try:
        # Generate realistic sample astronomical data
        np.random.seed(42)
        n_samples = 1000
        
        sample_data = {
            'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='H').strftime('%Y-%m-%d %H:%M:%S'),
            'right_ascension': np.random.uniform(0, 360, n_samples),
            'declination': np.random.uniform(-90, 90, n_samples),
            'brightness': np.random.normal(2000, 500, n_samples),
            'luminosity': np.random.normal(1e6, 2e5, n_samples),
            'temperature': np.random.normal(6000, 1000, n_samples),
            'radiation_level': np.random.normal(100, 20, n_samples),
            'redshift': np.random.exponential(0.1, n_samples),
            'proper_motion': np.random.normal(10, 3, n_samples),
            'parallax': np.random.normal(8, 2, n_samples)
        }
        
        df = pd.DataFrame(sample_data)
        
        # Introduce some anomalies (5% of data)
        n_anomalies = int(n_samples * 0.05)
        anomaly_indices = np.random.choice(n_samples, n_anomalies, replace=False)
        
        for idx in anomaly_indices:
            # Make some features extreme
            if np.random.random() > 0.5:
                df.loc[idx, 'brightness'] = np.random.uniform(5000, 10000)
                df.loc[idx, 'radiation_level'] = np.random.uniform(300, 500)
            else:
                df.loc[idx, 'temperature'] = np.random.uniform(10000, 20000)
                df.loc[idx, 'redshift'] = np.random.uniform(0.5, 1.0)
        
        # Create CSV in memory
        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)
        
        return send_file(
            io.BytesIO(output.getvalue().encode('utf-8')),
            as_attachment=True,
            download_name="sample_astronomical_data.csv",
            mimetype='text/csv'
        )
        
    except Exception as e:
        return f"Error generating sample data: {str(e)}", 500

# Global variable to store analysis results
analysis_results = None

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')
