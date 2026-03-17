# AstroVision

**Advanced Computational Framework for Astronomical Anomaly Detection**

[Python 3.8+](https://www.python.org/downloads/) | [Flask Framework](https://flask.palletsprojects.com/) | [Scikit-Learn ML](https://scikit-learn.org/) | [MIT License](https://opensource.org/licenses/MIT)

AstroVision is an enterprise-grade computational framework engineered for the high-precision detection of anomalous celestial phenomena. By integrating robust machine learning architectures with immersive data visualization, AstroVision empowers researchers to isolate rare cosmic events, identify instrument-borne artifacts, and discover novel astronomical entities within massive datasets.

---

## Core Capabilities

### Sophisticated Anomaly Detection Engine
AstroVision utilizes an ensemble of six high-performance machine learning algorithms, optimized for multidimensional astronomical data analysis:
- **Isolation Forest**: Leverages recursive partitioning to isolate anomalies in high-dimensional space with exceptional efficiency.
- **One-Class Support Vector Machines (SVM)**: Constructs high-dimensional boundaries around normative data patterns to identify novel observations.
- **Local Outlier Factor (LOF)**: Computes local density variations to detect observations that deviate significantly from their immediate celestial neighborhood.
- **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**: Isolates structural noise and outliers from dense cluster formations.
- **Robust Covariance (Elliptic Envelope)**: Employs Gaussian-based statistical modeling to identify extreme deviations in well-behaved datasets.
- **Ensemble Random Forest**: Provides a robust, multi-tree classification architecture for high-confidence anomaly scoring.

### High-Fidelity Data Visualization
- **Multidimensional Projections**: Employs Principal Component Analysis (PCA) and t-distributed Stochastic Neighbor Embedding (t-SNE) for 3D spatial mapping of complex feature sets.
- **Interactive Analytical Heatmaps**: Enables deep exploration of feature correlations and statistical distributions.
- **Real-time Heuristic Analysis**: Provides instantaneous feedback on anomaly metrics and clustering logic.

### Research-Grade Documentation and Reporting
- **Automated PDF Synthesis**: Generates publication-ready analytical reports featuring peer-reviewed formatting and comprehensive statistical summaries.
- **Extensible Data Export**: Supports high-fidelity data extraction in CSV and JSON formats for cross-platform research integration.
- **Synthetic Dataset Generation**: A built-in computational utility for producing realistic astronomical datasets for algorithm benchmarking.

---

## Immersive Space-Themed User Interface

AstroVision features a meticulously designed, space-themed user interface that aligns with the aesthetic and functional requirements of astronomical research. Utilizing a "Cosmic Design System," the UI employs glassmorphism, deep-space color palettes, and fluid animations to provide an immersive environment that minimizes cognitive load while maximizing data clarity.

---

## Technical Architecture and Stack

- **Backend Architecture**: Python-based Flask framework for robust request handling and computational scheduling.
- **Machine Learning Pipeline**: Scikit-Learn implementation of advanced manifold learning and outlier detection.
- **Scientific Computing**: Powered by NumPy and Pandas for high-speed matrix operations and data manipulation.
- **Visualization Engine**: Integration of Plotly.js and Chart.js for hardware-accelerated 3D rendering.
- **Reporting Engine**: ReportLab-based PDF generation for high-fidelity document synthesis.

---

## Deployment and Installation

### System Requirements
- Python 3.11 or higher
- Windows 10/11, macOS, or Linux
- Minimum 8GB RAM (16GB recommended for large datasets)

### Installation Procedure

1. **Repository Acquisition**
   ```powershell
   git clone https://github.com/organization/AstroVision.git
   cd AstroVision
   ```

2. **Environment Configuration**
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

3. **Dependency Deployment**
   ```powershell
   pip install -r requirements.txt
   ```

### Execution
To initiate the computational server:
```powershell
python app.py
```
The application will be accessible via the local host at `http://localhost:5000`.

---

## Project Structure

```text
AstroVision/
├── app.py              # Central Computational Hub and Flask Application
├── requirements.txt    # Comprehensive Dependency Manifest
├── static/             # Front-end Assets and Cosmic Design System
│   ├── style.css       # Core Aesthetic Framework
│   ├── script.js       # Asynchronous UI Logic
│   └── dashboard.js    # Advanced Analytical Visualization Logic
├── templates/          # HTML Reference Templates
│   ├── index.html      # Primary Ingress Point
│   ├── upload.html     # Data Ingestion Interface
│   └── analysis.html   # Analytical Results Dashboard
└── uploads/            # Secure Buffered Storage for Analysis
```

---

## Academic Integrity and Licensing

This project is released under the MIT License. Please refer to the [LICENSE](LICENSE) file for the full legal text.

---

## Acknowledgments

AstroVision is inspired by the data processing requirements of major sky surveys including Kepler, Gaia, and LSST. The development team acknowledges the contributions of the scikit-learn, Flask, and broader scientific Python communities.
