# AstroVision: AI-Powered Cosmic Anomaly Detection

AstroVision is a full-stack web application designed for astronomers and data scientists to upload astronomical datasets, perform advanced anomaly detection using various machine learning models, and explore the results through an interactive, cosmic-themed dashboard.

*The main analysis page (`analysis.html`) showing interactive 3D projections and anomaly insights.*

-----

## 🌟 Key Features

  * [cite\_start]**Multi-Algorithm Analysis:** Choose from five different anomaly detection algorithms[cite: 1]:
      * Isolation Forest
      * One-Class SVM
      * Local Outlier Factor (LOF)
      * DBSCAN
      * Elliptic Envelope
  * [cite\_start]**Interactive 3D Visualizations:** Rich, interactive 3D and 2D scatter plots of data clusters using PCA and t-SNE, built with Plotly.js. [cite: 1]
  * **Comprehensive Dashboard:** A dedicated dashboard (`dashboard.html`) and results page (`analysis.html`) displaying key metrics, anomaly distribution charts, and feature impact analysis.
  * **Dynamic UI/UX:** A beautiful, responsive "cosmic" theme with animated backgrounds, custom fonts (`Orbitron`, `Exo 2`), and a clean, modern interface.
  * [cite\_start]**Sample Data Generator:** Includes a route (`/generate_sample`) to generate a sample astronomical CSV dataset to quickly test the application's capabilities. [cite: 1]
  * **Full Data Export:**
      * [cite\_start]**PDF Report:** Dynamically generates a professional, multi-page PDF research report of the findings using ReportLab. [cite: 1]
      * [cite\_start]**CSV Export:** Download a CSV file containing only the detected anomalies. [cite: 1]
      * [cite\_start]**JSON Export:** Export the complete analysis results, including all summary stats and visualization data, as a JSON file. [cite: 1]

-----

## 🛠️ Tech Stack

  * [cite\_start]**Backend:** **Python 3**, **Flask** [cite: 1]
  * [cite\_start]**Data Science:** **scikit-learn** (for ML models), **pandas** (for data manipulation), **numpy** (for numerical operations) [cite: 1]
  * [cite\_start]**PDF Reporting:** **ReportLab** [cite: 1]
  * **Frontend:** HTML5, CSS3, JavaScript (class-based)
  * **Visualization:** **Plotly.js** (for 3D/2D interactive plots), **Chart.js** (for dashboard charts)

-----

## 🚀 Installation and Setup

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/astrovision.git
    cd astrovision
    ```

2.  **Create and activate a virtual environment:**

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install the required packages:**
    [cite\_start]The `requirements.txt` file includes all necessary dependencies[cite: 2]:

    ```bash
    pip install -r requirements.txt
    ```

      * [cite\_start]Flask==2.3.3 [cite: 2]
      * [cite\_start]pandas==2.0.3 [cite: 2]
      * [cite\_start]numpy==1.24.3 [cite: 2]
      * [cite\_start]scikit-learn==1.3.0 [cite: 2]
      * [cite\_start]reportlab==4.0.4 [cite: 2]
      * [cite\_start]werkzeug==2.3.7 [cite: 2]

4.  **Run the application:**

    ```bash
    python app.py
    ```

5.  Open your browser and navigate to `http://127.0.0.1:5000`.

-----

## 🛸 Usage / Workflow

1.  **Navigate Home:** Visit the homepage (`index.html`) to see an overview of the project's capabilities.
2.  **Upload Data:** Go to the 'Analyze' page (`upload.html`). [cite\_start]You can either drag-and-drop your own CSV file or download the provided sample data by clicking the "Download Sample Dataset" button (`/generate_sample`)[cite: 1]. A sample `kepler_exoplanet_data.csv` is also included.
3.  **Configure Analysis:** Select your preferred anomaly detection algorithm from the list. The parameters section below will dynamically update to show the relevant hyperparameters (e.g., "Number of Estimators" for Isolation Forest or "Kernel Type" for One-Class SVM). [cite\_start]Adjust the "Contamination" slider to set the expected proportion of anomalies. [cite: 1]
4.  **Run Analysis:** Click the "Launch Cosmic Analysis" button. [cite\_start]The frontend will show a loading state while the backend reads the CSV, preprocesses the data (scaling, handling NaNs), runs the selected ML model, and generates a comprehensive results object. [cite: 1]
5.  **View Results:** You will be redirected to the 'Results' page (`analysis.html`). Here you can:
      * View summary metrics (e.g., total anomalies, anomaly rate).
      * Explore interactive 3D PCA and 2D t-SNE plots.
      * Read AI-generated insights and key findings.
      * Browse tables of anomalous and normal data points.
6.  **Explore Dashboard:** Visit the 'Dashboard' page (`dashboard.html`) for a high-level overview with key metrics and summary charts (Note: In the current version, some dashboard charts use placeholder data).
7.  [cite\_start]**Export Findings:** From the results or insights tabs, use the download buttons to get your findings as a **PDF Report**, **Anomalies CSV**, or **Full JSON** dump. [cite: 1]

-----

## 🔮 Future Improvements

While the project is fully functional, here are some potential areas for enhancement:

  * **Support Concurrent Users:** Refactor the backend to move away from a global `analysis_results` variable. Store results in a Flask `session` or save them to a database/file with a unique ID to allow multiple users to run analyses simultaneously.
  * **Consolidate JavaScript:** Move all inline `<script>` blocks from `analysis.html` and `dashboard.html` into external `.js` files (`analysis.js`, `dashboard.js`) to improve code organization and maintainability.
  * **Live Dashboard Data:** Connect all charts on `dashboard.html` to the *actual* data from the `analysisResults` object, rather than using placeholder or randomly generated data.
  * **Advanced Feature Importance:** Implement a more robust feature importance method (e.g., SHAP) to provide more accurate insights into *why* a data point was flagged as an anomaly.

-----

## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for details.