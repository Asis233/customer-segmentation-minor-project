# 🛍️ Segmently — Customer Segmentation Dashboard Using K-Means

A complete, functional, and visually attractive **Customer Segmentation Dashboard** built with Python, Streamlit, Pandas, Scikit-learn, and Plotly. 

Designed for a **BTech CSE AI/ML Minor Project**, this dashboard enables businesses to analyze customer behavior, discover spending patterns, and identify high-value target groups using unsupervised machine learning (K-Means Clustering).

---

## 🌟 Key Features

1. **Overview Dashboard**
   - 4 Top Metric Cards (*Total Customers*, *Average Annual Income*, *Average Spending Score*, *Customer Segments*).
   - Interactive 2D Scatter Plot (Annual Income vs. Spending Score) with cluster centroid markers.
   - Cluster Share Donut Chart.
   - Interactive customer directory table with filtering capabilities.

2. **Customer Analytics Page**
   - Detailed univariate and bivariate distributions for Age, Annual Income, Spending Score, and Gender.
   - Narrative insights explaining customer behavior and spending trends.

3. **K-Means Segmentation & Model Evaluation**
   - Configurable $K$ slider (2 to 8 clusters).
   - Multi-feature selection for clustering.
   - Feature scaling using `StandardScaler`.
   - Live model evaluation metrics: **Inertia (WCSS)** and **Silhouette Score**.
   - Interactive **Elbow Method Curve** plot.
   - Exact mathematical cluster centroids display table.

4. **Cluster Insights & Behavioral Personas**
   - Automatic assignment of human-understandable customer personas based on dataset statistics (e.g., *Target / Premium Customers*, *Careful / Saver Customers*, *Impulse / Trendsetter Customers*, *Sensible / Budget Customers*).
   - Detailed statistical breakdown per cluster.
   - Actionable business strategies and marketing recommendations for each group.
   - Grouped bar chart comparing metrics across clusters.

5. **Customer Explorer**
   - Search by Customer ID.
   - Multi-select filter by Cluster assignment.
   - Income range slider filter.
   - Export filtered dataset to CSV (`segmented_customers_filtered.csv`).

6. **Dataset Management & CSV Upload**
   - Built-in realistic sample dataset (`data/sample_customers.csv`).
   - Custom CSV file uploader.
   - Auto-detection of column names with manual override column mapping.
   - Data validation, missing value handling, and duplicate removal summary.

7. **About Project Page**
   - Comprehensive project documentation ready for BTech minor project evaluation, detailing Problem Statement, ML Algorithm, Tech Stack, Applications, Limitations, and Future Scope.

---

## 🎨 Color Palette & Visual Theme

The dashboard follows a warm, professional, pastel-accented aesthetic:

- **Main Background:** Warm Cream (`#F8F3E7`)
- **Sidebar Background:** Compact Dark (`#171717`)
- **Cards Background:** Pure White (`#FFFFFF`) with subtle border (`#E7E1D7`)
- **Pastel Accents:**
  - Pastel Blue: `#C8D9F0`
  - Pastel Pink: `#F2C6DE`
  - Pastel Yellow: `#F7DF78`
  - Pastel Green: `#D4E8B8`
  - Pastel Lavender: `#DCD1F3`

---

## 📂 Project Architecture

```
customer-segmentation/
├── app.py                     # Main Streamlit application entry point
├── requirements.txt           # Project Python dependencies
├── README.md                  # Complete documentation & deployment guide
├── .gitignore                 # Files ignored by git
├── data/
│   └── sample_customers.csv   # Built-in synthetic customer dataset
└── utils/
    ├── __init__.py
    ├── data_processing.py      # Data loading, column detection, cleaning & stats
    ├── clustering.py           # StandardScaler, KMeans, Silhouette score & personas
    └── visualizations.py      # Plotly interactive charts & pastel styling
```

---

## 💻 Local Installation & Setup Guide

Follow these steps to run the application on your local machine:

### Prerequisites
- Python 3.9+ (Python 3.14 supported)
- Git

### 1. Clone or Open the Repository
Navigate to your project directory:
```bash
cd "customer-segmentation"
```

### 2. Create a Virtual Environment (Optional but Recommended)
On Windows:
```bash
python -m venv .venv
.venv\Scripts\activate
```

On macOS / Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Required Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Application
```bash
streamlit run app.py
```
The application will automatically open in your default browser at `http://localhost:8501`.

---

## 🚀 Render Cloud Deployment Guide

This project is fully ready for zero-configuration deployment on **Render** as a Python Web Service.

### Step 1: Push Code to GitHub
Create a new GitHub repository and push your project files:
```bash
git init
git add .
git commit -m "Initial commit of Customer Segmentation Dashboard"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/customer-segmentation.git
git push -u origin main
```

### Step 2: Deploy on Render
1. Log in to [Render.com](https://render.com).
2. Click **New +** and select **Web Service**.
3. Connect your GitHub account and select your `customer-segmentation` repository.
4. Configure the Web Service settings:
   - **Name:** `segmently-customer-dashboard` (or preferred name)
   - **Environment:** `Python 3`
   - **Region:** Choose closest region
   - **Branch:** `main`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
5. Click **Create Web Service**.

Render will automatically build and deploy your app. Once deployed, you will receive a public URL (e.g., `https://segmently-customer-dashboard.onrender.com`).

---

## 🎓 College Presentation & Viva Guide

Key points to highlight during your minor project presentation:

1. **Objective:** To automate customer segmentation for targeted marketing using unsupervised machine learning without manual data labeling.
2. **Algorithm Choice:** K-Means clustering was chosen for its computational efficiency $O(t \cdot k \cdot n \cdot d)$ and interpretability.
3. **Data Preprocessing:** Features are scaled using `StandardScaler` (Z-score normalization) to prevent higher-magnitude features (e.g., $100,000 Income vs 100 Spending Score) from dominating Euclidean distance calculations.
4. **Optimal Cluster Selection:** Demonstrated using the **Elbow Method** (minimizing Within-Cluster Sum of Squares - WCSS) and validated using the **Silhouette Coefficient** (measuring cluster cohesion vs separation).
5. **Business Utility:** Personas are generated dynamically based on statistical thresholds to provide actionable business strategies for marketing teams.
