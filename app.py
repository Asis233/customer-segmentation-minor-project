"""
Segmently - Customer Segmentation Dashboard
Built using Python, Streamlit, Pandas, Scikit-learn, and Plotly.
BTech CSE AI/ML Minor Project.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
from typing import List, Dict, Optional, Any

from utils.data_processing import (
    load_dataset, auto_detect_columns, validate_and_clean_data, calculate_dataset_stats
)
from utils.clustering import (
    perform_kmeans_clustering, evaluate_kmeans_model, compute_elbow_curve, generate_cluster_insights
)
from utils.visualizations import (
    plot_customer_segmentation_scatter, plot_cluster_distribution_donut,
    plot_feature_distribution, plot_cluster_comparison_bars, plot_elbow_curve,
    PASTEL_COLORS, THEME_BG, CARD_BG, PRIMARY_TEXT, SECONDARY_TEXT, BORDER_COLOR
)

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Segmently - Customer Segmentation Dashboard",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Theme CSS matching reference design specifications
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: #292929;
    }
    
    /* Main Background */
    .stApp {
        background-color: #F8F3E7;
    }
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1280px;
    }
    
    /* Dark Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #171717 !important;
        width: 260px !important;
    }
    section[data-testid="stSidebar"] * {
        color: #EAEAEA !important;
    }
    section[data-testid="stSidebar"] .stRadio label {
        font-size: 14px;
        padding: 8px 12px;
        border-radius: 8px;
        transition: all 0.2s ease;
    }
    
    /* Active Nav Item Styling */
    div[role="radiogroup"] > label {
        margin-bottom: 4px;
        border-radius: 8px;
    }
    div[role="radiogroup"] > label:hover {
        background-color: #292828 !important;
    }
    
    /* Card Container */
    .dashboard-card {
        background-color: #FFFFFF;
        border: 1px solid #E7E1D7;
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        box-shadow: 0px 2px 6px rgba(0, 0, 0, 0.02);
    }
    
    /* Metric Cards */
    .metric-card {
        background-color: #FFFFFF;
        border: 1px solid #E7E1D7;
        border-radius: 12px;
        padding: 1.1rem 1.25rem;
        box-shadow: 0px 2px 6px rgba(0,0,0,0.02);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        height: 100%;
        position: relative;
        overflow: hidden;
    }
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 5px;
        height: 100%;
    }
    .metric-blue::before { background-color: #C8D9F0; }
    .metric-pink::before { background-color: #F2C6DE; }
    .metric-yellow::before { background-color: #F7DF78; }
    .metric-green::before { background-color: #D4E8B8; }
    .metric-lavender::before { background-color: #DCD1F3; }
    
    .metric-title {
        font-size: 13px;
        font-weight: 500;
        color: #77736E;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .metric-value {
        font-size: 26px;
        font-weight: 700;
        color: #292929;
        margin: 6px 0 2px 0;
    }
    .metric-subtitle {
        font-size: 12px;
        color: #77736E;
    }
    
    /* Badges */
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        color: #292929;
        margin-right: 6px;
        border: 1px solid rgba(0,0,0,0.08);
    }
    
    /* Section Titles */
    .page-title {
        font-size: 26px;
        font-weight: 700;
        color: #292929;
        margin-bottom: 2px;
    }
    .page-subtitle {
        font-size: 14px;
        color: #77736E;
        margin-bottom: 1.25rem;
    }
    .section-header {
        font-size: 18px;
        font-weight: 600;
        color: #292929;
        margin-top: 1rem;
        margin-bottom: 0.75rem;
    }
    
    /* Custom Buttons & Inputs */
    .stButton > button {
        background-color: #FFFFFF;
        color: #292929;
        border: 1px solid #E7E1D7;
        border-radius: 8px;
        font-weight: 500;
        padding: 0.4rem 1rem;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        background-color: #F8F3E7;
        border-color: #292929;
        color: #292929;
    }
    
    /* Table Styling */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid #E7E1D7;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# Initialize Session State Variables
def init_session_state():
    if "dataset_source" not in st.session_state:
        st.session_state.dataset_source = "sample"
    if "raw_df" not in st.session_state:
        sample_path = os.path.join("data", "sample_customers.csv")
        df, err = load_dataset(sample_path)
        st.session_state.raw_df = df if df is not None else pd.DataFrame()
    if "uploaded_file" not in st.session_state:
        st.session_state.uploaded_file = None
    if "n_clusters" not in st.session_state:
        st.session_state.n_clusters = 5
    if "column_mapping" not in st.session_state:
        if not st.session_state.raw_df.empty:
            st.session_state.column_mapping = auto_detect_columns(st.session_state.raw_df)
        else:
            st.session_state.column_mapping = {"id": None, "income": None, "spending": None, "age": None, "gender": None}
    if "selected_features" not in st.session_state:
        m = st.session_state.column_mapping
        feats = []
        if m.get("income"): feats.append(m["income"])
        if m.get("spending"): feats.append(m["spending"])
        if not feats:
            num_cols = st.session_state.raw_df.select_dtypes(include=[np.number]).columns.tolist()
            feats = num_cols[:2] if len(num_cols) >= 2 else num_cols
        st.session_state.selected_features = feats

init_session_state()


# Helper function to get clustered dataframe and model results
@st.cache_data(show_spinner=False)
def get_processed_and_clustered_data(df: pd.DataFrame, feature_cols: List[str], id_col: Optional[str], n_clusters: int):
    cleaned_df, summary = validate_and_clean_data(df, feature_cols, id_col)
    if cleaned_df.empty or len(feature_cols) == 0:
        return cleaned_df, summary, None, None, None, {}, []
    
    df_clustered, kmeans_model, scaler, X_scaled = perform_kmeans_clustering(
        cleaned_df, feature_cols, n_clusters=n_clusters, random_state=42
    )
    
    eval_metrics = evaluate_kmeans_model(df_clustered, X_scaled, kmeans_model)
    
    # Auto-detect income & spending cols for persona naming
    mapping = auto_detect_columns(cleaned_df)
    income_c = mapping.get("income") or (feature_cols[0] if len(feature_cols) >= 1 else None)
    spending_c = mapping.get("spending") or (feature_cols[1] if len(feature_cols) >= 2 else None)
    
    insights = generate_cluster_insights(df_clustered, income_c, spending_c)
    
    # Map categories back into dataframe
    cat_map = {ins["cluster_name"]: ins["category"] for ins in insights}
    persona_map = {ins["cluster_name"]: ins["persona"] for ins in insights}
    
    df_clustered["Category"] = df_clustered["Cluster"].map(cat_map)
    df_clustered["Persona"] = df_clustered["Cluster"].map(persona_map)
    
    return df_clustered, summary, kmeans_model, scaler, X_scaled, eval_metrics, insights


# Execute Data Processing and Clustering Pipeline
raw_df = st.session_state.raw_df
column_mapping = auto_detect_columns(raw_df) if not raw_df.empty else st.session_state.column_mapping
id_col = column_mapping.get("id")
income_col = column_mapping.get("income")
spending_col = column_mapping.get("spending")
age_col = column_mapping.get("age")
gender_col = column_mapping.get("gender")

selected_features = st.session_state.selected_features
n_clusters = st.session_state.n_clusters

if not raw_df.empty and selected_features:
    df_clustered, summary, kmeans_model, scaler, X_scaled, eval_metrics, insights = get_processed_and_clustered_data(
        raw_df, selected_features, id_col, n_clusters
    )
else:
    df_clustered = pd.DataFrame()
    summary, kmeans_model, scaler, X_scaled, eval_metrics, insights = {}, None, None, None, {}, []


# ==========================================
# SIDEBAR NAVIGATION & CONTROL PANEL
# ==========================================
with st.sidebar:
    # Project Brand Logo & Title
    st.markdown("""
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 1.5rem; padding-bottom: 0.75rem; border-bottom: 1px solid #333333;">
            <div style="background-color: #F2C6DE; color: #171717; width: 36px; height: 36px; border-radius: 8px; display: flex; align-items: center; justify-content: center; font-weight: bold; font-size: 20px;">S</div>
            <div>
                <div style="font-size: 18px; font-weight: 700; color: #FFFFFF; letter-spacing: 0.5px;">Segmently</div>
                <div style="font-size: 11px; color: #999999;">Customer Analytics Hub</div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<div style='font-size: 11px; font-weight: 600; color: #888888; text-transform: uppercase; margin-bottom: 6px;'>NAVIGATION</div>", unsafe_allow_html=True)
    
    nav_page = st.radio(
        label="Navigation Menu",
        options=[
            "Overview",
            "Customer Analytics",
            "Segmentation",
            "Cluster Insights",
            "Customer Explorer",
            "Upload Dataset",
            "About Project"
        ],
        label_visibility="collapsed"
    )
    
    st.markdown("<hr style='border-color: #333333; margin: 1.25rem 0;'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size: 11px; font-weight: 600; color: #888888; text-transform: uppercase; margin-bottom: 8px;'>CLUSTERING CONTROLS</div>", unsafe_allow_html=True)
    
    # Cluster Count Slider
    new_k = st.slider("Number of Clusters (K)", min_value=2, max_value=8, value=n_clusters, step=1)
    if new_k != n_clusters:
        st.session_state.n_clusters = new_k
        st.rerun()
        
    # Active Dataset Status Badge
    source_badge = "Sample Dataset" if st.session_state.dataset_source == "sample" else "Custom Upload"
    st.markdown(f"""
        <div style="background-color: #222222; border: 1px solid #333333; border-radius: 8px; padding: 10px; margin-top: 10px;">
            <div style="font-size: 11px; color: #888888;">Active Dataset:</div>
            <div style="font-size: 13px; font-weight: 600; color: #F7DF78; display: flex; align-items: center; gap: 6px;">
                <span>📁</span> {source_badge} ({len(df_clustered)} rows)
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Reset to Default Sample Data", use_container_width=True):
        st.session_state.dataset_source = "sample"
        sample_path = os.path.join("data", "sample_customers.csv")
        df, _ = load_dataset(sample_path)
        st.session_state.raw_df = df
        st.session_state.uploaded_file = None
        st.session_state.n_clusters = 5
        st.session_state.column_mapping = auto_detect_columns(df)
        m = st.session_state.column_mapping
        st.session_state.selected_features = [m["income"], m["spending"]] if m.get("income") and m.get("spending") else list(df.select_dtypes(include=[np.number]).columns[:2])
        st.rerun()


# ==========================================
# PAGE 1: OVERVIEW DASHBOARD
# ==========================================
if nav_page == "Overview":
    st.markdown("<div class='page-title'>Customer Segmentation Dashboard</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-subtitle'>Understand customer behavior, discover spending patterns, and identify valuable customer groups.</div>", unsafe_allow_html=True)
    
    if df_clustered.empty:
        st.warning("No valid dataset loaded. Please upload a dataset or reset to sample data.")
    else:
        # Calculate summary stats
        stats = calculate_dataset_stats(df_clustered, income_col, spending_col)
        
        # Top 4 Compact Metric Cards
        c1, c2, c3, c4 = st.columns(4)
        
        with c1:
            st.markdown(f"""
                <div class='metric-card metric-blue'>
                    <div class='metric-title'>👥 TOTAL CUSTOMERS</div>
                    <div class='metric-value'>{stats['total_customers']:,}</div>
                    <div class='metric-subtitle'>Cleaned records analyzed</div>
                </div>
            """, unsafe_allow_html=True)
            
        with c2:
            inc_val = f"${stats['avg_income']:.1f}k" if stats['avg_income'] is not None else "N/A"
            st.markdown(f"""
                <div class='metric-card metric-pink'>
                    <div class='metric-title'>💰 AVG ANNUAL INCOME</div>
                    <div class='metric-value'>{inc_val}</div>
                    <div class='metric-subtitle'>Average customer earnings</div>
                </div>
            """, unsafe_allow_html=True)
            
        with c3:
            spend_val = f"{stats['avg_spending']:.1f}" if stats['avg_spending'] is not None else "N/A"
            st.markdown(f"""
                <div class='metric-card metric-yellow'>
                    <div class='metric-title'>🛍️ AVG SPENDING SCORE</div>
                    <div class='metric-value'>{spend_val} / 100</div>
                    <div class='metric-subtitle'>Customer spending index</div>
                </div>
            """, unsafe_allow_html=True)
            
        with c4:
            st.markdown(f"""
                <div class='metric-card metric-green'>
                    <div class='metric-title'>🎯 CUSTOMER SEGMENTS</div>
                    <div class='metric-value'>{n_clusters}</div>
                    <div class='metric-subtitle'>K-Means clusters fitted</div>
                </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Main Analytics Section (2 Columns: Scatter plot & Donut distribution)
        col_left, col_right = st.columns([1.5, 1])
        
        with col_left:
            if income_col and spending_col and income_col in df_clustered.columns and spending_col in df_clustered.columns:
                fig_scatter = plot_customer_segmentation_scatter(
                    df_clustered, income_col, spending_col, kmeans_model, scaler, selected_features
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
            else:
                st.info("Please select Annual Income and Spending Score features to render the 2D cluster scatter plot.")
                
        with col_right:
            fig_donut = plot_cluster_distribution_donut(df_clustered)
            st.plotly_chart(fig_donut, use_container_width=True)
            
        st.markdown("<hr style='border-color: #E7E1D7; margin: 1.5rem 0;'>", unsafe_allow_html=True)
        
        # Bottom Customer Data Table Section
        st.markdown("<div class='section-header'>📋 Customer Segment Directory</div>", unsafe_allow_html=True)
        
        # Filters bar for table
        tb_col1, tb_col2 = st.columns([1, 2])
        with tb_col1:
            selected_cluster_filter = st.selectbox(
                "Filter by Cluster",
                options=["All Clusters"] + sorted(df_clustered["Cluster"].unique().tolist(), key=lambda x: int(x.split()[-1]))
            )
            
        display_df = df_clustered.copy()
        if selected_cluster_filter != "All Clusters":
            display_df = display_df[display_df["Cluster"] == selected_cluster_filter]
            
        # Select key columns for neat clean display
        show_cols = []
        if "__display_id__" in display_df.columns: show_cols.append("__display_id__")
        if gender_col and gender_col in display_df.columns: show_cols.append(gender_col)
        if age_col and age_col in display_df.columns: show_cols.append(age_col)
        if income_col and income_col in display_df.columns: show_cols.append(income_col)
        if spending_col and spending_col in display_df.columns: show_cols.append(spending_col)
        if "Cluster" in display_df.columns: show_cols.append("Cluster")
        if "Category" in display_df.columns: show_cols.append("Category")
        
        st.dataframe(
            display_df[show_cols].rename(columns={"__display_id__": "Customer ID"}),
            use_container_width=True,
            hide_index=True,
            height=320
        )


# ==========================================
# PAGE 2: CUSTOMER ANALYTICS
# ==========================================
elif nav_page == "Customer Analytics":
    st.markdown("<div class='page-title'>Customer Feature Analytics</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-subtitle'>Deep-dive into demographic distributions, income patterns, and spending behaviors.</div>", unsafe_allow_html=True)
    
    if df_clustered.empty:
        st.warning("No dataset loaded.")
    else:
        # Row 1: Income and Spending Score distributions
        r1_left, r1_right = st.columns(2)
        
        with r1_left:
            if income_col and income_col in df_clustered.columns:
                fig_inc = plot_feature_distribution(df_clustered, income_col, "Annual Income (k$)", '#C8D9F0')
                st.plotly_chart(fig_inc, use_container_width=True)
                st.caption("ℹ️ **Income Analysis:** Displays how annual earnings are spread across customers. Multi-modal shapes indicate distinct income brackets.")
            else:
                st.info("Annual Income column unavailable.")
                
        with r1_right:
            if spending_col and spending_col in df_clustered.columns:
                fig_spd = plot_feature_distribution(df_clustered, spending_col, "Spending Score (1-100)", '#F2C6DE')
                st.plotly_chart(fig_spd, use_container_width=True)
                st.caption("ℹ️ **Spending Score Analysis:** Measures purchasing willingness. Uniform or bimodal patterns highlight different customer engagement levels.")
                
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Row 2: Age & Gender distributions if present
        r2_left, r2_right = st.columns(2)
        
        with r2_left:
            if age_col and age_col in df_clustered.columns:
                fig_age = plot_feature_distribution(df_clustered, age_col, "Customer Age", '#F7DF78')
                st.plotly_chart(fig_age, use_container_width=True)
                st.caption("ℹ️ **Age Distribution:** Analyzes customer demographics across age brackets.")
            else:
                st.info("Age column not found in dataset.")
                
        with r2_right:
            if gender_col and gender_col in df_clustered.columns:
                fig_gen = plot_feature_distribution(df_clustered, gender_col, "Gender", '#D4E8B8')
                st.plotly_chart(fig_gen, use_container_width=True)
                st.caption("ℹ️ **Gender Ratio:** Shows gender proportion across the customer population.")
            else:
                st.info("Gender column not found in dataset.")


# ==========================================
# PAGE 3: K-MEANS SEGMENTATION & EVALUATION
# ==========================================
elif nav_page == "Segmentation":
    st.markdown("<div class='page-title'>K-Means Segmentation & Model Evaluation</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-subtitle'>Configure feature inputs, inspect model parameters, and evaluate clustering quality using Silhouette Score & Elbow Method.</div>", unsafe_allow_html=True)
    
    if df_clustered.empty:
        st.warning("No dataset loaded.")
    else:
        # Configuration Controls Card
        st.markdown("""
            <div class='dashboard-card'>
                <div style='font-weight: 600; font-size: 16px; margin-bottom: 10px; color: #292929;'>⚙️ Model Parameters & Feature Selection</div>
            </div>
        """, unsafe_allow_html=True)
        
        cfg_col1, cfg_col2 = st.columns(2)
        with cfg_col1:
            all_num_cols = raw_df.select_dtypes(include=[np.number]).columns.tolist()
            new_feats = st.multiselect(
                "Select Numerical Features for Clustering:",
                options=all_num_cols,
                default=selected_features if set(selected_features).issubset(all_num_cols) else all_num_cols[:2]
            )
            if new_feats != selected_features and len(new_feats) >= 1:
                st.session_state.selected_features = new_feats
                st.rerun()
                
        with cfg_col2:
            st.markdown(f"**Current Number of Clusters ($K$):** `{n_clusters}`")
            st.markdown(f"**Feature Scaler:** `StandardScaler (Mean=0, Variance=1)`")
            st.markdown(f"**Algorithm:** `K-Means (scikit-learn)`")
            
        st.markdown("<hr style='border-color: #E7E1D7; margin: 1.5rem 0;'>", unsafe_allow_html=True)
        
        # Model Evaluation Metrics Cards
        ev1, ev2, ev3 = st.columns(3)
        
        with ev1:
            st.markdown(f"""
                <div class='metric-card metric-blue'>
                    <div class='metric-title'>📊 INERTIA (WCSS)</div>
                    <div class='metric-value'>{eval_metrics.get('inertia', 0.0):,.1f}</div>
                    <div class='metric-subtitle'>Within-cluster sum of squares</div>
                </div>
            """, unsafe_allow_html=True)
            
        with ev2:
            sil_val = eval_metrics.get('silhouette_score')
            sil_str = f"{sil_val:.3f}" if sil_val is not None else "N/A"
            st.markdown(f"""
                <div class='metric-card metric-pink'>
                    <div class='metric-title'>✨ SILHOUETTE SCORE</div>
                    <div class='metric-value'>{sil_str}</div>
                    <div class='metric-subtitle'>Measures cluster separation (-1 to +1)</div>
                </div>
            """, unsafe_allow_html=True)
            
        with ev3:
            st.markdown(f"""
                <div class='metric-card metric-lavender'>
                    <div class='metric-title'>📐 TOTAL SAMPLES</div>
                    <div class='metric-value'>{eval_metrics.get('n_samples', 0)}</div>
                    <div class='metric-subtitle'>Evaluated data points</div>
                </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Elbow Method Plot
        col_elbow, col_centroids = st.columns([1.2, 1])
        
        with col_elbow:
            if X_scaled is not None and len(X_scaled) > 2:
                elbow_dict = compute_elbow_curve(X_scaled, max_k=10)
                fig_elbow = plot_elbow_curve(elbow_dict, current_k=n_clusters)
                st.plotly_chart(fig_elbow, use_container_width=True)
                st.caption("💡 **Elbow Method:** Look for the 'elbow' point where inertia reduction slows down. This indicates the optimal number of clusters.")
            else:
                st.info("Insufficient data to generate Elbow plot.")
                
        with col_centroids:
            st.markdown("<div class='section-header'>🎯 Cluster Centroids (Mean Values)</div>", unsafe_allow_html=True)
            if kmeans_model is not None and scaler is not None:
                centroids_orig = scaler.inverse_transform(kmeans_model.cluster_centers_)
                cent_df = pd.DataFrame(
                    centroids_orig,
                    columns=selected_features,
                    index=[f"Cluster {i+1}" for i in range(n_clusters)]
                )
                st.dataframe(cent_df.style.highlight_max(axis=0, color='#D4E8B8'), use_container_width=True)
                st.caption("ℹ️ **Centroids** represent the mathematical center of each cluster in original feature units.")


# ==========================================
# PAGE 4: CLUSTER INSIGHTS
# ==========================================
elif nav_page == "Cluster Insights":
    st.markdown("<div class='page-title'>Cluster Insights & Behavioral Personas</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-subtitle'>Automated behavioral categorization, statistical breakdown, and strategic action recommendations for each customer segment.</div>", unsafe_allow_html=True)
    
    if not insights:
        st.warning("No cluster insights available.")
    else:
        # Cluster Comparison Bar Chart
        fig_comp = plot_cluster_comparison_bars(insights)
        st.plotly_chart(fig_comp, use_container_width=True)
        
        st.markdown("<hr style='border-color: #E7E1D7; margin: 1.5rem 0;'>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>💡 Detailed Segment Personas & Business Strategies</div>", unsafe_allow_html=True)
        
        # Grid of Cluster Insight Cards
        for idx, ins in enumerate(insights):
            accent_colors = ['#C8D9F0', '#F2C6DE', '#F7DF78', '#D4E8B8', '#DCD1F3', '#FFD1B3', '#BCE6FF', '#E2D0F9']
            card_accent = accent_colors[idx % len(accent_colors)]
            
            st.markdown(f"""
                <div class='dashboard-card' style='border-left: 6px solid {card_accent};'>
                    <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;'>
                        <div style='font-size: 18px; font-weight: 700; color: #292929;'>
                            {ins['cluster_name']} — <span style='color: #555555;'>{ins['persona']}</span>
                        </div>
                        <div>
                            <span class='badge' style='background-color: {card_accent};'>{ins['count']} Customers ({ins['percentage']:.1f}%)</span>
                        </div>
                    </div>
                    <div style='font-size: 13px; font-weight: 600; color: #77736E; margin-bottom: 10px;'>
                        Category: <span style='color: #292929;'>{ins['category']}</span>
                    </div>
                    <div style='display: flex; gap: 20px; background-color: #FCFAF5; padding: 10px; border-radius: 8px; margin-bottom: 10px;'>
                        <div><strong>Avg Income:</strong> ${ins['avg_income']:.1f}k</div>
                        <div><strong>Avg Spending Score:</strong> {ins['avg_spending']:.1f} / 100</div>
                        {'<div><strong>Avg Age:</strong> ' + f"{ins['avg_age']:.1f} yrs</div>" if ins.get('avg_age') else ''}
                    </div>
                    <div style='font-size: 14px; color: #292929; margin-bottom: 8px;'>
                        <strong>Behavioral Profile:</strong> {ins['description']}
                    </div>
                    <div style='font-size: 14px; color: #171717; background-color: #F8F3E7; padding: 10px; border-radius: 8px; border: 1px solid #E7E1D7;'>
                        🎯 <strong>Recommended Strategy:</strong> {ins['strategy']}
                    </div>
                </div>
            """, unsafe_allow_html=True)


# ==========================================
# PAGE 5: CUSTOMER EXPLORER
# ==========================================
elif nav_page == "Customer Explorer":
    st.markdown("<div class='page-title'>Customer Record Explorer</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-subtitle'>Search, filter, and export individual customer profiles based on income, spending score, and cluster assignment.</div>", unsafe_allow_html=True)
    
    if df_clustered.empty:
        st.warning("No dataset loaded.")
    else:
        # Filter Controls Card
        st.markdown("<div class='dashboard-card'>", unsafe_allow_html=True)
        st.markdown("<div style='font-weight: 600; margin-bottom: 10px;'>🔍 Search & Filter Controls</div>", unsafe_allow_html=True)
        
        f1, f2, f3 = st.columns([1, 1, 1])
        
        with f1:
            search_id = st.text_input("Search by Customer ID:", placeholder="e.g. CUST-1005")
            
        with f2:
            cluster_opts = sorted(df_clustered["Cluster"].unique().tolist(), key=lambda x: int(x.split()[-1]))
            selected_clusters = st.multiselect("Filter by Cluster:", options=cluster_opts, default=cluster_opts)
            
        with f3:
            if income_col and income_col in df_clustered.columns:
                min_inc = float(df_clustered[income_col].min())
                max_inc = float(df_clustered[income_col].max())
                inc_range = st.slider("Income Range ($k):", min_value=min_inc, max_value=max_inc, value=(min_inc, max_inc))
            else:
                inc_range = (0.0, 1000.0)
                
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Apply Filters
        filtered_df = df_clustered.copy()
        
        if search_id.strip():
            id_str = filtered_df["__display_id__"].astype(str)
            filtered_df = filtered_df[id_str.str.contains(search_id.strip(), case=False)]
            
        if selected_clusters:
            filtered_df = filtered_df[filtered_df["Cluster"].isin(selected_clusters)]
            
        if income_col and income_col in filtered_df.columns:
            filtered_df = filtered_df[(filtered_df[income_col] >= inc_range[0]) & (filtered_df[income_col] <= inc_range[1])]
            
        # Display Results Info & Download Button
        res_col1, res_col2 = st.columns([2, 1])
        with res_col1:
            st.markdown(f"Showing **{len(filtered_df)}** of **{len(df_clustered)}** matching customer records.")
            
        with res_col2:
            csv_data = filtered_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Filtered Data (CSV)",
                data=csv_data,
                file_name="segmented_customers_filtered.csv",
                mime="text/csv",
                use_container_width=True
            )
            
        # Filtered Table
        st.dataframe(filtered_df, use_container_width=True, hide_index=True)


# ==========================================
# PAGE 6: UPLOAD DATASET
# ==========================================
elif nav_page == "Upload Dataset":
    st.markdown("<div class='page-title'>Dataset Management & File Upload</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-subtitle'>Upload your custom CSV file, configure column mappings, and validate data quality.</div>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Choose a CSV file to upload", type=["csv"])
    
    if uploaded_file is not None:
        df_up, err = load_dataset(uploaded_file)
        if err:
            st.error(err)
        else:
            st.success(f"File successfully loaded! Found **{len(df_up)}** rows and **{len(df_up.columns)}** columns.")
            
            # Preview uploaded data
            st.markdown("<div class='section-header'>Preview Uploaded Dataset</div>", unsafe_allow_html=True)
            st.dataframe(df_up.head(10), use_container_width=True)
            
            # Auto-detected Column Mapping & Manual Override
            st.markdown("<div class='section-header'>Column Mapping</div>", unsafe_allow_html=True)
            st.caption("Verify or adjust column assignments for segmentation:")
            
            auto_map = auto_detect_columns(df_up)
            cols = ["-- None / Generate --"] + df_up.columns.tolist()
            
            m1, m2, m3 = st.columns(3)
            with m1:
                sel_id = st.selectbox("Customer ID Column:", options=cols, index=cols.index(auto_map["id"]) if auto_map["id"] in cols else 0)
            with m2:
                sel_inc = st.selectbox("Annual Income Column:", options=cols, index=cols.index(auto_map["income"]) if auto_map["income"] in cols else 0)
            with m3:
                sel_spd = st.selectbox("Spending Score Column:", options=cols, index=cols.index(auto_map["spending"]) if auto_map["spending"] in cols else 0)
                
            if st.button("Apply Custom Dataset to Dashboard", use_container_width=True):
                st.session_state.raw_df = df_up
                st.session_state.dataset_source = "uploaded"
                st.session_state.column_mapping = {
                    "id": sel_id if sel_id != "-- None / Generate --" else None,
                    "income": sel_inc if sel_inc != "-- None / Generate --" else None,
                    "spending": sel_spd if sel_spd != "-- None / Generate --" else None,
                    "age": auto_map.get("age"),
                    "gender": auto_map.get("gender")
                }
                
                # Update selected features
                feats = []
                if sel_inc != "-- None / Generate --": feats.append(sel_inc)
                if sel_spd != "-- None / Generate --": feats.append(sel_spd)
                if not feats:
                    num_c = df_up.select_dtypes(include=[np.number]).columns.tolist()
                    feats = num_c[:2]
                st.session_state.selected_features = feats
                
                st.success("Custom dataset applied successfully! Navigate to Overview or Segmentation to inspect results.")
                st.rerun()


# ==========================================
# PAGE 7: ABOUT PROJECT
# ==========================================
elif nav_page == "About Project":
    st.markdown("<div class='page-title'>About Segmently — Project Overview</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-subtitle'>BTech CSE AI/ML Minor Project Documentation & Algorithm Explanation.</div>", unsafe_allow_html=True)
    
    col_a1, col_a2 = st.columns(2)
    
    with col_a1:
        st.markdown("""
            <div class='dashboard-card'>
                <div style='font-size: 16px; font-weight: 700; margin-bottom: 8px;'>🎯 Problem Statement & Objective</div>
                <p style='font-size: 14px; color: #555555;'>
                    Businesses often struggle to personalize marketing campaigns for diverse customer bases. Unsupervised Machine Learning allows automated grouping of customers based on spending patterns and income levels without requiring prior manual labeling.
                </p>
                <div style='font-size: 16px; font-weight: 700; margin-top: 16px; margin-bottom: 8px;'>🤖 Machine Learning Algorithm</div>
                <p style='font-size: 14px; color: #555555;'>
                    <strong>K-Means Clustering:</strong> Partitioning $N$ observations into $K$ clusters in which each observation belongs to the cluster with the nearest mean centroid. Features are normalized using <code>StandardScaler</code> to ensure equal weighting across dimensions.
                </p>
            </div>
        """, unsafe_allow_html=True)
        
    with col_a2:
        st.markdown("""
            <div class='dashboard-card'>
                <div style='font-size: 16px; font-weight: 700; margin-bottom: 8px;'>🛠️ Technology Stack</div>
                <ul style='font-size: 14px; color: #555555;'>
                    <li><strong>Language:</strong> Python 3.14</li>
                    <li><strong>Web Framework:</strong> Streamlit</li>
                    <li><strong>Data Processing:</strong> Pandas, NumPy</li>
                    <li><strong>Machine Learning:</strong> Scikit-learn (KMeans, StandardScaler, Silhouette Score)</li>
                    <li><strong>Visualization:</strong> Plotly Express & Plotly Graph Objects</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
        
    st.markdown("""
        <div class='dashboard-card'>
            <div style='font-size: 16px; font-weight: 700; margin-bottom: 8px;'>💡 Real-World Applications & Future Scope</div>
            <p style='font-size: 14px; color: #555555;'>
                <strong>Applications:</strong> Targeted promotional campaigns, personalized product recommendations, customer churn prevention, loyalty program structuring, and retail inventory planning.<br><br>
                <strong>Future Scope:</strong> Integration with real-time transactional databases, Hierarchical Clustering comparisons, and Automated Recommendation engines.
            </p>
        </div>
    """, unsafe_allow_html=True)
