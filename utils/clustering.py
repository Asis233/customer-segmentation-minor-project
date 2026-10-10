"""
Clustering Module for Customer Segmentation Dashboard.
Implements StandardScaler scaling, K-Means model training, metrics evaluation, elbow method,
and dynamic cluster persona / insight generation based on dataset statistics.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from typing import Tuple, Dict, List, Optional, Any


def perform_kmeans_clustering(
    df: pd.DataFrame,
    feature_cols: List[str],
    n_clusters: int = 5,
    random_state: int = 42
) -> Tuple[pd.DataFrame, KMeans, StandardScaler, np.ndarray]:
    """
    Scales features, fits K-Means model, and assigns cluster labels to the DataFrame.
    """
    df_clustered = df.copy()
    
    # Extract numerical features
    X = df_clustered[feature_cols].values
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit K-Means
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10
    )
    cluster_indices = kmeans.fit_predict(X_scaled)
    
    # Add cluster labels (1-indexed for display)
    df_clustered["Cluster_Id"] = cluster_indices
    df_clustered["Cluster"] = [f"Cluster {i+1}" for i in cluster_indices]
    
    return df_clustered, kmeans, scaler, X_scaled


def evaluate_kmeans_model(
    df_clustered: pd.DataFrame,
    X_scaled: np.ndarray,
    kmeans_model: KMeans
) -> Dict[str, Any]:
    """
    Calculates K-Means performance metrics (Inertia, Silhouette Score).
    Handles edge cases like single cluster or insufficient samples gracefully.
    """
    n_clusters = kmeans_model.n_clusters
    n_samples = len(df_clustered)
    
    inertia = float(kmeans_model.inertia_)
    
    sil_score = None
    sil_message = ""
    
    if n_clusters < 2:
        sil_message = "Silhouette score requires at least 2 clusters."
    elif n_samples <= n_clusters:
        sil_message = "Insufficient data points for silhouette score calculation."
    else:
        try:
            sil_score = float(silhouette_score(X_scaled, kmeans_model.labels_))
            sil_message = "Calculated successfully."
        except Exception as e:
            sil_message = f"Could not calculate silhouette score: {str(e)}"
            
    return {
        "n_clusters": n_clusters,
        "n_samples": n_samples,
        "inertia": inertia,
        "silhouette_score": sil_score,
        "silhouette_message": sil_message
    }


def compute_elbow_curve(
    X_scaled: np.ndarray,
    max_k: int = 10,
    random_state: int = 42
) -> Dict[int, float]:
    """
    Computes inertia values for k ranging from 2 up to max_k for the Elbow Method plot.
    """
    elbow_data = {}
    n_samples = len(X_scaled)
    limit_k = min(max_k, n_samples - 1)
    
    for k in range(1, limit_k + 1):
        try:
            km = KMeans(n_clusters=k, random_state=random_state, n_init=5)
            km.fit(X_scaled)
            elbow_data[k] = float(km.inertia_)
        except Exception:
            pass
            
    return elbow_data


def classify_cluster_persona(
    avg_income: float,
    med_income: float,
    avg_spending: float,
    med_spending: float
) -> Tuple[str, str, str, str]:
    """
    Dynamically classifies a cluster based on average income and spending score relative to dataset medians/means.
    
    Returns:
        Tuple[str, str, str, str]: (Category Name, Persona Title, Spending Behavior Description, Recommended Strategy)
    """
    income_high = avg_income >= med_income
    spending_high = avg_spending >= med_spending
    
    # Check if close to median (moderate group)
    inc_diff_pct = abs(avg_income - med_income) / max(med_income, 1.0)
    spend_diff_pct = abs(avg_spending - med_spending) / max(med_spending, 1.0)
    
    if inc_diff_pct < 0.15 and spend_diff_pct < 0.15:
        category = "Average Earners & Moderate Spenders"
        persona = "Core Standard Customers"
        desc = "Customers with average annual income who exhibit balanced and predictable spending habits."
        strategy = "Maintain engagement with rewards programs, standard product recommendations, and steady promotional updates."
    elif income_high and spending_high:
        category = "High Income & High Spending"
        persona = "Target / Premium Customers"
        desc = "High annual income paired with high spending score. These are your most lucrative and high-value customers."
        strategy = "Focus on premium loyalty tiers, exclusive early access, luxury offerings, and personalized VIP service."
    elif income_high and not spending_high:
        category = "High Income & Low Spending"
        persona = "Careful / Saver Customers"
        desc = "Substantial income but low spending score. They have high purchasing power but are cautious or selective."
        strategy = "Target with high-value propositions, premium quality assurances, competitive discounts, and targeted value campaigns."
    elif not income_high and spending_high:
        category = "Low Income & High Spending"
        persona = "Impulse / Trendsetter Customers"
        desc = "Lower annual income paired with a high spending score. Enthusiastic buyers driven by trends and immediate satisfaction."
        strategy = "Promote flexible payment options, seasonal flash sales, trendy visual marketing, and rewards incentives."
    else: # Low Income, Low Spending
        category = "Low Income & Low Spending"
        persona = "Sensible / Budget Customers"
        desc = "Lower income and low spending score. Highly price-sensitive shoppers looking for essential items and discounts."
        strategy = "Offer bundle deals, essential discount promotions, budget recommendations, and value-for-money items."
        
    return category, persona, desc, strategy


def generate_cluster_insights(
    df_clustered: pd.DataFrame,
    income_col: Optional[str],
    spending_col: Optional[str]
) -> List[Dict[str, Any]]:
    """
    Generates detailed insights, personas, and strategic recommendations for each cluster.
    """
    insights = []
    total_customers = len(df_clustered)
    
    if total_customers == 0:
        return insights
        
    # Determine reference dataset benchmarks
    if income_col and income_col in df_clustered.columns:
        med_income = float(df_clustered[income_col].median())
    else:
        med_income = 50.0
        
    if spending_col and spending_col in df_clustered.columns:
        med_spending = float(df_clustered[spending_col].median())
    else:
        med_spending = 50.0
        
    cluster_labels = sorted(df_clustered["Cluster"].unique(), key=lambda x: int(x.split()[-1]))
    
    for cluster_name in cluster_labels:
        c_df = df_clustered[df_clustered["Cluster"] == cluster_name]
        count = len(c_df)
        pct = (count / total_customers) * 100.0
        
        avg_inc = float(c_df[income_col].mean()) if income_col and income_col in c_df.columns else 0.0
        avg_spend = float(c_df[spending_col].mean()) if spending_col and spending_col in c_df.columns else 0.0
        avg_age = float(c_df["Age"].mean()) if "Age" in c_df.columns and not c_df["Age"].isna().all() else None
        
        category, persona, desc, strategy = classify_cluster_persona(avg_inc, med_income, avg_spend, med_spending)
        
        insights.append({
            "cluster_name": cluster_name,
            "count": count,
            "percentage": pct,
            "avg_income": avg_inc,
            "avg_spending": avg_spend,
            "avg_age": avg_age,
            "category": category,
            "persona": persona,
            "description": desc,
            "strategy": strategy
        })
        
    return insights
