"""
Visualization Module for Customer Segmentation Dashboard.
Uses Plotly to generate interactive charts adhering to the pastel color palette and warm cream design system.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Optional, Any

# Pastel Color Palette
PASTEL_COLORS = [
    '#C8D9F0',  # Pastel Blue
    '#F2C6DE',  # Pastel Pink
    '#F7DF78',  # Pastel Yellow
    '#D4E8B8',  # Pastel Green
    '#DCD1F3',  # Pastel Lavender
    '#FFD1B3',  # Pastel Peach
    '#BCE6FF',  # Soft Sky
    '#E2D0F9',  # Soft Violet
    '#C5E0B4',  # Soft Mint
    '#F9C7C7'   # Soft Rose
]

THEME_BG = '#F8F3E7'
CARD_BG = '#FFFFFF'
PRIMARY_TEXT = '#292929'
SECONDARY_TEXT = '#77736E'
BORDER_COLOR = '#E7E1D7'


def get_cluster_color_map(clusters: List[str]) -> Dict[str, str]:
    """
    Returns a consistent mapping from Cluster names (e.g. Cluster 1) to pastel hex colors.
    """
    color_map = {}
    sorted_clusters = sorted(list(clusters), key=lambda x: int(x.split()[-1]) if ' ' in x and x.split()[-1].isdigit() else x)
    for idx, c in enumerate(sorted_clusters):
        color_map[c] = PASTEL_COLORS[idx % len(PASTEL_COLORS)]
    return color_map


def plot_customer_segmentation_scatter(
    df_clustered: pd.DataFrame,
    income_col: str,
    spending_col: str,
    kmeans_model: Optional[Any] = None,
    scaler: Optional[Any] = None,
    feature_cols: Optional[List[str]] = None
) -> go.Figure:
    """
    Generates an interactive scatter plot of Annual Income vs Spending Score colored by cluster.
    Includes centroid markers if valid.
    """
    clusters = sorted(df_clustered["Cluster"].unique(), key=lambda x: int(x.split()[-1]))
    color_map = get_cluster_color_map(clusters)
    
    hover_id = "__display_id__" if "__display_id__" in df_clustered.columns else df_clustered.columns[0]
    
    hover_dict = {
        income_col: ':.1f',
        spending_col: ':.1f',
        "Cluster": True
    }
    if "Category" in df_clustered.columns:
        hover_dict["Category"] = True
    if "Persona" in df_clustered.columns:
        hover_dict["Persona"] = True
    
    fig = px.scatter(
        df_clustered,
        x=income_col,
        y=spending_col,
        color="Cluster",
        color_discrete_map=color_map,
        hover_name=hover_id,
        hover_data=hover_dict,
        title="Customer Segments (Income vs. Spending)",
        category_orders={"Cluster": clusters}
    )
    
    # Add Cluster Centroids if available
    if kmeans_model is not None and scaler is not None and feature_cols is not None:
        if income_col in feature_cols and spending_col in feature_cols:
            idx_inc = feature_cols.index(income_col)
            idx_spd = feature_cols.index(spending_col)
            
            # Inverse transform centroids
            centroids_unscaled = scaler.inverse_transform(kmeans_model.cluster_centers_)
            
            fig.add_trace(
                go.Scatter(
                    x=centroids_unscaled[:, idx_inc],
                    y=centroids_unscaled[:, idx_spd],
                    mode='markers+text',
                    marker=dict(
                        symbol='star',
                        size=18,
                        color='#171717',
                        line=dict(color='#FFFFFF', width=1.5)
                    ),
                    text=[f"C{i+1}" for i in range(len(centroids_unscaled))],
                    textposition="top center",
                    name="Centroids",
                    hoverinfo="text",
                    hovertext=[f"Centroid Cluster {i+1}" for i in range(len(centroids_unscaled))]
                )
            )
            
    fig.update_traces(marker=dict(size=10, opacity=0.85, line=dict(width=1, color='#292929')))
    
    fig.update_layout(
        paper_bgcolor=CARD_BG,
        plot_bgcolor='#FCFAF5',
        font=dict(family="Inter, sans-serif", color=PRIMARY_TEXT, size=13),
        title=dict(text="<b>Customer Segments (Income vs. Spending)</b>", font=dict(size=17, color=PRIMARY_TEXT)),
        xaxis=dict(
            title=f"<b>{income_col}</b>",
            gridcolor='#EAE5D9',
            zerolinecolor='#EAE5D9',
            showline=True,
            linecolor=BORDER_COLOR
        ),
        yaxis=dict(
            title=f"<b>{spending_col}</b>",
            gridcolor='#EAE5D9',
            zerolinecolor='#EAE5D9',
            showline=True,
            linecolor=BORDER_COLOR
        ),
        legend=dict(
            title=dict(text="<b>Clusters</b>"),
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor=BORDER_COLOR,
            borderwidth=1
        ),
        margin=dict(l=40, r=40, t=50, b=40),
        height=450
    )
    
    return fig


def plot_cluster_distribution_donut(df_clustered: pd.DataFrame) -> go.Figure:
    """
    Generates a donut chart displaying customer count and percentage per cluster.
    """
    counts = df_clustered["Cluster"].value_counts().reset_index()
    counts.columns = ["Cluster", "Count"]
    counts = counts.sort_values(by="Cluster", key=lambda x: x.map(lambda v: int(v.split()[-1])))
    
    color_map = get_cluster_color_map(counts["Cluster"].tolist())
    colors = [color_map[c] for c in counts["Cluster"]]
    
    fig = go.Figure(data=[go.Pie(
        labels=counts["Cluster"],
        values=counts["Count"],
        hole=0.55,
        marker=dict(colors=colors, line=dict(color='#FFFFFF', width=2)),
        textinfo='label+percent',
        hoverinfo='label+value+percent',
        textfont=dict(size=12, color=PRIMARY_TEXT)
    )])
    
    fig.update_layout(
        paper_bgcolor=CARD_BG,
        plot_bgcolor=CARD_BG,
        font=dict(family="Inter, sans-serif", color=PRIMARY_TEXT, size=13),
        title=dict(text="<b>Cluster Share & Size</b>", font=dict(size=17, color=PRIMARY_TEXT)),
        annotations=[dict(
            text=f"<b>{len(df_clustered)}</b><br><span style='font-size:11px;color:{SECONDARY_TEXT}'>Customers</span>",
            x=0.5, y=0.5,
            font_size=18,
            showarrow=False
        )],
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.25,
            xanchor="center",
            x=0.5,
            font=dict(size=11)
        ),
        margin=dict(l=20, r=20, t=50, b=50),
        height=450
    )
    
    return fig


def plot_feature_distribution(
    df: pd.DataFrame,
    col: str,
    title_label: str,
    color: str = '#C8D9F0'
) -> go.Figure:
    """
    Generates a clean histogram with marginal box plot for a numerical column or bar chart for categorical.
    """
    if pd.api.types.is_numeric_dtype(df[col]):
        fig = px.histogram(
            df,
            x=col,
            marginal="box",
            nbins=20,
            color_discrete_sequence=[color],
            title=f"Distribution of {title_label}"
        )
        fig.update_traces(marker=dict(line=dict(width=1, color='#292929')))
    else:
        counts = df[col].value_counts().reset_index()
        counts.columns = [col, "Count"]
        fig = px.bar(
            counts,
            x=col,
            y="Count",
            color_discrete_sequence=[color],
            title=f"Distribution of {title_label}",
            text_auto=True
        )
        fig.update_traces(marker=dict(line=dict(width=1, color='#292929')))

    fig.update_layout(
        paper_bgcolor=CARD_BG,
        plot_bgcolor='#FCFAF5',
        font=dict(family="Inter, sans-serif", color=PRIMARY_TEXT, size=12),
        title=dict(text=f"<b>{title_label} Distribution</b>", font=dict(size=15, color=PRIMARY_TEXT)),
        xaxis=dict(gridcolor='#EAE5D9', showline=True, linecolor=BORDER_COLOR),
        yaxis=dict(gridcolor='#EAE5D9', showline=True, linecolor=BORDER_COLOR),
        margin=dict(l=30, r=30, t=45, b=30),
        height=320
    )
    return fig


def plot_cluster_comparison_bars(insights_list: List[Dict[str, Any]]) -> go.Figure:
    """
    Generates a grouped bar chart comparing Average Annual Income and Average Spending Score for each cluster.
    """
    df_ins = pd.DataFrame(insights_list)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df_ins["cluster_name"],
        y=df_ins["avg_income"],
        name="Avg Income",
        marker_color='#C8D9F0',
        marker_line=dict(width=1, color='#292929'),
        text=[f"${v:.1f}k" for v in df_ins["avg_income"]],
        textposition="auto"
    ))
    
    fig.add_trace(go.Bar(
        x=df_ins["cluster_name"],
        y=df_ins["avg_spending"],
        name="Avg Spending Score",
        marker_color='#F2C6DE',
        marker_line=dict(width=1, color='#292929'),
        text=[f"{v:.1f}" for v in df_ins["avg_spending"]],
        textposition="auto"
    ))
    
    fig.update_layout(
        barmode='group',
        paper_bgcolor=CARD_BG,
        plot_bgcolor='#FCFAF5',
        font=dict(family="Inter, sans-serif", color=PRIMARY_TEXT, size=12),
        title=dict(text="<b>Average Metrics Comparison Across Clusters</b>", font=dict(size=16, color=PRIMARY_TEXT)),
        xaxis=dict(title="<b>Cluster</b>", showline=True, linecolor=BORDER_COLOR),
        yaxis=dict(title="<b>Value</b>", gridcolor='#EAE5D9', showline=True, linecolor=BORDER_COLOR),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=30, r=30, t=50, b=30),
        height=380
    )
    
    return fig


def plot_elbow_curve(elbow_data: Dict[int, float], current_k: int = 5) -> go.Figure:
    """
    Generates the Elbow Method line chart to evaluate the optimal number of clusters.
    """
    ks = list(elbow_data.keys())
    inertias = list(elbow_data.values())
    
    fig = go.Figure()
    
    # Line trace
    fig.add_trace(go.Scatter(
        x=ks,
        y=inertias,
        mode='lines+markers',
        line=dict(color='#292929', width=2.5),
        marker=dict(size=8, color='#C8D9F0', line=dict(width=1.5, color='#292929')),
        name="Inertia"
    ))
    
    # Highlight current k
    if current_k in elbow_data:
        fig.add_trace(go.Scatter(
            x=[current_k],
            y=[elbow_data[current_k]],
            mode='markers',
            marker=dict(size=14, color='#F2C6DE', symbol='star', line=dict(width=2, color='#171717')),
            name=f"Current K = {current_k}"
        ))
        
    fig.update_layout(
        paper_bgcolor=CARD_BG,
        plot_bgcolor='#FCFAF5',
        font=dict(family="Inter, sans-serif", color=PRIMARY_TEXT, size=12),
        title=dict(text="<b>Elbow Method (Inertia vs. Number of Clusters K)</b>", font=dict(size=16, color=PRIMARY_TEXT)),
        xaxis=dict(title="<b>Number of Clusters (K)</b>", dtick=1, gridcolor='#EAE5D9', showline=True, linecolor=BORDER_COLOR),
        yaxis=dict(title="<b>Inertia (WCSS)</b>", gridcolor='#EAE5D9', showline=True, linecolor=BORDER_COLOR),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=50, b=40),
        height=360
    )
    
    return fig
