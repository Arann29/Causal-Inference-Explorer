"""
Data visualization panel - Connected to helpers/visualization_segmentation.py
"""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import sys
sys.path.append('/app/helpers')

def render():
    """Render the visualization panel component"""
    st.header("📈 Data Visualization")
    
    if st.session_state.get('current_dataframe') is None:
        st.info("Please load a dataset first")
        return
    
    df = st.session_state.current_dataframe
    
    if df.empty:
        st.warning("Dataset is empty")
        return
    
    # Get column names
    columns = df.columns.tolist()
    
    if len(columns) < 2:
        st.warning("Dataset needs at least 2 columns for visualization")
        return
    
    # Check if segmentation has been applied
    segmentation_applied = st.session_state.get('segmentation_applied', False)
    segment_labels = st.session_state.get('segmentation_result')
    
    col1, col2 = columns[0], columns[1]
    
    if segmentation_applied and segment_labels is not None:
        st.subheader("📊 Regime-Colored Visualization")
        
        # Convert to string labels for better visualization
        segment_labels_str = np.array(segment_labels).astype(str)
        
        # Create figure with regime coloring
        fig = px.scatter(
            df,
            x=col1,
            y=col2,
            color=segment_labels_str,
            title=f"{col1} vs {col2} (Regime Segmentation)",
            labels={"color": "Regime"},
            opacity=0.7,
            color_discrete_sequence=px.colors.qualitative.Set1
        )
        
        fig.update_layout(
            showlegend=True,
            width=800,
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add segment statistics
        unique_segments = np.unique(segment_labels)
        n_segments = len(unique_segments)
        
        st.info(f"📊 Showing {n_segments} regimes with {len(df)} total points")
        
        # Show segment sizes
        cols = st.columns(n_segments)
        for idx, segment in enumerate(unique_segments):
            mask = segment_labels == segment
            count = np.sum(mask)
            pct = (count / len(segment_labels)) * 100
            
            with cols[idx]:
                st.metric(
                    f"Regime {segment}",
                    f"{count} points",
                    f"{pct:.1f}%"
                )
        
        # Show statistics per regime
        with st.expander("📊 Regime Statistics"):
            for segment in unique_segments:
                mask = segment_labels == segment
                segment_data = df.loc[mask, [col1, col2]]
                
                st.write(f"**Regime {segment}:**")
                st.write(segment_data.describe())
                st.write("---")
    
    else:
        st.subheader("📊 Basic Visualization")
        st.info("💡 Apply segmentation from the sidebar to see regime-colored plots")
        
        # Create base scatter plot
        fig = px.scatter(
            df, 
            x=col1, 
            y=col2,
            title=f"{col1} vs {col2}",
            opacity=0.7
        )
        
        fig.update_layout(
            showlegend=True,
            width=800,
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Additional visualization options
    with st.expander("📊 Additional Visualizations"):
        viz_type = st.selectbox(
            "Visualization Type:",
            ["Correlation Heatmap", "Distribution Plot", "Box Plot"]
        )
        
        if viz_type == "Correlation Heatmap":
            # Only show numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) > 1:
                corr_matrix = df[numeric_cols].corr()
                fig_corr = px.imshow(
                    corr_matrix,
                    text_auto=True,
                    aspect="auto",
                    title="Correlation Matrix"
                )
                st.plotly_chart(fig_corr, use_container_width=True)
            else:
                st.warning("Need at least 2 numeric columns for correlation heatmap")
        
        elif viz_type == "Distribution Plot":
            selected_col = st.selectbox("Select column:", columns)
            fig_dist = px.histogram(
                df,
                x=selected_col,
                nbins=30,
                title=f"Distribution of {selected_col}"
            )
            st.plotly_chart(fig_dist, use_container_width=True)
        
        elif viz_type == "Box Plot":
            selected_col = st.selectbox("Select column:", columns)
            fig_box = px.box(
                df,
                y=selected_col,
                title=f"Box Plot of {selected_col}"
            )
            st.plotly_chart(fig_box, use_container_width=True)
