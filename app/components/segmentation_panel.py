"""
Segmentation configuration and execution panel
Connected to helpers/visualization_segmentation.py
"""
import streamlit as st
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

sys.path.append("/app/helpers")
sys.path.append("/app")
from helpers.visualization_segmentation import segment_data
from app.utils.ground_truth import get_ground_truth


def render():
    """Render the segmentation panel component"""
    st.sidebar.header("🔗 Regime Segmentation")

    if "current_dataframe" not in st.session_state or st.session_state.current_dataframe is None:
        st.sidebar.info("📂 Load a dataset first.")
        return

    df = st.session_state.current_dataframe
    
    # Get ground truth if available
    dataset_name = st.session_state.get('selected_dataset', {}).get('id', '')
    ground_truth = get_ground_truth(dataset_name) if dataset_name else None
    
    st.sidebar.subheader("Literature-Based Threshold")
    
    # Determine default column and threshold
    default_column = df.columns.tolist()[0]
    default_threshold = float(df[default_column].mean())
    
    if ground_truth:
        # Use threshold from pairmeta file
        threshold_var = ground_truth['threshold_var']
        threshold_val = ground_truth['threshold_val']
        
        # Find matching column
        matching_cols = [col for col in df.columns if threshold_var.lower() in col.lower() or col.lower() in threshold_var.lower()]
        if matching_cols:
            default_column = matching_cols[0]
            default_threshold = threshold_val
            st.sidebar.success(f"📋 Using ground truth threshold from pairmeta")
    
    column = st.sidebar.selectbox("Column for thresholding:", df.columns.tolist(), index=df.columns.tolist().index(default_column))
    
    threshold_value = st.sidebar.number_input(
        "Threshold value:", 
        value=default_threshold,
        format="%.2f",
        help="Threshold value from pairmeta file" if ground_truth else "Enter threshold value"
    )

    if st.sidebar.button("Apply Segmentation", type="primary"):
        if column and threshold_value is not None:
            with st.spinner("Applying segmentation..."):
                try:
                    # ✅ Step 1: Normalize ENTIRE dataset FIRST
                    columns = df.columns.tolist()
                    var1, var2 = columns[0], columns[1]
                    
                    scaler = StandardScaler()
                    df_normalized = df[[var1, var2]].copy()
                    df_normalized[[var1, var2]] = scaler.fit_transform(df_normalized[[var1, var2]])

                    # ✅ Step 2: Transform threshold to normalized scale
                    threshold_idx = 0 if column == var1 else 1
                    dummy = np.zeros((1, 2))
                    dummy[0, threshold_idx] = threshold_value
                    threshold_normalized = scaler.transform(dummy)[0, threshold_idx]
                    
                    st.sidebar.info(f"Threshold: {threshold_value:.2f} → {threshold_normalized:.4f} (normalized)")
                    
                    # ✅ Step 3: Apply segmentation on NORMALIZED data
                    labels = segment_data(
                        df_normalized,
                        strategy="threshold",
                        column=column,
                        threshold=threshold_normalized,
                        plot_results=False
                    )
                    
                    # ✅ Step 4: Store results in session state
                    st.session_state.segmentation_result = labels
                    st.session_state.segmentation_applied = True
                    st.session_state.normalized_dataframe = df_normalized
                    st.session_state.scaler = scaler
                    
                    n_segments = len(np.unique(labels[labels >= 0]))
                    
                    st.session_state.segmentation_metadata = {
                        'method': 'threshold',
                        'column': column,
                        'threshold': threshold_value,
                        'threshold_normalized': threshold_normalized,
                        'n_segments': n_segments
                    }
                    st.session_state.segmentation_method = "threshold"
                    st.session_state.segmentation_kwargs = {
                        'column': column, 
                        'threshold': threshold_value
                    }
                    
                    # Display statistics
                    unique_segments, counts = np.unique(labels, return_counts=True)
                    for segment_id, count in zip(unique_segments, counts):
                        pct = (count / len(labels)) * 100
                        st.sidebar.write(f"Regime {segment_id}: {count} samples ({pct:.1f}%)")
                    
                    st.sidebar.success(f"✅ Segmentation applied: {n_segments} regime(s) created")
                    
                except Exception as e:
                    st.sidebar.error(f"❌ Error: {str(e)}")
                    import traceback
                    st.sidebar.code(traceback.format_exc())
        else:
            st.sidebar.warning("⚠️ Please specify a column and threshold.")

    # Display current segmentation status
    if "segmentation_result" in st.session_state and st.session_state.segmentation_result is not None:
        metadata = st.session_state.get('segmentation_metadata', {})
        n_segments = metadata.get('n_segments', 'Unknown')
        st.sidebar.info(f"✅ Active: {n_segments} regime(s)")
        
        with st.sidebar.expander("📋 Segmentation Details"):
            st.json(metadata)
