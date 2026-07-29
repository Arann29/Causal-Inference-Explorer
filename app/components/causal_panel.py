"""
Causal discovery panel - Connected to helpers/causal_param_optimization.py
"""
import streamlit as st
import sys
import torch
sys.path.append('/app/helpers')
from helpers.causal_param_optimization import run_segmented_causal_switch

def render():
    """Render the causal discovery panel component"""
    st.header("🧠 Causal Discovery")
    
    if st.session_state.get('segmentation_result') is None:
        st.info("← Please load a dataset and apply segmentation first.")
        return
    
    df = st.session_state.current_dataframe
    
    # Validate dataset has at least 2 columns
    if df is None or len(df.columns) < 2:
        st.error("ERROR: Dataset must have at least 2 columns for causal analysis.")
        return
    
    # Display which variables will be analyzed
    columns = df.columns.tolist()
    var1, var2 = columns[0], columns[1]
    st.info(f"→ Analyzing causal relationship: **{var1}** ↔ **{var2}**")
    
    col1, col2 = st.columns(2)
    with col1:
        causal_method = st.selectbox(
            "Causal Method:",
            ["roche", "loci", "lcube", "lingam"],
            help="Choose causal discovery method"
        )
    with col2:
        device = st.selectbox(
            "Device:",
            ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"],
            disabled=not torch.cuda.is_available(),
            help="CUDA is required for GPU acceleration."
        )

    if st.button("▶ Run Causal Analysis", type="primary", use_container_width=True):
        try:
            with st.spinner(f"Running {causal_method.upper()} analysis..."):
                segmentation_strategy = st.session_state.get('segmentation_method', 'threshold')
                segmentation_kwargs = st.session_state.get('segmentation_kwargs', {})
                
                # Get variable names (already validated above)
                columns = df.columns.tolist()
                cause_col, effect_col = columns[0], columns[1]
                
                # Call the causal discovery function
                result = run_segmented_causal_switch(
                    df=df.copy(),
                    method=causal_method,
                    segmentation_strategy=segmentation_strategy,
                    segmentation_kwargs=segmentation_kwargs,
                    device=device,
                    cause_col=cause_col,
                    effect_col=effect_col
                )
                
                # Store results by method name
                if 'causal_results_by_method' not in st.session_state:
                    st.session_state.causal_results_by_method = {}
                st.session_state.causal_results_by_method[causal_method] = result
                
        except Exception as e:
            st.error(f"An error occurred during causal analysis: {e}")
            import traceback
            st.code(traceback.format_exc())

    # Display results for the selected method if available
    causal_results_by_method = st.session_state.get('causal_results_by_method', {})
    if causal_method in causal_results_by_method:
        st.subheader("🎯 Causal Discovery Results")
        
        results = causal_results_by_method[causal_method]
        method = causal_method.upper()
        
        if not results:
            st.warning("WARNING: Analysis completed but produced no results.")
            return

        # Display method and variable info
        columns = df.columns.tolist()
        var1, var2 = columns[0], columns[1]
        st.info(f"**Method:** {method} | **Variables:** {var1} ↔ {var2}")
        
        # Display results by regime
        for i, (segment_id, direction, score) in enumerate(results):
            # Direct mapping: segment 0=low, 1=high
            regime_name = "Low" if segment_id == 0 else "High"
            regime_label = f"Regime {segment_id} ({regime_name})"
            
            st.markdown(f"**{regime_label}:** `{direction}` (Score: {score:.3f})")
        
        st.success("✓ Causal analysis complete.")
