"""Results comparison panel - Compare ROCHE, LOCI, and LLM results with ground truth"""
import streamlit as st
import pandas as pd
import json
import sys
import os

# Add utils to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))

from app.utils.ground_truth import (
    get_ground_truth, 
    map_prediction_to_numeric, 
    map_numeric_to_string,
    get_ground_truth_summary
)
from app.utils.evaluation import (
    calculate_regime_accuracy,
    detect_regime_switch,
    generate_comparison_table
)


def _parse_llm_conclusion_from_trace() -> dict | None:
    """Fallback parser for legacy sessions where llm_conclusion is missing."""
    llm_results = st.session_state.get('llm_results')
    if not llm_results:
        return None

    try:
        final_response = llm_results[-1]['response']
        if '```json' in final_response:
            final_response = final_response.split('```json')[1].split('```')[0].strip()
        elif '```' in final_response:
            final_response = final_response.split('```')[1].split('```')[0].strip()

        return json.loads(final_response)
    except (IndexError, KeyError, TypeError, json.JSONDecodeError):
        return None

def render():
    """Render the comparison panel component"""
    st.header("📊 Results Comparison")

    causal_results_by_method = st.session_state.get('causal_results_by_method', {})
    llm_conclusion = st.session_state.get('llm_conclusion')
    if not llm_conclusion:
        llm_conclusion = _parse_llm_conclusion_from_trace()
    selected_dataset = st.session_state.get('selected_dataset', {})

    if not causal_results_by_method and not llm_conclusion:
        st.info("← Run Causal Analysis and/or LLM Analysis to see a comparison.")
        return

    # Try to load ground truth for current dataset
    dataset_name = (
        selected_dataset.get('id')
        or selected_dataset.get('file', '').replace('.txt', '')
        or selected_dataset.get('name', '').replace('.txt', '')
    )
    ground_truth = get_ground_truth(dataset_name)
    
    # Get dataframe for variable names
    df = st.session_state.get('current_dataframe')
    
    if ground_truth:
        # Get actual variable names
        from app.utils.ground_truth import get_variable_names
        cause_var, effect_var = get_variable_names(dataset_name, df)
        
        # Display ground truth information
        with st.expander("📋 Ground Truth Information", expanded=True):
            st.info(get_ground_truth_summary(dataset_name, df))
            
            gt_low = map_numeric_to_string(ground_truth['regime_low_dir'], cause_var, effect_var)
            gt_high = map_numeric_to_string(ground_truth['regime_high_dir'], cause_var, effect_var)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Regime Low (Expected)", gt_low)
            with col2:
                st.metric("Regime High (Expected)", gt_high)
            with col3:
                has_switch = ground_truth['regime_low_dir'] != ground_truth['regime_high_dir']
                st.metric("Regime Switch", "Yes" if has_switch else "No")
    
    # Collect all method results for evaluation
    all_method_predictions = {}
    comparison_data = []
    
    # Get variable names for evaluation (fallback to X/Y if df not available)
    cause_var: str = df.columns[0] if df is not None and len(df.columns) >= 2 else "X"
    effect_var: str = df.columns[1] if df is not None and len(df.columns) >= 2 else "Y"
    
    # Process Causal Algorithm Results
    if causal_results_by_method:
        for method_name, results in causal_results_by_method.items():
            method_display = method_name.upper()
            if results:
                if ground_truth and len(results) >= 2:
                    # Store for ground truth evaluation - direct mapping: 0=low, 1=high
                    sorted_results = sorted(results, key=lambda x: x[0])
                    _, dir_low, _ = sorted_results[0]  # Segment 0
                    _, dir_high, _ = sorted_results[1] if len(sorted_results) > 1 else (None, '', None)  # Segment 1
                    all_method_predictions[method_display] = {
                        'regime_low': map_prediction_to_numeric(dir_low, cause_var, effect_var),
                        'regime_high': map_prediction_to_numeric(dir_high, cause_var, effect_var)
                    }
                
                # Add to comparison table
                for segment_id, direction, score in results:
                    regime_name = "Low" if segment_id == 0 else "High"
                    comparison_data.append({
                        "Method": f"{method_display} ({regime_name})",
                        "Predicted Direction": direction,
                        "Score": f"{score:.3f}",
                    })
            else:
                comparison_data.append({
                    "Method": method_display,
                    "Predicted Direction": "No results",
                    "Score": "N/A",
                })

    # Process LLM Results
    if llm_conclusion:
        directions = llm_conclusion.get('directions', [])
        
        if directions:
            if ground_truth and len(directions) >= 2:
                # Store for ground truth evaluation (regime 0=low, regime 1=high)
                # Use 'regime' key first, fall back to 'cluster' for legacy, then position index
                regime_directions = {
                    d.get('regime', d.get('cluster', idx)): d.get('direction', '')
                    for idx, d in enumerate(directions)
                }
                dir_low = regime_directions.get(0, directions[0].get('direction', ''))
                dir_high = regime_directions.get(1, directions[1].get('direction', ''))
                # Check if directions are still in anonymous Variable_X/Variable_Y form
                is_anonymized = (
                    'variable_x' in dir_low.lower() or 'variable_x' in dir_high.lower()
                )
                all_method_predictions['LLM'] = {
                    'regime_low': map_prediction_to_numeric(dir_low, cause_var, effect_var, use_anonymized=is_anonymized),
                    'regime_high': map_prediction_to_numeric(dir_high, cause_var, effect_var, use_anonymized=is_anonymized)
                }
            
            # Add to comparison table
            for idx, regime_info in enumerate(directions):
                regime_num = regime_info.get('regime', regime_info.get('cluster', idx))
                regime_label = 'Low' if regime_num == 0 else ('High' if regime_num == 1 else f'Regime {regime_num}')
                direction = regime_info.get('direction', 'N/A')
                confidence = regime_info.get('confidence', 'N/A')
                
                comparison_data.append({
                    "Method": f"LLM ({regime_label})",
                    "Predicted Direction": direction,
                    "Score": confidence,
                })
        else:
            if st.session_state.get('llm_results'):
                comparison_data.append({
                    "Method": "LLM (OpenRouter)",
                    "Predicted Direction": "See detailed reasoning below",
                    "Score": "See analysis",
                })

    # Show comparison table
    if comparison_data:
        st.subheader("Side-by-Side Comparison")
        df_comparison = pd.DataFrame(comparison_data)
        st.dataframe(df_comparison, use_container_width=True)
    
    # Generate evaluation table if ground truth exists
    if ground_truth and all_method_predictions:
        st.subheader("🎯 Evaluation Against Ground Truth")
        comparison_table = generate_comparison_table(
            all_method_predictions, 
            ground_truth, 
            cause_var=cause_var if cause_var else "X",
            effect_var=effect_var if effect_var else "Y"
        )
        
        st.dataframe(
            comparison_table,
            use_container_width=True,
            hide_index=True
        )
        
        # Summary statistics
        st.subheader("📈 Summary Statistics")
        col1, col2, col3 = st.columns(3)
        
        n_correct_overall = sum(1 for _, pred in all_method_predictions.items()
                               if pred['regime_low'] == ground_truth['regime_low_dir'] 
                               and pred['regime_high'] == ground_truth['regime_high_dir'])
        overall_accuracy = (n_correct_overall / len(all_method_predictions)) * 100
        
        with col1:
            st.metric("Overall Accuracy", f"{overall_accuracy:.0f}%")
        
        with col2:
            n_switch_detected = sum(1 for _, pred in all_method_predictions.items()
                                   if pred['regime_low'] != pred['regime_high'])
            st.metric("Methods Detecting Switch", f"{n_switch_detected}/{len(all_method_predictions)}")
        
        with col3:
            st.metric("Methods Evaluated", len(all_method_predictions))

    # Detailed results expanders
    if causal_results_by_method:
        with st.expander("View Detailed Causal Results (All Methods)"):
            for method_name, results in causal_results_by_method.items():
                st.markdown(f"**{method_name.upper()}:**")
                if results:
                    for i, (cluster_id, direction, score) in enumerate(results):
                        st.markdown(f"  - Regime {i+1}: `{direction}` (Score: {score:.3f})")
                else:
                    st.markdown("  - No results")
                st.markdown("")
            
    if st.session_state.get('llm_results'):
        with st.expander("View Detailed LLM Reasoning"):
            for result in st.session_state.llm_results:
                st.markdown(f"**{result['step']}**")
                st.markdown(result['response'])
                st.divider()
