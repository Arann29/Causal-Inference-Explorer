import streamlit as st
import sys
import os
import pandas as pd

# Add paths for imports
sys.path.append('/app')
sys.path.append('/app/helpers')
sys.path.append('/app/LOCI')
sys.path.append('/app/ROCHE')
sys.path.append('/app/DATA')

# Import components
from app.components import dataset_panel, visualization_panel, segmentation_panel
from app.components import causal_panel, llm_panel, comparison_panel, batch_panel

st.set_page_config(page_title="Causal Inference Explorer", layout="wide")

def main():
    st.title("🔍 Causal Inference Explorer")
    
    # Initialize session state for the entire app
    if 'app_initialized' not in st.session_state:
        st.session_state.app_initialized = True
        st.session_state.selected_dataset = {}
        st.session_state.current_dataframe = None
        st.session_state.segmentation_result = None
        st.session_state.segmentation_metadata = {}
        st.session_state.segmentation_method = None
        st.session_state.segmentation_kwargs = {}
        st.session_state.causal_results_by_method = {}  # Store all method results
        st.session_state.llm_results = None
        st.session_state.llm_conclusion = None

    # Left sidebar with persistent controls
    with st.sidebar:
        dataset_panel.render()
        segmentation_panel.render()
        
        st.sidebar.write("---")
        if st.sidebar.button("Clear Session State"):
            for key in st.session_state.keys():
                del st.session_state[key]
            st.rerun()
    
    # Main content area with tabs (aligned with title)
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📈 Visualization", "🔬 Causal Analysis", "🤖 LLM Analysis", "📊 Comparison", "🚀 Batch Run", "📄 Results"])
    
    with tab1:
        visualization_panel.render()
    
    with tab2:
        causal_panel.render()
    
    with tab3:
        llm_panel.render()
    
    with tab4:
        comparison_panel.render()
    
    with tab5:
        batch_panel.render()
    
    with tab6:
        st.write("### 📄 Results Export & Evaluation")
        
        # Check if we have results to export
        causal_results = st.session_state.get('causal_results_by_method', {})
        llm_conclusion = st.session_state.get('llm_conclusion')
        selected_dataset = st.session_state.get('selected_dataset', {})
        
        if not causal_results and not llm_conclusion:
            st.info("Run analyses first to export results")
        else:
            from app.utils.ground_truth import get_ground_truth, map_prediction_to_numeric
            from app.utils.evaluation import generate_comparison_table
            import json
            import datetime
            
            dataset_name = (
                selected_dataset.get('id')
                or selected_dataset.get('file', '').replace('.txt', '')
                or selected_dataset.get('name', 'unknown').replace('.txt', '')
            )
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            ground_truth = get_ground_truth(dataset_name)
            
            # Prepare export data
            export_data = {
                'dataset': dataset_name,
                'timestamp': timestamp,
                'causal_methods': {},
                'llm_analysis': {},
                'has_ground_truth': ground_truth is not None
            }
            
            # Add causal results
            for method_name, results in causal_results.items():
                export_data['causal_methods'][method_name] = [
                    {'regime': i+1, 'direction': direction, 'score': float(score)}
                    for i, (segment_id, direction, score) in enumerate(results)
                ]
            
            # Add LLM results
            if llm_conclusion:
                export_data['llm_analysis'] = llm_conclusion
            
            # Add ground truth if available
            if ground_truth:
                export_data['ground_truth'] = {
                    'threshold_var': ground_truth['threshold_var'],
                    'threshold_val': ground_truth['threshold_val'],
                    'regime_low_dir': ground_truth['regime_low_dir'],
                    'regime_high_dir': ground_truth['regime_high_dir']
                }
            
            # Display summary
            st.subheader("📊 Results Summary")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Causal Methods Run", len(causal_results))
            with col2:
                st.metric("LLM Analysis", "Complete" if llm_conclusion else "Not Run")
            with col3:
                st.metric("Ground Truth", "Available" if ground_truth else "Not Available")
            
            # Show evaluation if ground truth exists
            if ground_truth:
                st.subheader("🎯 Evaluation Against Ground Truth")
                
                all_predictions = {}
                
                # Get variable names for evaluation
                df = st.session_state.get('current_dataframe')
                cause_var, effect_var = None, None
                if df is not None and len(df.columns) >= 2:
                    cause_var, effect_var = df.columns[0], df.columns[1]
                
                # Collect causal algorithm predictions
                for method_name, results in causal_results.items():
                    if results and len(results) >= 2:
                        # Sort by segment ID: 0=Low, 1=High
                        sorted_results = sorted(results, key=lambda x: x[0])
                        _, dir_low, _ = sorted_results[0]
                        _, dir_high, _ = sorted_results[1]
                        all_predictions[method_name.upper()] = {
                            'regime_low': map_prediction_to_numeric(dir_low, cause_var, effect_var),
                            'regime_high': map_prediction_to_numeric(dir_high, cause_var, effect_var)
                        }
                
                # Add LLM predictions
                if llm_conclusion:
                    directions = llm_conclusion.get('directions', [])
                    if len(directions) >= 2:
                        # Look up by regime key (0=Low, 1=High), with positional fallback
                        regime_directions = {
                            d.get('regime', d.get('cluster', idx)): d.get('direction', '')
                            for idx, d in enumerate(directions)
                        }
                        dir_low = regime_directions.get(0, directions[0].get('direction', ''))
                        dir_high = regime_directions.get(1, directions[1].get('direction', ''))
                        # Detect anonymized Variable_X/Variable_Y form
                        is_anonymized = 'variable_x' in dir_low.lower() or 'variable_x' in dir_high.lower()
                        all_predictions['LLM'] = {
                            'regime_low': map_prediction_to_numeric(dir_low, cause_var, effect_var, use_anonymized=is_anonymized),
                            'regime_high': map_prediction_to_numeric(dir_high, cause_var, effect_var, use_anonymized=is_anonymized)
                        }
                
                if all_predictions:
                    comparison_df = generate_comparison_table(all_predictions, ground_truth)
                    st.dataframe(comparison_df, use_container_width=True)
            
            # Export options
            st.subheader("💾 Export Options")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # JSON export
                json_str = json.dumps(export_data, indent=2)
                st.download_button(
                    label="📥 Download JSON",
                    data=json_str,
                    file_name=f"{dataset_name}_results_{timestamp}.json",
                    mime="application/json",
                    help="Download complete results as JSON"
                )
            
            with col2:
                # CSV export (simplified table)
                csv_rows = []
                for method_name, results in causal_results.items():
                    for i, (segment_id, direction, score) in enumerate(results):
                        csv_rows.append({
                            'Dataset': dataset_name,
                            'Method': method_name.upper(),
                            'Regime': i+1,
                            'Direction': direction,
                            'Score': f"{score:.3f}"
                        })
                
                if llm_conclusion and llm_conclusion.get('directions'):
                    for regime_info in llm_conclusion['directions']:
                        csv_rows.append({
                            'Dataset': dataset_name,
                            'Method': 'LLM',
                            'Regime': regime_info.get('segment', 'N/A'),
                            'Direction': regime_info.get('direction', 'N/A'),
                            'Score': regime_info.get('confidence', 'N/A')
                        })
                
                if csv_rows:
                    import pandas as pd
                    csv_df = pd.DataFrame(csv_rows)
                    csv = csv_df.to_csv(index=False)
                    
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv,
                        file_name=f"{dataset_name}_results_{timestamp}.csv",
                        mime="text/csv",
                        help="Download results table as CSV"
                    )
    
    # Footer
    st.write("---")
    st.write("**Causal Inference Explorer v1.0** - Built with Streamlit + FastAPI + OpenRouter")

if __name__ == "__main__":
    main()
