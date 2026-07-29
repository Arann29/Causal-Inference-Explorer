"""Batch Run Panel - Run multiple datasets with multiple methods"""
import streamlit as st
import pandas as pd
import sys
import os
from typing import Dict, List, Tuple, Optional
import json
import datetime

# Add paths for imports
sys.path.append('/app')
sys.path.append('/app/helpers')

from app.utils.data_loader import DatasetService
from app.utils.ground_truth import (
    get_ground_truth,
    get_variable_names,
    map_prediction_to_numeric,
    map_numeric_to_string
)
from app.utils.evaluation import calculate_regime_accuracy, generate_comparison_table
from helpers.causal_param_optimization import run_segmented_causal_switch
from helpers.visualization_segmentation import segment_data
from app.utils.sampling import choose_llm_sample_size, get_llm_sampling_plan, sample_llm_rows
import numpy as np


def run_causal_method(method_name: str, df: pd.DataFrame, segmentation_strategy: str, segmentation_kwargs: dict) -> List[Tuple]:
    """Run a single causal method and return results"""
    try:
        # Use the run_segmented_causal_switch function
        method_lower = method_name.lower()
        
        # Get actual variable names (first two numeric columns)
        numeric_cols = df.select_dtypes(include=['number']).columns[:2]
        cause_col = numeric_cols[0]
        effect_col = numeric_cols[1]
        
        # Run causal analysis - function will handle segmentation internally
        results = run_segmented_causal_switch(
            df=df.copy(),
            method=method_lower,
            segmentation_strategy=segmentation_strategy,
            segmentation_kwargs=segmentation_kwargs,
            device='cpu',
            cause_col=cause_col,
            effect_col=effect_col
        )
        
        return results if results else []
        
    except Exception as e:
        st.error(f"Error running {method_name}: {str(e)}")
        return []


def run_llm_analysis(
    df: pd.DataFrame,
    dataset_name: str,
    config=None,
    already_sampled: bool = False,
    segmented_max_rows_per_regime: Optional[int] = None,
    segmented_seed: int = 42,
) -> Optional[Dict]:
    """Run LLM analysis for a dataset with optional configuration.
    
    Args:
        df: Input dataframe
        dataset_name: Name of dataset
        config: ExperimentConfig instance (optional)
        
    Returns:
        Dictionary with LLM conclusions
    """
    try:
        from app.utils.llm_client import get_llm_client
        from app.utils.prompts import get_prompt_chain, _get_canonical_variable_labels, normalize_direction
        from app.utils.data_summarizer import generate_summary
        from app.utils.llm_config import ExperimentConfig
        
        # Use provided config or create default
        if config is None:
            config = ExperimentConfig()  # Default configuration
        
        # Generate summary using new modular function
        summary = generate_summary(
            df,
            config,
            dataset_name,
            already_sampled=already_sampled,
            segmented_max_rows_per_regime=segmented_max_rows_per_regime,
            segmented_seed=segmented_seed,
        )
        
        # Get threshold from ground truth
        from app.utils.ground_truth import get_ground_truth
        ground_truth = get_ground_truth(dataset_name)
        threshold_var = ground_truth.get('threshold_var') if ground_truth else None
        threshold_value = ground_truth.get('threshold_val') if ground_truth else None
        
        # Run prompt chain with configuration
        llm_client = get_llm_client()
        model_name = config.model_name if hasattr(config, 'model_name') else "gpt-5.2"
        temperature = config.temperature if hasattr(config, 'temperature') else 0.0
        
        # Get dataframe columns for anonymization mapping
        df_columns = df.columns[:2].tolist() if df is not None else None
        
        prompt_chain = get_prompt_chain(config, threshold_var, threshold_value, df_columns)

        def _inject_prompt_text(prompt_text: str, summary_text: str, history_text: str) -> str:
            return prompt_text.replace('{summary}', summary_text).replace('{history}', history_text)
        
        history = ""
        results = []
        
        for step in prompt_chain:
            prompt = _inject_prompt_text(step["prompt"], summary, history)
            response = llm_client.ask(prompt, model=model_name, temperature=temperature)
            
            if response:
                results.append({
                    "step": step["name"], 
                    "response": response,
                    "prompt_sent": prompt  # Store prompt for display
                })
                history += f"**{step['name']}**: {response}\n\n"
            else:
                return None
        
        # Parse final JSON response
        if results:
            final_response = results[-1]['response']
            
            # Strip markdown code blocks if present
            if '```json' in final_response:
                final_response = final_response.split('```json')[1].split('```')[0].strip()
            elif '```' in final_response:
                final_response = final_response.split('```')[1].split('```')[0].strip()
            
            try:
                conclusion = json.loads(final_response)
                var_x, var_y = _get_canonical_variable_labels(config, df.columns[:2].tolist() if df is not None else None)
                normalized_directions = []
                for direction_info in conclusion.get('directions', []):
                    normalized_direction = normalize_direction(direction_info.get('direction', ''), var_x, var_y)
                    if normalized_direction is None:
                        continue
                    normalized_item = dict(direction_info)
                    normalized_item['direction'] = normalized_direction
                    normalized_directions.append(normalized_item)

                conclusion['directions'] = normalized_directions

                # Reject invalid/malformed conclusions with fewer than 2 directions
                if len(normalized_directions) < 2:
                    return None

                # For batch runs we only return the parsed conclusion (compact)
                return {
                    'conclusion': conclusion
                }
            except json.JSONDecodeError as je:
                st.error(f"JSON parsing error: {str(je)}")
                st.error(f"Problematic JSON: {final_response}")
                return None
        
        return None
        
    except Exception as e:
        st.error(f"Error running LLM analysis for {dataset_name}: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return None


def format_step_to_txt(step_name: str, prompt_sent: str, response: str) -> str:
    """Format a single prompt step as a text file content.
    
    Args:
        step_name: Name of the step (e.g., "Step 1: Domain Understanding")
        prompt_sent: The prompt that was sent to the LLM
        response: The LLM's response
        
    Returns:
        Formatted string for the text file
    """
    content = f"{'='*80}\n"
    content += f"{step_name}\n"
    content += f"{'='*80}\n\n"
    content += f"📤 PROMPT SENT TO LLM:\n"
    content += f"{'-'*40}\n"
    content += f"{prompt_sent}\n\n"
    content += f"📥 LLM RESPONSE:\n"
    content += f"{'-'*40}\n"
    content += f"{response}\n"
    return content


def sanitize_filename(name: str) -> str:
    """Sanitize a string for use as a filename."""
    # Replace problematic characters
    for char in [':', '/', '\\', '?', '*', '"', '<', '>', '|']:
        name = name.replace(char, '_')
    return name


 


def render():
    """Render the batch run panel"""
    st.header("🚀 Batch Run")
    st.write("Run multiple datasets with selected causal methods and generate comparison reports.")
    
    # Initialize dataset service
    dataset_service = DatasetService(data_path='/app/DATA/custom_pairs')
    available_datasets = dataset_service.get_available_datasets()
    
    if not available_datasets:
        st.warning("No datasets found in DATA/custom_pairs/")
        return
    
    # Configuration section
    st.subheader("⚙️ Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Select Datasets:**")
        
        # Simple select all checkbox
        select_all = st.checkbox("Select All Datasets")
        
        # If select all is checked, update all dataset checkboxes in session state
        if select_all:
            for dataset in available_datasets:
                st.session_state[f"dataset_{dataset['id']}"] = True
        
        selected_datasets = []
        for dataset in available_datasets:
            dataset_id = dataset['id']
            dataset_name = dataset['name']
            
            # Simple checkbox - state controlled by key
            if st.checkbox(dataset_name, key=f"dataset_{dataset_id}"):
                selected_datasets.append(dataset_id)
    
    with col2:
        st.write("**Select Methods:**")
        
        run_roche_batch = st.checkbox("ROCHE (Heteroscedastic)", value=True)
        run_loci_batch = st.checkbox("LOCI (Location-Scale)", value=True)
        run_lcube_batch = st.checkbox("LCUBE (Dense Detection)", value=True)
        run_lingam_batch = st.checkbox("LiNGAM (DirectLiNGAM)", value=True)
        run_llm_batch = st.checkbox("LLM Analysis (OpenRouter)", value=False)
    
    # LLM Configuration (only show if LLM is selected)
    llm_config = None
    run_all_llm_configs = False
    selected_llm_configs = []
    
    if run_llm_batch:
        with st.expander("🤖 LLM Configuration", expanded=True):
            from app.utils.llm_config import (
                get_preset_config,
                get_available_presets,
                get_available_model_labels,
                get_default_model_spec,
                get_model_spec_by_label,
            )
            
            # Model selection
            available_model_labels = get_available_model_labels()
            default_model_label = get_default_model_spec().display_name
            selected_model_label = st.selectbox(
                "Model:",
                available_model_labels,
                index=available_model_labels.index(default_model_label),
                key="batch_model",
            )
            selected_model = get_model_spec_by_label(selected_model_label)
            
            # Run all configs checkbox
            run_all_llm_configs = st.checkbox(
                "Run All LLM Configurations",
                value=False,
                help="Run all 4 configuration presets (Raw Named, Raw Anonymous, Segmented Named, Segmented Anonymous)"
            )
            
            # Get all available presets (all are now _hint_explicit variants)
            all_presets = get_available_presets()
            
            # Create display names for the dropdown
            preset_display_names = {p: get_preset_config(p).scenario_name for p in all_presets}
            
            if run_all_llm_configs:
                # Show which configs will run
                st.info("**Will run all configurations:**\n" + "\n".join([f"- {v}" for v in preset_display_names.values()]))
                selected_llm_configs = all_presets
            else:
                # Preset selection dropdown (disabled when run_all is checked)
                selected_display_name = st.selectbox(
                    "Configuration Preset:",
                    options=list(preset_display_names.values()),
                    index=list(preset_display_names.values()).index('Raw+Stats | Named | Hint | Explicit'),  # Default
                    key="batch_preset",
                    disabled=run_all_llm_configs
                )
                
                # Get the actual preset name from display name
                preset_name = [k for k, v in preset_display_names.items() if v == selected_display_name][0]
                
                llm_config = get_preset_config(preset_name)
                
                # Update model from dropdown selection
                llm_config.model_name = selected_model.model_id
                selected_llm_configs = [preset_name]
                
                # Display configuration summary
                st.info(f"""
                **Active Configuration:** `{preset_name}`
                
                {llm_config.get_summary_description()}
                
                - 🤖 **Model**: {selected_model.display_name} (`{selected_model.model_id}`)
                - 🌡️ **Temperature**: {llm_config.temperature}
                """)
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("---")
        st.info("ℹ️ Thresholds are read from pairmeta_with_ground_truth.txt for each dataset")
    with col2:
        pass
    
    # Summary
    st.write("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Datasets Selected", len(selected_datasets))
    with col2:
        methods_count = sum([run_roche_batch, run_loci_batch, run_lcube_batch, run_lingam_batch, run_llm_batch])
        st.metric("Methods Selected", methods_count)
    with col3:
        total_runs = len(selected_datasets) * methods_count
        st.metric("Total Runs", total_runs)
    
    # Run button
    if st.button("▶ Start Batch Run", type="primary", disabled=(len(selected_datasets) == 0 or methods_count == 0)):
        run_batch_analysis(
            selected_datasets=selected_datasets,
            dataset_service=dataset_service,
            run_roche=run_roche_batch,
            run_loci=run_loci_batch,
            run_lcube=run_lcube_batch,
            run_lingam=run_lingam_batch,
            run_llm=run_llm_batch,
            llm_config=llm_config,
            run_all_llm_configs=run_all_llm_configs,
            selected_llm_configs=selected_llm_configs,
            selected_model=selected_model.model_id if run_llm_batch else None
        )
    
    # Display previous batch results if available
    if 'batch_results' in st.session_state and st.session_state.batch_results:
        display_batch_results(st.session_state.batch_results)


def run_batch_analysis(selected_datasets, dataset_service, run_roche, run_loci, run_lcube, run_lingam, run_llm, 
                       llm_config=None, run_all_llm_configs=False, selected_llm_configs=None, selected_model=None):
    """Run batch analysis on selected datasets with optional LLM configuration.
    
    Args:
        selected_datasets: List of dataset IDs to process
        dataset_service: DatasetService instance
        run_roche: Whether to run ROCHE algorithm
        run_loci: Whether to run LOCI algorithm
        run_lcube: Whether to run LCUBE algorithm
        run_lingam: Whether to run LiNGAM algorithm
        run_llm: Whether to run LLM analysis
        llm_config: Single LLM config (used when run_all_llm_configs=False)
        run_all_llm_configs: Whether to run all LLM configurations
        selected_llm_configs: List of preset names to run
        selected_model: Model name to use for LLM
    """
    from app.utils.llm_config import get_preset_config
    
    st.write("---")
    st.subheader("📊 Running Batch Analysis...")
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_datasets = len(selected_datasets)
    batch_results = []
    
    for idx, dataset_id in enumerate(selected_datasets):
        status_text.text(f"Processing: {dataset_id} ({idx + 1}/{total_datasets})")
        progress_bar.progress((idx) / total_datasets)
        
        try:
            # Load dataset
            df = dataset_service.load_dataset(dataset_id)
            
            # Get variable names first
            cause_var, effect_var = get_variable_names(dataset_id, df)
            
            # Get ground truth to retrieve threshold information
            ground_truth = get_ground_truth(dataset_id)
            
            if not ground_truth:
                st.warning(f"⚠️ No ground truth found for {dataset_id} in pairmeta file. Skipping...")
                continue
            
            # Ensure only 2 columns
            numeric_cols = df.select_dtypes(include=['number']).columns[:2]
            df_numeric = df[numeric_cols].copy()
            
            # Get threshold from ground truth
            threshold_var = ground_truth['threshold_var']
            threshold_value = ground_truth['threshold_val']
            
            # Find the column that matches the threshold variable
            threshold_col = None
            for col in df.columns:
                if threshold_var.lower() in col.lower() or col.lower() in threshold_var.lower():
                    threshold_col = col
                    break
            
            if not threshold_col:
                st.warning(f"⚠️ Could not find column matching threshold variable '{threshold_var}' for {dataset_id}. Using first numeric column.")
                threshold_col = numeric_cols[0]
            
            st.write(f"📍 {dataset_id}: Using threshold {threshold_col} @ {threshold_value}")
            
            # Prepare segmentation parameters
            segmentation_strategy = 'threshold'
            segmentation_kwargs = {
                'column': threshold_col,
                'threshold': threshold_value,
                'plot_results': False
            }
            
            # Initialize result for this dataset
            dataset_result = {
                'dataset_name': dataset_id,
                'cause_var': cause_var,
                'effect_var': effect_var,
                'threshold_col': threshold_col,
                'threshold_value': threshold_value,
                'methods': {},
                'ground_truth': ground_truth,
                'evaluation': {},
                'llm_configs_used': [],  # Track which LLM configs were used
                'run_all_llm_configs': run_all_llm_configs  # Flag for export formatting
            }
            
            # Run causal methods
            if run_roche:
                roche_results = run_causal_method('ROCHE', df_numeric, segmentation_strategy, segmentation_kwargs)
                if roche_results:
                    # Sort by segment ID: 0=low, 1=high
                    roche_results = sorted(roche_results, key=lambda x: x[0])
                    dataset_result['methods']['ROCHE'] = roche_results
                    st.write(f"✓ ROCHE - Segment 0 (Low): {roche_results[0][1]}, Segment 1 (High): {roche_results[1][1] if len(roche_results) > 1 else 'N/A'}")
            
            if run_loci:
                loci_results = run_causal_method('LOCI', df_numeric, segmentation_strategy, segmentation_kwargs)
                if loci_results:
                    loci_results = sorted(loci_results, key=lambda x: x[0])
                    dataset_result['methods']['LOCI'] = loci_results
                    st.write(f"✓ LOCI - Segment 0 (Low): {loci_results[0][1]}, Segment 1 (High): {loci_results[1][1] if len(loci_results) > 1 else 'N/A'}")
            
            if run_lcube:
                lcube_results = run_causal_method('LCUBE', df_numeric, segmentation_strategy, segmentation_kwargs)
                if lcube_results:
                    lcube_results = sorted(lcube_results, key=lambda x: x[0])
                    dataset_result['methods']['LCUBE'] = lcube_results
                    st.write(f"✓ LCUBE - Segment 0 (Low): {lcube_results[0][1]}, Segment 1 (High): {lcube_results[1][1] if len(lcube_results) > 1 else 'N/A'}")

            if run_lingam:
                lingam_results = run_causal_method('LINGAM', df_numeric, segmentation_strategy, segmentation_kwargs)
                if lingam_results:
                    lingam_results = sorted(lingam_results, key=lambda x: x[0])
                    dataset_result['methods']['LINGAM'] = lingam_results
                    st.write(f"✓ LiNGAM - Segment 0 (Low): {lingam_results[0][1]}, Segment 1 (High): {lingam_results[1][1] if len(lingam_results) > 1 else 'N/A'}")
            
            # Run LLM analysis
            if run_llm:
                # Determine which configs to run
                configs_to_run = selected_llm_configs if selected_llm_configs else []
                
                for preset_name in configs_to_run:
                    config = get_preset_config(preset_name)
                    if selected_model:
                        config.model_name = selected_model
                    
                    # Create a unique method key for this config
                    if run_all_llm_configs:
                        method_key = f"LLM_{preset_name}"
                    else:
                        method_key = "LLM"
                    st.write(f"🤖 Running LLM with config: {config.scenario_name}")

                    sampling_plan = get_llm_sampling_plan(df_numeric, config.data_format)

                    batch_n_samples = sampling_plan["n_samples"]
                    batch_sample_seed = sampling_plan["seed"]

                    actual_sample_sizes = []
                    
                    # Run dataset-aware stratified sampling K times and collect per-sample LLM outputs
                    per_sample_outputs = []
                    per_sample_conclusions = []

                    for sample_idx in range(int(batch_n_samples)):
                        seed = int(batch_sample_seed) + sample_idx

                        if config.data_format == "raw_data":
                            sample_size = choose_llm_sample_size(
                                df_numeric,
                                max_rows=sampling_plan["max_rows"],
                            )
                            sampled_df = sample_llm_rows(
                                df_numeric,
                                sample_size=sample_size,
                                seed=seed,
                            )
                            actual_sample_sizes.append(len(sampled_df))
                            st.write(f"  - LLM sample {sample_idx+1}/{batch_n_samples}: {len(sampled_df)} rows")
                            llm_results = run_llm_analysis(sampled_df, dataset_id, config, already_sampled=True)
                        elif config.data_format == "segmented_regimes":
                            actual_sample_sizes.append(f"up to 2 * {sampling_plan['max_rows_per_regime']} rows")
                            st.write(f"  - LLM sample {sample_idx+1}/{batch_n_samples}: segmented summary")
                            llm_results = run_llm_analysis(
                                df_numeric,
                                dataset_id,
                                config,
                                segmented_max_rows_per_regime=sampling_plan["max_rows_per_regime"],
                                segmented_seed=seed,
                            )
                        else:
                            st.error(f"Unsupported data_format for repeated sampling: {config.data_format}")
                            continue

                        if llm_results:
                            per_sample_outputs.append(llm_results)
                            if 'conclusion' in llm_results and llm_results['conclusion']:
                                per_sample_conclusions.append(llm_results['conclusion'])

                    if per_sample_outputs:
                        # Aggregate per-sample conclusions via majority vote per regime
                        def _aggregate(conclusions_list):
                            from collections import Counter
                            if not conclusions_list:
                                return None
                            regime_counters = {}
                            for c in conclusions_list:
                                if not c or 'directions' not in c:
                                    continue
                                for d in c['directions']:
                                    # Use 'regime' key, fall back to 'cluster' for legacy compatibility
                                    regime = d.get('regime', d.get('cluster', None))
                                    if regime is None:
                                        continue
                                    direction = d.get('direction', 'Uncertain')
                                    if not direction:
                                        continue
                                    regime_counters.setdefault(regime, Counter())[direction] += 1

                            aggregated = {'directions': []}
                            for regime, counter in sorted(regime_counters.items()):
                                total = sum(counter.values())
                                if total == 0:
                                    aggregated_dir = 'Uncertain'
                                    confidence = 0.0
                                else:
                                    most_common, count = counter.most_common(1)[0]
                                    confidence = 100.0 * (count / total)
                                    aggregated_dir = most_common
                                aggregated['directions'].append({'regime': regime, 'direction': aggregated_dir, 'confidence': f"{confidence:.0f}%"})
                            return aggregated

                        aggregated_conclusion = _aggregate(per_sample_conclusions)

                        # Store aggregated results and metadata as specified
                        first_valid = next((p for p in per_sample_outputs if p and 'conclusion' in p), per_sample_outputs[0])

                        dataset_result['methods'][method_key] = {
                            'conclusion': aggregated_conclusion,
                            'all_sample_conclusions': per_sample_conclusions,
                            'sampling': {
                                'mode': sampling_plan.get('mode', 'repeated_sampling'),
                                'n_samples': int(batch_n_samples),
                                'sample_size': sampling_plan.get('max_rows'),
                                'max_rows_per_regime': sampling_plan.get('max_rows_per_regime'),
                                'seed': int(batch_sample_seed),
                                'actual_sample_sizes': actual_sample_sizes,
                            },
                            'config': config.to_dict() if hasattr(config, 'to_dict') else {}
                        }

                        dataset_result['llm_configs_used'].append(preset_name)
                        st.write(f"✓ {method_key} produced {len(per_sample_outputs)} sample outputs for {dataset_id}")
                    else:
                        st.warning(f"{method_key} analysis returned no results for {dataset_id}")
            
            # Evaluate against ground truth if available
            if ground_truth:
                all_predictions = {}
                
                for method_name, results in dataset_result['methods'].items():
                    # Check if this is a non-LLM method (list of tuples)
                    if not method_name.startswith('LLM') and isinstance(results, list) and len(results) >= 2:
                        # ✅ DIRECT MAPPING: Segment 0 = regime_low, Segment 1 = regime_high
                        segment_0_dir = results[0][1]  # (segment_id, direction, score)
                        segment_1_dir = results[1][1] if len(results) > 1 else ''
                        
                        all_predictions[method_name] = {
                            'regime_low': map_prediction_to_numeric(segment_0_dir, cause_var, effect_var),
                            'regime_high': map_prediction_to_numeric(segment_1_dir, cause_var, effect_var)
                        }
                        
                        st.write(f"📊 {method_name} - Low: {segment_0_dir}, High: {segment_1_dir}")
                        
                    elif method_name.startswith('LLM'):
                        # LLM returns dict with conclusion and metadata; normalize to dicts
                        llm_data = results if isinstance(results, dict) else {}
                        conclusion = llm_data.get('conclusion', {}) if isinstance(llm_data.get('conclusion', {}), dict) else {}
                        config_used = llm_data.get('config', {}) if isinstance(llm_data.get('config', {}), dict) else {}
                        directions = conclusion.get('directions', []) if isinstance(conclusion.get('directions', []), list) else []
                        if len(directions) >= 2:
                            # ✅ FIXED: Map LLM regime numbers (0=low, 1=high) to regime_low and regime_high
                            regime_directions = {d.get('regime', d.get('cluster', idx)): d.get('direction', '') for idx, d in enumerate(directions)}
                            
                            # Get directions for regime 0 (low) and regime 1 (high)
                            dir_low = regime_directions.get(0, '')
                            dir_high = regime_directions.get(1, '')
                            
                            # Fallback: if regime numbers don't match, use order
                            if not dir_low and len(directions) > 0:
                                dir_low = directions[0].get('direction', '')
                            if not dir_high and len(directions) > 1:
                                dir_high = directions[1].get('direction', '')
                            
                            # Handle anonymized variable mapping based on config used for this run
                            is_anonymized = config_used.get('raw_data_options', {}).get('anonymize_names', False) if isinstance(config_used, dict) else False
                            if is_anonymized:
                                # For anonymized data, LLM uses Variable_X and Variable_Y
                                # Map them back to actual variable names for evaluation
                                all_predictions[method_name] = {
                                    'regime_low': map_prediction_to_numeric(dir_low, cause_var, effect_var, use_anonymized=True),
                                    'regime_high': map_prediction_to_numeric(dir_high, cause_var, effect_var, use_anonymized=True)
                                }
                                st.write(f"📊 {method_name} (anonymized): Low=C0 ({dir_low}), High=C1 ({dir_high})")
                            else:
                                # Non-anonymized: use original mapping
                                all_predictions[method_name] = {
                                    'regime_low': map_prediction_to_numeric(dir_low, cause_var, effect_var),
                                    'regime_high': map_prediction_to_numeric(dir_high, cause_var, effect_var)
                                }
                                st.write(f"📊 {method_name}: Low=C0 ({dir_low}), High=C1 ({dir_high})")
                
                # Calculate accuracy for each method
                for method_name, predictions in all_predictions.items():
                    accuracy_result = calculate_regime_accuracy(predictions, ground_truth)
                    dataset_result['evaluation'][method_name] = accuracy_result['overall_accuracy']
                    st.write(f"📈 {method_name} accuracy: {accuracy_result['overall_accuracy']*100:.0f}%")
            
            batch_results.append(dataset_result)
            
        except Exception as e:
            st.error(f"Error processing {dataset_id}: {str(e)}")
            continue
    
    progress_bar.progress(1.0)
    status_text.text(f"✓ Completed {total_datasets} datasets")
    
    # Store results in session state
    st.session_state.batch_results = batch_results
    
    st.success(f"✓ Batch analysis complete! Processed {len(batch_results)} datasets.")


def display_batch_results(batch_results):
    """Display batch results in a comprehensive table"""
    
    st.write("---")
    st.subheader("📈 Batch Results")
    
    # Create summary table
    summary_rows = []
    
    for result in batch_results:
        dataset_name = result['dataset_name']
        methods = result['methods']
        ground_truth = result['ground_truth']
        evaluation = result['evaluation']
        
        row = {
            'Dataset': dataset_name,
            'Has Ground Truth': '✓' if ground_truth else '✗',
            'Threshold': f"{result.get('threshold_col', 'N/A')} @ {result.get('threshold_value', 0):.2f}"
        }
        
        # Add method predictions (Segment 0 = Low, Segment 1 = High)
        for method_name, method_results in methods.items():
            # Check for non-LLM methods (list of tuples)
            if not method_name.startswith('LLM') and isinstance(method_results, list):
                # Sort by segment ID
                sorted_results = sorted(method_results, key=lambda x: x[0])
                if len(sorted_results) >= 1:
                    row[f'{method_name} Low'] = sorted_results[0][1]  # Segment 0
                if len(sorted_results) >= 2:
                    row[f'{method_name} High'] = sorted_results[1][1]  # Segment 1
            elif method_name.startswith('LLM'):
                # LLM returns dict with conclusion and full_results; normalize
                llm_data = method_results if isinstance(method_results, dict) else {}
                conclusion = llm_data.get('conclusion', {}) if isinstance(llm_data.get('conclusion', {}), dict) else {}
                directions = conclusion.get('directions', []) if isinstance(conclusion.get('directions', []), list) else []

                # Use short display name for column headers
                display_name = method_name  # e.g., "LLM" or "LLM_raw_named_hint_explicit"
                if method_name.startswith('LLM_'):
                    # Shorten config name for display
                    config_suffix = method_name[4:]  # Remove "LLM_" prefix
                    display_name = f"LLM ({config_suffix.replace('_hint_explicit', '')})"

                if len(directions) >= 2:
                    for idx, dir_info in enumerate(directions):
                        regime_num = dir_info.get('regime', dir_info.get('cluster', idx))
                        direction = dir_info.get('direction', 'N/A')
                        confidence = dir_info.get('confidence', 'N/A')
                        if regime_num == 0:
                            regime_name = 'Low'
                        elif regime_num == 1:
                            regime_name = 'High'
                        else:
                            regime_name = f"R{regime_num}"
                        row[f'{display_name} {regime_name}'] = f"{direction} ({confidence})"
                else:
                    row[f'{display_name}'] = 'No directions found'

                # Detailed LLM fields (per-sample conclusions and sampling metadata)
                # are intentionally omitted from the visible batch table to keep it compact.
                # Full details remain available in the JSON export stored in session state.
        
        # Add accuracy if ground truth available
        if evaluation:
            for method_name, accuracy in evaluation.items():
                # Shorten LLM config names in accuracy columns
                if method_name.startswith('LLM_'):
                    config_suffix = method_name[4:]
                    display_name = f"LLM ({config_suffix.replace('_hint_explicit', '')})"
                else:
                    display_name = method_name
                row[f'{display_name} Acc'] = f"{accuracy * 100:.0f}%"
        
        summary_rows.append(row)
    
    if summary_rows:
        df_summary = pd.DataFrame(summary_rows)
        st.dataframe(df_summary, use_container_width=True)
        
        # LLM prompt chains and raw outputs are intentionally omitted from batch view
        # to keep batch exports compact. Use the LLM panel for detailed prompt/response inspection.
        
        # Calculate overall statistics
        if any(result['ground_truth'] for result in batch_results):
            st.subheader("📊 Overall Statistics")
            
            all_methods = set()
            for result in batch_results:
                all_methods.update(result['evaluation'].keys())
            
            stats_cols = st.columns(len(all_methods) if all_methods else 1)
            
            for idx, method in enumerate(sorted(all_methods)):
                accuracies = [
                    result['evaluation'].get(method, 0) 
                    for result in batch_results 
                    if result['ground_truth'] and method in result['evaluation']
                ]
                
                if accuracies:
                    avg_accuracy = sum(accuracies) / len(accuracies) * 100
                    with stats_cols[idx]:
                        st.metric(
                                f"{method} Avg",
                                f"{avg_accuracy:.1f}%",
                                help=f"Average accuracy across {len(accuracies)} datasets"
                            )
        
        # Export options
        st.subheader("💾 Export Results")
        
        col1, col2, col3 = st.columns(3)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        with col1:
            # JSON export
            json_str = json.dumps(batch_results, indent=2, default=str)
            st.download_button(
                label="📥 Download JSON",
                data=json_str,
                file_name=f"batch_results_{timestamp}.json",
                mime="application/json"
            )
        
        with col2:
            # CSV export
            csv = df_summary.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"batch_results_{timestamp}.csv",
                mime="text/csv"
            )
        
        with col3:
            st.info("LLM prompt logs are not exported in batch mode. Use the LLM panel to save individual runs.")
