import streamlit as st
import pandas as pd
import json
import datetime
from app.utils.llm_client import get_llm_client
from app.utils.prompts import get_prompt_chain, _get_canonical_variable_labels, normalize_direction
from app.utils.llm_config import (
    ExperimentConfig,
    get_preset_config,
    get_available_presets,
    get_available_model_labels,
    get_default_model_spec,
    get_model_spec_by_label,
)
from app.utils.data_summarizer import generate_summary
from app.utils.sampling import (
    choose_llm_sample_size,
    get_llm_sampling_plan,
    sample_llm_rows,
)


from typing import Optional, List, Dict
 


def render_config_panel() -> ExperimentConfig:
    """Render configuration UI and return ExperimentConfig."""
    st.subheader("⚙️ Experiment Configuration")
    
    # Model selection
    available_model_labels = get_available_model_labels()
    default_model_label = get_default_model_spec().display_name
    selected_model_label = st.selectbox(
        "Model:",
        available_model_labels,
        index=available_model_labels.index(default_model_label),
    )
    selected_model = get_model_spec_by_label(selected_model_label)
    
    # Get all available presets (all are now _hint_explicit variants)
    all_presets = get_available_presets()
    
    # Create display names for the dropdown
    preset_display_names = {p: get_preset_config(p).scenario_name for p in all_presets}
    
    # Preset selection dropdown
    selected_display_name = st.selectbox(
        "Configuration Preset:",
        options=list(preset_display_names.values()),
        index=list(preset_display_names.values()).index('Raw+Stats | Named | Hint | Explicit')  # Default
    )
    
    # Get the actual preset name from display name
    preset_name = [k for k, v in preset_display_names.items() if v == selected_display_name][0]
    
    config = get_preset_config(preset_name)
    
    # Update model from dropdown selection
    config.model_name = selected_model.model_id
    
    # Display configuration details
    with st.expander("📋 Configuration Details", expanded=False):
        st.markdown(f"""
        **Active Configuration:** `{preset_name}`
        
        {config.get_summary_description()}
        
        - 🤖 **Model**: {selected_model.display_name} (`{selected_model.model_id}`)
        - 🌡️ **Temperature**: {config.temperature}
        - 🎲 **Random Seed**: {config.raw_data_options.get('random_seed', 'N/A')}
        """)
    
    # Store in session state
    st.session_state['experiment_config'] = config
    
    return config


def summarize_data(df: pd.DataFrame) -> str:
    """Generate a textual statistical summary for the LLM."""
    if df.shape[1] < 2:
        return "Dataset must have at least two columns."

    x, y = df.columns[:2]
    desc = df[[x, y]].describe().T
    corr = df[[x, y]].corr().iloc[0, 1]

    summary = (
        f"The dataset contains two variables: '{x}' and '{y}'.\n\n"
        f"Variable '{x}':\n"
        f"  - Mean: {desc.loc[x, 'mean']:.2f}, Std Dev: {desc.loc[x, 'std']:.2f}\n"
        f"  - Range: [{desc.loc[x, 'min']:.2f}, {desc.loc[x, 'max']:.2f}]\n\n"
        f"Variable '{y}':\n"
        f"  - Mean: {desc.loc[y, 'mean']:.2f}, Std Dev: {desc.loc[y, 'std']:.2f}\n"
        f"  - Range: [{desc.loc[y, 'min']:.2f}, {desc.loc[y, 'max']:.2f}]\n\n"
        f"The Pearson correlation between '{x}' and '{y}' is {corr:.2f}."
    )
    return summary

def run_prompt_chain(summary, config: Optional[ExperimentConfig] = None, threshold_var: Optional[str] = None, threshold_value: Optional[float] = None, df_columns: Optional[List[str]] = None):
    """Runs the optimized 4-step prompt chain with optional configuration.
    
    The summary is injected into ALL steps to maintain grounding in original data.
    This prevents telephone-game effects where later steps only reason over prior interpretations.
    """
    llm_client = get_llm_client()
    
    # Get model and temperature from config if provided, otherwise use defaults
    # model_name = config.model_name if config else "gpt-4.1"
    model_name = config.model_name if config else get_default_model_spec().model_id
    temperature = config.temperature if config else 0.0
    
    prompt_chain = get_prompt_chain(config, threshold_var, threshold_value, df_columns)
    
    history = ""
    results = []

    def _inject_prompt_text(prompt_text: str, summary_text: str, history_text: str) -> str:
        return prompt_text.replace("{summary}", summary_text).replace("{history}", history_text)

    for i, step in enumerate(prompt_chain):
        # Inject summary into ALL steps to maintain grounding in original data
        prompt = _inject_prompt_text(step["prompt"], summary, history)
        
        with st.spinner(f"Running {step['name']}..."):
            response = llm_client.ask(prompt, model=model_name, temperature=temperature)
        
        if response:
            results.append({
                "step": step["name"], 
                "response": response,
                "prompt_sent": prompt  # Store the actual prompt for debugging
            })
            history += f"**{step['name']}**: {response}\n\n"
        else:
            st.error(f"Failed to get response for {step['name']}")
            return None
            
    return results


def aggregate_llm_conclusions(conclusions: List[dict]) -> Optional[dict]:
    """Aggregate multiple LLM JSON conclusions via majority vote per regime.

    Returns a dict with 'directions' list containing regime, direction and confidence (%).
    """
    if not conclusions:
        return None

    # Collect per-regime direction counters
    from collections import Counter

    regime_counters = {}
    valid_counts = {}

    for c in conclusions:
        if not c or 'directions' not in c:
            continue
        for d in c['directions']:
            regime = d.get('regime', d.get('cluster', None))
            if regime is None:
                continue
            direction = d.get('direction', 'Uncertain')
            regime_counters.setdefault(regime, Counter())[direction] += 1
            valid_counts[regime] = valid_counts.get(regime, 0) + 1

    if not regime_counters:
        return None

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

        aggregated['directions'].append({
            'regime': regime,
            'direction': aggregated_dir,
            'confidence': f"{confidence:.0f}%",
        })

    return aggregated


def _normalize_conclusion_directions(conclusion: dict, var_x: str, var_y: str) -> Optional[dict]:
    if not isinstance(conclusion, dict):
        return None

    normalized = dict(conclusion)
    normalized_directions = []

    for direction_info in conclusion.get("directions", []):
        normalized_direction = normalize_direction(direction_info.get("direction", ""), var_x, var_y)
        if normalized_direction is None:
            continue

        normalized_item = dict(direction_info)
        normalized_item["direction"] = normalized_direction
        normalized_directions.append(normalized_item)

    normalized["directions"] = normalized_directions

    if len(normalized_directions) < 2:
        return None

    return normalized

def render():
    st.header("🤖 LLM-Based Causal Reasoning")

    if "current_dataframe" not in st.session_state or st.session_state.current_dataframe is None:
        st.info("Please load a dataset and apply segmentation first.")
        return

    df = st.session_state.current_dataframe
    selected_dataset = st.session_state.get('selected_dataset', {})
    dataset_name = (
        selected_dataset.get('id')
        or selected_dataset.get('file', '').replace('.txt', '')
        or selected_dataset.get('name', '').replace('.txt', '')
    )
    
    # Render configuration panel
    config = render_config_panel()
    
    # Display configuration summary
    st.write("---")
    st.subheader("📋 Current Configuration")
    st.markdown(config.get_summary_description())
    
    st.write("---")
    st.subheader("📊 Preview Summary")
    st.caption("This is only a preview. In repeated sampling mode, each run receives its own sampled summary.")
    summary = generate_summary(df, config, dataset_name)
    st.code(summary, language="markdown")

    from app.utils.ground_truth import get_ground_truth
    ground_truth = get_ground_truth(dataset_name)
    threshold_var = ground_truth.get('threshold_var') if ground_truth else None
    threshold_value = ground_truth.get('threshold_val') if ground_truth else None

    llm_run_mode = st.radio(
        "LLM run mode",
        options=["Single run", "Repeated sampling"],
        index=0,
        horizontal=True,
    )

    sampling_plan = None
    if llm_run_mode == "Repeated sampling":
        sampling_plan = get_llm_sampling_plan(df, config.data_format)
        if config.data_format == "raw_data":
            st.info(
                f"Repeated LLM sampling will use {sampling_plan['n_samples']} random samples of up to {sampling_plan['max_rows']} rows, sampled directly from the full datapair each run."
            )
        elif config.data_format == "segmented_regimes":
            st.info(
                f"Repeated LLM sampling will use {sampling_plan['n_samples']} segmented samples, each sampled directly from the full low/high regimes with up to {sampling_plan['max_rows_per_regime']} rows per regime."
            )
        else:
            st.info(
                f"Repeated LLM sampling will use {sampling_plan['n_samples']} samples using the normal summary path."
            )

    run_button = st.button("🚀 Run LLM Analysis", type="primary", use_container_width=True)

    if run_button:
        st.session_state.llm_results = None  # Clear previous results
        st.session_state.llm_all_trials = []  # Store all trials
        st.session_state.llm_conclusion = None
        st.session_state.llm_sampling = None
        st.session_state.llm_sample_conclusions = []
        st.session_state.llm_config = config.to_dict()  # Store configuration
        
        # Progress tracking for repeated sampling
        if llm_run_mode == "Repeated sampling":
            progress_bar = st.progress(0)
            status_text = st.empty()
        
        all_trial_results = []
        all_trial_conclusions = []
        actual_sample_sizes = []
        all_trial_summaries = []

        if llm_run_mode == "Single run":
            df_columns = df.columns[:2].tolist() if df is not None else None
            results = run_prompt_chain(summary, config, threshold_var, threshold_value, df_columns)
            var_x, var_y = _get_canonical_variable_labels(config, df_columns)

            if results:
                all_trial_results.append(results)
                # Store the actual summary used for this single run
                all_trial_summaries.append(summary)

                try:
                    final_conclusion_str = results[-1]['response']
                    if '```json' in final_conclusion_str:
                        final_conclusion_str = final_conclusion_str.split('```json')[1].split('```')[0]
                    elif '```' in final_conclusion_str:
                        final_conclusion_str = final_conclusion_str.split('```')[1].split('```')[0]

                    final_conclusion_json = json.loads(final_conclusion_str.strip())
                    final_conclusion_json = _normalize_conclusion_directions(final_conclusion_json, var_x, var_y)
                    if final_conclusion_json is None:
                        raise ValueError("No normalized directions found")
                    all_trial_conclusions.append(final_conclusion_json)
                except (json.JSONDecodeError, KeyError, IndexError, ValueError) as e:
                    st.warning(f"Could not parse JSON conclusion: {e}")

        else:
            sampling_plan = sampling_plan or get_llm_sampling_plan(df, config.data_format)
            n_samples = sampling_plan["n_samples"]
            sample_seed = sampling_plan["seed"]

            for sample_idx in range(n_samples):
                if n_samples > 1:
                    status_text.text(f"Running run {sample_idx + 1} of {n_samples}...")
                    progress_bar.progress((sample_idx) / n_samples)

                seed = sample_seed + sample_idx

                if config.data_format == "raw_data":
                    sample_size = choose_llm_sample_size(
                        df,
                        max_rows=sampling_plan["max_rows"],
                    )
                    sampled_df = sample_llm_rows(
                        df,
                        sample_size=sample_size,
                        seed=seed,
                    )
                    actual_sample_sizes.append(len(sampled_df))
                    summary_run = generate_summary(
                        sampled_df,
                        config,
                        dataset_name,
                        already_sampled=True,
                    )
                    df_columns = sampled_df.columns[:2].tolist()
                    # Record the exact summary used for this trial
                    all_trial_summaries.append(summary_run)
                elif config.data_format == "segmented_regimes":
                    actual_sample_sizes.append(f"up to 2 * {sampling_plan['max_rows_per_regime']} rows")
                    summary_run = generate_summary(
                        df,
                        config,
                        dataset_name,
                        segmented_max_rows_per_regime=sampling_plan["max_rows_per_regime"],
                        segmented_seed=seed,
                    )
                    df_columns = df.columns[:2].tolist() if df is not None else None
                    # Record the exact segmented summary used for this trial
                    all_trial_summaries.append(summary_run)
                else:
                    st.error(f"Unsupported data_format for repeated sampling: {config.data_format}")
                    return

                var_x, var_y = _get_canonical_variable_labels(config, df_columns)
                results = run_prompt_chain(summary_run, config, threshold_var, threshold_value, df_columns)
            
                if results:
                    all_trial_results.append(results)

                    final_text = results[-1]["response"]

                    if "```json" in final_text:
                        final_text = final_text.split("```json")[1].split("```")[0]
                    elif "```" in final_text:
                        final_text = final_text.split("```")[1].split("```")[0]

                    try:
                        parsed_conclusion = json.loads(final_text.strip())
                        normalized_conclusion = _normalize_conclusion_directions(parsed_conclusion, var_x, var_y)
                        if normalized_conclusion is not None:
                            all_trial_conclusions.append(normalized_conclusion)
                    except json.JSONDecodeError:
                        pass

            progress_bar.progress(1.0)
            status_text.text(f"✅ Completed {n_samples} trials")
        
        # Store all trials and the actual summaries used in session state
        st.session_state.llm_all_trials = all_trial_results
        st.session_state.llm_all_summaries = all_trial_summaries
        
        if all_trial_results:
            # Used only to display one prompt-chain trace.
            # The final prediction is stored in st.session_state.llm_conclusion.
            st.session_state.llm_results = all_trial_results[0]

            if llm_run_mode == "Repeated sampling":
                st.session_state.llm_sampling = {
                    "mode": "repeated_sampling",
                    "n_samples": n_samples,
                    "sample_size": (sampling_plan.get("max_rows") if sampling_plan else None),
                    "max_rows_per_regime": (sampling_plan.get("max_rows_per_regime") if sampling_plan else None),
                    "actual_sample_sizes": actual_sample_sizes,
                }

            # Aggregate conclusions if we have multiple parsed conclusions
            valid_conclusions = [c for c in all_trial_conclusions if c is not None]
            st.session_state.llm_sample_conclusions = valid_conclusions
            if len(valid_conclusions) > 1:
                aggregated = aggregate_llm_conclusions(valid_conclusions)
                st.session_state.llm_conclusion = aggregated
            else:
                st.session_state.llm_conclusion = valid_conclusions[0] if valid_conclusions else None

    # Display results
    if 'llm_results' in st.session_state and st.session_state.llm_results:
        # Show which configuration was used
        if 'llm_config' in st.session_state:
            st.info(f"🎯 **Scenario**: {st.session_state.llm_config.get('scenario_name', 'Custom')}")
        
        # Check if single or multiple trials
        n_trials_run = len(st.session_state.get('llm_all_trials', []))
        
        if n_trials_run == 1:
            st.subheader("🔍 LLM Reasoning Steps")
        else:
            st.subheader("🔍 LLM Reasoning Steps (sample 1 shown; aggregated result below)")
        
        for result in st.session_state.llm_results:
            with st.expander(f"**{result['step']}**", expanded=False):
                # Show the prompt that was sent (for debugging)
                if 'prompt_sent' in result:
                    st.markdown("#### 📤 Prompt Sent to LLM:")
                    st.code(result['prompt_sent'], language="markdown")
                    st.markdown("---")
                
                # Show the response
                st.markdown("#### 📥 LLM Response:")
                st.markdown(result['response'])

        if st.session_state.get("llm_sampling"):
            st.subheader("Repeated Sampling Results")

            sampling = st.session_state.get("llm_sampling", {})
            st.info(
                f"Samples: {sampling.get('n_samples')} | "
                f"Sample size: {sampling.get('sample_size')} | "
                f"Max rows per regime: {sampling.get('max_rows_per_regime')} | "
                f"Actual sample sizes: {sampling.get('actual_sample_sizes')}"
            )

            sample_conclusions = st.session_state.get("llm_sample_conclusions", [])

            if sample_conclusions:
                rows = []
                for i, conclusion in enumerate(sample_conclusions, start=1):
                    directions = conclusion.get("directions", []) if isinstance(conclusion, dict) else []
                    row: Dict[str, object] = {"Sample": i}

                    for idx, d in enumerate(directions):
                        regime = d.get("regime", d.get("cluster", idx))
                        direction = d.get("direction", "N/A")
                        confidence = d.get("confidence", "N/A")

                        label = "Low" if regime == 0 else ("High" if regime == 1 else f"Regime {regime}")
                        row[label] = f"{direction} ({confidence})"

                    rows.append(row)

                if rows:
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)

            aggregated = st.session_state.get("llm_conclusion")

            if aggregated:
                st.markdown("#### Aggregated result")
                st.json(aggregated)

        # Show the actual summaries that were sent to the LLM for each run
        summaries = st.session_state.get("llm_all_summaries", [])
        if summaries:
            st.subheader("📊 Actual Summaries Sent to LLM")
            for i, summary_text in enumerate(summaries, start=1):
                with st.expander(f"Summary sent to run {i}", expanded=False):
                    st.code(summary_text, language="markdown")
        
        # Display consistency metrics ONLY if multiple trials were run
        if 'llm_consistency' in st.session_state and n_trials_run > 1:
            st.subheader("📊 Consistency Across Trials")
            consistency = st.session_state.llm_consistency
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Regime Low Consistency", f"{consistency['regime_low_consistency']:.0f}%")
            with col2:
                st.metric("Regime High Consistency", f"{consistency['regime_high_consistency']:.0f}%")
            with col3:
                st.metric("Overall Consistency", f"{consistency['overall_consistency']:.0f}%")
            
            st.info(f"💡 Majority vote from {consistency['n_trials']} trials used for final prediction")
            
            # Show all trial results in expander
            if 'llm_all_trials' in st.session_state and len(st.session_state.llm_all_trials) > 1:
                with st.expander("🔬 View All Trial Results"):
                    for i, trial_results in enumerate(st.session_state.llm_all_trials):
                        st.markdown(f"### Trial {i+1}")
                        if trial_results:
                            # Show only final conclusion for brevity
                            final_step = trial_results[-1]
                            st.markdown(f"**{final_step['step']}**")
                            st.markdown(final_step['response'][:500] + "..." if len(final_step['response']) > 500 else final_step['response'])
                        st.divider()
        
        st.success("✅ LLM analysis complete.")

        # Allow saving a single-run LLM trace (compact) for offline inspection
        if n_trials_run == 1:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            saved_run = {
                "dataset_name": dataset_name,
                "model": config.model_name,
                "config": config.to_dict() if hasattr(config, 'to_dict') else {},
                "summary": summary,
                "full_results": st.session_state.llm_results,
                "final_conclusion": st.session_state.llm_conclusion,
            }

            st.download_button(
                label="Save LLM run",
                data=json.dumps(saved_run, indent=2, default=str),
                file_name=f"llm_run_{dataset_name}_{timestamp}.json",
                mime="application/json",
            )
