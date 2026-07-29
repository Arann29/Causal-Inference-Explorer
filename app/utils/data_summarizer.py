# app/utils/data_summarizer.py
"""Data summarization factory for LLM input generation."""

import pandas as pd
import numpy as np
from typing import Optional, Union, Tuple
from app.utils.llm_config import ExperimentConfig
from app.utils.ground_truth import get_ground_truth
from helpers.visualization_segmentation import segment_data

# ============================================================================
# DISPLAY CONFIGURATION: cap raw rows shown in preview/single-summary text.
# Repeated LLM experiments sample directly from the full source dataframe in
# app.utils.sampling instead of sampling from this preview subset.
# ============================================================================
DEFAULT_RAW_DATA_SAMPLE_SIZE = 800  # Maximum rows to show from raw data summaries
                                     # If dataset has fewer rows, uses full dataset
DEFAULT_SEGMENTED_ROWS_PER_REGIME = 400  # Maximum rows sampled from each threshold regime
# ============================================================================


def _minimal_stats(df: pd.DataFrame) -> str:
    """Generate minimal statistical summary (mean, min, max only)."""
    if df.shape[1] < 2:
        return "Dataset must have at least two columns."
    
    x, y = df.columns[:2]
    desc = df[[x, y]].describe().T
    
    summary = (
        f"The dataset contains two variables: '{x}' and '{y}'.\n\n"
        f"Variable '{x}':\n"
        f"  - Mean: {desc.loc[x, 'mean']:.2f}\n"
        f"  - Range: [{desc.loc[x, 'min']:.2f}, {desc.loc[x, 'max']:.2f}]\n\n"
        f"Variable '{y}':\n"
        f"  - Mean: {desc.loc[y, 'mean']:.2f}\n"
        f"  - Range: [{desc.loc[y, 'min']:.2f}, {desc.loc[y, 'max']:.2f}]"
    )
    return summary


def _full_stats(df: pd.DataFrame, config: ExperimentConfig) -> str:
    """Generate full statistical summary based on config."""
    if df.shape[1] < 2:
        return "Dataset must have at least two columns."
    
    x, y = df.columns[:2]
    desc = df[[x, y]].describe().T
    
    summary_lines = [f"The dataset contains two variables: '{x}' and '{y}'."]
    
    if config.stats_included.get('sample_size', False):
        summary_lines.append(f"\nSample size: {len(df)} observations")
    
    summary_lines.append("")
    
    # Variable X statistics
    x_stats = [f"Variable '{x}':"]
    if config.stats_included.get('mean', True):
        x_stats.append(f"  - Mean: {desc.loc[x, 'mean']:.2f}")
    if config.stats_included.get('std_dev', True):
        x_stats.append(f"  - Std Dev: {desc.loc[x, 'std']:.2f}")
    if config.stats_included.get('min_max', True):
        x_stats.append(f"  - Range: [{desc.loc[x, 'min']:.2f}, {desc.loc[x, 'max']:.2f}]")
    if config.stats_included.get('quartiles', False):
        x_stats.append(f"  - Quartiles: Q1={desc.loc[x, '25%']:.2f}, Median={desc.loc[x, '50%']:.2f}, Q3={desc.loc[x, '75%']:.2f}")
    
    summary_lines.extend(x_stats)
    summary_lines.append("")
    
    # Variable Y statistics
    y_stats = [f"Variable '{y}':"]
    if config.stats_included.get('mean', True):
        y_stats.append(f"  - Mean: {desc.loc[y, 'mean']:.2f}")
    if config.stats_included.get('std_dev', True):
        y_stats.append(f"  - Std Dev: {desc.loc[y, 'std']:.2f}")
    if config.stats_included.get('min_max', True):
        y_stats.append(f"  - Range: [{desc.loc[y, 'min']:.2f}, {desc.loc[y, 'max']:.2f}]")
    if config.stats_included.get('quartiles', False):
        y_stats.append(f"  - Quartiles: Q1={desc.loc[y, '25%']:.2f}, Median={desc.loc[y, '50%']:.2f}, Q3={desc.loc[y, '75%']:.2f}")
    
    summary_lines.extend(y_stats)
    
    # Correlation
    if config.stats_included.get('correlation', True):
        corr = df[[x, y]].corr().iloc[0, 1]
        summary_lines.append(f"\nThe Pearson correlation between '{x}' and '{y}' is {corr:.2f}.")
    
    return '\n'.join(summary_lines)


def _raw_data_with_names(df: pd.DataFrame, config: ExperimentConfig) -> str:
    """Format raw data with actual variable names."""
    if df.shape[1] < 2:
        return "Dataset must have at least two columns."
    
    x, y = df.columns[:2]
    desc = df[[x, y]].describe().T
    
    # Start with basic statistics
    summary = f"The dataset contains {len(df)} observations of two variables: '{x}' and '{y}'.\n\n"
    summary += f"Variable '{x}':\n"
    # summary += f"  - Mean: {desc.loc[x, 'mean']:.2f}, Std Dev: {desc.loc[x, 'std']:.2f}\n"  # Disabled: config has mean=False, std_dev=False
    summary += f"  - Range: [{desc.loc[x, 'min']:.2f}, {desc.loc[x, 'max']:.2f}]\n\n"
    summary += f"Variable '{y}':\n"
    # summary += f"  - Mean: {desc.loc[y, 'mean']:.2f}, Std Dev: {desc.loc[y, 'std']:.2f}\n"  # Disabled: config has mean=False, std_dev=False
    summary += f"  - Range: [{desc.loc[y, 'min']:.2f}, {desc.loc[y, 'max']:.2f}]\n\n"
    
    # Use min(DEFAULT_RAW_DATA_SAMPLE_SIZE, dataset_length) for automatic sizing
    max_rows = min(DEFAULT_RAW_DATA_SAMPLE_SIZE, len(df))
    
    # Always use random sampling with seed=42 for reproducibility
    if len(df) > max_rows:
        sample_df = df[[x, y]].sample(n=max_rows, random_state=42)
        sample_type = "randomly sampled"
    else:
        sample_df = df[[x, y]]
        sample_type = "full dataset"
    
    summary += f"Sample data ({sample_type} {min(max_rows, len(df))} rows):\n"
    summary += sample_df.to_string(index=False)
    
    if len(df) > max_rows:
        summary += f"\n\n... ({len(df) - max_rows} more rows)"
    
    if config.raw_data_options.get('include_description', False):
        summary += f"\n\nThe data shows the relationship between {x} and {y}."
    
    return summary


def _raw_data_anonymous(df: pd.DataFrame, config: ExperimentConfig) -> str:
    """Format raw data with anonymous variable labels (X, Y)."""
    if df.shape[1] < 2:
        return "Dataset must have at least two columns."
    
    desc = df.iloc[:, :2].describe().T
    
    # Start with basic statistics (anonymous labels)
    summary = f"The dataset contains {len(df)} observations of two variables.\n\n"
    summary += f"Variable_X:\n"
    # summary += f"  - Mean: {desc.iloc[0]['mean']:.2f}, Std Dev: {desc.iloc[0]['std']:.2f}\n"  # Disabled: config has mean=False, std_dev=False
    summary += f"  - Range: [{desc.iloc[0]['min']:.2f}, {desc.iloc[0]['max']:.2f}]\n\n"
    summary += f"Variable_Y:\n"
    # summary += f"  - Mean: {desc.iloc[1]['mean']:.2f}, Std Dev: {desc.iloc[1]['std']:.2f}\n"  # Disabled: config has mean=False, std_dev=False
    summary += f"  - Range: [{desc.iloc[1]['min']:.2f}, {desc.iloc[1]['max']:.2f}]\n\n"
    
    # Use min(DEFAULT_RAW_DATA_SAMPLE_SIZE, dataset_length) for automatic sizing
    max_rows = min(DEFAULT_RAW_DATA_SAMPLE_SIZE, len(df))
    
    # Always use random sampling with seed=42 for reproducibility
    if len(df) > max_rows:
        sample_df = df.iloc[:, :2].sample(n=max_rows, random_state=42).copy()
        sample_type = "randomly sampled"
    else:
        sample_df = df.iloc[:, :2].copy()
        sample_type = "full dataset"
    
    sample_df.columns = ['Variable_X', 'Variable_Y']
    
    summary += f"Sample data ({sample_type} {min(max_rows, len(df))} rows):\n"
    summary += sample_df.to_string(index=False)
    
    if len(df) > max_rows:
        summary += f"\n\n... ({len(df) - max_rows} more rows)"
    
    return summary


def _raw_data_from_provided_df(df: pd.DataFrame, config: ExperimentConfig) -> str:
    """Format a summary for a dataframe that is already a sampled subset.

    This avoids re-subsampling when the caller has already provided the sampled df.
    """
    if df.shape[1] < 2:
        return "Dataset must have at least two columns."

    anonymize = config.raw_data_options.get("anonymize_names", False)

    if anonymize:
        sample_df = df.iloc[:, :2].copy()
        sample_df.columns = ["Variable_X", "Variable_Y"]

        desc = df.iloc[:, :2].describe().T

        summary = f"This LLM input contains {len(df)} randomly sampled observations of two variables.\n\n"
        summary += "Variable_X:\n"
        summary += f"  - Range: [{desc.iloc[0]['min']:.2f}, {desc.iloc[0]['max']:.2f}]\n\n"
        summary += "Variable_Y:\n"
        summary += f"  - Range: [{desc.iloc[1]['min']:.2f}, {desc.iloc[1]['max']:.2f}]\n\n"
        summary += f"Sample data ({len(sample_df)} rows):\n"
        summary += sample_df.to_string(index=False)
        return summary

    x, y = df.columns[:2]
    sample_df = df[[x, y]].copy()
    desc = sample_df.describe().T

    summary = f"This LLM input contains {len(df)} randomly sampled observations from the datapair.\n\n"
    summary += f"Variable '{x}':\n"
    summary += f"  - Range: [{desc.loc[x, 'min']:.2f}, {desc.loc[x, 'max']:.2f}]\n\n"
    summary += f"Variable '{y}':\n"
    summary += f"  - Range: [{desc.loc[y, 'min']:.2f}, {desc.loc[y, 'max']:.2f}]\n\n"
    summary += f"Sample data ({len(sample_df)} rows):\n"
    summary += sample_df.to_string(index=False)
    return summary


def _segmented_regime_data(
    df: pd.DataFrame,
    config: ExperimentConfig,
    dataset_name: Optional[str] = None,
    max_rows_per_regime: Optional[int] = None,
    seed: int = 42,
) -> str:
    """Format segmented regime data (low/high) with each regime shown separately.
    
    Each regime is capped at DEFAULT_SEGMENTED_ROWS_PER_REGIME unless overridden.
    If a regime has more rows, they are randomly sampled.
    """
    if df.shape[1] < 2:
        return "Dataset must have at least two columns."
    
    # Get threshold from ground truth
    ground_truth = get_ground_truth(dataset_name) if dataset_name else None
    if not ground_truth or 'threshold_var' not in ground_truth or 'threshold_val' not in ground_truth:
        return "ERROR: Segmented regime data requires threshold information in ground truth metadata."
    
    threshold_var = ground_truth['threshold_var']
    threshold_val = ground_truth['threshold_val']

    # Resolve threshold column name ( tolerate case/substring mismatches )
    threshold_col = threshold_var
    if threshold_col not in df.columns:
        threshold_col = None
        for col in df.columns:
            if threshold_var.lower() in col.lower() or col.lower() in threshold_var.lower():
                threshold_col = col
                break

        if threshold_col is None:
            numeric_cols = df.select_dtypes(include=["number"]).columns
            if len(numeric_cols) == 0:
                return "ERROR: Could not resolve threshold column for segmented regime data."
            threshold_col = numeric_cols[0]

    # Segment the data
    try:
        segmentation_kwargs = {'threshold': threshold_val, 'column': threshold_col, 'plot_results': False}
        labels = segment_data(df, strategy='threshold', **segmentation_kwargs)

        # Split dataframe by labels (0 = low regime, 1 = high regime)
        regime_low_df = df[labels == 0].copy()
        regime_high_df = df[labels == 1].copy()

    except Exception as e:
        return f"ERROR segmenting data: {str(e)}"
    
    # Check if anonymization is requested
    anonymize = config.raw_data_options.get('anonymize_names', False)
    
    if anonymize:
        x_label = 'Variable_X'
        y_label = 'Variable_Y'
        x, y = df.columns[:2]  # Actual column names for data access
    else:
        x, y = df.columns[:2]
        x_label = x
        y_label = y
    
    # Max rows per regime uses the LLM experiment cap unless overridden by caller
    if max_rows_per_regime is None:
        max_rows_per_regime = DEFAULT_SEGMENTED_ROWS_PER_REGIME
    
    summary = f"The dataset contains {len(df)} observations of two variables"
    if not anonymize:
        summary += f": '{x}' and '{y}'"
    summary += ".\n"
    
    if not anonymize:
        summary += f"Data is segmented by **{threshold_col}** at threshold **{threshold_val}** into two regimes:\n\n"
    else:
        # Map threshold_col to anonymized name
        if threshold_col == x:
            threshold_label = 'Variable_X'
        elif threshold_col == y:
            threshold_label = 'Variable_Y'
        else:
            threshold_label = 'the segmentation variable'
        summary += f"Data is segmented by **{threshold_label}** at threshold **{threshold_val}** into two regimes:\n\n"
    
    threshold_label = threshold_label if anonymize else threshold_var
    
    # === Regime 0 (Low) ===
    summary += f"## REGIME 0 (Low): {threshold_label} ≤ {threshold_val}\n"
    summary += f"Observations: {len(regime_low_df)}\n\n"
    
    if len(regime_low_df) > 0:
        desc_low = regime_low_df[[x, y]].describe().T
        summary += f"Variable '{x_label}' (Low regime):\n"
        # summary += f"  - Mean: {desc_low.loc[x, 'mean']:.2f}, Std Dev: {desc_low.loc[x, 'std']:.2f}\n"  # Disabled: config has mean=False, std_dev=False
        summary += f"  - Range: [{desc_low.loc[x, 'min']:.2f}, {desc_low.loc[x, 'max']:.2f}]\n\n"
        summary += f"Variable '{y_label}' (Low regime):\n"
        # summary += f"  - Mean: {desc_low.loc[y, 'mean']:.2f}, Std Dev: {desc_low.loc[y, 'std']:.2f}\n"  # Disabled: config has mean=False, std_dev=False
        summary += f"  - Range: [{desc_low.loc[y, 'min']:.2f}, {desc_low.loc[y, 'max']:.2f}]\n\n"
        
        # Sample low regime data (always random sample with seed=42)
        n_low = min(len(regime_low_df), max_rows_per_regime)
        if len(regime_low_df) > max_rows_per_regime:
            sample_low = regime_low_df[[x, y]].sample(n=n_low, random_state=seed)
            sample_type_low = "randomly sampled"
        else:
            sample_low = regime_low_df[[x, y]]
            sample_type_low = "full regime"
        
        # Rename columns if anonymizing
        if anonymize:
            sample_low = sample_low.copy()
            # ensure DataFrame
            sample_low = pd.DataFrame(sample_low)
            sample_low.columns = [x_label, y_label]
        
        summary += f"Sample data from Low regime ({sample_type_low} {len(sample_low)} rows):\n"
        summary += sample_low.to_string(index=False)
        
        if len(regime_low_df) > max_rows_per_regime:
            summary += f"\n... ({len(regime_low_df) - max_rows_per_regime} more rows in Low regime)"
    else:
        summary += "No observations in Low regime.\n"
    
    summary += "\n\n"
    
    # === Regime 1 (High) ===
    summary += f"## REGIME 1 (High): {threshold_label} > {threshold_val}\n"
    summary += f"Observations: {len(regime_high_df)}\n\n"
    
    if len(regime_high_df) > 0:
        desc_high = regime_high_df[[x, y]].describe().T
        summary += f"Variable '{x_label}' (High regime):\n"
        # summary += f"  - Mean: {desc_high.loc[x, 'mean']:.2f}, Std Dev: {desc_high.loc[x, 'std']:.2f}\n"  # Disabled: config has mean=False, std_dev=False
        summary += f"  - Range: [{desc_high.loc[x, 'min']:.2f}, {desc_high.loc[x, 'max']:.2f}]\n\n"
        summary += f"Variable '{y_label}' (High regime):\n"
        # summary += f"  - Mean: {desc_high.loc[y, 'mean']:.2f}, Std Dev: {desc_high.loc[y, 'std']:.2f}\n"  # Disabled: config has mean=False, std_dev=False
        summary += f"  - Range: [{desc_high.loc[y, 'min']:.2f}, {desc_high.loc[y, 'max']:.2f}]\n\n"
        
        # Sample high regime data (always random sample with seed=42)
        n_high = min(len(regime_high_df), max_rows_per_regime)
        if len(regime_high_df) > max_rows_per_regime:
            sample_high = regime_high_df[[x, y]].sample(n=n_high, random_state=seed + 1)
            sample_type_high = "randomly sampled"
        else:
            sample_high = regime_high_df[[x, y]]
            sample_type_high = "full regime"
        
        # Rename columns if anonymizing
        if anonymize:
            sample_high = sample_high.copy()
            # ensure DataFrame
            sample_high = pd.DataFrame(sample_high)
            sample_high.columns = [x_label, y_label]
        
        summary += f"Sample data from High regime ({sample_type_high} {len(sample_high)} rows):\n"
        summary += sample_high.to_string(index=False)
        
        if len(regime_high_df) > max_rows_per_regime:
            summary += f"\n... ({len(regime_high_df) - max_rows_per_regime} more rows in High regime)"
    else:
        summary += "No observations in High regime.\n"
    
    return summary


def add_context_hint(summary: str, config: ExperimentConfig, dataset_name: Optional[str] = None, df: Optional[pd.DataFrame] = None) -> str:
    """Add context information to the summary based on configuration.
    
    Args:
        summary: Base data summary
        config: Experiment configuration
        dataset_name: Dataset name for ground truth lookup (optional)
        df: Dataframe to determine variable mapping for anonymization (optional)
        
    Returns:
        Summary with context hint appended
    """
    if config.context_hint_level == 'none':
        return summary
    
    elif config.context_hint_level == 'threshold_var_name' and dataset_name:
        # Get threshold variable from ground truth
        ground_truth = get_ground_truth(dataset_name)
        if ground_truth and 'threshold_var' in ground_truth:
            threshold_var = ground_truth['threshold_var']
            
            # Check if variables are anonymized
            if config.data_format == 'raw_data' and config.raw_data_options.get('anonymize_names', False) and df is not None:
                # Map actual variable name to anonymized name (Variable_X or Variable_Y)
                if df.shape[1] >= 2:
                    var_names = df.columns[:2].tolist()
                    if threshold_var == var_names[0]:
                        threshold_var = 'Variable_X'
                    elif threshold_var == var_names[1]:
                        threshold_var = 'Variable_Y'
                    else:
                        # Threshold variable doesn't match first two columns - skip hint to maintain anonymization
                        return summary
            
            context = (
                f"\n\n**Domain Context**: Research suggests that the causal "
                f"relationship between these variables switches based on the value of '{threshold_var}'. "
            )
            return summary + context
        else:
            # No ground truth available, return summary without context hint
            return summary
    
    return summary


def generate_summary(
    df: pd.DataFrame,
    config: ExperimentConfig,
    dataset_name: Optional[str] = None,
    already_sampled: bool = False,
    segmented_max_rows_per_regime: Optional[int] = None,
    segmented_seed: int = 42,
)-> str:
    """Generate data summary for LLM input based on configuration.
    
    Args:
        df: Input dataframe
        config: Experiment configuration
        dataset_name: Dataset name for context hints (optional)
        
    Returns:
        Formatted data summary string
    """
    # Generate base summary based on data format
    if config.data_format == 'segmented_regimes':
        base_summary = _segmented_regime_data(
            df,
            config,
            dataset_name,
            max_rows_per_regime=segmented_max_rows_per_regime,
            seed=segmented_seed,
        )
    elif config.data_format == 'raw_data':
        if already_sampled:
            base_summary = _raw_data_from_provided_df(df, config)
        elif config.raw_data_options.get('anonymize_names', False):
            base_summary = _raw_data_anonymous(df, config)
        else:
            base_summary = _raw_data_with_names(df, config)
    else:  # statistical_summary
        # Check if it's minimal (only mean, min, max)
        is_minimal = (
            config.stats_included.get('mean', True) and
            config.stats_included.get('min_max', True) and
            not config.stats_included.get('std_dev', False) and
            not config.stats_included.get('correlation', False) and
            not config.stats_included.get('quartiles', False)
        )
        
        if is_minimal and not config.stats_included.get('sample_size', False):
            base_summary = _minimal_stats(df)
        else:
            base_summary = _full_stats(df, config)
    
    # Add context hints if configured (skip for segmented_regimes as threshold is already explicit)
    if config.data_format == 'segmented_regimes':
        final_summary = base_summary  # Threshold info already included
    else:
        final_summary = add_context_hint(base_summary, config, dataset_name, df)
    
    # Add dataset name prefix if provided (but not when anonymizing variable names)
    # because dataset names often reveal variable identities
    if dataset_name and config.data_format not in ['segmented_regimes'] and not (config.data_format == 'raw_data' and config.raw_data_options.get('anonymize_names', False)):
        final_summary = f"Dataset: {dataset_name}\n\n" + final_summary
    
    return final_summary
