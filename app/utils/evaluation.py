"""
Evaluation Metrics for Regime-Dependent Causal Discovery
Calculates accuracy, regime-switch detection, and generates comparison tables
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import Counter


def calculate_regime_accuracy(predictions: Dict[str, int], ground_truth: Dict) -> Dict:
    """
    Calculate per-regime accuracy for a single method
    
    Args:
        predictions: {'regime_low': direction, 'regime_high': direction}
                    directions are numeric: 1 (X->Y), -1 (Y->X), 0 (uncertain)
        ground_truth: Ground truth config dict from ground_truth.py
    
    Returns:
        dict: {'regime_low_correct': bool, 'regime_high_correct': bool, 
               'regime_low_accuracy': float, 'regime_high_accuracy': float,
               'overall_accuracy': float}
    """
    regime_low_pred = predictions.get('regime_low', 0)
    regime_high_pred = predictions.get('regime_high', 0)
    
    regime_low_gt = ground_truth['regime_low_dir']
    regime_high_gt = ground_truth['regime_high_dir']
    
    regime_low_correct = (regime_low_pred == regime_low_gt)
    regime_high_correct = (regime_high_pred == regime_high_gt)
    
    # Per-regime accuracy (0 or 1)
    regime_low_accuracy = 1.0 if regime_low_correct else 0.0
    regime_high_accuracy = 1.0 if regime_high_correct else 0.0
    
    # Overall accuracy: average of both regimes (partial credit per regime)
    # 0% if both wrong, 50% if one correct, 100% if both correct
    overall_accuracy = (regime_low_accuracy + regime_high_accuracy) / 2.0
    
    return {
        'regime_low_correct': regime_low_correct,
        'regime_high_correct': regime_high_correct,
        'regime_low_accuracy': regime_low_accuracy,
        'regime_high_accuracy': regime_high_accuracy,
        'overall_accuracy': overall_accuracy,
        'regime_low_pred': regime_low_pred,
        'regime_high_pred': regime_high_pred,
        'regime_low_gt': regime_low_gt,
        'regime_high_gt': regime_high_gt
    }


def detect_regime_switch(predictions: Dict[str, int], ground_truth: Dict) -> Dict:
    """
    Detect if method correctly identified regime-dependent causal direction switch
    
    Args:
        predictions: {'regime_low': direction, 'regime_high': direction}
        ground_truth: Ground truth config dict
    
    Returns:
        dict: {'switch_exists_gt': bool, 'switch_detected': bool, 
               'switch_correct': bool}
    """
    regime_low_pred = predictions.get('regime_low', 0)
    regime_high_pred = predictions.get('regime_high', 0)
    
    regime_low_gt = ground_truth['regime_low_dir']
    regime_high_gt = ground_truth['regime_high_dir']
    
    # Does ground truth indicate a switch?
    switch_exists_gt = (regime_low_gt != regime_high_gt)
    
    # Did method predict a switch?
    switch_detected = (regime_low_pred != regime_high_pred) and (regime_low_pred != 0) and (regime_high_pred != 0)
    
    # Is the switch detection correct?
    if switch_exists_gt:
        # Should detect switch and both directions should be correct
        switch_correct = switch_detected and (regime_low_pred == regime_low_gt) and (regime_high_pred == regime_high_gt)
    else:
        # Should NOT detect switch (both directions same)
        switch_correct = not switch_detected
    
    return {
        'switch_exists_gt': switch_exists_gt,
        'switch_detected': switch_detected,
        'switch_correct': switch_correct
    }


def calculate_llm_consistency(trial_results: List[Dict]) -> Dict:
    """
    Calculate consistency metrics for repeated LLM trials
    
    Args:
        trial_results: List of dicts, each with {'regime_low': dir, 'regime_high': dir}
    
    Returns:
        dict: {'regime_low_consistency': float, 'regime_high_consistency': float,
               'regime_low_majority': int, 'regime_high_majority': int,
               'overall_consistency': float}
    """
    if not trial_results:
        return {
            'regime_low_consistency': 0.0,
            'regime_high_consistency': 0.0,
            'regime_low_majority': 0,
            'regime_high_majority': 0,
            'overall_consistency': 0.0,
            'n_trials': 0
        }
    
    n_trials = len(trial_results)
    
    # Extract predictions per regime
    regime_low_preds = [t.get('regime_low', 0) for t in trial_results]
    regime_high_preds = [t.get('regime_high', 0) for t in trial_results]
    
    # Find majority vote
    regime_low_counter = Counter(regime_low_preds)
    regime_high_counter = Counter(regime_high_preds)
    
    regime_low_majority, low_count = regime_low_counter.most_common(1)[0]
    regime_high_majority, high_count = regime_high_counter.most_common(1)[0]
    
    # Calculate consistency (percentage agreeing with majority)
    regime_low_consistency = (low_count / n_trials) * 100
    regime_high_consistency = (high_count / n_trials) * 100
    
    # Overall consistency (both regimes agree with majority)
    both_agree_count = sum(1 for t in trial_results 
                          if t.get('regime_low') == regime_low_majority 
                          and t.get('regime_high') == regime_high_majority)
    overall_consistency = (both_agree_count / n_trials) * 100
    
    return {
        'regime_low_consistency': regime_low_consistency,
        'regime_high_consistency': regime_high_consistency,
        'regime_low_majority': regime_low_majority,
        'regime_high_majority': regime_high_majority,
        'overall_consistency': overall_consistency,
        'n_trials': n_trials
    }


def generate_comparison_table(all_results: Dict[str, Dict], ground_truth: Dict, 
                             cause_var: str = "X", effect_var: str = "Y") -> pd.DataFrame:
    """
    Generate comparison table for all methods against ground truth
    
    Args:
        all_results: {method_name: {'regime_low': dir, 'regime_high': dir}}
        ground_truth: Ground truth config dict
        cause_var: Name of cause variable (default "X")
        effect_var: Name of effect variable (default "Y")
    
    Returns:
        pd.DataFrame: Comparison table with accuracy metrics
    """
    from .ground_truth import map_numeric_to_string
    
    rows = []
    
    for method_name, predictions in all_results.items():
        # Calculate accuracy
        accuracy_metrics = calculate_regime_accuracy(predictions, ground_truth)
        
        # Calculate switch detection
        switch_metrics = detect_regime_switch(predictions, ground_truth)
        
        row = {
            'Method': method_name,
            'Regime Low Predicted': map_numeric_to_string(predictions.get('regime_low', 0), cause_var, effect_var),
            'Regime Low Correct': '✓' if accuracy_metrics['regime_low_correct'] else '✗',
            'Regime High Predicted': map_numeric_to_string(predictions.get('regime_high', 0), cause_var, effect_var),
            'Regime High Correct': '✓' if accuracy_metrics['regime_high_correct'] else '✗',
            'Switch Detected': '✓' if switch_metrics['switch_detected'] else '✗',
            'Switch Correct': '✓' if switch_metrics['switch_correct'] else '✗',
            'Overall Accuracy': f"{accuracy_metrics['overall_accuracy']*100:.0f}%"
        }
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    return df


def generate_metrics_summary(all_results: Dict[str, Dict], ground_truth: Dict) -> Dict:
    """
    Generate summary statistics across all methods
    
    Args:
        all_results: {method_name: {'regime_low': dir, 'regime_high': dir}}
        ground_truth: Ground truth config dict
    
    Returns:
        dict: Summary statistics
    """
    metrics_per_method = {}
    
    for method_name, predictions in all_results.items():
        accuracy_metrics = calculate_regime_accuracy(predictions, ground_truth)
        switch_metrics = detect_regime_switch(predictions, ground_truth)
        
        metrics_per_method[method_name] = {
            'regime_low_accuracy': accuracy_metrics['regime_low_accuracy'],
            'regime_high_accuracy': accuracy_metrics['regime_high_accuracy'],
            'overall_accuracy': accuracy_metrics['overall_accuracy'],
            'switch_correct': 1.0 if switch_metrics['switch_correct'] else 0.0
        }
    
    # Calculate aggregate statistics
    all_accuracies = [m['overall_accuracy'] for m in metrics_per_method.values()]
    all_switch_correct = [m['switch_correct'] for m in metrics_per_method.values()]
    
    summary = {
        'n_methods': len(metrics_per_method),
        'mean_overall_accuracy': np.mean(all_accuracies) if all_accuracies else 0.0,
        'mean_switch_detection_accuracy': np.mean(all_switch_correct) if all_switch_correct else 0.0,
        'best_method': max(metrics_per_method.items(), key=lambda x: x[1]['overall_accuracy'])[0] if metrics_per_method else None,
        'per_method_metrics': metrics_per_method
    }
    
    return summary


def compute_confusion_matrix(all_predictions: List[int], ground_truth_label: int) -> Dict:
    """
    Compute confusion matrix statistics for a single regime
    
    Args:
        all_predictions: List of predicted directions (1, -1, or 0)
        ground_truth_label: True direction (1 or -1)
    
    Returns:
        dict: Confusion matrix statistics
    """
    predictions_array = np.array(all_predictions)
    
    # Count predictions
    n_forward = np.sum(predictions_array == 1)
    n_reverse = np.sum(predictions_array == -1)
    n_uncertain = np.sum(predictions_array == 0)
    n_total = len(predictions_array)
    
    # Calculate correct predictions
    n_correct = np.sum(predictions_array == ground_truth_label)
    accuracy = (n_correct / n_total * 100) if n_total > 0 else 0.0
    
    return {
        'n_forward': n_forward,
        'n_reverse': n_reverse,
        'n_uncertain': n_uncertain,
        'n_total': n_total,
        'n_correct': n_correct,
        'accuracy': accuracy,
        'ground_truth': ground_truth_label
    }


def export_results_csv(comparison_df: pd.DataFrame, filepath: str):
    """
    Export comparison results to CSV file
    
    Args:
        comparison_df: DataFrame from generate_comparison_table
        filepath: Output CSV path
    """
    comparison_df.to_csv(filepath, index=False)
    print(f"Results exported to: {filepath}")
