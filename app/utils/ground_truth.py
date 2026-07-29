"""
Ground Truth Management for Regime-Dependent Causal Discovery
Parses pairmeta.txt and provides ground truth for evaluation
"""
import os
import pandas as pd
from typing import Dict, Tuple, Optional, List


def parse_pairmeta(filepath: Optional[str] = None) -> Dict:
    """
    Parse pairmeta file with ground truth information
    
    Format: dataset_name|cause_cols|effect_cols|weight|threshold_var|threshold_val|regime_low_dir|regime_high_dir
    Directions: 1 = X->Y, -1 = Y->X
    
    Returns:
        dict: {dataset_name: {cause_cols, effect_cols, weight, threshold_var, threshold_val, 
                              regime_low_dir, regime_high_dir}}
    """
    if filepath is None:
        # Default path relative to app structure
        filepath = os.path.join(os.path.dirname(__file__), '..', '..', 'DATA', 'pairmeta_with_ground_truth.txt')
    
    ground_truth_db = {}
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split('|')
                if len(parts) != 8:
                    print(f"Warning: Skipping malformed line: {line}")
                    continue
                
                dataset_name = parts[0].strip()
                cause_cols = parts[1].strip()
                effect_cols = parts[2].strip()
                weight = float(parts[3].strip())
                threshold_var = parts[4].strip()
                threshold_val = float(parts[5].strip())
                regime_low_dir = int(parts[6].strip())
                regime_high_dir = int(parts[7].strip())
                
                ground_truth_db[dataset_name] = {
                    'cause_cols': cause_cols,
                    'effect_cols': effect_cols,
                    'weight': weight,
                    'threshold_var': threshold_var,
                    'threshold_val': threshold_val,
                    'regime_low_dir': regime_low_dir,
                    'regime_high_dir': regime_high_dir
                }
    
    except FileNotFoundError:
        print(f"Warning: Ground truth file not found at {filepath}")
        return {}
    except Exception as e:
        print(f"Error parsing ground truth file: {e}")
        return {}
    
    return ground_truth_db


def get_ground_truth(dataset_name: str) -> Optional[Dict]:
    """
    Get ground truth configuration for a specific dataset
    
    Args:
        dataset_name: Name of the dataset (without .txt extension)
    
    Returns:
        dict or None: Ground truth configuration
    """
    ground_truth_db = parse_pairmeta()
    
    # Try exact match first
    if dataset_name in ground_truth_db:
        return ground_truth_db[dataset_name]
    
    # Try without common suffixes
    clean_name = dataset_name.replace('_cleaned', '').replace('.txt', '')
    if clean_name in ground_truth_db:
        return ground_truth_db[clean_name]
    
    return None


def segment_data(df: pd.DataFrame, threshold_config: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Segment dataframe into two regimes based on threshold
    
    Args:
        df: DataFrame with data
        threshold_config: Dict with threshold_var and threshold_val
    
    Returns:
        tuple: (regime_low_df, regime_high_df)
    """
    threshold_var = threshold_config['threshold_var']
    threshold_val = threshold_config['threshold_val']
    
    # Find column that matches threshold variable
    matching_cols = [col for col in df.columns if threshold_var.lower() in col.lower() or col.lower() in threshold_var.lower()]
    
    if not matching_cols:
        raise ValueError(f"Could not find column matching threshold variable: {threshold_var}")
    
    threshold_col = matching_cols[0]
    
    regime_low: pd.DataFrame = pd.DataFrame(df[df[threshold_col] <= threshold_val])
    regime_high: pd.DataFrame = pd.DataFrame(df[df[threshold_col] > threshold_val])
    
    return regime_low, regime_high


def map_prediction_to_numeric(prediction: str, cause_var: Optional[str] = None, effect_var: Optional[str] = None, use_anonymized: bool = False) -> int:
    """
    Convert string prediction to numeric direction
    
    Args:
        prediction: String like "X->Y", "horsepower->mpg", "Variable_X->Variable_Y", etc.
        cause_var: Name of cause variable (e.g., "horsepower")
        effect_var: Name of effect variable (e.g., "mpg")
        use_anonymized: If True, treat prediction as using Variable_X/Variable_Y format
    
    Returns:
        int: 1 for cause->effect, -1 for effect->cause, 0 for uncertain/unknown
    """
    pred_str = str(prediction).strip()
    pred_lower = pred_str.lower()
    
    # Handle anonymized variable format (Variable_X/Variable_Y)
    if use_anonymized:
        # For anonymized data: Variable_X = cause_var, Variable_Y = effect_var
        if "variable_x" in pred_lower and "variable_y" in pred_lower:
            if "variable_x->variable_y" in pred_lower.replace(" ", "") or "variable_x → variable_y" in pred_lower.replace(" ", ""):
                return 1  # cause_var -> effect_var
            elif "variable_y->variable_x" in pred_lower.replace(" ", "") or "variable_y → variable_x" in pred_lower.replace(" ", ""):
                return -1  # effect_var -> cause_var
    
    # If we have actual variable names, try to match them
    if cause_var and effect_var:
        cause_lower = cause_var.lower()
        effect_lower = effect_var.lower()
        
        # Check if prediction contains both variables
        has_cause = cause_lower in pred_lower
        has_effect = effect_lower in pred_lower
        
        if has_cause and has_effect:
            # Find positions to determine direction
            cause_pos = pred_lower.find(cause_lower)
            effect_pos = pred_lower.find(effect_lower)
            
            # Check for arrow symbols
            has_arrow = '→' in pred_str or '->' in pred_str
            
            if has_arrow:
                # If cause comes before effect: cause->effect (direction = 1)
                if cause_pos < effect_pos:
                    return 1
                # If effect comes before cause: effect->cause (direction = -1)
                else:
                    return -1
    
    # Fallback to generic X/Y patterns
    if 'x->y' in pred_lower or 'x → y' in pred_lower or pred_lower == '1':
        return 1
    elif 'y->x' in pred_lower or 'y → x' in pred_lower or pred_lower == '-1':
        return -1
    
    # Check for uncertain/unknown
    if 'uncertain' in pred_lower or 'unknown' in pred_lower or 'ambiguous' in pred_lower:
        return 0
    
    return 0  # Default to uncertain


def get_variable_names(dataset_name: str, df: Optional[pd.DataFrame] = None) -> Tuple[str, str]:
    """
    Get actual variable names (cause, effect) for a dataset from pairmeta
    
    Args:
        dataset_name: Name of dataset
        df: Optional dataframe to get column names from
    
    Returns:
        tuple: (cause_var, effect_var) or ("X", "Y") if not found
    """
    gt = get_ground_truth(dataset_name)
    
    if not gt:
        # Fallback: use dataframe columns if available
        if df is not None and len(df.columns) >= 2:
            return df.columns[0], df.columns[1]
        return "X", "Y"
    
    # Parse cause/effect column specifications (e.g., "1 1" means column 1)
    # For now, if we have a dataframe, use actual columns
    if df is not None and len(df.columns) >= 2:
        # Sanity check: warn if pairmeta cause_cols doesn't look like column 1
        # (all current datasets have X as col 0, Y as col 1; alert if that ever changes)
        cause_cols_raw = gt.get('cause_cols', '').strip()
        if cause_cols_raw and cause_cols_raw not in ('1', '1 1', df.columns[0]):
            print(
                f"WARNING: get_variable_names({dataset_name}): pairmeta cause_cols='{cause_cols_raw}' "
                f"does not match df.columns[0]='{df.columns[0]}'. "
                "Check column order to avoid silent direction-mapping errors."
            )
        return df.columns[0], df.columns[1]
    
    # Fallback to X/Y
    return "X", "Y"


def map_numeric_to_string(direction: int, cause_var: str = "X", effect_var: str = "Y") -> str:
    """
    Convert numeric direction to readable string with actual variable names
    
    Args:
        direction: 1, -1, or 0
        cause_var: Name of cause variable (default "X")
        effect_var: Name of effect variable (default "Y")
    
    Returns:
        str: "cause→effect", "effect→cause", or "Uncertain"
    """
    if direction == 1:
        return f"{cause_var}→{effect_var}"
    elif direction == -1:
        return f"{effect_var}→{cause_var}"
    else:
        return "Uncertain"


def check_regime_switch(ground_truth: Dict) -> bool:
    """
    Check if ground truth indicates a regime switch (causal direction flip)
    
    Args:
        ground_truth: Ground truth configuration dict
    
    Returns:
        bool: True if directions differ across regimes
    """
    return ground_truth['regime_low_dir'] != ground_truth['regime_high_dir']


def get_all_datasets_with_ground_truth() -> List[str]:
    """
    Get list of all datasets that have ground truth defined
    
    Returns:
        list: Dataset names
    """
    ground_truth_db = parse_pairmeta()
    return list(ground_truth_db.keys())


def get_ground_truth_summary(dataset_name: str, df: Optional[pd.DataFrame] = None) -> str:
    """
    Get human-readable summary of ground truth for a dataset
    
    Args:
        dataset_name: Name of dataset
        df: Optional dataframe to get actual variable names
    
    Returns:
        str: Formatted summary
    """
    gt = get_ground_truth(dataset_name)
    if not gt:
        return f"No ground truth available for {dataset_name}"
    
    # Get actual variable names
    cause_var, effect_var = get_variable_names(dataset_name, df)
    
    low_dir = map_numeric_to_string(gt['regime_low_dir'], cause_var, effect_var)
    high_dir = map_numeric_to_string(gt['regime_high_dir'], cause_var, effect_var)
    
    has_switch = check_regime_switch(gt)
    switch_text = "✓ Direction switches" if has_switch else "✗ No switch"
    
    summary = f"""
Ground Truth for {dataset_name}:
- Threshold: {gt['threshold_var']} = {gt['threshold_val']}
- Regime Low (≤ {gt['threshold_val']}): {low_dir}
- Regime High (> {gt['threshold_val']}): {high_dir}
- {switch_text}
"""
    return summary.strip()
