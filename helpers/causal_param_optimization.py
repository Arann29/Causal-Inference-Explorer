import json
import sys
import os
import importlib.util
import pandas as pd
import psutil
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch

from visualization_segmentation import segment_data

def check_memory_usage():
    """Check current memory usage for debugging"""
    # System RAM
    ram = psutil.virtual_memory()
    print(f"💾 RAM: {ram.percent}% used ({ram.available // (1024**3)}GB free)")
    
    # GPU memory
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.memory_allocated() / (1024**3)
        gpu_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"🖥️  GPU: {gpu_mem:.1f}GB / {gpu_total:.1f}GB used")

def make_json_serializable(obj):
    """Convert numpy/pandas types to JSON-serializable Python types"""
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    elif hasattr(obj, 'item'):  # numpy scalar
        return obj.item()
    elif hasattr(obj, 'tolist'):  # numpy array
        return obj.tolist()
    else:
        return obj

import os
import json

def safe_save_results(results, filename):
    """Safely merge results into JSON file, replacing previous entries."""
    try:
        filepath = os.path.abspath(filename)

        # Load previous results if they exist
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                previous = json.load(f)
        else:
            previous = {}

        # Merge results (convert int keys to str to match JSON format)
        for key, value in results.items():
            str_key = str(key)
            previous[str_key] = make_json_serializable(value)  # ✅ FIXED LINE

        # Overwrite file
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(previous, f, indent=2)

        return True

    except Exception as e:
        print(f"❌ Error saving results: {e}")
        return False




def run_segmented_causal_switch(df, method='roche', segmentation_strategy='threshold', segmentation_kwargs=None, device='cpu', cause_col=None, effect_col=None, causal_kwargs=None):
    """Segment data using threshold, run supported causal methods per segment, print and return directions."""
    method_registry = {
        'loci': {
            'package_dir': 'LOCI',
            'submodule': 'loci',
            'callable_name': 'loci',
            'return_type': 'score',
        },
        'roche': {
            'package_dir': 'ROCHE',
            'submodule': 'roche',
            'callable_name': 'roche',
            'return_type': 'score',
        },
        'lcube': {
            'package_dir': 'LCUBE',
            'submodule': 'LCube',
            'callable_name': 'infer_causal_direction',
            'return_type': 'direction_score',
        },
        'lingam': {
            'package_dir': 'LINGAM',
            'submodule': 'lingam_wrapper',
            'callable_name': 'lingam_wrapper',
            'return_type': 'score',
        },
    }

    if method not in method_registry:
        raise ValueError(f'Unknown method for dynamic import: {method}')

    method_config = method_registry[method]
    causa_dir = os.path.join(os.getcwd(), method_config['package_dir'], 'causa')
    submodule = method_config['submodule']

    causa_init = os.path.join(causa_dir, '__init__.py')
    spec = importlib.util.spec_from_file_location('causa', causa_init)
    causa = importlib.util.module_from_spec(spec)
    sys.modules['causa'] = causa
    spec.loader.exec_module(causa)

    sub_path = os.path.join(causa_dir, f'{submodule}.py')
    if os.path.exists(sub_path):
        sub_spec = importlib.util.spec_from_file_location(f'causa.{submodule}', sub_path)
        sub_mod = importlib.util.module_from_spec(sub_spec)
        sys.modules[f'causa.{submodule}'] = sub_mod
        sub_spec.loader.exec_module(sub_mod)
        setattr(causa, submodule, sub_mod)
    
    score_func = getattr(getattr(causa, submodule), method_config['callable_name'])

    segmentation_kwargs = segmentation_kwargs or {}
    segment_labels = segment_data(df, strategy=segmentation_strategy, **segmentation_kwargs)

    if cause_col is None or effect_col is None:
        print("ERROR: cause_col and effect_col must be specified in non-interactive mode")
        print(f"Available columns: {df.columns.tolist()}")
        return None

    # Scale entire dataset before splitting by segments
    scaler = StandardScaler()
    df_scaled = df[[cause_col, effect_col]].copy()
    df_scaled[[cause_col, effect_col]] = scaler.fit_transform(df_scaled[[cause_col, effect_col]])

    results = []

    for segment in np.unique(segment_labels):
        mask = (segment_labels == segment)
        df_segment = df_scaled.loc[mask, [cause_col, effect_col]]
        x_subset = df_segment[cause_col].values
        y_subset = df_segment[effect_col].values

        if method_config['return_type'] == 'direction_score':
            # LCUBE-style return: (direction_str, strength)
            direction_str, strength = score_func(x_subset, y_subset)
            score = strength
            if direction_str == "->":
                direction = f"{cause_col} → {effect_col}"
            elif direction_str == "<-":
                direction = f"{effect_col} → {cause_col}"
            else:
                direction = "Undecided"
        else:
            # Score-based methods return a signed float
            score = score_func(x_subset, y_subset, device=device)
            direction = f"{cause_col} → {effect_col}" if score > 0 else f"{effect_col} → {cause_col}"
            
        print(f"Segment {segment}: causal direction: {direction}, score: {score}")
        results.append((segment, direction, score))

    print(f"\n💾 Device: {device.upper()}")
    print(f"✅ Detected Causal Directions ({method.upper()}):", results)
    return results

def run_comprehensive_analysis(dataset_configs, dataset_loader_func, methods=('roche', 'loci')):
    """
    Minimal driver:
    - load dataset
    - (optionally) add Date for Tübingen 49–51
    - run clustering+causality per method with method defaults
    - simple CUDA→CPU fallback
    - light logging and basic errors
    - return dict: {dataset: {method: results or {'error': ...}}}
    """
    import torch
    

    all_results = {}
    print(f"Processing {len(dataset_configs)} datasets")

    for dataset_name, config in dataset_configs.items():
        print(f"\n=== {config.get('description', dataset_name)} ({dataset_name}) ===")
        all_results[dataset_name] = {}

        # --- Load dataset (basic error handling) ---
        try:
            loader_kwargs = config.get('loader_kwargs', {'pair_id': dataset_name})
            data = dataset_loader_func(**loader_kwargs)
            cause_col, effect_col = data.labels
            df = data.dataframe

            # Add a synthetic Date column only for Tübingen pairs 49–51 if missing
            try:
                pid = int(dataset_name)
                if (dataset_loader_func.__name__ == 'Tuebingen'
                        and pid in {49, 50, 51} and 'Date' not in df.columns):
                    df = df.copy()
                    df['Date'] = pd.date_range(start='2009-01-01', periods=len(df), freq='D')
            except Exception:
                pass  # dataset_name may not be an int; ignore silently

            print(f"Data shape: {df.shape} | Cause→Effect: {cause_col}→{effect_col} | "
                  f"Clustering: {config['clustering_strategy']}")
        except Exception as e:
            msg = f"load failed: {e}"
            print(msg)
            all_results[dataset_name] = {'error': msg}
            continue

        # --- Run methods with defaults; simple CUDA→CPU fallback ---
        for method in methods:
            print(f"- {method.upper()} ... ", end="", flush=True)

            def _run(dev):
                return run_segmented_causal_switch(
                    df=df,
                    method=method,
                    segmentation_strategy=config['clustering_strategy'],
                    segmentation_kwargs=config.get('clustering_kwargs', {}),
                    device=dev,
                    cause_col=cause_col,
                    effect_col=effect_col,
                    causal_kwargs=None  # let methods use their defaults
                )

            preferred = 'cuda' if torch.cuda.is_available() else 'cpu'
            try:
                res = _run(preferred)
                all_results[dataset_name][method] = res
                print(f"done on {preferred.upper()} ({len(res)} clusters)")
            except Exception as e:
                if preferred == 'cuda':
                    print("CUDA failed → retry CPU ... ", end="", flush=True)
                    try:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        res = _run('cpu')
                        all_results[dataset_name][method] = res
                        print(f"done on CPU ({len(res)} clusters)")
                    except Exception as e2:
                        err = f"failed on CPU: {e2}"
                        print(err)
                        all_results[dataset_name][method] = {'error': str(e2)}
                else:
                    err = f"failed: {e}"
                    print(err)
                    all_results[dataset_name][method] = {'error': str(e)}

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return all_results
