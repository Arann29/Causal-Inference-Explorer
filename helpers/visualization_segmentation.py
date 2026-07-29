
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


def plot_selected_pairs(df, segment_labels=None):
    """Plot scatter or time series plots for variable pairs automatically (non-interactive version).
    
    Args:
        df (pd.DataFrame): DataFrame containing the data to plot
        segment_labels (array-like, optional): Regime/segment labels for coloring points
        
    Returns:
        None: Displays plots using matplotlib
    """
    columns = df.columns.tolist()
    pairs = [(a, b) for i, a in enumerate(columns) for b in columns[i+1:]]
    if not pairs:
        print("No variable pairs available for plotting.")
        return

    print(f"Plotting variable pairs for dataset with {len(pairs)} total combinations")
    n_pairs = min(len(pairs), 8)  # Limit to first 8 pairs

    # Calculate subplot layout
    n_cols = min(4, n_pairs)
    n_rows = (n_pairs + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    # Normalize axes to 2D for consistent indexing
    if n_pairs == 1:
        axes = np.array([[axes]])
    elif isinstance(axes, np.ndarray) and axes.ndim == 1:
        axes = axes.reshape(n_rows, n_cols)

    for i, (a, b) in enumerate(pairs[:n_pairs]):
        row, col = divmod(i, n_cols)
        ax = axes[row, col]

        if a == 'Date' or b == 'Date':
            time_col = a if a == 'Date' else b
            val_col  = b if a == 'Date' else a
            x = df[time_col]
            y = df[val_col]
            # Try to convert to datetime if not already
            if not pd.api.types.is_datetime64_any_dtype(x):
                try:
                    x = pd.to_datetime(x, errors='raise')
                except Exception:
                    print(f"Warning: Could not convert column '{time_col}' to datetime. Using integer index instead.")
                    x = range(len(df))
            if segment_labels is not None:
                scatter = ax.scatter(x, y, c=segment_labels, cmap='viridis', alpha=0.7)
                ax.set_title(f"{val_col} over {time_col} (by regime)", fontsize=10)
                plt.colorbar(scatter, ax=ax, label='Regime')
            else:
                ax.plot(x, y, marker='o', linestyle='-', alpha=0.7, markersize=2)
                ax.set_title(f"{val_col} over {time_col}", fontsize=10)
            ax.set_xlabel(time_col if pd.api.types.is_datetime64_any_dtype(x) else f"{time_col} (index)", fontsize=8)
            ax.set_ylabel(val_col, fontsize=8)
        else:
            x = df[a]
            y = df[b]
            if segment_labels is not None:
                scatter = ax.scatter(x, y, c=segment_labels, cmap='viridis', alpha=0.7)
                ax.set_title(f"{a} vs {b} (by regime)", fontsize=10)
                plt.colorbar(scatter, ax=ax, label='Regime')
            else:
                ax.scatter(x, y, alpha=0.7)
                ax.set_title(f"{a} vs {b}", fontsize=10)
            ax.set_xlabel(a, fontsize=8)
            ax.set_ylabel(b, fontsize=8)

        ax.grid(True, alpha=0.3)

    # Hide empty subplots
    for i in range(n_pairs, n_rows * n_cols):
        row, col = divmod(i, n_cols)
        axes[row, col].set_visible(False)

    plt.tight_layout()
    plt.show()
    print(f"✅ Displayed plots for {n_pairs} variable pairs out of {len(pairs)} total pairs")


def segment_data(df, strategy='threshold', **kwargs):
    """Threshold-based data segmentation for regime-dependent causal analysis.

    Args:
        df (pd.DataFrame): Input DataFrame (should be normalized).
        strategy (str): Segmentation strategy. Only 'threshold' is supported.
        **kwargs: Optional parameters (column/threshold_var, threshold/threshold_value, plot_results).

    Returns:
        np.array: Segment labels (same length as df). Binary: 0 for low regime, 1 for high regime.
    """
    import numpy as np
    import pandas as pd

    plot_results = kwargs.pop('plot_results', True)

    # Only threshold-based segmentation is used in this submission package.
    if strategy == 'threshold':
        column = kwargs.get('column') or kwargs.get('threshold_var')
        threshold = kwargs.get('threshold') or kwargs.get('threshold_value')
        column = kwargs.get('column') or kwargs.get('threshold_var')
        threshold = kwargs.get('threshold') or kwargs.get('threshold_value')

        if column is None or column not in df.columns:
            raise ValueError(f"Must specify a valid 'column' for threshold segmentation. Available columns: {df.columns.tolist()}")
        if threshold is None:
            raise ValueError("Must specify a 'threshold' value for threshold segmentation.")

        # Apply threshold: regime 0 (Low) = <= threshold, regime 1 (High) = > threshold
        labels = np.zeros(len(df), dtype=int)
        values = df[column]
        labels[values > threshold] = 1
        
        print(f"[DEBUG] Threshold Segmentation:")
        print(f"  Column: {column}")
        print(f"  Threshold: {threshold:.4f}")
        print(f"  Regime 0 (Low, <= threshold): {np.sum(labels == 0)} samples")
        print(f"  Regime 1 (High, > threshold): {np.sum(labels == 1)} samples")

    else:
        raise ValueError(f"Unknown segmentation strategy: {strategy}. Only 'threshold' is supported.")

    if plot_results:
        try:
            plot_selected_pairs(df, segment_labels=labels)
        except Exception as e:
            print(f"Warning: Could not plot results: {e}")

    return labels
