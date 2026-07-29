import os
import pandas as pd
import numpy as np
import torch

class CausalDataset:
    pass

class Custom(CausalDataset):
    def __init__(self,
                 pair_id=None,
                 path="DATA/custom_pairs",
                 columns=None,
                 double=False,
                 preprocessor=None):
        assert pair_id is not None, "You must specify a pair_id"
        file_path = os.path.join(path, f"{pair_id}.txt")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Could not find file: {file_path}")

        # Read with headers, let pandas infer separator and types
        df = pd.read_csv(file_path, sep=r'\s+', header=0, engine="python")



        # Auto-select columns if not specified
        if columns is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) < 2:
                raise ValueError(f"{file_path} must contain at least two numeric columns")
            x_col, y_col = numeric_cols[:2]
        else:
            x_col, y_col = columns

        # Preprocess cause/effect only
        X = df[[x_col, y_col]].to_numpy()
        if preprocessor is not None:
            X = preprocessor.fit_transform(X)

        dtype = torch.float64 if double else torch.float32
        self.cause = torch.tensor(X[:, 0].reshape(-1, 1), dtype=dtype)
        self.effect = torch.tensor(X[:, 1].reshape(-1, 1), dtype=dtype)

        self.dataframe = df
        self.labels = [x_col, y_col]
