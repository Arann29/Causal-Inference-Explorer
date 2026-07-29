#!/usr/bin/env python3
"""Generate regime-switching synthetic causal pairs for the app.

The active app receives two sample-size variants for each design:
- base filename: 400 observations, balanced 200/200 by regime
- *_n2000 filename: 2000 observations, balanced 1000/1000 by regime
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ACTIVE_DIR = ROOT / "DATA" / "custom_pairs"
DOC_DIR = ACTIVE_DIR / "papers_and_files" / "synthetic_regime_switching"


def rbf_kernel_matrix(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float).reshape(-1, 1)
    d2 = (z - z.T) ** 2
    nonzero = d2[d2 > 0]
    gamma = 1.0 / (2.0 * np.median(nonzero)) if len(nonzero) else 1.0
    return np.exp(-gamma * d2)


def hsic_statistic(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n < 2:
        return float("nan")
    kx = rbf_kernel_matrix(x)
    ky = rbf_kernel_matrix(y)
    h = np.eye(n) - np.ones((n, n)) / n
    return float(np.trace((h @ kx @ h) @ (h @ ky @ h)) / ((n - 1) ** 2))


def make_noise(rng: np.random.Generator, size: int, noise: dict) -> np.ndarray:
    kind = noise["type"]
    if kind == "gaussian":
        return rng.normal(0.0, noise["scale"], size)
    if kind == "laplace":
        return rng.laplace(0.0, noise["scale"], size)
    if kind == "student_t":
        return noise["scale"] * rng.standard_t(noise["df"], size)
    raise ValueError(f"Unknown noise type: {kind}")


def rejection_sample_regime(
    rng: np.random.Generator,
    n_target: int,
    direction: str,
    function: Callable[[np.ndarray], np.ndarray],
    noise: dict,
    tau: float,
    regime: str,
    batch_size: int = 2000,
    max_attempts: int = 1_000_000,
) -> dict:
    x_out: list[float] = []
    y_out: list[float] = []
    attempts = 0

    while len(x_out) < n_target:
        if attempts >= max_attempts:
            raise RuntimeError(
                f"Could not generate enough samples for regime={regime}, "
                f"direction={direction}, tau={tau}. "
                f"Accepted {len(x_out)} / {n_target} after {attempts} attempts."
            )

        remaining = n_target - len(x_out)
        size = max(batch_size, remaining * 20)

        if direction == "X->Y":
            x = rng.uniform(0.0, 1.0, size)
            y = function(x) + make_noise(rng, size, noise)
        elif direction == "Y->X":
            y = rng.uniform(0.0, 1.0, size)
            x = function(y) + make_noise(rng, size, noise)
        else:
            raise ValueError(f"Unknown direction: {direction}")

        attempts += size
        valid = (x >= 0.0) & (x <= 1.0) & (y >= 0.0) & (y <= 1.0)
        if regime == "low":
            valid &= x <= tau
        elif regime == "high":
            valid &= x > tau
        else:
            raise ValueError(f"Unknown regime: {regime}")

        take = min(remaining, int(np.sum(valid)))
        if take:
            x_out.extend(x[valid][:take])
            y_out.extend(y[valid][:take])

    return {
        "X": np.asarray(x_out, dtype=float),
        "Y": np.asarray(y_out, dtype=float),
        "attempts": int(attempts),
        "accepted": int(n_target),
        "acceptance_rate": float(n_target / attempts),
    }


def residuals_for_regime(
    x: np.ndarray,
    y: np.ndarray,
    direction: str,
    function: Callable[[np.ndarray], np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    if direction == "X->Y":
        return x, y - function(x)
    if direction == "Y->X":
        return y, x - function(y)
    raise ValueError(f"Unknown direction: {direction}")


def specs() -> list[dict]:
    return [
        {
            "name": "Linear-to-Cubic Switch with Gaussian Noise",
            "stem": "synthetic_regime_switch_linear_cubic_gaussian",
            "tau": 0.50,
            "low_function": lambda x: 0.10 + 0.70 * x,
            "low_noise": {"type": "gaussian", "scale": 0.04},
            "high_function": lambda y: 0.55 + 0.35 * (3 * y**3 - 2 * y**2 + 0.5 * y),
            "high_noise": {"type": "gaussian", "scale": 0.04},
        },
        {
            "name": "Quadratic-to-Linear Switch with Student-t Noise",
            "stem": "synthetic_regime_switch_quadratic_linear_student",
            "tau": 0.60,
            "low_function": lambda x: 0.15 + 0.35 * x + 0.45 * x**2,
            "low_noise": {"type": "student_t", "df": 3, "scale": 0.035},
            "high_function": lambda y: 0.62 + 0.30 * y,
            "high_noise": {"type": "student_t", "df": 2, "scale": 0.030},
        },
        {
            "name": "Linear-Gaussian to Linear-Laplace Switch (scale 0.1)",
            "stem": "synthetic_regime_switch_linear_gaussian_to_linear_laplace_scale_0_1",
            "tau": 0.40,
            "low_function": lambda x: 0.20 + 0.90 * x,
            "low_noise": {"type": "gaussian", "scale": 0.035},
            "high_function": lambda y: 0.42 + 0.45 * y,
            "high_noise": {"type": "laplace", "scale": 0.1},
        },
        {
            "name": "Linear-Gaussian to Linear-Laplace Switch (scale 1.0)",
            "stem": "synthetic_regime_switch_linear_gaussian_to_linear_laplace_scale_1_0",
            "tau": 0.40,
            "low_function": lambda x: 0.20 + 0.90 * x,
            "low_noise": {"type": "gaussian", "scale": 0.035},
            "high_function": lambda y: 0.42 + 0.45 * y,
            "high_noise": {"type": "laplace", "scale": 1.0},
        },
        {
            "name": "Variance Magnitude Switch",
            "stem": "synthetic_regime_switch_variance_magnitude",
            "tau": 0.55,
            "low_function": lambda x: 0.10 + 0.65 * x + 0.10 * x**2,
            "low_noise": {"type": "gaussian", "scale": 0.055},
            "high_function": lambda y: 0.58 + 0.30 * y,
            "high_noise": {"type": "gaussian", "scale": 0.020},
        },
        {
            "name": "Quartic-to-Quadratic Polynomial Switch",
            "stem": "synthetic_regime_switch_quartic_quadratic",
            "tau": 0.50,
            "low_function": lambda x: 0.15 + 0.35 * x - 0.20 * x**2 + 0.55 * x**4,
            "low_noise": {"type": "gaussian", "scale": 0.030},
            "high_function": lambda y: 0.52 + 0.38 * y**2,
            "high_noise": {"type": "gaussian", "scale": 0.030},
        },
        {
            "name": "Monotonic-to-Nonmonotonic Switch",
            "stem": "synthetic_regime_switch_monotonic_nonmonotonic",
            "tau": 0.45,
            "low_function": lambda x: 0.15 + 0.75 * x,
            "low_noise": {"type": "gaussian", "scale": 0.035},
            "high_function": lambda y: 0.50 + 0.38 * (4 * (y - 0.5) ** 2),
            "high_noise": {"type": "gaussian", "scale": 0.035},
        },
    ]


def filename_for(stem: str, n_samples: int) -> str:
    suffix = "" if n_samples == 400 else f"_n{n_samples}"
    return f"{stem}{suffix}.txt"


def generate_one(spec: dict, n_samples: int, seed: int) -> tuple[dict, str]:
    if n_samples % 2:
        raise ValueError("n_samples must be even to keep balanced low/high regimes.")

    rng = np.random.default_rng(seed)
    n_low = n_high = n_samples // 2
    tau = float(spec["tau"])
    low = rejection_sample_regime(
        rng, n_low, "X->Y", spec["low_function"], spec["low_noise"], tau, "low"
    )
    high = rejection_sample_regime(
        rng, n_high, "Y->X", spec["high_function"], spec["high_noise"], tau, "high"
    )

    x = np.concatenate([low["X"], high["X"]])
    y = np.concatenate([low["Y"], high["Y"]])
    shuffle_idx = np.arange(n_samples)
    rng.shuffle(shuffle_idx)
    x = x[shuffle_idx]
    y = y[shuffle_idx]
    data = np.column_stack([x, y])

    assert data.shape == (n_samples, 2)
    assert np.all((data >= 0.0) & (data <= 1.0))
    assert np.sum(x <= tau) == n_low
    assert np.sum(x > tau) == n_high
    assert not np.isnan(data).any()
    assert not np.isinf(data).any()

    file_name = filename_for(spec["stem"], n_samples)
    for output_dir in (ACTIVE_DIR, DOC_DIR):
        np.savetxt(
            output_dir / file_name,
            data,
            header="Variable_X Variable_Y",
            comments="",
            fmt="%.6f",
        )

    low_mask = x <= tau
    high_mask = x > tau
    low_cause, low_residuals = residuals_for_regime(
        x[low_mask], y[low_mask], "X->Y", spec["low_function"]
    )
    high_cause, high_residuals = residuals_for_regime(
        x[high_mask], y[high_mask], "Y->X", spec["high_function"]
    )

    dataset_id = Path(file_name).stem
    pairmeta = f"{dataset_id}|Variable_X|Variable_Y|1.0|Variable_X|{tau:g}|1|-1"
    metadata = {
        "dataset_name": spec["name"],
        "dataset_id": dataset_id,
        "file_name": file_name,
        "n_samples": int(n_samples),
        "threshold_variable": "Variable_X",
        "threshold_value": tau,
        "generation": (
            "Samples are generated by rejection sampling. Candidate pairs outside "
            "[0,1]^2 or outside their intended Variable_X threshold regime are "
            "discarded. No clipping is applied."
        ),
        "regime_low": {
            "direction": "X->Y",
            "direction_code": 1,
            "n_samples": int(np.sum(low_mask)),
            "x_range": [float(x[low_mask].min()), float(x[low_mask].max())],
            "y_range": [float(y[low_mask].min()), float(y[low_mask].max())],
            "hsic_cause_residual": hsic_statistic(low_cause, low_residuals),
            "attempts": low["attempts"],
            "accepted": low["accepted"],
            "acceptance_rate": low["acceptance_rate"],
        },
        "regime_high": {
            "direction": "Y->X",
            "direction_code": -1,
            "n_samples": int(np.sum(high_mask)),
            "x_range": [float(x[high_mask].min()), float(x[high_mask].max())],
            "y_range": [float(y[high_mask].min()), float(y[high_mask].max())],
            "hsic_cause_residual": hsic_statistic(high_cause, high_residuals),
            "attempts": high["attempts"],
            "accepted": high["accepted"],
            "acceptance_rate": high["acceptance_rate"],
        },
        "seed_used": int(seed),
        "has_nan": False,
        "has_inf": False,
        "direction_reversal": True,
    }
    return metadata, pairmeta


def write_readme(pairmeta_entries: list[str]) -> None:
    lines = [
        "# Synthetic Regime-Switching Datasets",
        "",
        "Generated with `scripts/generate_synthetic_regime_switching.py`.",
        "",
        "The active app includes two variants for each design:",
        "- base filename: 400 observations, 200 low-regime and 200 high-regime",
        "- `_n2000` filename: 2000 observations, 1000 low-regime and 1000 high-regime",
        "",
        "The threshold variable is `Variable_X`. Low regime uses direction code `1` (`X->Y`); high regime uses direction code `-1` (`Y->X`).",
        "",
        "## Active Synthetic IDs",
        "",
    ]
    lines.extend(f"- `{entry.split('|', 1)[0]}`" for entry in pairmeta_entries)
    lines.extend(
        [
            "",
            "## Generated Outputs",
            "",
            "- `synthetic_regime_switch_metadata.json`: metadata for all 14 active synthetic pairs.",
            "- `synthetic_regime_switch_pairmeta.txt`: synthetic-only pairmeta rows.",
            "- `*.txt`: copies of the active generated datapair files.",
            "",
        ]
    )
    (DOC_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--sizes", type=int, nargs="+", default=[400, 2000])
    args = parser.parse_args()

    ACTIVE_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)

    metadata: list[dict] = []
    pairmeta_entries: list[str] = []
    for size_idx, n_samples in enumerate(args.sizes):
        for spec_idx, spec in enumerate(specs()):
            seed = args.seed + size_idx * 1000 + spec_idx
            item, pairmeta = generate_one(spec, n_samples, seed)
            metadata.append(item)
            pairmeta_entries.append(pairmeta)
            print(
                f"{item['file_name']} | low={item['regime_low']['n_samples']} "
                f"high={item['regime_high']['n_samples']}"
            )

    metadata_text = json.dumps(metadata, indent=2)
    pairmeta_text = "\n".join(pairmeta_entries) + "\n"
    (DOC_DIR / "synthetic_regime_switch_metadata.json").write_text(
        metadata_text + "\n", encoding="utf-8"
    )
    (DOC_DIR / "synthetic_regime_switch_pairmeta.txt").write_text(
        pairmeta_text, encoding="utf-8"
    )
    (ROOT / "DATA" / "synthetic_regime_switch_pairmeta.txt").write_text(
        pairmeta_text, encoding="utf-8"
    )
    write_readme(pairmeta_entries)


if __name__ == "__main__":
    main()
