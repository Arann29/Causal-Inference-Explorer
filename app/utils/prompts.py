# app/utils/prompts.py
"""Dynamic prompt generation for LLM causal analysis experiments."""

from typing import Optional, List, Dict
from app.utils.llm_config import ExperimentConfig


def _get_canonical_variable_labels(
    config: Optional[ExperimentConfig] = None,
    df_columns: Optional[List[str]] = None,
) -> tuple[str, str]:
    if config and config.raw_data_options.get("anonymize_names", False):
        return "Variable_X", "Variable_Y"

    if df_columns and len(df_columns) >= 2:
        return df_columns[0], df_columns[1]

    return "Variable_X", "Variable_Y"


def normalize_direction(direction: str, var_x: str, var_y: str) -> Optional[str]:
    if not direction:
        return None

    d = str(direction).strip().replace(" ", "").replace("→", "->")

    xy = f"{var_x}->{var_y}"
    yx = f"{var_y}->{var_x}"

    if d == xy:
        return xy
    if d == yx:
        return yx

    if d in {"X->Y", "Variable_X->Variable_Y"}:
        return xy
    if d in {"Y->X", "Variable_Y->Variable_X"}:
        return yx

    return None


def _build_step2_explicit(
    config: Optional[ExperimentConfig] = None,
    threshold_var: Optional[str] = None,
    threshold_value: Optional[float] = None,
    df_columns: Optional[List[str]] = None,
) -> str:
    """Build Step 2 prompt for regime characterisation.

    Args:
        config: Experiment configuration (optional)
        threshold_var: Name of the threshold variable (always provided)
        threshold_value: Numeric threshold value (provided when explicit_threshold=True)
        df_columns: First two column names from dataframe for anonymization mapping

    Returns:
        Formatted prompt string for Step 2
    """

    threshold_display = threshold_var

    # Preserve anonymisation mapping
    if (
        config
        and config.raw_data_options.get("anonymize_names", False)
        and df_columns
        and threshold_var
        and len(df_columns) >= 2
    ):
        if threshold_var == df_columns[0]:
            threshold_display = "Variable_X"
        elif threshold_var == df_columns[1]:
            threshold_display = "Variable_Y"
        else:
            threshold_display = "the segmentation variable"

    if threshold_var and threshold_value is not None:
        instructions = [
            "Step 2: Regime Characterisation",
            "",
            "Role: Data Scientist",
            "",
            "Input:",
            "{summary}",
            "",
            "Previous analysis:",
            "{history}",
            "",
            "The data are split by a literature-based threshold:",
            f"- Regime 0: {threshold_display} <= {threshold_value}",
            f"- Regime 1: {threshold_display} > {threshold_value}",
            "",
            "Using only the dataset, variable descriptions if available, and Step 1, describe each regime.",
            "",
            "Report:",
            "1. sample size, if available;",
            "2. range, center, and spread of X and Y, if available;",
            "3. X-Y association: sign and strength;",
            "4. functional type, heteroscedasticity, outliers, or clusters, if visible;",
            "5. regime differences in center, spread, range, association sign, association strength, or functional type.",
            "",
            "Do not infer causal direction.",
        ]
    else:
        # Fallback for when threshold is not explicitly provided
        instructions = [
            "Step 2: Regime Characterisation",
            "",
            "Role: Data Scientist",
            "",
            "Input:",
            "{summary}",
            "",
            "Previous analysis:",
            "{history}",
            "",
            f"The segmentation variable is {threshold_display}.",
            "",
            "Using only the dataset, variable descriptions if available, and Step 1, describe each regime.",
            "",
            "Report:",
            "1. sample size, if available;",
            "2. range, center, and spread of X and Y, if available;",
            "3. X-Y association: sign and strength;",
            "4. functional type, heteroscedasticity, outliers, or clusters, if visible;",
            "5. regime differences in center, spread, range, association sign, association strength, or functional type.",
            "",
            "Do not infer causal direction.",
        ]

    return "\n".join(instructions)


def get_prompt_chain(config: Optional[ExperimentConfig] = None, threshold_var: Optional[str] = None, threshold_value: Optional[float] = None, df_columns: Optional[List[str]] = None) -> List[Dict[str, str]]:
    """
    Returns the optimized 4-step prompt chain for causal reasoning.
    
    Args:
        config: Optional experiment configuration for dynamic prompt modification
        threshold_var: Name of the threshold variable (optional, from ground truth)
        threshold_value: Numeric threshold value (optional, from ground truth)
        df_columns: First two column names from dataframe for anonymization mapping
        
    Returns:
        List of prompt steps with name and prompt text
    """
    var_x, var_y = _get_canonical_variable_labels(config, df_columns)
    allowed_direction_xy = f"{var_x}->{var_y}"
    allowed_direction_yx = f"{var_y}->{var_x}"

    return [
          {
                "name": "Step 1: Domain Understanding and Descriptive Analysis",
                "prompt": """Step 1: Domain Understanding and Descriptive Analysis

Role: Data Scientist

Task: Describe the variables and their statistical relationship in a bivariate numerical dataset.

Input:
{summary}

Instructions:

1. Variable description
For X and Y, report only what is available from the data sample, statistical summary, or variable description:

- approximate range;
- center, if available;
- spread, if available;
- visible skewness, outliers, clusters, or bounded ranges.

2. Statistical relationship
Describe the relationship between X and Y using the data sample and the variable descriptions from step 1. Report:

- correlation sign: positive, negative, or unclear;
- correlation strength: weak, moderate, strong, or unclear;
- functional properties: linear, nonlinear, monotonic, step-wise, or no concrete functional type;
- heteroscedasticity, if the spread of at least one variable changes with the other.

Do not choose or suggest a causal direction. This step is descriptive only.
"""
          },
        {
            "name": "Step 2: Regime Identification",
            "prompt": _build_step2_explicit(config, threshold_var, threshold_value, df_columns)
        },
        {
            "name": "Step 3: Regime-Specific Causal Analysis",
            "prompt": """Step 3: Regime-Specific Causal Analysis

    Role: Data Scientist

    Task: Determine one causal direction per regime.

    Input:
    {summary}

    Previous analysis:
    {history}

    Determine one causal direction per regime.

    For Regime 0 and Regime 1, report:
    1. Direction: X→Y or Y→X.
    2. Evidence used: data, variable description, Step 1, Step 2, domain knowledge, threshold, or weak/conflicting evidence.
    3. Reason: Provide 2–3 short sentences.
    4. Confidence: High, Moderate, or Low.

    Use Low confidence when evidence is weak, mixed, or mainly based on variable names.

    Then state whether there is a direction change:
    - yes, if the two regimes have opposite directions;
    - no, if both regimes have the same direction."""
        },
        {
            "name": "Step 4: JSON Output",
            "prompt": f"""Step 4: JSON Output

Role: Data Scientist

Task: Return the final causal direction results as valid JSON.

Dataset Summary:
{{summary}}

Previous Analysis:
---
{{history}}
---

Output ONLY valid JSON. Do not include explanations before or after the JSON.

Use exactly this structure:
{{
    "threshold_variable": "<threshold variable label>",
    "threshold_value": <numeric threshold>,
    "regimes": 2,
    "directions": [
        {{"regime": 0, "direction": "<allowed direction>", "confidence": "<High|Moderate|Low>"}},
        {{"regime": 1, "direction": "<allowed direction>", "confidence": "<High|Moderate|Low>"}}
    ],
    "direction_change": "<Yes|No>"
}}

Allowed direction values:
- "{allowed_direction_xy}"
- "{allowed_direction_yx}"

Rules:
- Use the threshold variable label shown in the dataset summary.
- Regime 0 means threshold_variable <= threshold_value.
- Regime 1 means threshold_variable > threshold_value.
- The direction value must be exactly one of the allowed direction values above.
- Use ASCII arrows only: "->". Do not use the Unicode arrow character "→".
- Confidence must be exactly one of: "High", "Moderate", or "Low".
- direction_change must be "Yes" only if the two regimes have opposite directions.
- direction_change must be "No" if both regimes have the same direction.
- Do not abbreviate direction labels.

- Do not use X, Y, Variable A, Variable B, cause, effect, low, high, or arrows with spaces in the direction field.
- Avoid informal synonyms for 'direction change'.

Return the JSON object now."""
        }
    ]
