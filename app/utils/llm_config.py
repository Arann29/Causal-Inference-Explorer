"""Configuration management for LLM experiment scenarios."""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Iterable
import json


DEFAULT_LLM_PROVIDER = "openrouter"
DEFAULT_MODEL_ID = "openai/gpt-oss-20b:free"


@dataclass(frozen=True)
class ModelSpec:
    """Metadata for a selectable LLM model."""

    key: str
    display_name: str
    model_id: str
    provider: str = DEFAULT_LLM_PROVIDER
    is_default: bool = False
    supported_parameters: tuple[str, ...] = ()


MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec(
        key="openai_gpt_oss_20b_free",
        display_name="OpenAI GPT-OSS 20B (free)",
        model_id="openai/gpt-oss-20b:free",
        is_default=True,
        supported_parameters=(
            "reasoning",
            "max_tokens",
            "temperature",
            "response_format",
            "structured_outputs",
        ),
    ),
    ModelSpec(
        key="nemotron_3_ultra_550b_a55b_free",
        display_name="NVIDIA Nemotron 3 Ultra 550B A55B (free)",
        model_id="nvidia/nemotron-3-ultra-550b-a55b:free",
        supported_parameters=(
            "reasoning",
            "max_tokens",
        ),
    ),
    ModelSpec(
        key="gpt_5_5",
        display_name="OpenAI GPT-5.5",
        model_id="openai/gpt-5.5",
        supported_parameters=(
            "reasoning",
            "reasoning_effort",
            "include_reasoning",
            "seed",
            "max_completion_tokens",
            "max_tokens",
            "response_format",
            "structured_outputs",
            "tools",
            "tool_choice",
        ),
    ),
    ModelSpec(
        key="claude_opus_4_7",
        display_name="Anthropic Claude Opus 4.7",
        model_id="anthropic/claude-opus-4.7",
        supported_parameters=(
            "max_tokens",
            "stop",
            "reasoning",
            "include_reasoning",
            "tool_choice",
            "tools",
            "response_format",
            "verbosity",
        ),
    ),
)


def get_model_specs() -> List[ModelSpec]:
    """Return all selectable model specifications."""
    return list(MODEL_SPECS)


def get_available_model_labels() -> List[str]:
    """Return display labels for the UI."""
    return [spec.display_name for spec in MODEL_SPECS]


def get_default_model_spec() -> ModelSpec:
    """Return the default model specification."""
    for spec in MODEL_SPECS:
        if spec.is_default:
            return spec
    return MODEL_SPECS[0]


def get_model_spec_by_label(display_name: str) -> ModelSpec:
    """Look up a model spec by its display label."""
    for spec in MODEL_SPECS:
        if spec.display_name == display_name:
            return spec
    raise ValueError(f"Unknown model label: {display_name}")


def get_model_spec_by_id(model_id: str) -> ModelSpec:
    """Look up a model spec by its provider model id."""
    for spec in MODEL_SPECS:
        if spec.model_id == model_id:
            return spec
    raise ValueError(f"Unknown model id: {model_id}")



@dataclass
class ExperimentConfig:
    """Configuration for LLM causal analysis experiments.
    
    Attributes:
        data_format: Type of data representation ('statistical_summary' or 'raw_data')
        stats_included: Dictionary of statistical measures to include
        raw_data_options: Options for raw data display
        context_hint_level: Level of context hints ('none', 'threshold_var_name')
        threshold_detection_mode: How to ask LLM about switching points
        regime_mode: How many regimes to assume
        model_name: LLM model to use
        temperature: Temperature parameter for LLM
        config_version: Version identifier for tracking schema changes
    """
    # Data representation
    data_format: str = 'statistical_summary'  # 'statistical_summary' or 'raw_data'
    
    stats_included: Dict[str, bool] = field(default_factory=lambda: {
        'mean': True,
        'std_dev': True,
        'min_max': True,
        'correlation': True,
        'sample_size': False,
        'quartiles': False
    })
    
    raw_data_options: Dict[str, Any] = field(default_factory=lambda: {

        'anonymize_names': False,  # Use X/Y instead of actual variable names
        'include_description': False,  # Add textual description of patterns
        'random_sample': True,  # Randomly sample rows instead of taking first N
        'random_seed': 42  # Seed for reproducibility
    })
    
    # Context hints
    context_hint_level: str = 'none'  # 'none', 'threshold_var_name'
    
    # Prompt chain modifications
    threshold_detection_mode: Dict[str, bool] = field(default_factory=lambda: {
        'explicit_threshold': False,
        'request_numeric_value': False,
        'ask_threshold_var': False,
        'confidence_interval': False
    })
    
    regime_mode: str = 'force_2_regimes'  # 'force_2_regimes', 'let_llm_decide', 'force_uniform'
    
    # LLM parameters
    model_name: str = DEFAULT_MODEL_ID
    temperature: float = 0.0  # Requested low-variance sampling; only used for models that support temperature.
    
    # Metadata
    config_version: str = '1.0'
    scenario_name: str = 'custom'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentConfig':
        """Create configuration from dictionary."""
        return cls(**data)
    
    def to_json(self) -> str:
        """Convert configuration to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'ExperimentConfig':
        """Create configuration from JSON string."""
        return cls.from_dict(json.loads(json_str))
    
    def validate(self) -> tuple[bool, Optional[str]]:
        """Validate configuration consistency.
        
        Returns:
            (is_valid, error_message)
        """
        # Check data format
        if self.data_format not in ['statistical_summary', 'raw_data', 'segmented_regimes']:
            return False, f"Invalid data_format: {self.data_format}"
        
        # Check context hint level
        valid_hints = ['none', 'threshold_var_name']
        if self.context_hint_level not in valid_hints:
            return False, f"Invalid context_hint_level: {self.context_hint_level}"
        
        # Check regime mode
        valid_regimes = ['force_2_regimes', 'let_llm_decide', 'force_uniform']
        if self.regime_mode not in valid_regimes:
            return False, f"Invalid regime_mode: {self.regime_mode}"
        
        # Check logical consistency
        if self.data_format == 'raw_data' and any(self.stats_included.values()):
            pass  # Warning but valid - stats are ignored for raw data
        
        if self.threshold_detection_mode['request_numeric_value'] and not self.threshold_detection_mode['explicit_threshold']:
            return False, "Cannot request numeric value without explicit_threshold enabled"
        
        try:
            get_model_spec_by_id(self.model_name)
        except ValueError:
            return False, f"Invalid model_name: {self.model_name}"

        return True, None
    
    def get_summary_description(self) -> str:
        """Get human-readable description of configuration."""
        lines = []
        
        # Data format
        if self.data_format == 'statistical_summary':
            stats = [k for k, v in self.stats_included.items() if v]
            lines.append(f"📊 **Data**: {', '.join(stats)}")
        else:
            anon = "anonymous variables" if self.raw_data_options['anonymize_names'] else "with variable names"
            lines.append(f"📊 **Data**: Raw data + statistics ({anon})")
        
        # Context hints
        context_map = {
            'none': 'No hints',

            'threshold_var_name': 'Threshold variable provided'
        }
        lines.append(f"🧠 **Context**: {context_map.get(self.context_hint_level, 'Unknown')}")
        
        # Model
        try:
            model_spec = get_model_spec_by_id(self.model_name)
            lines.append(f"🤖 **Model**: {model_spec.display_name} (`{model_spec.model_id}`)")
        except ValueError:
            lines.append(f"🤖 **Model**: {self.model_name}")

        # Threshold detection
        if self.threshold_detection_mode['explicit_threshold']:
            details = []
            if self.threshold_detection_mode['request_numeric_value']:
                details.append("numeric value")
            if self.threshold_detection_mode['ask_threshold_var']:
                details.append("variable name")
            if self.threshold_detection_mode['confidence_interval']:
                details.append("confidence interval")
            lines.append(f"🔍 **Threshold**: Explicit request ({', '.join(details)})")
        else:
            lines.append(f"🔍 **Threshold**: Organic discovery")
        
        # Regime mode
        regime_map = {
            'force_2_regimes': 'Force 2 regimes',
            'let_llm_decide': 'Let LLM decide',
            'force_uniform': 'No switching'
        }
        lines.append(f"📈 **Regimes**: {regime_map.get(self.regime_mode, 'Unknown')}")
        
        return '\n'.join(lines)


# Preset configurations - Only the 4 active configurations used in the UI
# All use fixed stats: min_max, sample_size (mean/std_dev disabled for fair comparison)
# All use force_2_regimes (bivariate data with single threshold)
PRESET_CONFIGS = {
    
    # Raw Data (Named) + Threshold Hint + Explicit Threshold
    'raw_named_hint_explicit': ExperimentConfig(
        scenario_name='Raw+Stats | Named | Hint | Explicit',
        data_format='raw_data',
        stats_included={
            'mean': False,
            'std_dev': False,
            'min_max': True,
            'sample_size': True,
            'correlation': False,
            'quartiles': False
        },
        raw_data_options={
            'anonymize_names': False,
            'random_sample': True,
            'random_seed': 42
        },
        context_hint_level='threshold_var_name',
        threshold_detection_mode={
            'explicit_threshold': True,
            'request_numeric_value': True,
            'ask_threshold_var': True,
            'confidence_interval': True
        },
        regime_mode='force_2_regimes',
    ),
    
    # Raw Data (Anonymous) + Threshold Hint + Explicit Threshold
    'raw_anon_hint_explicit': ExperimentConfig(
        scenario_name='Raw+Stats | Anonymous | Hint | Explicit',
        data_format='raw_data',
        stats_included={
            'mean': False,
            'std_dev': False,
            'min_max': True,
            'sample_size': True,
            'correlation': False,
            'quartiles': False
        },
        raw_data_options={
            'anonymize_names': True,
            'random_sample': True,
            'random_seed': 42
        },
        context_hint_level='threshold_var_name',
        threshold_detection_mode={
            'explicit_threshold': True,
            'request_numeric_value': True,
            'ask_threshold_var': True,
            'confidence_interval': True
        },
        regime_mode='force_2_regimes',
    ),
    
    # Segmented Regimes (Named) + Hint + Explicit
    'segmented_named_hint_explicit': ExperimentConfig(
        scenario_name='Segmented | Named | Hint | Explicit',
        data_format='segmented_regimes',
        stats_included={
            'mean': False,
            'std_dev': False,
            'min_max': True,
            'sample_size': True,
            'correlation': False,
            'quartiles': False
        },
        raw_data_options={
            'anonymize_names': False,
            'random_sample': True,
            'random_seed': 42
        },
        context_hint_level='threshold_var_name',
        threshold_detection_mode={
            'explicit_threshold': True,
            'request_numeric_value': True,
            'ask_threshold_var': True,
            'confidence_interval': True
        },
        regime_mode='force_2_regimes',
    ),
    
    # Segmented Regimes (Anonymous) + Hint + Explicit
    'segmented_anon_hint_explicit': ExperimentConfig(
        scenario_name='Segmented | Anonymous | Hint | Explicit',
        data_format='segmented_regimes',
        stats_included={
            'mean': False,
            'std_dev': False,
            'min_max': True,
            'sample_size': True,
            'correlation': False,
            'quartiles': False
        },
        raw_data_options={
            'anonymize_names': True,
            'random_sample': True,
            'random_seed': 42
        },
        context_hint_level='threshold_var_name',
        threshold_detection_mode={
            'explicit_threshold': True,
            'request_numeric_value': True,
            'ask_threshold_var': True,
            'confidence_interval': True
        },
        regime_mode='force_2_regimes',
    ),
}


def get_preset_config(preset_name: str) -> ExperimentConfig:
    """Get a preset configuration by name.
    
    Args:
        preset_name: Name of preset, such as 'raw_anon_hint_explicit'.
        
    Returns:
        ExperimentConfig instance
        
    Raises:
        ValueError: If preset name is not found
    """
    if preset_name not in PRESET_CONFIGS:
        raise ValueError(f"Unknown preset: {preset_name}. Available: {list(PRESET_CONFIGS.keys())}")
    
    return PRESET_CONFIGS[preset_name]


def get_available_presets() -> List[str]:
    """Get list of available preset names."""
    return list(PRESET_CONFIGS.keys())
