"""
BarkSSL Configuration Module
Configuration loading and management with YAML support and CLI overrides.
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional, Union
from dataclasses import dataclass, field


@dataclass
class Config:
    """
    Configuration manager for BarkSSL.

    Supports loading from YAML files, merging configs, and CLI overrides.

    Usage:
        config = Config('configs/config.yaml')
        config = Config.from_dict({'model': {'hidden_dim': 512}})
        config.get('model.hidden_dim')  # dot notation access
        config.set('training.learning_rate', 0.001)
    """
    _config: Dict[str, Any] = field(default_factory=dict)
    _raw: Dict[str, Any] = field(default_factory=dict)

    def __init__(self, config_path: Optional[str] = None, config_dict: Optional[Dict] = None):
        """
        Initialize config from file or dictionary.

        Args:
            config_path: Path to YAML config file
            config_dict: Dictionary with config values
        """
        if config_path:
            self.load(config_path)
        elif config_dict:
            self._config = config_dict
            self._raw = config_dict.copy()

    def load(self, config_path: str) -> 'Config':
        """Load config from YAML file."""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, 'r') as f:
            self._raw = yaml.safe_load(f) or {}
            self._config = self._flatten_dict(self._raw)

        return self

    def _flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '.') -> Dict:
        """Flatten nested dict with dot notation keys."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def _unflatten_dict(self, d: Dict, sep: str = '.') -> Dict:
        """Unflatten dict with dot notation keys."""
        result = {}
        for key, value in d.items():
            parts = key.split(sep)
            current = result
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = value
        return result

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get config value using dot notation.

        Args:
            key: Config key (e.g., 'model.small.hidden_dim')
            default: Default value if key not found

        Returns:
            Config value or default
        """
        return self._config.get(key, default)

    def set(self, key: str, value: Any) -> 'Config':
        """
        Set config value using dot notation.

        Args:
            key: Config key (e.g., 'training.learning_rate')
            value: Value to set

        Returns:
            Self for chaining
        """
        self._config[key] = value
        return self

    def update(self, updates: Dict[str, Any]) -> 'Config':
        """
        Update config with dictionary.

        Args:
            updates: Dictionary of key-value pairs

        Returns:
            Self for chaining
        """
        self._config.update(updates)
        return self

    def override(self, **kwargs) -> 'Config':
        """
        Override config values from CLI arguments.

        Args:
            **kwargs: Key-value pairs to override

        Returns:
            Self for chaining
        """
        for key, value in kwargs.items():
            if value is not None:
                self._config[key] = value
        return self

    def get_model_config(self) -> Dict[str, Any]:
        """Get model config based on scale setting."""
        scale = self.get('model.scale', 'small')
        model_config = self.get(f'model.{scale}', {})
        return model_config

    def to_dict(self) -> Dict:
        """Convert config to nested dictionary."""
        return self._unflatten_dict(self._config)

    def __getitem__(self, key: str) -> Any:
        """Allow dict-style access."""
        return self._config[key]

    def __contains__(self, key: str) -> bool:
        """Allow 'in' operator."""
        return key in self._config

    def __repr__(self) -> str:
        return f"Config({len(self._config)} keys)"

    def __str__(self) -> str:
        """Pretty print config."""
        lines = []
        for key, value in sorted(self._config.items()):
            lines.append(f"  {key}: {value}")
        return "Config:\n" + "\n".join(lines)


def load_config(config_path: str, overrides: Optional[Dict] = None) -> Config:
    """
    Load config from file with optional overrides.

    Args:
        config_path: Path to YAML config file
        overrides: Optional dict of overrides

    Returns:
        Config instance
    """
    config = Config(config_path)
    if overrides:
        config.override(**overrides)
    return config


def merge_configs(*configs: Config) -> Config:
    """
    Merge multiple configs, later configs taking precedence.

    Args:
        *configs: Config instances to merge

    Returns:
        Merged Config instance
    """
    result = Config()
    for config in configs:
        result._config.update(config._config)
        result._raw.update(config._raw)
    return result