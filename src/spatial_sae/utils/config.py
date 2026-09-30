import os
import yaml
from typing import Any, Dict, Union
from pathlib import Path


class ConfigDict(dict):
    """Dot-accessible dictionary for configuration management."""

    def __init__(self, *args, **kwargs):
        super(ConfigDict, self).__init__(*args, **kwargs)
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = ConfigDict(value)

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"Configuration has no attribute '{key}'")

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = ConfigDict(value) if isinstance(value, dict) else value

    def __delattr__(self, key: str) -> None:
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"Configuration has no attribute '{key}'")

    def to_dict(self) -> Dict[str, Any]:
        """Convert dot-accessible ConfigDict back to a standard Python dictionary."""
        res = {}
        for k, v in self.items():
            if isinstance(v, ConfigDict):
                res[k] = v.to_dict()
            else:
                res[k] = v
        return res


def load_config(config_path: Union[str, Path]) -> ConfigDict:
    """Load a YAML configuration file into a ConfigDict.

    Supports nested dictionary loading and resolving sub-configs referenced in 'configs'.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f) or {}

    # If sub-configs are specified, resolve them relative to config_path parent or project root
    if "configs" in raw_config and isinstance(raw_config["configs"], dict):
        resolved_configs = {}
        for cfg_name, sub_path in raw_config["configs"].items():
            sub_path_obj = Path(sub_path)
            if not sub_path_obj.is_absolute():
                # Try relative to current config path first, then relative to working dir
                candidate = config_path.parent / sub_path_obj
                if candidate.exists():
                    sub_path_obj = candidate
                elif not sub_path_obj.exists():
                    sub_path_obj = Path.cwd() / sub_path_obj
            resolved_configs[cfg_name] = load_config(sub_path_obj)
        raw_config["sub_configs"] = resolved_configs

    return ConfigDict(raw_config)


def save_config(config: Union[ConfigDict, Dict[str, Any]], output_path: Union[str, Path]) -> None:
    """Save configuration dictionary to a YAML file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = config.to_dict() if isinstance(config, ConfigDict) else config
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)
