"""
Centralized configuration for ICASSP project
"""
from pathlib import Path
import json
from typing import Dict, Any, Optional

class Config:
    """Configuration management for ICASSP project"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or Path.home() / "ICASSP" / "config.json"
        self.config = self._load_or_create_config()
    
    def _load_or_create_config(self) -> Dict[str, Any]:
        """Load config from file or create default"""
        if Path(self.config_path).exists():
            with open(self.config_path, 'r') as f:
                return json.load(f)
        
        # Default configuration
        return {
            "paths": {
                "base_dir": str(Path.home() / "ICASSP"),
                "data_dir": str(Path.home() / "ICASSP" / "Depth_IR_PM"),
                "ir_dir": str(Path.home() / "ICASSP" / "Depth_IR_PM" / "IR_png"),
                "pm_dir": str(Path.home() / "ICASSP" / "Depth_IR_PM" / "PM_png"),
                "result_dir": str(Path.home() / "ICASSP" / "result"),
                "model_save_dir": str(Path.home() / "ICASSP" / "result" / "model_save")
            },
            "training": {
                "batch_size": 32,
                "num_epochs": 50,
                "learning_rate": 1e-3,
                "train_split": 0.8,
                "val_split": 0.1,
                "test_split": 0.1,
                "image_height": 192,
                "image_width": 96,
                "num_workers": 4
            },
            "model": {
                "encoder_name": "resnet50",
                "encoder_weights": "imagenet",
                "in_channels": 3,
                "classes": 1
            },
            "metrics": {
                "ema_alpha": 0.3,
                "max_val": 1.0,
                "resample_to": "gt"
            },
            "baseline": {
                "cal_frac": 0.2,
                "sample_pix_frac": 0.05,
                "ridge_alpha": 10.0
            }
        }
    
    def save(self):
        """Save configuration to file"""
        Path(self.config_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def get(self, key_path: str, default=None):
        """Get config value using dot notation (e.g., 'paths.base_dir')"""
        keys = key_path.split('.')
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value
    
    def set(self, key_path: str, value: Any):
        """Set config value using dot notation"""
        keys = key_path.split('.')
        config = self.config
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        config[keys[-1]] = value
        self.save()
    
    def get_path(self, key: str) -> Path:
        """Get path from config and ensure it's a Path object"""
        path_str = self.get(f"paths.{key}")
        return Path(path_str) if path_str else Path.home() / "ICASSP"