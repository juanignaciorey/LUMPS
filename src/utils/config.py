"""
Configuration Management
Load and validate configuration files
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class Config:
    """Configuration container."""
    evolution: Dict[str, Any]
    model: Dict[str, Any]
    training: Dict[str, Any]
    diagnostics: Dict[str, Any]
    use_wandb: bool = True


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    config_file = Path(config_path)

    if not config_file.exists():
        # Create default config if it doesn't exist
        create_default_config(config_path)

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    return config


def create_default_config(config_path: str):
    """
    Create default configuration file.

    Args:
        config_path: Path where to create config file
    """
    default_config = {
        'evolution': {
            'grid_size': 15,
            'population_size': 100,
            'generations': 1000,  # Reduced for faster testing
            'mutation_rate': 0.1,
            'crossover_rate': 0.7,
            'elite_size': 10,
            'num_fitness_types': 5,
            'tasks_per_fitness': 10000  # Reduced for faster testing
        },
        'model': {
            'd_model': 512,
            'n_layers': 12,
            'n_heads': 8,
            'dropout': 0.1,
            'max_grid_size': 30,
            'num_states': 10
        },
        'training': {
            'batch_size': 32,
            'learning_rate': 1e-4,
            'epochs': 50,  # Reduced for faster testing
            'num_candidates': 30,
            'checkpoint_every': 5
        },
        'diagnostics': {
            'eval_every': 10,
            'ood_tasks': 1000
        },
        'use_wandb': True
    }

    config_file = Path(config_path)
    config_file.parent.mkdir(parents=True, exist_ok=True)

    with open(config_file, 'w') as f:
        yaml.dump(default_config, f, default_flow_style=False, indent=2)

    print(f"Created default configuration at {config_path}")


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration parameters.

    Args:
        config: Configuration dictionary

    Returns:
        True if configuration is valid
    """
    required_sections = ['evolution', 'model', 'training', 'diagnostics']

    for section in required_sections:
        if section not in config:
            print(f"Missing required configuration section: {section}")
            return False

    # Validate evolution config
    evolution = config['evolution']
    required_evolution = ['grid_size', 'population_size', 'generations', 'tasks_per_fitness']
    for param in required_evolution:
        if param not in evolution:
            print(f"Missing required evolution parameter: {param}")
            return False

    # Validate model config
    model = config['model']
    required_model = ['d_model', 'n_layers', 'n_heads', 'max_grid_size', 'num_states']
    for param in required_model:
        if param not in model:
            print(f"Missing required model parameter: {param}")
            return False

    # Validate training config
    training = config['training']
    required_training = ['batch_size', 'learning_rate', 'epochs', 'num_candidates']
    for param in required_training:
        if param not in training:
            print(f"Missing required training parameter: {param}")
            return False

    return True
