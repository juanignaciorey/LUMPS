"""
Phase 0 Pipeline
Complete orchestration of data generation, training, and diagnostics
"""

import os
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
import numpy as np
from tqdm import tqdm
import yaml

from .model import LUMPSTransformer, EnergyFunction, OutputGenerator
from .trainer import Phase0Trainer
from .data_loader import create_dataloader
from ..cellular_automata.evolution import evolve_ca_batch
from ..cellular_automata.task_generator import ARCTaskGenerator
from ..diagnostics.metrics import evaluate_phase0_exit_criteria
from ..utils.config import load_config
from ..utils.logger import setup_logger


class Phase0Pipeline:
    """
    Complete Phase 0 pipeline orchestrating data generation, training, and diagnostics.

    Implements the full workflow from CA evolution to model training and evaluation.
    """

    def __init__(
        self,
        config_path: str = "configs/phase0.yaml",
        output_dir: str = "data",
        device: str = "cuda",
        seed: Optional[int] = 42
    ):
        """
        Initialize Phase 0 pipeline.

        Args:
            config_path: Path to configuration file
            output_dir: Output directory for data and models
            device: Device to use for training
            seed: Random seed for reproducibility
        """
        self.config = load_config(config_path)
        self.output_dir = Path(output_dir)
        self.device = device
        self.seed = seed

        # Set random seeds
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Create output directories
        self.raw_dir = self.output_dir / "raw"
        self.processed_dir = self.output_dir / "processed"
        self.checkpoints_dir = self.output_dir / "checkpoints"

        for dir_path in [self.raw_dir, self.processed_dir, self.checkpoints_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self.logger = setup_logger("phase0_pipeline")

        # Initialize components
        self.task_generator = ARCTaskGenerator(
            grid_size=(self.config['evolution']['grid_size'], self.config['evolution']['grid_size']),
            seed=seed
        )

        # Model components
        self.model = None
        self.energy_function = None
        self.output_generator = None
        self.trainer = None

    def generate_data(self) -> Dict[str, List[str]]:
        """
        Generate synthetic training data through CA evolution.

        Returns:
            Dictionary mapping fitness types to task file lists
        """
        self.logger.info("Starting data generation phase...")

        evolution_config = self.config['evolution']
        fitness_types = ['expand', 'symmetry', 'count', 'topology', 'replicate']

        all_task_files = {}

        for fitness_type in fitness_types:
            self.logger.info(f"Evolving CAs for {fitness_type} fitness...")

            # Create output directory for this fitness type
            fitness_raw_dir = self.raw_dir / fitness_type
            fitness_raw_dir.mkdir(exist_ok=True)

            # Evolve CAs
            ca_files = evolve_ca_batch(
                grid_size=(evolution_config['grid_size'], evolution_config['grid_size']),
                population_size=evolution_config['population_size'],
                generations=evolution_config['generations'],
                fitness_type=fitness_type,
                num_cas=evolution_config['tasks_per_fitness'] // 5,  # 5 tasks per CA
                output_dir=str(fitness_raw_dir),
                seed=self.seed
            )

            self.logger.info(f"Evolved {len(ca_files)} CAs for {fitness_type}")

            # Generate ARC tasks from evolved CAs
            self.logger.info(f"Generating ARC tasks for {fitness_type}...")

            task_files = self.task_generator.generate_tasks_from_evolved_cas(
                ca_files=ca_files,
                output_dir=str(self.processed_dir / fitness_type),
                tasks_per_ca=5,
                max_tasks=evolution_config['tasks_per_fitness']
            )

            all_task_files[fitness_type] = task_files
            self.logger.info(f"Generated {len(task_files)} tasks for {fitness_type}")

        # Create statistics
        total_tasks = sum(len(files) for files in all_task_files.values())
        self.logger.info(f"Total tasks generated: {total_tasks}")

        # Save task file lists
        with open(self.processed_dir / "task_files.json", 'w') as f:
            json.dump(all_task_files, f, indent=2)

        return all_task_files

    def setup_model(self):
        """Initialize model components."""
        self.logger.info("Setting up model components...")

        model_config = self.config['model']

        # Initialize model
        self.model = LUMPSTransformer(
            d_model=model_config['d_model'],
            n_layers=model_config['n_layers'],
            n_heads=model_config['n_heads'],
            max_grid_size=model_config['max_grid_size'],
            num_states=model_config['num_states'],
            dropout=model_config['dropout']
        )

        # Initialize energy function
        self.energy_function = EnergyFunction(
            d_model=model_config['d_model']
        )

        # Initialize output generator
        self.output_generator = OutputGenerator(
            d_model=model_config['d_model'],
            max_grid_size=model_config['max_grid_size'],
            num_states=model_config['num_states']
        )

        # Initialize trainer
        training_config = self.config['training']
        self.trainer = Phase0Trainer(
            model=self.model,
            energy_function=self.energy_function,
            output_generator=self.output_generator,
            device=self.device,
            learning_rate=training_config['learning_rate'],
            num_candidates=training_config['num_candidates'],
            use_wandb=self.config.get('use_wandb', True)
        )

        self.logger.info("Model components initialized")

    def create_dataloaders(self, task_files: Dict[str, List[str]]) -> Tuple:
        """
        Create training, validation, and OOD dataloaders.

        Args:
            task_files: Dictionary mapping fitness types to task files

        Returns:
            Tuple of (train_loader, val_loader, ood_loader)
        """
        self.logger.info("Creating data loaders...")

        # Combine all task files
        all_task_files = []
        for fitness_files in task_files.values():
            all_task_files.extend(fitness_files)

        # Split into train/val/OOD
        np.random.shuffle(all_task_files)

        n_total = len(all_task_files)
        n_train = int(0.7 * n_total)
        n_val = int(0.15 * n_total)

        train_files = all_task_files[:n_train]
        val_files = all_task_files[n_train:n_train + n_val]
        ood_files = all_task_files[n_train + n_val:]

        self.logger.info(f"Data split: {len(train_files)} train, {len(val_files)} val, {len(ood_files)} OOD")

        # Create data loaders
        training_config = self.config['training']

        train_loader = create_dataloader(
            task_files=train_files,
            batch_size=training_config['batch_size'],
            max_grid_size=self.config['model']['max_grid_size'],
            augment=True,
            shuffle=True,
            seed=self.seed
        )

        val_loader = create_dataloader(
            task_files=val_files,
            batch_size=training_config['batch_size'],
            max_grid_size=self.config['model']['max_grid_size'],
            augment=False,
            shuffle=False,
            seed=self.seed
        )

        ood_loader = create_dataloader(
            task_files=ood_files,
            batch_size=training_config['batch_size'],
            max_grid_size=self.config['model']['max_grid_size'],
            augment=False,
            shuffle=False,
            seed=self.seed
        )

        return train_loader, val_loader, ood_loader

    def train_model(self, train_loader, val_loader) -> Dict:
        """
        Train the model.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader

        Returns:
            Training history
        """
        self.logger.info("Starting model training...")

        training_config = self.config['training']

        history = self.trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=training_config['epochs'],
            checkpoint_every=training_config['checkpoint_every'],
            checkpoint_dir=str(self.checkpoints_dir)
        )

        self.logger.info("Model training completed")
        return history

    def run_diagnostics(self, train_loader, val_loader, ood_loader) -> Dict:
        """
        Run diagnostic evaluation.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            ood_loader: Out-of-distribution data loader

        Returns:
            Diagnostic results
        """
        self.logger.info("Running diagnostic evaluation...")

        results = evaluate_phase0_exit_criteria(
            model=self.model,
            train_loader=train_loader,
            val_loader=val_loader,
            ood_loader=ood_loader,
            device=self.device
        )

        # Save diagnostic results
        with open(self.checkpoints_dir / "diagnostic_results.json", 'w') as f:
            json.dump(results, f, indent=2)

        self.logger.info("Diagnostic evaluation completed")
        return results

    def run_full_pipeline(self) -> Dict:
        """
        Run the complete Phase 0 pipeline.

        Returns:
            Dictionary with pipeline results
        """
        self.logger.info("Starting Phase 0 pipeline...")

        results = {}

        try:
            # 1. Generate data
            self.logger.info("Step 1: Data generation")
            task_files = self.generate_data()
            results['data_generation'] = {
                'total_tasks': sum(len(files) for files in task_files.values()),
                'tasks_by_fitness': {k: len(v) for k, v in task_files.items()}
            }

            # 2. Setup model
            self.logger.info("Step 2: Model setup")
            self.setup_model()

            # 3. Create data loaders
            self.logger.info("Step 3: Data loader creation")
            train_loader, val_loader, ood_loader = self.create_dataloaders(task_files)

            # 4. Train model
            self.logger.info("Step 4: Model training")
            training_history = self.train_model(train_loader, val_loader)
            results['training'] = training_history

            # 5. Run diagnostics
            self.logger.info("Step 5: Diagnostic evaluation")
            diagnostic_results = self.run_diagnostics(train_loader, val_loader, ood_loader)
            results['diagnostics'] = diagnostic_results

            # 6. Final evaluation
            phase0_passed = diagnostic_results['phase0_passed']
            results['phase0_passed'] = phase0_passed

            if phase0_passed:
                self.logger.info("🎉 Phase 0 PASSED! Ready to proceed to Phase 1.")
            else:
                self.logger.warning("⚠️ Phase 0 FAILED. Consider adjusting parameters or continuing training.")

        except Exception as e:
            self.logger.error(f"Pipeline failed with error: {e}")
            results['error'] = str(e)
            raise

        # Save final results
        with open(self.checkpoints_dir / "pipeline_results.json", 'w') as f:
            json.dump(results, f, indent=2)

        self.logger.info("Phase 0 pipeline completed")
        return results

    def run_pilot_study(self, num_cas: int = 100) -> Dict:
        """
        Run a small pilot study to validate the system.

        Args:
            num_cas: Number of CAs to evolve for pilot

        Returns:
            Pilot study results
        """
        self.logger.info(f"Running pilot study with {num_cas} CAs...")

        # Temporarily modify config for pilot
        original_tasks_per_fitness = self.config['evolution']['tasks_per_fitness']
        self.config['evolution']['tasks_per_fitness'] = num_cas // 5  # 5 tasks per CA

        try:
            # Run reduced pipeline
            results = self.run_full_pipeline()

            # Add pilot-specific analysis
            results['pilot_study'] = True
            results['num_cas_used'] = num_cas

            self.logger.info("Pilot study completed successfully")

        finally:
            # Restore original config
            self.config['evolution']['tasks_per_fitness'] = original_tasks_per_fitness

        return results
