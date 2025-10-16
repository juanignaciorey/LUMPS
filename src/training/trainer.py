"""
Phase 0 Trainer
Energy-based training with contrastive learning
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Optional, Tuple
try:
    import wandb
except ImportError:
    wandb = None
from tqdm import tqdm
import os
from pathlib import Path

from .model import LUMPSTransformer, EnergyFunction, OutputGenerator
from .data_loader import ARCDataset


class Phase0Trainer:
    """
    Trainer for Phase 0 with energy-based contrastive learning.

    Uses energy function to distinguish good from bad solutions.
    """

    def __init__(
        self,
        model: LUMPSTransformer,
        energy_function: EnergyFunction,
        output_generator: OutputGenerator,
        device: str = "cuda",
        learning_rate: float = 1e-4,
        num_candidates: int = 30,
        temperature: float = 1.0,
        use_wandb: bool = True,
        project_name: str = "lumps-phase0"
    ):
        """
        Initialize trainer.

        Args:
            model: LUMPSTransformer model
            energy_function: Energy function for evaluation
            output_generator: Generator for candidate outputs
            device: Device to use for training
            learning_rate: Learning rate
            num_candidates: Number of negative candidates per example
            temperature: Temperature for contrastive loss
            use_wandb: Whether to use Weights & Biases logging
            project_name: W&B project name
        """
        self.model = model.to(device)
        self.energy_function = energy_function.to(device)
        self.output_generator = output_generator.to(device)
        self.device = device
        self.num_candidates = num_candidates
        self.temperature = temperature

        # Optimizers
        self.optimizer = optim.AdamW(
            list(self.model.parameters()) +
            list(self.energy_function.parameters()) +
            list(self.output_generator.parameters()),
            lr=learning_rate,
            weight_decay=0.01
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=1e-6
        )

        # Loss function
        self.contrastive_loss = nn.CrossEntropyLoss()

        # Initialize W&B
        if use_wandb and wandb is not None:
            try:
                wandb.init(project=project_name, name="phase0-training")
                wandb.watch(self.model)
            except Exception as e:
                print(f"Warning: wandb initialization failed: {e}")
                use_wandb = False
        elif use_wandb and wandb is None:
            print("Warning: wandb not available, continuing without logging")
            use_wandb = False

        self.use_wandb = use_wandb

    def generate_candidates(
        self,
        input_grid: torch.Tensor,
        target_output: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate positive and negative candidates.

        Args:
            input_grid: Input grid tensor
            target_output: Target output grid tensor

        Returns:
            (candidates, labels) where labels indicate positive/negative
        """
        batch_size, num_examples, height, width = input_grid.shape

        # Get model outputs
        with torch.no_grad():
            outputs = self.model(input_grid.view(-1, height, width))
            embeddings = outputs['embeddings']

        # Generate positive candidates (target outputs)
        positive_candidates = target_output.view(-1, height, width)

        # Generate negative candidates
        negative_candidates = []

        for _ in range(self.num_candidates):
            # Generate random negative candidates
            negative = torch.randint(
                0, self.model.num_states,
                size=(batch_size * num_examples, height, width),
                device=self.device
            )
            negative_candidates.append(negative)

        # Combine all candidates
        all_candidates = torch.cat([positive_candidates.unsqueeze(1)] +
                                  [neg.unsqueeze(1) for neg in negative_candidates], dim=1)

        # Create labels (0 for positive, 1 for negative)
        labels = torch.zeros(batch_size * num_examples, device=self.device, dtype=torch.long)

        return all_candidates, labels

    def compute_energy_loss(
        self,
        input_grid: torch.Tensor,
        candidates: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute energy-based contrastive loss.

        Args:
            input_grid: Input grid tensor
            candidates: Candidate output grids
            labels: Labels indicating positive/negative

        Returns:
            Contrastive loss
        """
        batch_size, num_candidates, height, width = candidates.shape

        # Flatten for processing
        input_flat = input_grid.view(-1, height, width)
        candidates_flat = candidates.view(-1, height, width)

        # Get embeddings for input
        with torch.no_grad():
            input_outputs = self.model(input_flat)
            input_embeddings = input_outputs['embeddings']

        # Get embeddings for candidates
        candidate_outputs = self.model(candidates_flat)
        candidate_embeddings = candidate_outputs['embeddings']

        # Compute energies
        input_energy = self.energy_function(input_embeddings)
        candidate_energy = self.energy_function(candidate_embeddings)

        # Reshape energies
        input_energy = input_energy.view(batch_size, 1)
        candidate_energy = candidate_energy.view(batch_size, num_candidates)

        # Compute energy differences
        energy_diff = candidate_energy - input_energy  # (batch_size, num_candidates)

        # Apply temperature
        energy_diff = energy_diff / self.temperature

        # Create logits (negative energy differences for contrastive loss)
        logits = -energy_diff

        # Compute contrastive loss
        loss = self.contrastive_loss(logits, labels)

        return loss

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            dataloader: Training data loader

        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        self.energy_function.train()
        self.output_generator.train()

        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(dataloader, desc="Training")

        for batch in pbar:
            input_grids = batch['input'].to(self.device)
            output_grids = batch['output'].to(self.device)

            # Generate candidates
            candidates, labels = self.generate_candidates(input_grids, output_grids)

            # Compute loss
            loss = self.compute_energy_loss(input_grids, candidates, labels)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                list(self.model.parameters()) +
                list(self.energy_function.parameters()) +
                list(self.output_generator.parameters()),
                max_norm=1.0
            )

            self.optimizer.step()

            # Update metrics
            total_loss += loss.item()
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_loss = total_loss / num_batches

        return {
            'train_loss': avg_loss,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }

    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Evaluate model on validation set.

        Args:
            dataloader: Validation data loader

        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        self.energy_function.eval()
        self.output_generator.eval()

        total_loss = 0.0
        num_batches = 0
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                input_grids = batch['input'].to(self.device)
                output_grids = batch['output'].to(self.device)

                # Generate candidates
                candidates, labels = self.generate_candidates(input_grids, output_grids)

                # Compute loss
                loss = self.compute_energy_loss(input_grids, candidates, labels)
                total_loss += loss.item()
                num_batches += 1

                # Compute accuracy (simplified)
                batch_size, num_candidates, height, width = candidates.shape
                input_flat = input_grids.view(-1, height, width)
                candidates_flat = candidates.view(-1, height, width)

                # Get embeddings
                input_outputs = self.model(input_flat)
                candidate_outputs = self.model(candidates_flat)

                input_embeddings = input_outputs['embeddings']
                candidate_embeddings = candidate_outputs['embeddings']

                # Compute energies
                input_energy = self.energy_function(input_embeddings)
                candidate_energy = self.energy_function(candidate_embeddings)

                # Reshape and find best candidates
                input_energy = input_energy.view(batch_size, 1)
                candidate_energy = candidate_energy.view(batch_size, num_candidates)

                # Find lowest energy candidate (should be positive)
                best_candidates = torch.argmin(candidate_energy, dim=1)
                correct_predictions += (best_candidates == 0).sum().item()
                total_predictions += batch_size

        avg_loss = total_loss / num_batches
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0

        return {
            'val_loss': avg_loss,
            'val_accuracy': accuracy
        }

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        checkpoint_every: int = 10,
        checkpoint_dir: str = "checkpoints"
    ) -> Dict[str, List[float]]:
        """
        Full training loop.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs to train
            checkpoint_every: Save checkpoint every N epochs
            checkpoint_dir: Directory to save checkpoints

        Returns:
            Dictionary with training history
        """
        # Create checkpoint directory
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'learning_rate': []
        }

        best_val_loss = float('inf')

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")

            # Train
            train_metrics = self.train_epoch(train_loader)

            # Validate
            val_metrics = self.evaluate(val_loader)

            # Update scheduler
            self.scheduler.step()

            # Log metrics
            if self.use_wandb and wandb is not None:
                try:
                    wandb.log({
                        'epoch': epoch,
                        **train_metrics,
                        **val_metrics
                    })
                except:
                    pass  # Continue without wandb logging

            # Update history
            history['train_loss'].append(train_metrics['train_loss'])
            history['val_loss'].append(val_metrics['val_loss'])
            history['val_accuracy'].append(val_metrics['val_accuracy'])
            history['learning_rate'].append(train_metrics['learning_rate'])

            # Print metrics
            print(f"Train Loss: {train_metrics['train_loss']:.4f}")
            print(f"Val Loss: {val_metrics['val_loss']:.4f}")
            print(f"Val Accuracy: {val_metrics['val_accuracy']:.4f}")
            print(f"Learning Rate: {train_metrics['learning_rate']:.6f}")

            # Save checkpoint
            if (epoch + 1) % checkpoint_every == 0:
                checkpoint_path = Path(checkpoint_dir) / f"checkpoint_epoch_{epoch + 1}.pt"
                self.save_checkpoint(checkpoint_path, epoch, history)

            # Save best model
            if val_metrics['val_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_loss']
                best_path = Path(checkpoint_dir) / "best_model.pt"
                self.save_checkpoint(best_path, epoch, history)
                print(f"New best model saved with val_loss: {best_val_loss:.4f}")

        return history

    def save_checkpoint(self, filepath: str, epoch: int, history: Dict):
        """
        Save model checkpoint.

        Args:
            filepath: Path to save checkpoint
            epoch: Current epoch
            history: Training history
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'energy_function_state_dict': self.energy_function.state_dict(),
            'output_generator_state_dict': self.output_generator.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': history
        }

        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath: str) -> int:
        """
        Load model checkpoint.

        Args:
            filepath: Path to checkpoint file

        Returns:
            Epoch number
        """
        checkpoint = torch.load(filepath, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.energy_function.load_state_dict(checkpoint['energy_function_state_dict'])
        self.output_generator.load_state_dict(checkpoint['output_generator_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        return checkpoint['epoch']
