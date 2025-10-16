"""
Data Loading Module
PyTorch Dataset for ARC tasks with batching and augmentation
"""

import torch
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import random
from torchvision import transforms


class ARCDataset(Dataset):
    """
    PyTorch Dataset for ARC tasks.

    Loads tasks from JSON files and provides batching with padding.
    """

    def __init__(
        self,
        task_files: List[str],
        max_grid_size: int = 30,
        max_examples: int = 5,
        augment: bool = True,
        seed: Optional[int] = None
    ):
        """
        Initialize ARC dataset.

        Args:
            task_files: List of paths to task JSON files
            max_grid_size: Maximum grid size to handle
            max_examples: Maximum number of examples per task
            augment: Whether to apply data augmentation
            seed: Random seed for reproducibility
        """
        self.task_files = task_files
        self.max_grid_size = max_grid_size
        self.max_examples = max_examples
        self.augment = augment

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Load and validate tasks
        self.tasks = []
        self._load_tasks()

        # Augmentation transforms
        if augment:
            self.transform = transforms.Compose([
                RandomRotation(),
                RandomFlip(),
                RandomNoise(noise_prob=0.1)
            ])
        else:
            self.transform = None

    def _load_tasks(self):
        """Load and validate all tasks."""
        print(f"Loading {len(self.task_files)} tasks...")

        for task_file in self.task_files:
            try:
                with open(task_file, 'r') as f:
                    task = json.load(f)

                if self._validate_task(task):
                    self.tasks.append(task)
                else:
                    print(f"Skipping invalid task: {task_file}")

            except Exception as e:
                print(f"Error loading {task_file}: {e}")
                continue

        print(f"Loaded {len(self.tasks)} valid tasks")

    def _validate_task(self, task: Dict) -> bool:
        """
        Validate a task.

        Args:
            task: Task dictionary

        Returns:
            True if task is valid
        """
        if 'train' not in task:
            return False

        train_examples = task['train']
        if not isinstance(train_examples, list) or len(train_examples) == 0:
            return False

        # Check each example
        for example in train_examples:
            if 'input' not in example or 'output' not in example:
                return False

            input_grid = np.array(example['input'])
            output_grid = np.array(example['output'])

            # Check dimensions
            if input_grid.shape[0] > self.max_grid_size or input_grid.shape[1] > self.max_grid_size:
                return False
            if output_grid.shape[0] > self.max_grid_size or output_grid.shape[1] > self.max_grid_size:
                return False

            # Check that grids are not empty
            if input_grid.size == 0 or output_grid.size == 0:
                return False

        return True

    def _pad_grid(self, grid: np.ndarray) -> np.ndarray:
        """
        Pad grid to max_grid_size.

        Args:
            grid: Input grid

        Returns:
            Padded grid
        """
        height, width = grid.shape

        # Calculate padding
        pad_h = max(0, self.max_grid_size - height)
        pad_w = max(0, self.max_grid_size - width)

        # Pad with zeros
        padded = np.pad(grid, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)

        return padded

    def _apply_augmentation(self, input_grid: np.ndarray, output_grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply data augmentation to grids.

        Args:
            input_grid: Input grid
            output_grid: Output grid

        Returns:
            Augmented (input_grid, output_grid)
        """
        if self.transform is None:
            return input_grid, output_grid

        # Stack grids for joint augmentation
        stacked = np.stack([input_grid, output_grid], axis=0)

        # Apply transform
        augmented = self.transform(stacked)

        return augmented[0], augmented[1]

    def __len__(self) -> int:
        """Return number of tasks."""
        return len(self.tasks)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single task.

        Args:
            idx: Task index

        Returns:
            Dictionary with task data
        """
        task = self.tasks[idx]
        train_examples = task['train']

        # Limit number of examples
        if len(train_examples) > self.max_examples:
            train_examples = random.sample(train_examples, self.max_examples)

        # Process examples
        input_grids = []
        output_grids = []

        for example in train_examples:
            input_grid = np.array(example['input'], dtype=np.int64)
            output_grid = np.array(example['output'], dtype=np.int64)

            # Apply augmentation
            if self.augment:
                input_grid, output_grid = self._apply_augmentation(input_grid, output_grid)

            # Pad grids
            input_grid = self._pad_grid(input_grid)
            output_grid = self._pad_grid(output_grid)

            input_grids.append(input_grid)
            output_grids.append(output_grid)

        # Pad to max_examples
        while len(input_grids) < self.max_examples:
            # Duplicate last example
            input_grids.append(input_grids[-1].copy())
            output_grids.append(output_grids[-1].copy())

        # Convert to tensors
        input_tensor = torch.tensor(np.stack(input_grids), dtype=torch.long)
        output_tensor = torch.tensor(np.stack(output_grids), dtype=torch.long)

        return {
            'input': input_tensor,
            'output': output_tensor,
            'num_examples': len(train_examples)
        }


class RandomRotation:
    """Random rotation augmentation."""

    def __init__(self, prob: float = 0.5):
        self.prob = prob

    def __call__(self, grids: np.ndarray) -> np.ndarray:
        """Apply random rotation."""
        if random.random() < self.prob:
            k = random.randint(1, 3)  # 90, 180, or 270 degrees
            grids = np.rot90(grids, k, axes=(1, 2))

        return grids


class RandomFlip:
    """Random flip augmentation."""

    def __init__(self, prob: float = 0.5):
        self.prob = prob

    def __call__(self, grids: np.ndarray) -> np.ndarray:
        """Apply random flip."""
        if random.random() < self.prob:
            # Random horizontal or vertical flip
            if random.random() < 0.5:
                grids = np.flip(grids, axis=2)  # Horizontal flip
            else:
                grids = np.flip(grids, axis=1)  # Vertical flip

        return grids


class RandomNoise:
    """Random noise augmentation."""

    def __init__(self, noise_prob: float = 0.1):
        self.noise_prob = noise_prob

    def __call__(self, grids: np.ndarray) -> np.ndarray:
        """Apply random noise."""
        if random.random() < self.noise_prob:
            # Add small amount of random noise
            noise = np.random.randint(0, 2, size=grids.shape)
            grids = (grids + noise) % grids.max()

        return grids


def create_dataloader(
    task_files: List[str],
    batch_size: int = 32,
    max_grid_size: int = 30,
    max_examples: int = 5,
    augment: bool = True,
    shuffle: bool = True,
    num_workers: int = 4,
    seed: Optional[int] = None
) -> DataLoader:
    """
    Create a DataLoader for ARC tasks.

    Args:
        task_files: List of task file paths
        batch_size: Batch size
        max_grid_size: Maximum grid size
        max_examples: Maximum examples per task
        augment: Whether to apply augmentation
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        seed: Random seed

    Returns:
        PyTorch DataLoader
    """
    dataset = ARCDataset(
        task_files=task_files,
        max_grid_size=max_grid_size,
        max_examples=max_examples,
        augment=augment,
        seed=seed
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )

    return dataloader


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for batching.

    Args:
        batch: List of task dictionaries

    Returns:
        Batched data dictionary
    """
    # Stack all tensors
    inputs = torch.stack([item['input'] for item in batch])
    outputs = torch.stack([item['output'] for item in batch])
    num_examples = torch.tensor([item['num_examples'] for item in batch])

    return {
        'input': inputs,
        'output': outputs,
        'num_examples': num_examples
    }
