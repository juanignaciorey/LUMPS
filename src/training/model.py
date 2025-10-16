"""
LUMPSTransformer Model Architecture
Energy-based model with specialized heads for different reasoning types
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import math


class GridEncoder(nn.Module):
    """
    Encodes 2D grids into embeddings.

    Uses patch-based encoding similar to Vision Transformer.
    """

    def __init__(
        self,
        max_grid_size: int = 30,
        num_states: int = 10,
        patch_size: int = 3,
        d_model: int = 512,
        dropout: float = 0.1
    ):
        """
        Initialize grid encoder.

        Args:
            max_grid_size: Maximum grid size to handle
            num_states: Number of possible cell states
            patch_size: Size of patches to extract
            d_model: Model dimension
            dropout: Dropout rate
        """
        super().__init__()

        self.max_grid_size = max_grid_size
        self.num_states = num_states
        self.patch_size = patch_size
        self.d_model = d_model

        # State embedding
        self.state_embedding = nn.Embedding(num_states, d_model)

        # Position embedding for patches
        max_patches = (max_grid_size // patch_size) ** 2
        self.pos_embedding = nn.Parameter(torch.randn(1, max_patches, d_model))

        # Patch projection
        self.patch_proj = nn.Linear(patch_size * patch_size * d_model, d_model)

        # Layer normalization
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def extract_patches(self, grid: torch.Tensor) -> torch.Tensor:
        """
        Extract patches from grid.

        Args:
            grid: Grid tensor of shape (batch_size, height, width)

        Returns:
            Patches tensor of shape (batch_size, num_patches, patch_size^2)
        """
        batch_size, height, width = grid.shape

        # Pad grid to be divisible by patch_size
        pad_h = (self.patch_size - height % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - width % self.patch_size) % self.patch_size

        if pad_h > 0 or pad_w > 0:
            grid = F.pad(grid, (0, pad_w, 0, pad_h), value=0)

        # Extract patches
        patches = grid.unfold(1, self.patch_size, self.patch_size).unfold(2, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(batch_size, -1, self.patch_size * self.patch_size)

        return patches

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        """
        Encode grid to embeddings.

        Args:
            grid: Grid tensor of shape (batch_size, height, width)

        Returns:
            Embeddings of shape (batch_size, num_patches, d_model)
        """
        batch_size = grid.shape[0]

        # Extract patches
        patches = self.extract_patches(grid)  # (batch_size, num_patches, patch_size^2)

        # Embed states in patches
        patches_flat = patches.view(-1, self.patch_size * self.patch_size)
        embedded_patches = self.state_embedding(patches_flat)  # (batch_size * num_patches, patch_size^2, d_model)
        embedded_patches = embedded_patches.view(batch_size, -1, self.patch_size * self.patch_size, self.d_model)

        # Flatten patch embeddings
        embedded_patches = embedded_patches.view(batch_size, -1, self.patch_size * self.patch_size * self.d_model)

        # Project to model dimension
        patch_embeddings = self.patch_proj(embedded_patches)  # (batch_size, num_patches, d_model)

        # Add position embeddings
        num_patches = patch_embeddings.shape[1]
        pos_emb = self.pos_embedding[:, :num_patches, :]
        patch_embeddings = patch_embeddings + pos_emb

        # Normalize and apply dropout
        patch_embeddings = self.norm(patch_embeddings)
        patch_embeddings = self.dropout(patch_embeddings)

        return patch_embeddings


class SpecializedHead(nn.Module):
    """Base class for specialized reasoning heads."""

    def __init__(self, d_model: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.output_dim = output_dim

        self.attention = nn.MultiheadAttention(d_model, num_heads=8, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )

        self.output_proj = nn.Linear(d_model, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through specialized head."""
        # Self-attention
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)

        # Feed-forward
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        # Output projection
        output = self.output_proj(x)

        return output


class ObjectHead(SpecializedHead):
    """Head for object detection and manipulation."""

    def __init__(self, d_model: int, max_objects: int = 20, dropout: float = 0.1):
        super().__init__(d_model, max_objects, dropout)
        self.max_objects = max_objects


class SymmetryHead(SpecializedHead):
    """Head for symmetry pattern detection."""

    def __init__(self, d_model: int, num_symmetry_types: int = 4, dropout: float = 0.1):
        super().__init__(d_model, num_symmetry_types, dropout)
        self.num_symmetry_types = num_symmetry_types


class CountHead(SpecializedHead):
    """Head for numerosity and counting."""

    def __init__(self, d_model: int, max_count: int = 50, dropout: float = 0.1):
        super().__init__(d_model, max_count, dropout)
        self.max_count = max_count


class TopologyHead(SpecializedHead):
    """Head for spatial relationships and topology."""

    def __init__(self, d_model: int, num_relations: int = 10, dropout: float = 0.1):
        super().__init__(d_model, num_relations, dropout)
        self.num_relations = num_relations


class GoalHead(SpecializedHead):
    """Head for goal prediction and output generation."""

    def __init__(self, d_model: int, num_states: int = 10, dropout: float = 0.1):
        super().__init__(d_model, num_states, dropout)
        self.num_states = num_states


class LUMPSTransformer(nn.Module):
    """
    Main transformer model with specialized heads.

    Processes grid inputs and produces multiple specialized outputs.
    """

    def __init__(
        self,
        d_model: int = 512,
        n_layers: int = 12,
        n_heads: int = 8,
        max_grid_size: int = 30,
        num_states: int = 10,
        dropout: float = 0.1
    ):
        """
        Initialize LUMPSTransformer.

        Args:
            d_model: Model dimension
            n_layers: Number of transformer layers
            n_heads: Number of attention heads
            max_grid_size: Maximum grid size
            num_states: Number of possible cell states
            dropout: Dropout rate
        """
        super().__init__()

        self.d_model = d_model
        self.n_layers = n_layers
        self.max_grid_size = max_grid_size
        self.num_states = num_states

        # Grid encoder
        self.grid_encoder = GridEncoder(
            max_grid_size=max_grid_size,
            num_states=num_states,
            d_model=d_model,
            dropout=dropout
        )

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Specialized heads
        self.object_head = ObjectHead(d_model, max_objects=20, dropout=dropout)
        self.symmetry_head = SymmetryHead(d_model, num_symmetry_types=4, dropout=dropout)
        self.count_head = CountHead(d_model, max_count=50, dropout=dropout)
        self.topology_head = TopologyHead(d_model, num_relations=10, dropout=dropout)
        self.goal_head = GoalHead(d_model, num_states=num_states, dropout=dropout)

        # Output projection for grid generation
        self.output_proj = nn.Linear(d_model, num_states)

    def forward(self, input_grid: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.

        Args:
            input_grid: Input grid tensor of shape (batch_size, height, width)

        Returns:
            Dictionary with outputs from all specialized heads
        """
        # Encode input grid
        embeddings = self.grid_encoder(input_grid)  # (batch_size, num_patches, d_model)

        # Process through transformer
        transformer_out = self.transformer(embeddings)  # (batch_size, num_patches, d_model)

        # Get outputs from specialized heads
        outputs = {
            'objects': self.object_head(transformer_out),
            'symmetry': self.symmetry_head(transformer_out),
            'count': self.count_head(transformer_out),
            'topology': self.topology_head(transformer_out),
            'goals': self.goal_head(transformer_out),
            'embeddings': transformer_out
        }

        return outputs


class EnergyFunction(nn.Module):
    """
    Energy function for evaluating solution quality.

    Assigns low energy to good solutions, high energy to bad ones.
    """

    def __init__(self, d_model: int = 512, hidden_dim: int = 256):
        """
        Initialize energy function.

        Args:
            d_model: Input dimension
            hidden_dim: Hidden layer dimension
        """
        super().__init__()

        self.energy_net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute energy for given embeddings.

        Args:
            embeddings: Embeddings tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Energy scores of shape (batch_size, seq_len)
        """
        # Pool embeddings (mean pooling)
        pooled = torch.mean(embeddings, dim=1)  # (batch_size, d_model)

        # Compute energy
        energy = self.energy_net(pooled)  # (batch_size, 1)

        return energy.squeeze(-1)  # (batch_size,)


class OutputGenerator(nn.Module):
    """
    Generates candidate output grids.

    Uses the model's goal predictions to generate possible solutions.
    """

    def __init__(
        self,
        d_model: int = 512,
        max_grid_size: int = 30,
        num_states: int = 10,
        patch_size: int = 3
    ):
        """
        Initialize output generator.

        Args:
            d_model: Model dimension
            max_grid_size: Maximum grid size
            num_states: Number of possible cell states
            patch_size: Patch size for reconstruction
        """
        super().__init__()

        self.d_model = d_model
        self.max_grid_size = max_grid_size
        self.num_states = num_states
        self.patch_size = patch_size

        # Generator network
        self.generator = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, patch_size * patch_size * num_states)
        )

    def forward(
        self,
        embeddings: torch.Tensor,
        target_size: Tuple[int, int]
    ) -> torch.Tensor:
        """
        Generate output grid from embeddings.

        Args:
            embeddings: Embeddings tensor of shape (batch_size, num_patches, d_model)
            target_size: Target grid size (height, width)

        Returns:
            Generated grid tensor of shape (batch_size, height, width)
        """
        batch_size, num_patches, d_model = embeddings.shape
        height, width = target_size

        # Generate patches
        patch_logits = self.generator(embeddings)  # (batch_size, num_patches, patch_size^2 * num_states)
        patch_logits = patch_logits.view(batch_size, num_patches, self.patch_size, self.patch_size, self.num_states)

        # Convert to probabilities
        patch_probs = F.softmax(patch_logits, dim=-1)

        # Sample states
        patch_states = torch.multinomial(patch_probs.view(-1, self.num_states), 1)
        patch_states = patch_states.view(batch_size, num_patches, self.patch_size, self.patch_size)

        # Reconstruct grid from patches
        grid = self._patches_to_grid(patch_states, target_size)

        return grid

    def _patches_to_grid(self, patches: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
        """
        Reconstruct grid from patches.

        Args:
            patches: Patch tensor of shape (batch_size, num_patches, patch_size, patch_size)
            target_size: Target grid size

        Returns:
            Reconstructed grid tensor
        """
        batch_size, num_patches, patch_h, patch_w = patches.shape
        height, width = target_size

        # Calculate grid dimensions
        grid_h = (height + self.patch_size - 1) // self.patch_size
        grid_w = (width + self.patch_size - 1) // self.patch_size

        # Reshape patches to grid
        patches_grid = patches.view(batch_size, grid_h, grid_w, patch_h, patch_w)

        # Reconstruct full grid
        grid = torch.zeros(batch_size, height, width, dtype=patches.dtype, device=patches.device)

        for i in range(grid_h):
            for j in range(grid_w):
                start_h = i * self.patch_size
                start_w = j * self.patch_size
                end_h = min(start_h + self.patch_size, height)
                end_w = min(start_w + self.patch_size, width)

                patch_h_actual = end_h - start_h
                patch_w_actual = end_w - start_w

                grid[:, start_h:end_h, start_w:end_w] = patches_grid[:, i, j, :patch_h_actual, :patch_w_actual]

        return grid
