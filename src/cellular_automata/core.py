"""
Core Cellular Automaton Implementation
Supports configurable rules and neighborhoods
"""

import numpy as np
from typing import Tuple, Optional, Callable
from enum import Enum


class Neighborhood(Enum):
    """Types of cellular automaton neighborhoods"""
    MOORE = "moore"  # 8 neighbors (including diagonals)
    VON_NEUMANN = "von_neumann"  # 4 neighbors (orthogonal only)


class CellularAutomaton:
    """
    Cellular Automaton with configurable rules and neighborhoods.

    Supports grids up to 30x30 and custom transition functions.
    """

    def __init__(
        self,
        grid_size: Tuple[int, int],
        num_states: int = 10,
        neighborhood: Neighborhood = Neighborhood.MOORE,
        rule: Optional[Callable] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize cellular automaton.

        Args:
            grid_size: (height, width) of the grid
            num_states: Number of possible cell states (0 to num_states-1)
            neighborhood: Type of neighborhood (Moore or Von Neumann)
            rule: Custom transition function (neighbor_counts, current_state) -> new_state
            seed: Random seed for reproducibility
        """
        self.height, self.width = grid_size
        self.num_states = num_states
        self.neighborhood = neighborhood
        self.rule = rule or self._default_rule

        if seed is not None:
            np.random.seed(seed)

        # Initialize grid with random states
        self.grid = np.random.randint(0, num_states, size=grid_size, dtype=np.int8)
        self.history = [self.grid.copy()]

    def _get_neighbors(self, i: int, j: int) -> np.ndarray:
        """
        Get neighbor states for cell at position (i, j).

        Args:
            i: Row index
            j: Column index

        Returns:
            Array of neighbor states
        """
        if self.neighborhood == Neighborhood.MOORE:
            # 8 neighbors (Moore neighborhood)
            offsets = [(-1, -1), (-1, 0), (-1, 1),
                      (0, -1),          (0, 1),
                      (1, -1),  (1, 0),  (1, 1)]
        else:
            # 4 neighbors (Von Neumann neighborhood)
            offsets = [(-1, 0), (0, -1), (0, 1), (1, 0)]

        neighbors = []
        for di, dj in offsets:
            ni, nj = i + di, j + dj
            # Periodic boundary conditions
            ni = ni % self.height
            nj = nj % self.width
            neighbors.append(self.grid[ni, nj])

        return np.array(neighbors, dtype=np.int8)

    def _default_rule(self, neighbors: np.ndarray, current_state: int) -> int:
        """
        Default transition rule (can be overridden).

        Args:
            neighbors: Array of neighbor states
            current_state: Current cell state

        Returns:
            New cell state
        """
        # Simple rule: majority voting with some randomness
        if len(neighbors) > 0:
            unique, counts = np.unique(neighbors, return_counts=True)
            most_common = unique[np.argmax(counts)]

            # 80% follow majority, 20% random
            if np.random.random() < 0.8:
                return most_common

        return np.random.randint(0, self.num_states)

    def step(self) -> np.ndarray:
        """
        Execute one time step of the cellular automaton.

        Returns:
            New grid state
        """
        new_grid = np.zeros_like(self.grid)

        for i in range(self.height):
            for j in range(self.width):
                neighbors = self._get_neighbors(i, j)
                current_state = self.grid[i, j]
                new_grid[i, j] = self.rule(neighbors, current_state)

        self.grid = new_grid
        self.history.append(self.grid.copy())

        return self.grid.copy()

    def run(self, steps: int) -> list:
        """
        Run the CA for multiple steps.

        Args:
            steps: Number of steps to execute

        Returns:
            List of grid states (including initial state)
        """
        for _ in range(steps):
            self.step()

        return self.history

    def get_state(self) -> np.ndarray:
        """Get current grid state."""
        return self.grid.copy()

    def reset(self, initial_state: Optional[np.ndarray] = None):
        """
        Reset the CA to initial or specified state.

        Args:
            initial_state: Optional initial grid state
        """
        if initial_state is not None:
            self.grid = initial_state.copy()
        else:
            self.grid = np.random.randint(0, self.num_states,
                                         size=(self.height, self.width),
                                         dtype=np.int8)

        self.history = [self.grid.copy()]

    def get_statistics(self) -> dict:
        """
        Compute statistics about current grid state.

        Returns:
            Dictionary with statistics
        """
        return {
            'num_active': np.sum(self.grid > 0),
            'num_states_used': len(np.unique(self.grid)),
            'entropy': self._compute_entropy(),
            'density': np.mean(self.grid > 0)
        }

    def _compute_entropy(self) -> float:
        """Compute Shannon entropy of current state."""
        unique, counts = np.unique(self.grid, return_counts=True)
        probabilities = counts / counts.sum()
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        return float(entropy)
