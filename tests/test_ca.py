"""
Tests for Cellular Automata Core
"""

import pytest
import numpy as np
from src.cellular_automata.core import CellularAutomaton, Neighborhood


def test_ca_initialization():
    """Test CA initialization."""
    ca = CellularAutomaton(grid_size=(10, 10), num_states=5, seed=42)

    assert ca.grid.shape == (10, 10)
    assert ca.num_states == 5
    assert ca.neighborhood == Neighborhood.MOORE
    assert len(ca.history) == 1


def test_ca_step():
    """Test single CA step."""
    ca = CellularAutomaton(grid_size=(5, 5), num_states=3, seed=42)
    initial_grid = ca.get_state().copy()

    new_grid = ca.step()

    assert new_grid.shape == (5, 5)
    assert len(ca.history) == 2
    assert not np.array_equal(initial_grid, new_grid)


def test_ca_run():
    """Test multiple CA steps."""
    ca = CellularAutomaton(grid_size=(5, 5), num_states=3, seed=42)

    history = ca.run(10)

    assert len(history) == 11  # Initial + 10 steps
    assert all(grid.shape == (5, 5) for grid in history)


def test_ca_reset():
    """Test CA reset."""
    ca = CellularAutomaton(grid_size=(5, 5), num_states=3, seed=42)
    ca.run(5)

    initial_len = len(ca.history)
    ca.reset()

    assert len(ca.history) == 1
    assert ca.grid.shape == (5, 5)


def test_ca_statistics():
    """Test CA statistics computation."""
    ca = CellularAutomaton(grid_size=(5, 5), num_states=3, seed=42)

    stats = ca.get_statistics()

    assert 'num_active' in stats
    assert 'num_states_used' in stats
    assert 'entropy' in stats
    assert 'density' in stats
    assert 0 <= stats['density'] <= 1


def test_ca_neighborhoods():
    """Test different neighborhood types."""
    # Moore neighborhood
    ca_moore = CellularAutomaton(grid_size=(5, 5), neighborhood=Neighborhood.MOORE)
    neighbors_moore = ca_moore._get_neighbors(2, 2)
    assert len(neighbors_moore) == 8

    # Von Neumann neighborhood
    ca_vn = CellularAutomaton(grid_size=(5, 5), neighborhood=Neighborhood.VON_NEUMANN)
    neighbors_vn = ca_vn._get_neighbors(2, 2)
    assert len(neighbors_vn) == 4


if __name__ == "__main__":
    pytest.main([__file__])
