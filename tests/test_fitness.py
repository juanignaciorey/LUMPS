"""
Tests for Fitness Functions
"""

import pytest
import numpy as np
from src.cellular_automata.fitness import (
    expand_fitness,
    symmetry_fitness,
    count_fitness,
    topology_fitness,
    replicate_fitness,
    evaluate_fitness,
    FITNESS_FUNCTIONS
)


def test_expand_fitness():
    """Test expand fitness function."""
    # Create a simple expanding pattern
    history = [
        np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]]),
        np.array([[1, 1, 0], [1, 0, 0], [0, 0, 0]]),
        np.array([[1, 1, 1], [1, 1, 0], [1, 0, 0]])
    ]

    fitness = expand_fitness(history)

    assert 0 <= fitness <= 1
    assert isinstance(fitness, float)


def test_symmetry_fitness():
    """Test symmetry fitness function."""
    # Create a symmetric pattern
    symmetric_grid = np.array([
        [1, 0, 1],
        [0, 1, 0],
        [1, 0, 1]
    ])

    history = [symmetric_grid]
    fitness = symmetry_fitness(history)

    assert 0 <= fitness <= 1
    assert isinstance(fitness, float)


def test_count_fitness():
    """Test count fitness function."""
    # Create pattern with distinct objects
    history = [
        np.array([
            [1, 1, 0, 2, 2],
            [1, 1, 0, 2, 2],
            [0, 0, 0, 0, 0],
            [3, 3, 3, 0, 0],
            [3, 3, 3, 0, 0]
        ])
    ]

    fitness = count_fitness(history)

    assert 0 <= fitness <= 1
    assert isinstance(fitness, float)


def test_topology_fitness():
    """Test topology fitness function."""
    # Create connected pattern
    history = [
        np.array([
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1]
        ])
    ]

    fitness = topology_fitness(history)

    assert 0 <= fitness <= 1
    assert isinstance(fitness, float)


def test_replicate_fitness():
    """Test replicate fitness function."""
    # Create repeating pattern
    pattern = np.array([
        [1, 0],
        [0, 1]
    ])

    history = [
        np.tile(pattern, (2, 2))  # 2x2 repetition
    ]

    fitness = replicate_fitness(history)

    assert 0 <= fitness <= 1
    assert isinstance(fitness, float)


def test_evaluate_fitness():
    """Test fitness evaluation function."""
    history = [np.random.randint(0, 3, size=(5, 5))]

    for fitness_type in FITNESS_FUNCTIONS.keys():
        fitness = evaluate_fitness(history, fitness_type)
        assert 0 <= fitness <= 1
        assert isinstance(fitness, float)


def test_fitness_with_empty_history():
    """Test fitness functions with empty history."""
    empty_history = []

    for fitness_type in FITNESS_FUNCTIONS.keys():
        fitness = evaluate_fitness(empty_history, fitness_type)
        assert fitness == 0.0


def test_fitness_with_single_state():
    """Test fitness functions with single state."""
    single_state = [np.array([[1, 0], [0, 1]])]

    for fitness_type in FITNESS_FUNCTIONS.keys():
        fitness = evaluate_fitness(single_state, fitness_type)
        assert 0 <= fitness <= 1


if __name__ == "__main__":
    pytest.main([__file__])
