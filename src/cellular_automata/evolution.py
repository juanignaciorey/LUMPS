"""
Cellular Automata Evolution Engine
Genetic algorithm for evolving CAs with different fitness functions
"""

import numpy as np
import random
from typing import List, Dict, Tuple, Optional, Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from tqdm import tqdm
import pickle
import os

from .core import CellularAutomaton, Neighborhood
from .fitness import FITNESS_FUNCTIONS, evaluate_fitness


class CAEvolver:
    """
    Genetic algorithm for evolving cellular automata.

    Evolves populations of CAs using different fitness functions to discover
    computational lumps that can be converted to ARC tasks.
    """

    def __init__(
        self,
        grid_size: Tuple[int, int] = (15, 15),
        population_size: int = 100,
        num_states: int = 10,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.7,
        elite_size: int = 10,
        max_generations: int = 10000,
        fitness_type: str = "expand",
        seed: Optional[int] = None
    ):
        """
        Initialize CA evolver.

        Args:
            grid_size: Size of CA grids
            population_size: Number of CAs per generation
            num_states: Number of possible cell states
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
            elite_size: Number of best CAs to preserve
            max_generations: Maximum generations to evolve
            fitness_type: Type of fitness function to use
            seed: Random seed for reproducibility
        """
        self.grid_size = grid_size
        self.population_size = population_size
        self.num_states = num_states
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        self.max_generations = max_generations
        self.fitness_type = fitness_type

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # Initialize population
        self.population = []
        self.fitness_scores = []
        self.generation = 0
        self.best_fitness_history = []
        self.avg_fitness_history = []

        # Create initial population
        self._initialize_population()

    def _initialize_population(self):
        """Create initial random population of CAs."""
        self.population = []
        for _ in range(self.population_size):
            ca = CellularAutomaton(
                grid_size=self.grid_size,
                num_states=self.num_states,
                neighborhood=Neighborhood.MOORE,
                rule=self._create_random_rule(),
                seed=None  # Let each CA have different random seed
            )
            self.population.append(ca)

    def _create_random_rule(self) -> Callable:
        """
        Create a random transition rule.

        Returns:
            Function that takes (neighbors, current_state) -> new_state
        """
        # Create a rule based on neighbor patterns
        rule_table = {}

        def rule(neighbors: np.ndarray, current_state: int) -> int:
            # Create a key from neighbor pattern
            key = tuple(neighbors.tolist()) + (current_state,)

            if key not in rule_table:
                # Generate random new state
                rule_table[key] = np.random.randint(0, self.num_states)

            return rule_table[key]

        return rule

    def _evaluate_population(self, steps: int = 50) -> List[float]:
        """
        Evaluate fitness of entire population.

        Args:
            steps: Number of CA steps to run for evaluation

        Returns:
            List of fitness scores
        """
        fitness_scores = []

        for ca in self.population:
            # Reset CA to random initial state
            ca.reset()

            # Run CA for specified steps
            history = ca.run(steps)

            # Evaluate fitness
            fitness = evaluate_fitness(history, self.fitness_type)
            fitness_scores.append(fitness)

        return fitness_scores

    def _select_parents(self, fitness_scores: List[float]) -> List[Tuple[int, int]]:
        """
        Select parent pairs using tournament selection.

        Args:
            fitness_scores: Fitness scores for current population

        Returns:
            List of (parent1_idx, parent2_idx) pairs
        """
        parents = []
        tournament_size = 3

        for _ in range(self.population_size - self.elite_size):
            # Tournament selection for parent 1
            candidates1 = random.sample(range(len(fitness_scores)), tournament_size)
            parent1 = max(candidates1, key=lambda i: fitness_scores[i])

            # Tournament selection for parent 2 (different from parent 1)
            candidates2 = [i for i in range(len(fitness_scores)) if i != parent1]
            candidates2 = random.sample(candidates2, min(tournament_size, len(candidates2)))
            parent2 = max(candidates2, key=lambda i: fitness_scores[i])

            parents.append((parent1, parent2))

        return parents

    def _crossover(self, parent1: CellularAutomaton, parent2: CellularAutomaton) -> CellularAutomaton:
        """
        Create offspring through crossover of two parent CAs.

        Args:
            parent1: First parent CA
            parent2: Second parent CA

        Returns:
            Offspring CA
        """
        # For now, create a new random rule (simplified crossover)
        # In a more sophisticated version, we could combine rule tables
        offspring = CellularAutomaton(
            grid_size=self.grid_size,
            num_states=self.num_states,
            neighborhood=Neighborhood.MOORE,
            rule=self._create_random_rule()
        )

        return offspring

    def _mutate(self, ca: CellularAutomaton) -> CellularAutomaton:
        """
        Apply mutation to a CA.

        Args:
            ca: CA to mutate

        Returns:
            Mutated CA
        """
        # Create new CA with slightly modified rule
        mutated_ca = CellularAutomaton(
            grid_size=self.grid_size,
            num_states=self.num_states,
            neighborhood=Neighborhood.MOORE,
            rule=self._create_random_rule()  # Simplified mutation
        )

        return mutated_ca

    def _evolve_generation(self, steps: int = 50) -> Tuple[List[float], float, float]:
        """
        Evolve one generation.

        Args:
            steps: Number of CA steps for evaluation

        Returns:
            (fitness_scores, best_fitness, avg_fitness)
        """
        # Evaluate current population
        fitness_scores = self._evaluate_population(steps)

        # Track statistics
        best_fitness = max(fitness_scores)
        avg_fitness = np.mean(fitness_scores)

        # Select elite individuals
        elite_indices = sorted(range(len(fitness_scores)),
                             key=lambda i: fitness_scores[i],
                             reverse=True)[:self.elite_size]

        # Create new population
        new_population = [self.population[i] for i in elite_indices]

        # Select parents for reproduction
        parents = self._select_parents(fitness_scores)

        # Create offspring
        for parent1_idx, parent2_idx in parents:
            parent1 = self.population[parent1_idx]
            parent2 = self.population[parent2_idx]

            # Crossover
            if random.random() < self.crossover_rate:
                offspring = self._crossover(parent1, parent2)
            else:
                offspring = parent1  # Clone parent 1

            # Mutation
            if random.random() < self.mutation_rate:
                offspring = self._mutate(offspring)

            new_population.append(offspring)

        # Update population
        self.population = new_population
        self.generation += 1

        return fitness_scores, best_fitness, avg_fitness

    def evolve(self, steps: int = 50, verbose: bool = True) -> Dict:
        """
        Run full evolution process.

        Args:
            steps: Number of CA steps for evaluation
            verbose: Whether to show progress

        Returns:
            Dictionary with evolution results
        """
        if verbose:
            pbar = tqdm(total=self.max_generations, desc=f"Evolving {self.fitness_type}")

        for gen in range(self.max_generations):
            fitness_scores, best_fitness, avg_fitness = self._evolve_generation(steps)

            # Track history
            self.best_fitness_history.append(best_fitness)
            self.avg_fitness_history.append(avg_fitness)

            if verbose:
                pbar.set_postfix({
                    'best': f"{best_fitness:.3f}",
                    'avg': f"{avg_fitness:.3f}"
                })
                pbar.update(1)

            # Early stopping if fitness plateaus
            if gen > 100 and len(set(self.best_fitness_history[-50:])) == 1:
                if verbose:
                    print(f"\nEarly stopping at generation {gen} (fitness plateau)")
                break

        if verbose:
            pbar.close()

        # Get best CA
        final_fitness = self._evaluate_population(steps)
        best_idx = np.argmax(final_fitness)
        best_ca = self.population[best_idx]

        return {
            'best_ca': best_ca,
            'best_fitness': max(final_fitness),
            'final_population': self.population,
            'fitness_history': {
                'best': self.best_fitness_history,
                'average': self.avg_fitness_history
            },
            'generations': self.generation
        }

    def save_population(self, filepath: str):
        """Save evolved population to file."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'population': self.population,
                'fitness_type': self.fitness_type,
                'generation': self.generation,
                'fitness_history': {
                    'best': self.best_fitness_history,
                    'average': self.avg_fitness_history
                }
            }, f)

    def load_population(self, filepath: str):
        """Load population from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.population = data['population']
            self.fitness_type = data['fitness_type']
            self.generation = data['generation']
            self.best_fitness_history = data['fitness_history']['best']
            self.avg_fitness_history = data['fitness_history']['average']


def evolve_ca_batch(
    grid_size: Tuple[int, int],
    population_size: int,
    generations: int,
    fitness_type: str,
    num_cas: int,
    output_dir: str,
    seed: Optional[int] = None
) -> List[str]:
    """
    Evolve multiple CAs in parallel.

    Args:
        grid_size: Size of CA grids
        population_size: Population size per evolution
        generations: Number of generations
        fitness_type: Type of fitness function
        num_cas: Number of CAs to evolve
        output_dir: Directory to save evolved CAs
        seed: Random seed

    Returns:
        List of filepaths to saved CAs
    """
    os.makedirs(output_dir, exist_ok=True)

    def evolve_single_ca(ca_id: int) -> str:
        """Evolve a single CA."""
        evolver = CAEvolver(
            grid_size=grid_size,
            population_size=population_size,
            max_generations=generations,
            fitness_type=fitness_type,
            seed=seed + ca_id if seed is not None else None
        )

        results = evolver.evolve(verbose=False)

        # Save best CA
        filepath = os.path.join(output_dir, f"ca_{fitness_type}_{ca_id:06d}.pkl")
        evolver.save_population(filepath)

        return filepath

    # Evolve CAs in parallel
    filepaths = []
    with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
        futures = [executor.submit(evolve_single_ca, i) for i in range(num_cas)]

        for future in tqdm(as_completed(futures), total=num_cas,
                          desc=f"Evolving {num_cas} {fitness_type} CAs"):
            filepath = future.result()
            filepaths.append(filepath)

    return filepaths
