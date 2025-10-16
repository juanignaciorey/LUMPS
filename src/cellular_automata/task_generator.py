"""
ARC Task Generator
Converts evolved cellular automata into ARC-format tasks
"""

import numpy as np
import json
import random
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import pickle

from .core import CellularAutomaton
from .evolution import CAEvolver


class ARCTaskGenerator:
    """
    Converts evolved cellular automata into ARC-format tasks.

    Generates 3-5 training examples per task and ensures solvability.
    """

    def __init__(
        self,
        grid_size: Tuple[int, int] = (15, 15),
        num_examples_per_task: int = 4,
        min_examples: int = 3,
        max_examples: int = 5,
        seed: Optional[int] = None
    ):
        """
        Initialize task generator.

        Args:
            grid_size: Size of grids for tasks
            num_examples_per_task: Default number of examples per task
            min_examples: Minimum examples per task
            max_examples: Maximum examples per task
            seed: Random seed for reproducibility
        """
        self.grid_size = grid_size
        self.num_examples_per_task = num_examples_per_task
        self.min_examples = min_examples
        self.max_examples = max_examples

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

    def _ca_to_grid(self, ca: CellularAutomaton, steps: int = 20) -> np.ndarray:
        """
        Run CA and extract final grid state.

        Args:
            ca: Cellular automaton
            steps: Number of steps to run

        Returns:
            Final grid state
        """
        ca.reset()
        history = ca.run(steps)
        return history[-1]

    def _create_task_examples(
        self,
        ca: CellularAutomaton,
        num_examples: int
    ) -> List[Dict[str, List[List[int]]]]:
        """
        Create training examples for a task.

        Args:
            ca: Cellular automaton to use
            num_examples: Number of examples to generate

        Returns:
            List of {input, output} examples
        """
        examples = []

        for _ in range(num_examples):
            # Reset CA to random initial state
            ca.reset()

            # Run for random number of steps (5-30)
            steps = random.randint(5, 30)
            history = ca.run(steps)

            # Use middle state as input, final state as output
            if len(history) >= 2:
                input_grid = history[len(history) // 2]
                output_grid = history[-1]
            else:
                input_grid = history[0]
                output_grid = history[0]

            # Convert to list format (ARC format)
            input_list = input_grid.tolist()
            output_list = output_grid.tolist()

            examples.append({
                "input": input_list,
                "output": output_list
            })

        return examples

    def _validate_task(self, examples: List[Dict]) -> bool:
        """
        Validate that a task is solvable and interesting.

        Args:
            examples: List of training examples

        Returns:
            True if task is valid
        """
        if len(examples) < self.min_examples:
            return False

        # Check that inputs and outputs are different
        for example in examples:
            input_grid = np.array(example["input"])
            output_grid = np.array(example["output"])

            # Task is too trivial if input == output
            if np.array_equal(input_grid, output_grid):
                return False

            # Check for reasonable complexity (not all zeros or all same value)
            if len(np.unique(input_grid)) < 2 or len(np.unique(output_grid)) < 2:
                return False

        # Check that examples show consistent pattern
        # (simplified check - in practice would be more sophisticated)
        return True

    def generate_task_from_ca(
        self,
        ca: CellularAutomaton,
        task_id: str,
        num_examples: Optional[int] = None
    ) -> Optional[Dict]:
        """
        Generate a single ARC task from a cellular automaton.

        Args:
            ca: Cellular automaton
            task_id: Unique identifier for task
            num_examples: Number of examples (if None, use default)

        Returns:
            ARC task dictionary or None if invalid
        """
        if num_examples is None:
            num_examples = self.num_examples_per_task

        # Ensure num_examples is within bounds
        num_examples = max(self.min_examples,
                          min(self.max_examples, num_examples))

        # Create examples
        examples = self._create_task_examples(ca, num_examples)

        # Validate task
        if not self._validate_task(examples):
            return None

        # Create ARC task format
        task = {
            "train": examples,
            "test": []  # Test examples would be added later
        }

        return task

    def generate_tasks_from_evolved_cas(
        self,
        ca_files: List[str],
        output_dir: str,
        tasks_per_ca: int = 5,
        max_tasks: Optional[int] = None
    ) -> List[str]:
        """
        Generate ARC tasks from evolved CA files.

        Args:
            ca_files: List of paths to evolved CA files
            output_dir: Directory to save generated tasks
            tasks_per_ca: Number of tasks to generate per CA
            max_tasks: Maximum total tasks to generate

        Returns:
            List of generated task file paths
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        generated_tasks = []
        task_counter = 0

        for ca_file in ca_files:
            if max_tasks and task_counter >= max_tasks:
                break

            try:
                # Load evolved CA
                with open(ca_file, 'rb') as f:
                    ca_data = pickle.load(f)

                # Get best CA from evolution
                best_ca = ca_data['population'][0]  # Assuming best is first

                # Generate tasks from this CA
                for i in range(tasks_per_ca):
                    if max_tasks and task_counter >= max_tasks:
                        break

                    task_id = f"task_{task_counter:06d}"
                    task = self.generate_task_from_ca(best_ca, task_id)

                    if task is not None:
                        # Save task
                        task_file = Path(output_dir) / f"{task_id}.json"
                        with open(task_file, 'w') as f:
                            json.dump(task, f, indent=2)

                        generated_tasks.append(str(task_file))
                        task_counter += 1

                        if task_counter % 1000 == 0:
                            print(f"Generated {task_counter} tasks...")

            except Exception as e:
                print(f"Error processing {ca_file}: {e}")
                continue

        print(f"Generated {len(generated_tasks)} valid tasks")
        return generated_tasks

    def generate_tasks_by_fitness_type(
        self,
        ca_dir: str,
        output_dir: str,
        fitness_types: List[str],
        tasks_per_fitness: int = 100000
    ) -> Dict[str, List[str]]:
        """
        Generate tasks organized by fitness type.

        Args:
            ca_dir: Directory containing evolved CA files
            output_dir: Directory to save tasks
            fitness_types: List of fitness types to process
            tasks_per_fitness: Number of tasks per fitness type

        Returns:
            Dictionary mapping fitness type to list of task files
        """
        all_tasks = {}

        for fitness_type in fitness_types:
            print(f"\nGenerating tasks for {fitness_type} fitness...")

            # Find CA files for this fitness type
            ca_files = list(Path(ca_dir).glob(f"ca_{fitness_type}_*.pkl"))

            if not ca_files:
                print(f"No CA files found for {fitness_type}")
                continue

            # Create output subdirectory
            fitness_output_dir = Path(output_dir) / fitness_type
            fitness_output_dir.mkdir(parents=True, exist_ok=True)

            # Generate tasks
            task_files = self.generate_tasks_from_evolved_cas(
                ca_files=[str(f) for f in ca_files],
                output_dir=str(fitness_output_dir),
                tasks_per_ca=5,
                max_tasks=tasks_per_fitness
            )

            all_tasks[fitness_type] = task_files
            print(f"Generated {len(task_files)} tasks for {fitness_type}")

        return all_tasks

    def validate_arc_format(self, task_file: str) -> bool:
        """
        Validate that a task file is in correct ARC format.

        Args:
            task_file: Path to task file

        Returns:
            True if format is valid
        """
        try:
            with open(task_file, 'r') as f:
                task = json.load(f)

            # Check required fields
            if 'train' not in task:
                return False

            train_examples = task['train']
            if not isinstance(train_examples, list) or len(train_examples) == 0:
                return False

            # Check each example
            for example in train_examples:
                if 'input' not in example or 'output' not in example:
                    return False

                input_grid = example['input']
                output_grid = example['output']

                if not isinstance(input_grid, list) or not isinstance(output_grid, list):
                    return False

                # Check grid dimensions
                if len(input_grid) == 0 or len(output_grid) == 0:
                    return False

                # Check that all rows have same length
                if len(set(len(row) for row in input_grid)) > 1:
                    return False
                if len(set(len(row) for row in output_grid)) > 1:
                    return False

            return True

        except Exception:
            return False

    def create_task_statistics(self, task_files: List[str]) -> Dict:
        """
        Create statistics about generated tasks.

        Args:
            task_files: List of task file paths

        Returns:
            Dictionary with statistics
        """
        stats = {
            'total_tasks': len(task_files),
            'valid_tasks': 0,
            'invalid_tasks': 0,
            'examples_per_task': [],
            'grid_sizes': [],
            'num_states_used': []
        }

        for task_file in task_files:
            if self.validate_arc_format(task_file):
                stats['valid_tasks'] += 1

                with open(task_file, 'r') as f:
                    task = json.load(f)

                # Count examples
                num_examples = len(task['train'])
                stats['examples_per_task'].append(num_examples)

                # Analyze first example
                if task['train']:
                    example = task['train'][0]
                    input_grid = np.array(example['input'])
                    output_grid = np.array(example['output'])

                    # Grid sizes
                    stats['grid_sizes'].append(input_grid.shape)

                    # Number of states
                    input_states = len(np.unique(input_grid))
                    output_states = len(np.unique(output_grid))
                    stats['num_states_used'].append(max(input_states, output_states))
            else:
                stats['invalid_tasks'] += 1

        # Convert lists to statistics
        if stats['examples_per_task']:
            stats['avg_examples_per_task'] = np.mean(stats['examples_per_task'])
            stats['min_examples'] = min(stats['examples_per_task'])
            stats['max_examples'] = max(stats['examples_per_task'])

        if stats['num_states_used']:
            stats['avg_states'] = np.mean(stats['num_states_used'])
            stats['max_states'] = max(stats['num_states_used'])

        return stats
