"""
Diagnostic Visualizer
Visualization tools for Phase 0 diagnostics
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import Dict, List, Optional
import json
from pathlib import Path

# Importaciones opcionales
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

try:
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class DiagnosticVisualizer:
    """
    Visualization tools for diagnostic analysis.
    """

    def __init__(self, output_dir: str = "diagnostics_plots"):
        """
        Initialize visualizer.

        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set style
        if HAS_SEABORN:
            plt.style.use('seaborn-v0_8')
            sns.set_palette("husl")
        else:
            plt.style.use('default')

    def plot_training_curves(self, history: Dict[str, List[float]], save: bool = True):
        """
        Plot training curves.

        Args:
            history: Training history dictionary
            save: Whether to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Curves', fontsize=16)

        # Loss curves
        axes[0, 0].plot(history['train_loss'], label='Train Loss', color='blue')
        axes[0, 0].plot(history['val_loss'], label='Val Loss', color='red')
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Accuracy
        axes[0, 1].plot(history['val_accuracy'], label='Val Accuracy', color='green')
        axes[0, 1].set_title('Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Learning rate
        axes[1, 0].plot(history['learning_rate'], label='Learning Rate', color='orange')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # Loss difference (overfitting indicator)
        if len(history['train_loss']) == len(history['val_loss']):
            loss_diff = np.array(history['val_loss']) - np.array(history['train_loss'])
            axes[1, 1].plot(loss_diff, label='Val - Train Loss', color='purple')
            axes[1, 1].set_title('Overfitting Indicator')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Loss Difference')
            axes[1, 1].legend()
            axes[1, 1].grid(True)

        plt.tight_layout()

        if save:
            plt.savefig(self.output_dir / 'training_curves.png', dpi=300, bbox_inches='tight')

        plt.show()

    def plot_embedding_tsne(self, embeddings: np.ndarray, labels: Optional[np.ndarray] = None,
                           save: bool = True):
        """
        Plot t-SNE visualization of embeddings.

        Args:
            embeddings: Embedding matrix
            labels: Optional labels for coloring
            save: Whether to save the plot
        """
        print("Computing t-SNE...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        embeddings_2d = tsne.fit_transform(embeddings)

        plt.figure(figsize=(10, 8))

        if labels is not None:
            scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                                c=labels, cmap='tab10', alpha=0.7)
            plt.colorbar(scatter)
        else:
            plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.7)

        plt.title('t-SNE Visualization of Learned Embeddings')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.grid(True, alpha=0.3)

        if save:
            plt.savefig(self.output_dir / 'embedding_tsne.png', dpi=300, bbox_inches='tight')

        plt.show()

    def plot_fitness_distribution(self, fitness_scores: Dict[str, List[float]], save: bool = True):
        """
        Plot distribution of fitness scores by type.

        Args:
            fitness_scores: Dictionary mapping fitness types to scores
            save: Whether to save the plot
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        fitness_types = list(fitness_scores.keys())

        for i, fitness_type in enumerate(fitness_types):
            if i < len(axes):
                scores = fitness_scores[fitness_type]

                axes[i].hist(scores, bins=30, alpha=0.7, edgecolor='black')
                axes[i].set_title(f'{fitness_type.title()} Fitness Distribution')
                axes[i].set_xlabel('Fitness Score')
                axes[i].set_ylabel('Frequency')
                axes[i].grid(True, alpha=0.3)

                # Add statistics
                mean_score = np.mean(scores)
                std_score = np.std(scores)
                axes[i].axvline(mean_score, color='red', linestyle='--',
                              label=f'Mean: {mean_score:.3f}')
                axes[i].legend()

        # Hide unused subplots
        for i in range(len(fitness_types), len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()

        if save:
            plt.savefig(self.output_dir / 'fitness_distribution.png', dpi=300, bbox_inches='tight')

        plt.show()

    def plot_exit_criteria(self, results: Dict, save: bool = True):
        """
        Plot Phase 0 exit criteria results.

        Args:
            results: Diagnostic results dictionary
            save: Whether to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Phase 0 Exit Criteria', fontsize=16)

        # Transfer gap
        transfer_gap = results['transfer_gap']
        axes[0, 0].bar(['Transfer Gap'], [transfer_gap],
                      color='green' if transfer_gap < 25 else 'red')
        axes[0, 0].axhline(25, color='red', linestyle='--', label='Threshold (25%)')
        axes[0, 0].set_title('Transfer Gap')
        axes[0, 0].set_ylabel('Percentage')
        axes[0, 0].legend()

        # Lump diversity
        lump_diversity = results['lump_diversity']['lump_diversity_score']
        axes[0, 1].bar(['Lump Diversity'], [lump_diversity],
                      color='green' if lump_diversity > 0.5 else 'red')
        axes[0, 1].axhline(0.5, color='red', linestyle='--', label='Threshold (0.5)')
        axes[0, 1].set_title('Lump Diversity Score')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].legend()

        # Size generalization
        size_gen = results['size_generalization']['overall_size_generalization']
        axes[1, 0].bar(['Size Generalization'], [size_gen],
                      color='green' if size_gen > 0.7 else 'red')
        axes[1, 0].axhline(0.7, color='red', linestyle='--', label='Threshold (0.7)')
        axes[1, 0].set_title('Size Generalization')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].legend()

        # Primitive emergence
        primitive_emergence = results['primitive_emergence']['overall_primitive_emergence']
        axes[1, 1].bar(['Primitive Emergence'], [primitive_emergence],
                      color='green' if primitive_emergence > 0.6 else 'red')
        axes[1, 1].axhline(0.6, color='red', linestyle='--', label='Threshold (0.6)')
        axes[1, 1].set_title('Primitive Emergence')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].legend()

        plt.tight_layout()

        if save:
            plt.savefig(self.output_dir / 'exit_criteria.png', dpi=300, bbox_inches='tight')

        plt.show()

    def plot_grid_examples(self, grids: List[np.ndarray], titles: Optional[List[str]] = None,
                          save: bool = True):
        """
        Plot example grids.

        Args:
            grids: List of grid arrays
            titles: Optional titles for each grid
            save: Whether to save the plot
        """
        n_grids = len(grids)
        cols = min(4, n_grids)
        rows = (n_grids + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()

        for i, grid in enumerate(grids):
            if i < len(axes):
                im = axes[i].imshow(grid, cmap='tab10', interpolation='nearest')
                axes[i].set_title(titles[i] if titles and i < len(titles) else f'Grid {i+1}')
                plt.colorbar(im, ax=axes[i])

        # Hide unused subplots
        for i in range(n_grids, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()

        if save:
            plt.savefig(self.output_dir / 'grid_examples.png', dpi=300, bbox_inches='tight')

        plt.show()

    def create_diagnostic_report(self, results: Dict, save: bool = True):
        """
        Create comprehensive diagnostic report.

        Args:
            results: Complete diagnostic results
            save: Whether to save the report
        """
        report = {
            'summary': {
                'phase0_passed': results['phase0_passed'],
                'transfer_gap': results['transfer_gap'],
                'lump_diversity_score': results['lump_diversity']['lump_diversity_score'],
                'size_generalization': results['size_generalization']['overall_size_generalization'],
                'primitive_emergence': results['primitive_emergence']['overall_primitive_emergence']
            },
            'detailed_results': results
        }

        if save:
            with open(self.output_dir / 'diagnostic_report.json', 'w') as f:
                json.dump(report, f, indent=2)

        return report
