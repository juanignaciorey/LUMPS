"""
Diagnostic Metrics for Meta-Learning
Transfer gap, lump diversity, and other Phase 0 exit criteria
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import json
from pathlib import Path

from ..training.model import LUMPSTransformer
from ..training.data_loader import ARCDataset


def compute_transfer_gap(
    model: LUMPSTransformer,
    in_dist_loader,
    ood_loader,
    device: str = "cuda"
) -> float:
    """
    Compute transfer gap between in-distribution and out-of-distribution tasks.

    Args:
        model: Trained model
        in_dist_loader: In-distribution data loader
        ood_loader: Out-of-distribution data loader
        device: Device to use

    Returns:
        Transfer gap (percentage difference)
    """
    model.eval()

    def evaluate_loader(loader):
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in loader:
                input_grids = batch['input'].to(device)
                output_grids = batch['output'].to(device)

                # Get model outputs
                outputs = model(input_grids.view(-1, input_grids.shape[-2], input_grids.shape[-1]))

                # Compute simple reconstruction loss
                predicted = outputs['goals']
                target = output_grids.view(-1, output_grids.shape[-2], output_grids.shape[-1])

                # Convert to probabilities and compute loss
                loss = torch.nn.functional.cross_entropy(
                    predicted.view(-1, predicted.shape[-1]),
                    target.view(-1)
                )

                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0.0

    # Evaluate on both distributions
    in_dist_loss = evaluate_loader(in_dist_loader)
    ood_loss = evaluate_loader(ood_loader)

    # Compute transfer gap
    transfer_gap = abs(ood_loss - in_dist_loss) / in_dist_loss * 100

    return transfer_gap


def measure_lump_diversity(
    model: LUMPSTransformer,
    dataloader,
    device: str = "cuda",
    num_samples: int = 1000
) -> Dict[str, float]:
    """
    Measure diversity of learned computational lumps.

    Args:
        model: Trained model
        dataloader: Data loader for analysis
        device: Device to use
        num_samples: Number of samples to analyze

    Returns:
        Dictionary with diversity metrics
    """
    model.eval()

    embeddings = []
    labels = []

    with torch.no_grad():
        sample_count = 0
        for batch in dataloader:
            if sample_count >= num_samples:
                break

            input_grids = batch['input'].to(device)

            # Get embeddings
            outputs = model(input_grids.view(-1, input_grids.shape[-2], input_grids.shape[-1]))
            batch_embeddings = outputs['embeddings']

            # Pool embeddings (mean pooling)
            pooled = torch.mean(batch_embeddings, dim=1)
            embeddings.append(pooled.cpu().numpy())

            # Use batch index as label for now
            batch_labels = [sample_count + i for i in range(len(pooled))]
            labels.extend(batch_labels)

            sample_count += len(pooled)

    # Concatenate all embeddings
    embeddings = np.concatenate(embeddings, axis=0)

    # Compute diversity metrics
    metrics = {}

    # 1. Embedding variance
    metrics['embedding_variance'] = float(np.var(embeddings))

    # 2. Number of distinct clusters (using K-means)
    if len(embeddings) > 10:
        # Try different numbers of clusters
        best_silhouette = -1
        best_k = 2

        for k in range(2, min(20, len(embeddings) // 10)):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings)
            silhouette = silhouette_score(embeddings, cluster_labels)

            if silhouette > best_silhouette:
                best_silhouette = silhouette
                best_k = k

        metrics['num_clusters'] = best_k
        metrics['silhouette_score'] = best_silhouette
    else:
        metrics['num_clusters'] = 1
        metrics['silhouette_score'] = 0.0

    # 3. Effective dimensionality (using PCA)
    from sklearn.decomposition import PCA
    pca = PCA()
    pca.fit(embeddings)

    # Count components that explain 95% of variance
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    effective_dims = np.argmax(cumsum >= 0.95) + 1
    metrics['effective_dimensionality'] = int(effective_dims)

    # 4. Lump diversity score (combination of metrics)
    diversity_score = (
        min(metrics['num_clusters'] / 20, 1.0) * 0.4 +
        min(metrics['effective_dimensionality'] / 50, 1.0) * 0.3 +
        min(metrics['silhouette_score'], 1.0) * 0.3
    )
    metrics['lump_diversity_score'] = float(diversity_score)

    return metrics


def test_size_generalization(
    model: LUMPSTransformer,
    dataloader,
    device: str = "cuda",
    size_ranges: List[Tuple[int, int]] = [(5, 10), (10, 15), (15, 20), (20, 25)]
) -> Dict[str, float]:
    """
    Test generalization to different grid sizes.

    Args:
        model: Trained model
        dataloader: Data loader
        device: Device to use
        size_ranges: List of (min_size, max_size) tuples

    Returns:
        Dictionary with size generalization metrics
    """
    model.eval()

    size_accuracies = {}

    for min_size, max_size in size_ranges:
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch in dataloader:
                input_grids = batch['input'].to(device)
                output_grids = batch['output'].to(device)

                # Filter by size
                batch_size, num_examples, height, width = input_grids.shape

                for i in range(batch_size):
                    for j in range(num_examples):
                        grid_h, grid_w = input_grids[i, j].shape
                        grid_size = max(grid_h, grid_w)

                        if min_size <= grid_size <= max_size:
                            # Get model prediction
                            single_input = input_grids[i, j].unsqueeze(0)
                            outputs = model(single_input)

                            # Simple accuracy check (simplified)
                            predicted = outputs['goals']
                            target = output_grids[i, j]

                            # Convert to predictions
                            pred_probs = torch.softmax(predicted, dim=-1)
                            pred_grid = torch.argmax(pred_probs, dim=-1)

                            # Compute accuracy (simplified)
                            accuracy = torch.mean((pred_grid == target).float())
                            total_correct += accuracy.item()
                            total_samples += 1

        if total_samples > 0:
            size_accuracies[f'size_{min_size}_{max_size}'] = total_correct / total_samples
        else:
            size_accuracies[f'size_{min_size}_{max_size}'] = 0.0

    # Overall size generalization score
    overall_score = np.mean(list(size_accuracies.values()))
    size_accuracies['overall_size_generalization'] = float(overall_score)

    return size_accuracies


def check_primitive_emergence(
    model: LUMPSTransformer,
    dataloader,
    device: str = "cuda"
) -> Dict[str, float]:
    """
    Check for emergence of core knowledge primitives.

    Args:
        model: Trained model
        dataloader: Data loader
        device: Device to use

    Returns:
        Dictionary with primitive emergence metrics
    """
    model.eval()

    primitive_scores = {
        'objectness': 0.0,
        'symmetry': 0.0,
        'numerosity': 0.0,
        'topology': 0.0,
        'goals': 0.0
    }

    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            input_grids = batch['input'].to(device)

            # Get model outputs
            outputs = model(input_grids.view(-1, input_grids.shape[-2], input_grids.shape[-1]))

            # Analyze each specialized head
            objects = outputs['objects']
            symmetry = outputs['symmetry']
            count = outputs['count']
            topology = outputs['topology']
            goals = outputs['goals']

            # Compute primitive scores (simplified)
            # Objectness: variance in object predictions
            primitive_scores['objectness'] += float(torch.var(objects))

            # Symmetry: variance in symmetry predictions
            primitive_scores['symmetry'] += float(torch.var(symmetry))

            # Numerosity: variance in count predictions
            primitive_scores['numerosity'] += float(torch.var(count))

            # Topology: variance in topology predictions
            primitive_scores['topology'] += float(torch.var(topology))

            # Goals: variance in goal predictions
            primitive_scores['goals'] += float(torch.var(goals))

            total_samples += 1

    # Normalize scores
    if total_samples > 0:
        for key in primitive_scores:
            primitive_scores[key] /= total_samples

    # Overall primitive emergence score
    overall_score = np.mean(list(primitive_scores.values()))
    primitive_scores['overall_primitive_emergence'] = float(overall_score)

    return primitive_scores


def evaluate_phase0_exit_criteria(
    model: LUMPSTransformer,
    train_loader,
    val_loader,
    ood_loader,
    device: str = "cuda"
) -> Dict[str, any]:
    """
    Evaluate all Phase 0 exit criteria.

    Args:
        model: Trained model
        train_loader: Training data loader
        val_loader: Validation data loader
        ood_loader: Out-of-distribution data loader
        device: Device to use

    Returns:
        Dictionary with all exit criteria results
    """
    print("Evaluating Phase 0 exit criteria...")

    results = {}

    # 1. Transfer gap
    print("Computing transfer gap...")
    transfer_gap = compute_transfer_gap(model, val_loader, ood_loader, device)
    results['transfer_gap'] = transfer_gap
    results['transfer_gap_passed'] = transfer_gap < 25.0

    # 2. Lump diversity
    print("Measuring lump diversity...")
    lump_diversity = measure_lump_diversity(model, train_loader, device)
    results['lump_diversity'] = lump_diversity
    results['lump_diversity_passed'] = lump_diversity['lump_diversity_score'] > 0.5  # Adjusted threshold

    # 3. Size generalization
    print("Testing size generalization...")
    size_gen = test_size_generalization(model, val_loader, device)
    results['size_generalization'] = size_gen
    results['size_generalization_passed'] = size_gen['overall_size_generalization'] > 0.7

    # 4. Primitive emergence
    print("Checking primitive emergence...")
    primitive_emergence = check_primitive_emergence(model, train_loader, device)
    results['primitive_emergence'] = primitive_emergence
    results['primitive_emergence_passed'] = primitive_emergence['overall_primitive_emergence'] > 0.6

    # Overall pass/fail
    all_passed = all([
        results['transfer_gap_passed'],
        results['lump_diversity_passed'],
        results['size_generalization_passed'],
        results['primitive_emergence_passed']
    ])
    results['phase0_passed'] = all_passed

    # Print results
    print("\n" + "="*50)
    print("PHASE 0 EXIT CRITERIA RESULTS")
    print("="*50)
    print(f"Transfer Gap: {transfer_gap:.2f}% {'✓' if results['transfer_gap_passed'] else '✗'}")
    print(f"Lump Diversity: {lump_diversity['lump_diversity_score']:.3f} {'✓' if results['lump_diversity_passed'] else '✗'}")
    print(f"Size Generalization: {size_gen['overall_size_generalization']:.3f} {'✓' if results['size_generalization_passed'] else '✗'}")
    print(f"Primitive Emergence: {primitive_emergence['overall_primitive_emergence']:.3f} {'✓' if results['primitive_emergence_passed'] else '✗'}")
    print(f"\nOVERALL: {'PASSED' if all_passed else 'FAILED'}")
    print("="*50)

    return results
