"""
Fitness Functions for CA Evolution
5 fitness types based on core knowledge priors
"""

import numpy as np
from scipy import ndimage
from typing import List


def expand_fitness(history: List[np.ndarray]) -> float:
    """
    Fitness based on pattern expansion/growth.

    Rewards CAs that show controlled growth of active cells.

    Args:
        history: List of grid states over time

    Returns:
        Fitness score in [0, 1]
    """
    if len(history) < 2:
        return 0.0

    # Measure growth rate
    initial_active = np.sum(history[0] > 0)
    final_active = np.sum(history[-1] > 0)

    if initial_active == 0:
        return 0.0

    growth_rate = final_active / initial_active

    # Reward moderate growth (2x to 5x)
    if 2.0 <= growth_rate <= 5.0:
        fitness = 1.0
    elif growth_rate > 5.0:
        # Penalize excessive growth
        fitness = max(0.0, 1.0 - (growth_rate - 5.0) / 10.0)
    else:
        # Penalize shrinkage or no growth
        fitness = growth_rate / 2.0

    # Bonus for smooth growth
    growth_trajectory = [np.sum(state > 0) for state in history]
    smoothness = 1.0 - np.std(np.diff(growth_trajectory)) / (np.mean(growth_trajectory) + 1)
    smoothness = np.clip(smoothness, 0, 1)

    return float(0.7 * fitness + 0.3 * smoothness)


def symmetry_fitness(history: List[np.ndarray]) -> float:
    """
    Fitness based on emergent symmetry.

    Rewards CAs that develop symmetric patterns.

    Args:
        history: List of grid states over time

    Returns:
        Fitness score in [0, 1]
    """
    if len(history) == 0:
        return 0.0

    final_state = history[-1]

    # Check multiple types of symmetry
    symmetries = []

    # Vertical symmetry
    vertical_sym = np.mean(final_state == np.flipud(final_state))
    symmetries.append(vertical_sym)

    # Horizontal symmetry
    horizontal_sym = np.mean(final_state == np.fliplr(final_state))
    symmetries.append(horizontal_sym)

    # Diagonal symmetry (if square grid)
    if final_state.shape[0] == final_state.shape[1]:
        diagonal_sym = np.mean(final_state == final_state.T)
        symmetries.append(diagonal_sym)

    # Rotational symmetry (180 degrees)
    rotational_sym = np.mean(final_state == np.rot90(final_state, 2))
    symmetries.append(rotational_sym)

    # Take maximum symmetry score
    max_symmetry = max(symmetries)

    # Bonus if symmetry emerged (wasn't there initially)
    initial_symmetries = []
    initial_state = history[0]
    initial_symmetries.append(np.mean(initial_state == np.flipud(initial_state)))
    initial_symmetries.append(np.mean(initial_state == np.fliplr(initial_state)))
    max_initial_symmetry = max(initial_symmetries)

    emergence_bonus = max(0, max_symmetry - max_initial_symmetry)

    return float(0.7 * max_symmetry + 0.3 * emergence_bonus)


def count_fitness(history: List[np.ndarray]) -> float:
    """
    Fitness based on numerosity (counting).

    Rewards CAs that create distinct, countable objects.

    Args:
        history: List of grid states over time

    Returns:
        Fitness score in [0, 1]
    """
    if len(history) == 0:
        return 0.0

    final_state = history[-1]

    # Convert to binary (active vs inactive)
    binary = (final_state > 0).astype(int)

    # Label connected components (objects)
    labeled, num_objects = ndimage.label(binary)

    if num_objects == 0:
        return 0.0

    # Reward 3-10 distinct objects
    if 3 <= num_objects <= 10:
        count_score = 1.0
    elif num_objects > 10:
        count_score = max(0.0, 1.0 - (num_objects - 10) / 20.0)
    else:
        count_score = num_objects / 3.0

    # Bonus for objects of similar size (easier to count)
    object_sizes = []
    for obj_id in range(1, num_objects + 1):
        size = np.sum(labeled == obj_id)
        object_sizes.append(size)

    if len(object_sizes) > 1:
        size_variance = np.std(object_sizes) / (np.mean(object_sizes) + 1)
        uniformity = np.exp(-size_variance)
    else:
        uniformity = 1.0

    return float(0.7 * count_score + 0.3 * uniformity)


def topology_fitness(history: List[np.ndarray]) -> float:
    """
    Fitness based on topological connectivity.

    Rewards CAs that create interesting spatial relationships.

    Args:
        history: List of grid states over time

    Returns:
        Fitness score in [0, 1]
    """
    if len(history) == 0:
        return 0.0

    final_state = history[-1]
    binary = (final_state > 0).astype(int)

    # Label connected components
    labeled, num_components = ndimage.label(binary)

    if num_components == 0:
        return 0.0

    # Compute connectivity metrics
    total_active = np.sum(binary)

    # Average component size
    avg_component_size = total_active / num_components if num_components > 0 else 0

    # Reward medium-sized connected components (5-20 cells)
    if 5 <= avg_component_size <= 20:
        size_score = 1.0
    elif avg_component_size > 20:
        size_score = max(0.0, 1.0 - (avg_component_size - 20) / 30.0)
    else:
        size_score = avg_component_size / 5.0

    # Compute perimeter-to-area ratio (shape complexity)
    total_perimeter = 0
    for obj_id in range(1, num_components + 1):
        obj_mask = (labeled == obj_id)
        # Erode and subtract to get perimeter
        eroded = ndimage.binary_erosion(obj_mask)
        perimeter = np.sum(obj_mask) - np.sum(eroded)
        total_perimeter += perimeter

    if total_active > 0:
        complexity = total_perimeter / total_active
        # Normalize to [0, 1]
        complexity_score = np.clip(complexity / 2.0, 0, 1)
    else:
        complexity_score = 0.0

    return float(0.6 * size_score + 0.4 * complexity_score)


def replicate_fitness(history: List[np.ndarray]) -> float:
    """
    Fitness based on self-replication patterns.

    Rewards CAs that create repeating patterns.

    Args:
        history: List of grid states over time

    Returns:
        Fitness score in [0, 1]
    """
    if len(history) < 3:
        return 0.0

    final_state = history[-1]

    # Look for repeating patterns in the grid
    h, w = final_state.shape

    # Try different pattern sizes
    best_replication_score = 0.0

    for pattern_h in [2, 3, 4, 5]:
        for pattern_w in [2, 3, 4, 5]:
            if pattern_h > h // 2 or pattern_w > w // 2:
                continue

            # Extract pattern from top-left
            pattern = final_state[:pattern_h, :pattern_w]

            # Count how many times this pattern appears
            matches = 0
            total_positions = 0

            for i in range(0, h - pattern_h + 1, pattern_h):
                for j in range(0, w - pattern_w + 1, pattern_w):
                    candidate = final_state[i:i+pattern_h, j:j+pattern_w]
                    if candidate.shape == pattern.shape:
                        similarity = np.mean(candidate == pattern)
                        if similarity > 0.8:
                            matches += 1
                        total_positions += 1

            if total_positions > 0:
                replication_score = matches / total_positions
                best_replication_score = max(best_replication_score, replication_score)

    # Bonus for temporal stability (pattern persists)
    if len(history) >= 5:
        recent_states = history[-5:]
        stability = np.mean([np.mean(state == final_state) for state in recent_states[:-1]])
    else:
        stability = 0.5

    return float(0.7 * best_replication_score + 0.3 * stability)


def pattern_match_fitness(history: List[np.ndarray]) -> float:
    """
    Fitness basado en detección y reproducción de patrones.

    Mide similitud entre patrones detectados en entrada y salida.
    """
    if len(history) < 2:
        return 0.0

    final_state = history[-1]
    initial_state = history[0]

    # Detectar patrones en estado inicial
    initial_patterns = _detect_patterns(initial_state)
    final_patterns = _detect_patterns(final_state)

    if not initial_patterns or not final_patterns:
        return 0.0

    # Calcular similitud de patrones
    pattern_similarity = _calculate_pattern_similarity(initial_patterns, final_patterns)

    # Bonus por preservación de patrones
    preservation_bonus = len(final_patterns) / max(len(initial_patterns), 1)
    preservation_bonus = min(preservation_bonus, 1.0)

    return float(0.7 * pattern_similarity + 0.3 * preservation_bonus)


def transform_consistency_fitness(history: List[np.ndarray]) -> float:
    """
    Fitness basado en consistencia de transformaciones.

    Evalúa si la misma regla se aplica de forma consistente.
    """
    if len(history) < 3:
        return 0.0

    # Analizar transiciones entre estados
    transitions = []
    for i in range(len(history) - 1):
        transition = _analyze_transition(history[i], history[i + 1])
        transitions.append(transition)

    if not transitions:
        return 0.0

    # Calcular consistencia de reglas
    consistency_score = _calculate_rule_consistency(transitions)

    # Bonus por estabilidad temporal
    stability = 1.0 - np.std([len(t) for t in transitions]) / (np.mean([len(t) for t in transitions]) + 1)
    stability = max(0, stability)

    return float(0.8 * consistency_score + 0.2 * stability)


def compression_fitness(history: List[np.ndarray]) -> float:
    """
    Fitness basado en compresión (principio de mínima complejidad).

    Premia soluciones con menor descripción (Kolmogorov-style).
    """
    if len(history) == 0:
        return 0.0

    final_state = history[-1]

    # Calcular complejidad descriptiva
    complexity = _calculate_descriptive_complexity(final_state)

    # Normalizar (menor complejidad = mayor fitness)
    max_complexity = final_state.size * np.log2(np.max(final_state) + 1)
    compression_score = 1.0 - (complexity / max_complexity) if max_complexity > 0 else 0.0

    # Bonus por evolución hacia menor complejidad
    if len(history) > 1:
        initial_complexity = _calculate_descriptive_complexity(history[0])
        evolution_bonus = max(0, (initial_complexity - complexity) / initial_complexity) if initial_complexity > 0 else 0
    else:
        evolution_bonus = 0.0

    return float(0.7 * compression_score + 0.3 * evolution_bonus)


def analogy_fitness(history: List[np.ndarray]) -> float:
    """
    Fitness basado en razonamiento analógico.

    Mide correspondencias relacionales (A→B :: C→D).
    """
    if len(history) < 2:
        return 0.0

    final_state = history[-1]
    initial_state = history[0]

    # Detectar relaciones en estado inicial
    initial_relations = _detect_relations(initial_state)
    final_relations = _detect_relations(final_state)

    if not initial_relations or not final_relations:
        return 0.0

    # Calcular correspondencias analógicas
    analogy_score = _calculate_analogical_correspondence(initial_relations, final_relations)

    # Bonus por preservación de estructura relacional
    structure_preservation = len(final_relations) / max(len(initial_relations), 1)
    structure_preservation = min(structure_preservation, 1.0)

    return float(0.6 * analogy_score + 0.4 * structure_preservation)


def objectness_fitness(history: List[np.ndarray]) -> float:
    """
    Fitness basado en segmentación conceptual de objetos.

    Evalúa si el modelo agrupa correctamente píxeles en objetos coherentes.
    """
    if len(history) == 0:
        return 0.0

    final_state = history[-1]

    # Detectar objetos
    objects = _detect_objects(final_state)

    if not objects:
        return 0.0

    # Calcular calidad de segmentación
    segmentation_quality = _calculate_segmentation_quality(objects)

    # Bonus por número óptimo de objetos (3-8)
    num_objects = len(objects)
    if 3 <= num_objects <= 8:
        object_count_score = 1.0
    else:
        object_count_score = max(0.0, 1.0 - abs(num_objects - 5.5) / 5.5)

    # Bonus por coherencia de objetos
    object_coherence = _calculate_object_coherence(objects)

    return float(0.5 * segmentation_quality + 0.3 * object_count_score + 0.2 * object_coherence)


def rule_entropy_fitness(history: List[np.ndarray]) -> float:
    """
    Fitness basado en estabilidad semántica de reglas.

    Penaliza reglas caóticas o no determinísticas.
    """
    if len(history) < 3:
        return 0.0

    # Analizar determinismo de transiciones
    deterministic_score = _calculate_determinism(history)

    # Calcular entropía de reglas
    rule_entropy = _calculate_rule_entropy(history)

    # Fitness inversamente proporcional a entropía
    entropy_fitness = max(0.0, 1.0 - rule_entropy)

    return float(0.7 * deterministic_score + 0.3 * entropy_fitness)


def invariance_fitness(history: List[np.ndarray]) -> float:
    """
    Fitness basado en abstracción e invariancia.

    Premia que la regla funcione sin depender de posiciones absolutas o colores.
    """
    if len(history) < 2:
        return 0.0

    final_state = history[-1]
    initial_state = history[0]

    # Calcular invariancia posicional
    positional_invariance = _calculate_positional_invariance(initial_state, final_state)

    # Calcular invariancia de color
    color_invariance = _calculate_color_invariance(initial_state, final_state)

    # Calcular invariancia de escala
    scale_invariance = _calculate_scale_invariance(initial_state, final_state)

    return float(0.4 * positional_invariance + 0.3 * color_invariance + 0.3 * scale_invariance)


def relational_distance_fitness(history: List[np.ndarray]) -> float:
    """
    Fitness basado en comprensión relacional.

    Cuantifica si la salida mantiene las relaciones entre objetos.
    """
    if len(history) < 2:
        return 0.0

    final_state = history[-1]
    initial_state = history[0]

    # Detectar objetos y sus relaciones
    initial_objects = _detect_objects(initial_state)
    final_objects = _detect_objects(final_state)

    if not initial_objects or not final_objects:
        return 0.0

    # Calcular preservación de distancias
    distance_preservation = _calculate_distance_preservation(initial_objects, final_objects)

    # Calcular preservación de simetrías
    symmetry_preservation = _calculate_symmetry_preservation(initial_objects, final_objects)

    return float(0.6 * distance_preservation + 0.4 * symmetry_preservation)


def causal_score_fitness(history: List[np.ndarray]) -> float:
    """
    Fitness basado en razonamiento causal.

    Evalúa si los cambios en entrada → cambios esperados en salida.
    """
    if len(history) < 2:
        return 0.0

    # Analizar causalidad en transiciones
    causal_score = _calculate_causal_consistency(history)

    # Bonus por predictibilidad
    predictability = _calculate_predictability(history)

    return float(0.7 * causal_score + 0.3 * predictability)


def divergence_penalty_fitness(history: List[np.ndarray]) -> float:
    """
    Fitness basado en coherencia perceptual.

    Penaliza soluciones que divergen visualmente sin razón funcional.
    """
    if len(history) < 2:
        return 0.0

    # Calcular divergencia visual
    visual_divergence = _calculate_visual_divergence(history)

    # Calcular coherencia funcional
    functional_coherence = _calculate_functional_coherence(history)

    # Fitness inversamente proporcional a divergencia
    divergence_fitness = max(0.0, 1.0 - visual_divergence)

    return float(0.6 * divergence_fitness + 0.4 * functional_coherence)


def compositionality_fitness(history: List[np.ndarray]) -> float:
    """
    Fitness basado en composición jerárquica de conceptos.

    Premia soluciones que se pueden expresar como composición de sub-reglas simples.
    """
    if len(history) < 3:
        return 0.0

    # Analizar composicionalidad de reglas
    compositionality_score = _calculate_rule_compositionality(history)

    # Bonus por modularidad
    modularity = _calculate_rule_modularity(history)

    # Bonus por jerarquía
    hierarchy = _calculate_rule_hierarchy(history)

    return float(0.5 * compositionality_score + 0.3 * modularity + 0.2 * hierarchy)


# Funciones auxiliares para las nuevas métricas de fitness

def _detect_patterns(grid: np.ndarray) -> List[dict]:
    """Detectar patrones en una grilla."""
    patterns = []
    h, w = grid.shape

    # Detectar patrones 2x2
    for i in range(h - 1):
        for j in range(w - 1):
            pattern = grid[i:i+2, j:j+2]
            if not np.all(pattern == 0):  # Ignorar patrones vacíos
                patterns.append({
                    'type': '2x2',
                    'pattern': pattern,
                    'position': (i, j)
                })

    return patterns


def _calculate_pattern_similarity(patterns1: List[dict], patterns2: List[dict]) -> float:
    """Calcular similitud entre conjuntos de patrones."""
    if not patterns1 or not patterns2:
        return 0.0

    similarities = []
    for p1 in patterns1:
        max_sim = 0.0
        for p2 in patterns2:
            if p1['type'] == p2['type']:
                sim = np.mean(p1['pattern'] == p2['pattern'])
                max_sim = max(max_sim, sim)
        similarities.append(max_sim)

    return np.mean(similarities) if similarities else 0.0


def _analyze_transition(state1: np.ndarray, state2: np.ndarray) -> dict:
    """Analizar transición entre dos estados."""
    changes = state2 - state1
    return {
        'num_changes': np.sum(changes != 0),
        'change_positions': np.where(changes != 0),
        'change_values': changes[changes != 0]
    }


def _calculate_rule_consistency(transitions: List[dict]) -> float:
    """Calcular consistencia de reglas en transiciones."""
    if len(transitions) < 2:
        return 1.0

    # Calcular variabilidad en número de cambios
    num_changes = [t['num_changes'] for t in transitions]
    consistency = 1.0 - (np.std(num_changes) / (np.mean(num_changes) + 1))

    return max(0.0, consistency)


def _calculate_descriptive_complexity(grid: np.ndarray) -> float:
    """Calcular complejidad descriptiva de una grilla."""
    # Usar compresión simple como proxy de complejidad
    unique_values = len(np.unique(grid))
    total_cells = grid.size

    # Complejidad basada en entropía
    if unique_values > 1:
        probs = np.bincount(grid.flatten()) / total_cells
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
    else:
        entropy = 0.0

    return entropy * total_cells


def _detect_relations(grid: np.ndarray) -> List[dict]:
    """Detectar relaciones entre elementos en una grilla."""
    relations = []
    h, w = grid.shape

    # Detectar relaciones de adyacencia
    for i in range(h):
        for j in range(w):
            if grid[i, j] != 0:
                neighbors = []
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue
                        ni, nj = i + di, j + dj
                        if 0 <= ni < h and 0 <= nj < w and grid[ni, nj] != 0:
                            neighbors.append((ni, nj))

                if neighbors:
                    relations.append({
                        'center': (i, j),
                        'neighbors': neighbors,
                        'value': grid[i, j]
                    })

    return relations


def _calculate_analogical_correspondence(relations1: List[dict], relations2: List[dict]) -> float:
    """Calcular correspondencia analógica entre conjuntos de relaciones."""
    if not relations1 or not relations2:
        return 0.0

    # Calcular similitud estructural
    structural_similarity = len(relations2) / max(len(relations1), 1)
    structural_similarity = min(structural_similarity, 1.0)

    return structural_similarity


def _detect_objects(grid: np.ndarray) -> List[dict]:
    """Detectar objetos en una grilla."""
    try:
        from scipy import ndimage
        # Etiquetar componentes conectados
        labeled, num_objects = ndimage.label(grid > 0)
    except ImportError:
        # Fallback simple sin scipy
        return _detect_objects_simple(grid)

    objects = []
    for obj_id in range(1, num_objects + 1):
        obj_mask = (labeled == obj_id)
        positions = np.where(obj_mask)

        if len(positions[0]) > 0:
            objects.append({
                'id': obj_id,
                'positions': list(zip(positions[0], positions[1])),
                'size': len(positions[0]),
                'center': (np.mean(positions[0]), np.mean(positions[1]))
            })

    return objects


def _calculate_segmentation_quality(objects: List[dict]) -> float:
    """Calcular calidad de segmentación de objetos."""
    if not objects:
        return 0.0

    # Calcular uniformidad de tamaños
    sizes = [obj['size'] for obj in objects]
    size_uniformity = 1.0 - (np.std(sizes) / (np.mean(sizes) + 1))

    # Calcular separación entre objetos
    separation_score = _calculate_object_separation(objects)

    return float(0.6 * size_uniformity + 0.4 * separation_score)


def _calculate_object_separation(objects: List[dict]) -> float:
    """Calcular separación entre objetos."""
    if len(objects) < 2:
        return 1.0

    min_distances = []
    for i, obj1 in enumerate(objects):
        min_dist = float('inf')
        for j, obj2 in enumerate(objects):
            if i != j:
                dist = np.sqrt((obj1['center'][0] - obj2['center'][0])**2 +
                             (obj1['center'][1] - obj2['center'][1])**2)
                min_dist = min(min_dist, dist)
        min_distances.append(min_dist)

    # Normalizar distancias
    avg_min_distance = np.mean(min_distances)
    separation_score = min(1.0, avg_min_distance / 5.0)  # Normalizar a distancia 5

    return separation_score


def _calculate_object_coherence(objects: List[dict]) -> float:
    """Calcular coherencia de objetos."""
    if not objects:
        return 0.0

    coherence_scores = []
    for obj in objects:
        # Calcular compacidad del objeto
        positions = np.array(obj['positions'])
        if len(positions) > 1:
            center = np.mean(positions, axis=0)
            distances = np.linalg.norm(positions - center, axis=1)
            compactness = 1.0 - (np.std(distances) / (np.mean(distances) + 1))
            coherence_scores.append(compactness)
        else:
            coherence_scores.append(1.0)

    return np.mean(coherence_scores)


def _calculate_determinism(history: List[np.ndarray]) -> float:
    """Calcular determinismo en la historia."""
    if len(history) < 2:
        return 1.0

    # Analizar si las transiciones son determinísticas
    deterministic_count = 0
    total_transitions = len(history) - 1

    for i in range(total_transitions):
        # Verificar si la transición es predecible
        if _is_transition_deterministic(history[i], history[i + 1]):
            deterministic_count += 1

    return deterministic_count / total_transitions


def _is_transition_deterministic(state1: np.ndarray, state2: np.ndarray) -> bool:
    """Verificar si una transición es determinística."""
    # Simplificado: verificar si hay un patrón claro en los cambios
    changes = state2 - state1
    num_changes = np.sum(changes != 0)

    # Si hay pocos cambios, es más probable que sea determinístico
    return num_changes < state1.size * 0.3


def _calculate_rule_entropy(history: List[np.ndarray]) -> float:
    """Calcular entropía de reglas."""
    if len(history) < 2:
        return 0.0

    # Calcular entropía de transiciones
    transition_entropies = []
    for i in range(len(history) - 1):
        state1, state2 = history[i], history[i + 1]
        changes = state2 - state1

        # Calcular entropía de cambios
        if np.sum(changes != 0) > 0:
            change_values = changes[changes != 0]
            unique_changes, counts = np.unique(change_values, return_counts=True)
            probs = counts / len(change_values)
            entropy = -np.sum(probs * np.log2(probs + 1e-10))
            transition_entropies.append(entropy)

    return np.mean(transition_entropies) if transition_entropies else 0.0


def _calculate_positional_invariance(state1: np.ndarray, state2: np.ndarray) -> float:
    """Calcular invariancia posicional."""
    # Verificar si los patrones se mantienen independientemente de posición
    # Simplificado: verificar similitud estructural
    return np.mean(state1 == state2)


def _calculate_color_invariance(state1: np.ndarray, state2: np.ndarray) -> float:
    """Calcular invariancia de color."""
    # Verificar si la estructura se mantiene con diferentes colores
    # Simplificado: verificar si los patrones no-cero son similares
    mask1 = state1 > 0
    mask2 = state2 > 0
    return np.mean(mask1 == mask2)


def _calculate_scale_invariance(state1: np.ndarray, state2: np.ndarray) -> float:
    """Calcular invariancia de escala."""
    # Verificar si los patrones se mantienen a diferentes escalas
    # Simplificado: verificar proporción de elementos activos
    density1 = np.mean(state1 > 0)
    density2 = np.mean(state2 > 0)
    return 1.0 - abs(density1 - density2)


def _calculate_distance_preservation(objects1: List[dict], objects2: List[dict]) -> float:
    """Calcular preservación de distancias entre objetos."""
    if len(objects1) != len(objects2) or len(objects1) < 2:
        return 0.0

    # Calcular distancias en ambos conjuntos
    distances1 = _calculate_object_distances(objects1)
    distances2 = _calculate_object_distances(objects2)

    if not distances1 or not distances2:
        return 0.0

    # Calcular correlación de distancias
    correlation = np.corrcoef(distances1, distances2)[0, 1]
    return max(0.0, correlation) if not np.isnan(correlation) else 0.0


def _calculate_object_distances(objects: List[dict]) -> List[float]:
    """Calcular distancias entre objetos."""
    distances = []
    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            dist = np.sqrt((objects[i]['center'][0] - objects[j]['center'][0])**2 +
                          (objects[i]['center'][1] - objects[j]['center'][1])**2)
            distances.append(dist)
    return distances


def _calculate_symmetry_preservation(objects1: List[dict], objects2: List[dict]) -> float:
    """Calcular preservación de simetrías."""
    if len(objects1) != len(objects2):
        return 0.0

    # Verificar si las simetrías se mantienen
    # Simplificado: verificar si los centros mantienen simetría
    centers1 = [obj['center'] for obj in objects1]
    centers2 = [obj['center'] for obj in objects2]

    # Calcular similitud de patrones de centros
    if centers1 and centers2:
        centers1 = np.array(centers1)
        centers2 = np.array(centers2)

        # Normalizar centros
        centers1 = centers1 - np.mean(centers1, axis=0)
        centers2 = centers2 - np.mean(centers2, axis=0)

        # Calcular similitud
        similarity = np.mean(np.linalg.norm(centers1 - centers2, axis=1))
        return max(0.0, 1.0 - similarity / 10.0)  # Normalizar

    return 0.0


def _calculate_causal_consistency(history: List[np.ndarray]) -> float:
    """Calcular consistencia causal."""
    if len(history) < 3:
        return 0.0

    # Verificar si los cambios siguen patrones causales
    causal_score = 0.0
    for i in range(len(history) - 2):
        state1, state2, state3 = history[i], history[i + 1], history[i + 2]

        # Verificar si los cambios son predecibles
        changes1 = state2 - state1
        changes2 = state3 - state2

        # Calcular similitud en patrones de cambio
        if np.sum(changes1 != 0) > 0 and np.sum(changes2 != 0) > 0:
            similarity = np.mean((changes1 != 0) == (changes2 != 0))
            causal_score += similarity

    return causal_score / max(1, len(history) - 2)


def _calculate_predictability(history: List[np.ndarray]) -> float:
    """Calcular predictibilidad de la secuencia."""
    if len(history) < 3:
        return 0.0

    # Calcular qué tan predecible es la secuencia
    prediction_accuracy = 0.0
    for i in range(1, len(history) - 1):
        # Predecir siguiente estado basado en tendencia
        predicted = history[i] + (history[i] - history[i - 1])
        actual = history[i + 1]

        # Calcular precisión de predicción
        accuracy = np.mean(predicted == actual)
        prediction_accuracy += accuracy

    return prediction_accuracy / max(1, len(history) - 2)


def _calculate_visual_divergence(history: List[np.ndarray]) -> float:
    """Calcular divergencia visual."""
    if len(history) < 2:
        return 0.0

    # Calcular divergencia entre estados consecutivos
    divergences = []
    for i in range(len(history) - 1):
        state1, state2 = history[i], history[i + 1]
        divergence = np.mean(state1 != state2)
        divergences.append(divergence)

    return np.mean(divergences)


def _calculate_functional_coherence(history: List[np.ndarray]) -> float:
    """Calcular coherencia funcional."""
    if len(history) < 2:
        return 1.0

    # Verificar si los cambios tienen propósito funcional
    coherence_scores = []
    for i in range(len(history) - 1):
        state1, state2 = history[i], history[i + 1]

        # Calcular coherencia basada en patrones
        coherence = _calculate_pattern_coherence(state1, state2)
        coherence_scores.append(coherence)

    return np.mean(coherence_scores)


def _calculate_pattern_coherence(state1: np.ndarray, state2: np.ndarray) -> float:
    """Calcular coherencia de patrones entre dos estados."""
    # Verificar si los cambios mantienen la estructura
    mask1 = state1 > 0
    mask2 = state2 > 0

    # Coherencia basada en preservación de estructura
    structure_preservation = np.mean(mask1 == mask2)

    return structure_preservation


def _calculate_rule_compositionality(history: List[np.ndarray]) -> float:
    """Calcular composicionalidad de reglas."""
    if len(history) < 3:
        return 0.0

    # Analizar si las reglas se pueden descomponer en sub-reglas
    sub_rules = _identify_sub_rules(history)

    if not sub_rules:
        return 0.0

    # Calcular modularidad de sub-reglas
    modularity = len(sub_rules) / max(1, len(history) - 1)
    modularity = min(modularity, 1.0)

    return modularity


def _identify_sub_rules(history: List[np.ndarray]) -> List[dict]:
    """Identificar sub-reglas en la historia."""
    sub_rules = []

    for i in range(len(history) - 1):
        state1, state2 = history[i], history[i + 1]
        changes = state2 - state1

        # Identificar patrones de cambio
        if np.sum(changes != 0) > 0:
            change_positions = np.where(changes != 0)
            if len(change_positions[0]) > 0:
                sub_rules.append({
                    'positions': list(zip(change_positions[0], change_positions[1])),
                    'values': changes[changes != 0]
                })

    return sub_rules


def _calculate_rule_modularity(history: List[np.ndarray]) -> float:
    """Calcular modularidad de reglas."""
    sub_rules = _identify_sub_rules(history)

    if not sub_rules:
        return 0.0

    # Calcular independencia de sub-reglas
    independence_scores = []
    for i, rule1 in enumerate(sub_rules):
        for j, rule2 in enumerate(sub_rules):
            if i != j:
                # Calcular solapamiento
                overlap = _calculate_rule_overlap(rule1, rule2)
                independence = 1.0 - overlap
                independence_scores.append(independence)

    return np.mean(independence_scores) if independence_scores else 0.0


def _calculate_rule_overlap(rule1: dict, rule2: dict) -> float:
    """Calcular solapamiento entre dos reglas."""
    positions1 = set(rule1['positions'])
    positions2 = set(rule2['positions'])

    if not positions1 or not positions2:
        return 0.0

    overlap = len(positions1.intersection(positions2))
    total = len(positions1.union(positions2))

    return overlap / total if total > 0 else 0.0


def _calculate_rule_hierarchy(history: List[np.ndarray]) -> float:
    """Calcular jerarquía de reglas."""
    if len(history) < 3:
        return 0.0

    # Analizar si hay reglas de diferentes niveles
    sub_rules = _identify_sub_rules(history)

    if len(sub_rules) < 2:
        return 0.0

    # Calcular variabilidad en tamaño de reglas
    rule_sizes = [len(rule['positions']) for rule in sub_rules]
    size_variability = np.std(rule_sizes) / (np.mean(rule_sizes) + 1)

    # Mayor variabilidad indica jerarquía
    hierarchy_score = min(1.0, size_variability)

    return hierarchy_score


def _detect_objects_simple(grid: np.ndarray) -> List[dict]:
    """Detectar objetos en una grilla sin scipy (fallback simple)."""
    objects = []
    h, w = grid.shape
    visited = np.zeros_like(grid, dtype=bool)

    def flood_fill(start_i, start_j, obj_id):
        """Flood fill para encontrar componentes conectados."""
        stack = [(start_i, start_j)]
        positions = []

        while stack:
            i, j = stack.pop()
            if (i < 0 or i >= h or j < 0 or j >= w or
                visited[i, j] or grid[i, j] == 0):
                continue

            visited[i, j] = True
            positions.append((i, j))

            # Agregar vecinos
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0:
                        continue
                    stack.append((i + di, j + dj))

        return positions

    obj_id = 1
    for i in range(h):
        for j in range(w):
            if grid[i, j] != 0 and not visited[i, j]:
                positions = flood_fill(i, j, obj_id)
                if positions:
                    objects.append({
                        'id': obj_id,
                        'positions': positions,
                        'size': len(positions),
                        'center': (np.mean([p[0] for p in positions]),
                                 np.mean([p[1] for p in positions]))
                    })
                    obj_id += 1

    return objects


# Fitness function registry
FITNESS_FUNCTIONS = {
    'expand': expand_fitness,
    'symmetry': symmetry_fitness,
    'count': count_fitness,
    'topology': topology_fitness,
    'replicate': replicate_fitness,
    # Nuevas funciones específicas para ARC
    'pattern_match': pattern_match_fitness,
    'transform_consistency': transform_consistency_fitness,
    'compression': compression_fitness,
    'analogy': analogy_fitness,
    'objectness': objectness_fitness,
    'rule_entropy': rule_entropy_fitness,
    'invariance': invariance_fitness,
    'relational_distance': relational_distance_fitness,
    'causal_score': causal_score_fitness,
    'divergence_penalty': divergence_penalty_fitness,
    'compositionality': compositionality_fitness
}


def evaluate_fitness(history: List[np.ndarray], fitness_type: str) -> float:
    """
    Evaluate fitness of a CA history.

    Args:
        history: List of grid states
        fitness_type: Type of fitness to evaluate

    Returns:
        Fitness score in [0, 1]
    """
    if fitness_type not in FITNESS_FUNCTIONS:
        raise ValueError(f"Unknown fitness type: {fitness_type}")

    return FITNESS_FUNCTIONS[fitness_type](history)
