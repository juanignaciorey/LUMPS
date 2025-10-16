"""
Script de alto rendimiento para generar dataset de Fase 0 con sistema de logging persistente y checkpoints
"""

import sys
import argparse
import pickle
import json
import time
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def setup_logging(log_dir: str = "logs") -> None:
    """Configurar sistema de logging persistente."""
    from src.utils.logger import setup_logger

    # Crear directorio de logs si no existe
    Path(log_dir).mkdir(exist_ok=True)

    # Configurar logger con timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"generation_full_{timestamp}.log"

    logger = setup_logger(
        name="dataset_generation",
        log_file=str(log_file),
        level="INFO"
    )

    return logger

def load_config(config_file: str = None) -> Dict[str, Any]:
    """Cargar configuración desde archivo o usar defaults."""
    if config_file and Path(config_file).exists():
        import yaml
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)

    # Configuración por defecto optimizada para alto rendimiento
    return {
        'evolution': {
            'grid_size': (12, 12),
            'population_size': 50,
            'max_generations': 500,
            'num_cas_per_fitness': 2000,
            'steps': 8,
            'mutation_rate': 0.1,
            'crossover_rate': 0.7,
            'elite_size': 10
        },
        'output': {
            'dataset_dir': 'dataset_phase0_hp',
            'num_examples_per_task': 3
        },
        'checkpoints': {
            'enabled': True,
            'checkpoint_dir': 'checkpoints',
            'save_every': 100,  # Guardar cada 100 CAs evolucionados
            'max_checkpoints': 5  # Mantener solo los últimos 5 checkpoints
        },
        'parallel': {
            'enabled': True,
            'max_workers': None,  # Usar todos los cores disponibles
            'chunk_size': 10  # Procesar en chunks de 10 CAs
        },
        'fitness_types': {
            'all': [
                # Funciones originales
                'expand', 'symmetry', 'count', 'topology', 'replicate',
                # Funciones cognitivas
                'pattern_match', 'transform_consistency', 'compression', 'analogy', 'objectness',
                # Funciones estructurales
                'rule_entropy', 'invariance', 'relational_distance', 'causal_score',
                # Funciones de coherencia
                'divergence_penalty', 'compositionality'
            ],
            'quick': ['expand', 'pattern_match', 'compression', 'objectness', 'compositionality'],
            'original': ['expand', 'symmetry', 'count', 'topology', 'replicate']
        }
    }

def save_checkpoint(checkpoint_data: Dict[str, Any], checkpoint_dir: str, checkpoint_name: str) -> None:
    """Guardar checkpoint en formato pickle y JSON."""
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(exist_ok=True)

    # Guardar en formato pickle (más eficiente)
    pickle_file = checkpoint_path / f"{checkpoint_name}.pkl"
    with open(pickle_file, 'wb') as f:
        pickle.dump(checkpoint_data, f)

    # Guardar en formato JSON (legible)
    json_file = checkpoint_path / f"{checkpoint_name}.json"
    with open(json_file, 'w') as f:
        json.dump(checkpoint_data, f, indent=2, default=str)

    print(f"💾 Checkpoint guardado: {checkpoint_name}")

def load_checkpoint(checkpoint_dir: str, checkpoint_name: str) -> Optional[Dict[str, Any]]:
    """Cargar checkpoint desde archivo pickle."""
    checkpoint_path = Path(checkpoint_dir) / f"{checkpoint_name}.pkl"

    if not checkpoint_path.exists():
        return None

    try:
        with open(checkpoint_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"⚠️ Error cargando checkpoint {checkpoint_name}: {e}")
        return None

def cleanup_old_checkpoints(checkpoint_dir: str, max_checkpoints: int = 5) -> None:
    """Limpiar checkpoints antiguos manteniendo solo los más recientes."""
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        return

    # Obtener todos los archivos de checkpoint
    checkpoint_files = list(checkpoint_path.glob("checkpoint_*.pkl"))
    checkpoint_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    # Eliminar checkpoints antiguos
    for old_checkpoint in checkpoint_files[max_checkpoints:]:
        try:
            old_checkpoint.unlink()
            # También eliminar el archivo JSON correspondiente
            json_file = old_checkpoint.with_suffix('.json')
            if json_file.exists():
                json_file.unlink()
            print(f"🗑️ Checkpoint antiguo eliminado: {old_checkpoint.name}")
        except Exception as e:
            print(f"⚠️ Error eliminando checkpoint {old_checkpoint.name}: {e}")

def evolve_ca_parallel(fitness_type: str, config: Dict[str, Any], seed: int, logger) -> Tuple[str, Dict[str, Any]]:
    """Evolucionar un CA en paralelo."""
    try:
        from src.cellular_automata.evolution import CAEvolver

        logger.info(f"Iniciando evolución para {fitness_type} (seed: {seed})")

        evolver = CAEvolver(
            grid_size=config['evolution']['grid_size'],
            population_size=config['evolution']['population_size'],
            max_generations=config['evolution']['max_generations'],
            fitness_type=fitness_type,
            seed=seed
        )

        results = evolver.evolve(
            steps=config['evolution']['steps'],
            verbose=False
        )

        logger.info(f"Evolución completada para {fitness_type}: fitness={results['best_fitness']:.3f}")

        return fitness_type, {
            'ca': results['best_ca'],
            'fitness': results['best_fitness'],
            'generations': results.get('generations', 0),
            'seed': seed,
            'timestamp': datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error evolucionando {fitness_type}: {e}")
        logger.error(traceback.format_exc())
        return fitness_type, None

def generate_tasks_from_ca(ca_data: Dict[str, Any], fitness_type: str, config: Dict[str, Any],
                          start_idx: int, num_tasks: int, logger) -> List[Dict[str, Any]]:
    """Generar tareas desde un CA evolucionado."""
    try:
        from src.cellular_automata.task_generator import ARCTaskGenerator

        generator = ARCTaskGenerator(
            grid_size=config['evolution']['grid_size'],
            num_examples_per_task=config['output']['num_examples_per_task'],
            seed=42 + start_idx
        )

        tasks = []
        for i in range(num_tasks):
            try:
                task = generator.generate_task_from_ca(
                    ca_data['ca'],
                    f"{fitness_type}_{start_idx + i}"
                )

                if task:
                    tasks.append({
                        'task_id': f"{fitness_type}_{start_idx + i}",
                        'fitness_type': fitness_type,
                        'task': task,
                        'fitness_score': ca_data['fitness'],
                        'generation_config': {
                            'grid_size': config['evolution']['grid_size'],
                            'population_size': config['evolution']['population_size'],
                            'max_generations': config['evolution']['max_generations'],
                            'steps': config['evolution']['steps']
                        },
                        'ca_metadata': {
                            'generations': ca_data['generations'],
                            'seed': ca_data['seed'],
                            'timestamp': ca_data['timestamp']
                        }
                    })

            except Exception as e:
                logger.warning(f"Error generando tarea {i} para {fitness_type}: {e}")
                continue

        logger.info(f"Generadas {len(tasks)} tareas para {fitness_type}")
        return tasks

    except Exception as e:
        logger.error(f"Error generando tareas para {fitness_type}: {e}")
        return []

def generate_dataset_high_performance(config: Dict[str, Any], mode: str = 'full',
                                    fitness_subset: List[str] = None,
                                    output_dir: str = None,
                                    resume_from_checkpoint: bool = True) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Generar dataset con funcionalidades de alto rendimiento."""

    # Configurar logging
    logger = setup_logging()
    logger.info(f"🚀 Iniciando generación de dataset (modo: {mode})")

    # Ajustar configuración según modo
    if mode == 'quick':
        config['evolution']['max_generations'] = 100
        config['evolution']['population_size'] = 20
        config['evolution']['num_cas_per_fitness'] = 50
        config['evolution']['steps'] = 5
    elif mode == 'debug':
        config['evolution']['max_generations'] = 20
        config['evolution']['population_size'] = 10
        config['evolution']['num_cas_per_fitness'] = 5
        config['evolution']['steps'] = 3

    # Determinar funciones de fitness a usar
    if fitness_subset:
        fitness_types = fitness_subset
    else:
        fitness_types = config['fitness_types'].get(mode, config['fitness_types']['all'])

    logger.info(f"📊 Funciones de fitness: {len(fitness_types)}")
    logger.info(f"🔍 Funciones: {fitness_types}")

    # Configurar directorios
    dataset_dir = Path(output_dir or config['output']['dataset_dir'])
    dataset_dir.mkdir(exist_ok=True)

    checkpoint_dir = config['checkpoints']['checkpoint_dir']
    Path(checkpoint_dir).mkdir(exist_ok=True)

    # Verificar checkpoint existente
    checkpoint_name = f"checkpoint_full_{int(time.time())}"
    checkpoint_data = None

    if resume_from_checkpoint:
        # Buscar el checkpoint más reciente
        checkpoint_files = list(Path(checkpoint_dir).glob("checkpoint_*.pkl"))
        if checkpoint_files:
            latest_checkpoint = max(checkpoint_files, key=lambda x: x.stat().st_mtime)
            checkpoint_data = load_checkpoint(checkpoint_dir, latest_checkpoint.stem)
            if checkpoint_data:
                logger.info(f"📂 Checkpoint encontrado: {latest_checkpoint.name}")
                checkpoint_name = latest_checkpoint.stem

    # Inicializar datos
    if checkpoint_data:
        all_tasks = checkpoint_data.get('all_tasks', [])
        successful_fitness_types = checkpoint_data.get('successful_fitness_types', [])
        failed_fitness_types = checkpoint_data.get('failed_fitness_types', [])
        processed_cas = checkpoint_data.get('processed_cas', {})
        start_time = checkpoint_data.get('start_time', time.time())
    else:
        all_tasks = []
        successful_fitness_types = []
        failed_fitness_types = []
        processed_cas = {}
        start_time = time.time()

    # Procesar cada función de fitness
    for i, fitness_type in enumerate(fitness_types):
        if fitness_type in successful_fitness_types:
            logger.info(f"⏭️ Saltando {fitness_type} (ya procesado)")
            continue

        logger.info(f"\n📊 [{i+1}/{len(fitness_types)}] Procesando {fitness_type}...")

        try:
            # Evolucionar CAs en paralelo
            if config['parallel']['enabled']:
                max_workers = config['parallel']['max_workers'] or mp.cpu_count()
                chunk_size = config['parallel']['chunk_size']

                logger.info(f"🔄 Evolucionando CAs en paralelo (workers: {max_workers})")

                # Dividir en chunks para procesamiento paralelo
                num_cas = config['evolution']['num_cas_per_fitness']
                chunks = [(fitness_type, config, 42 + i * num_cas + j, logger)
                         for j in range(0, num_cas, chunk_size)]

                evolved_cas = []
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    future_to_chunk = {executor.submit(evolve_ca_parallel, *chunk): chunk
                                     for chunk in chunks}

                    for future in as_completed(future_to_chunk):
                        fitness_type_result, ca_data = future.result()
                        if ca_data:
                            evolved_cas.append(ca_data)
                            logger.info(f"✅ CA evolucionado: {fitness_type_result} (fitness: {ca_data['fitness']:.3f})")
                        else:
                            logger.error(f"❌ Falló evolución: {fitness_type_result}")

                if not evolved_cas:
                    logger.error(f"❌ No se pudieron evolucionar CAs para {fitness_type}")
                    failed_fitness_types.append(fitness_type)
                    continue

                # Usar el mejor CA evolucionado
                best_ca = max(evolved_cas, key=lambda x: x['fitness'])
                logger.info(f"🏆 Mejor CA para {fitness_type}: fitness={best_ca['fitness']:.3f}")

            else:
                # Evolución secuencial
                from src.cellular_automata.evolution import CAEvolver

                evolver = CAEvolver(
                    grid_size=config['evolution']['grid_size'],
                    population_size=config['evolution']['population_size'],
                    max_generations=config['evolution']['max_generations'],
                    fitness_type=fitness_type,
                    seed=42 + i
                )

                results = evolver.evolve(
                    steps=config['evolution']['steps'],
                    verbose=False
                )

                best_ca = {
                    'ca': results['best_ca'],
                    'fitness': results['best_fitness'],
                    'generations': results.get('generations', 0),
                    'seed': 42 + i,
                    'timestamp': datetime.now().isoformat()
                }

            # Generar tareas desde el CA
            tasks = generate_tasks_from_ca(
                best_ca, fitness_type, config,
                len(all_tasks), config['evolution']['num_cas_per_fitness'], logger
            )

            all_tasks.extend(tasks)
            successful_fitness_types.append(fitness_type)
            processed_cas[fitness_type] = best_ca

            logger.info(f"✅ {fitness_type}: {len(tasks)} tareas generadas")

            # Guardar checkpoint
            if config['checkpoints']['enabled'] and len(all_tasks) % config['checkpoints']['save_every'] == 0:
                checkpoint_data = {
                    'all_tasks': all_tasks,
                    'successful_fitness_types': successful_fitness_types,
                    'failed_fitness_types': failed_fitness_types,
                    'processed_cas': processed_cas,
                    'start_time': start_time,
                    'config': config,
                    'mode': mode
                }

                save_checkpoint(checkpoint_data, checkpoint_dir, checkpoint_name)
                cleanup_old_checkpoints(checkpoint_dir, config['checkpoints']['max_checkpoints'])

        except Exception as e:
            logger.error(f"❌ Error procesando {fitness_type}: {e}")
            logger.error(traceback.format_exc())
            failed_fitness_types.append(fitness_type)
            continue

    # Guardar dataset final
    logger.info("💾 Guardando dataset final...")

    with open(dataset_dir / "all_tasks.json", 'w') as f:
        json.dump(all_tasks, f, indent=2)

    # Calcular estadísticas detalladas
    tasks_by_fitness = {}
    fitness_scores = {}

    for task in all_tasks:
        fitness_type = task['fitness_type']
        if fitness_type not in tasks_by_fitness:
            tasks_by_fitness[fitness_type] = 0
            fitness_scores[fitness_type] = []

        tasks_by_fitness[fitness_type] += 1
        fitness_scores[fitness_type].append(task['fitness_score'])

    # Crear estadísticas consolidadas
    stats = {
        'total_tasks': len(all_tasks),
        'mode': mode,
        'successful_fitness_types': successful_fitness_types,
        'failed_fitness_types': failed_fitness_types,
        'tasks_by_fitness': tasks_by_fitness,
        'fitness_scores': fitness_scores,
        'grid_size': config['evolution']['grid_size'],
        'num_examples_per_task': config['output']['num_examples_per_task'],
        'generation_config': config['evolution'],
        'fitness_categories': {
            'original': ['expand', 'symmetry', 'count', 'topology', 'replicate'],
            'cognitive': ['pattern_match', 'transform_consistency', 'compression', 'analogy', 'objectness'],
            'structural': ['rule_entropy', 'invariance', 'relational_distance', 'causal_score'],
            'coherence': ['divergence_penalty', 'compositionality']
        },
        'performance': {
            'total_time': time.time() - start_time,
            'tasks_per_minute': len(all_tasks) / ((time.time() - start_time) / 60),
            'parallel_processing': config['parallel']['enabled'],
            'checkpoints_used': config['checkpoints']['enabled']
        }
    }

    with open(dataset_dir / "dataset_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)

    # Guardar checkpoint final
    if config['checkpoints']['enabled']:
        checkpoint_data = {
            'all_tasks': all_tasks,
            'successful_fitness_types': successful_fitness_types,
            'failed_fitness_types': failed_fitness_types,
            'processed_cas': processed_cas,
            'start_time': start_time,
            'config': config,
            'mode': mode,
            'stats': stats
        }

        save_checkpoint(checkpoint_data, checkpoint_dir, f"{checkpoint_name}_final")

    logger.info(f"✅ Dataset generado exitosamente!")
    logger.info(f"   Total tareas: {len(all_tasks)}")
    logger.info(f"   Funciones exitosas: {len(successful_fitness_types)}")
    logger.info(f"   Funciones fallidas: {len(failed_fitness_types)}")
    logger.info(f"   Directorio: {dataset_dir}")
    logger.info(f"   Tiempo total: {(time.time() - start_time)/60:.1f} minutos")

    return all_tasks, stats

def main():
    """Función principal con argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(description='Generar dataset de alto rendimiento de Fase 0')
    parser.add_argument('--mode', choices=['full', 'quick', 'debug'], default='full',
                       help='Modo de generación (default: full)')
    parser.add_argument('--fitness-types', nargs='+',
                       help='Funciones de fitness específicas a usar')
    parser.add_argument('--output-dir', default='dataset_phase0_hp',
                       help='Directorio de salida (default: dataset_phase0_hp)')
    parser.add_argument('--config',
                       help='Archivo de configuración YAML')
    parser.add_argument('--no-resume', action='store_true',
                       help='No reanudar desde checkpoint existente')
    parser.add_argument('--no-parallel', action='store_true',
                       help='Deshabilitar procesamiento paralelo')
    parser.add_argument('--no-checkpoints', action='store_true',
                       help='Deshabilitar sistema de checkpoints')

    args = parser.parse_args()

    print("LUMPS - Generación de Dataset de Alto Rendimiento")
    print("=" * 60)

    # Cargar configuración
    config = load_config(args.config)

    # Aplicar argumentos de línea de comandos
    if args.no_parallel:
        config['parallel']['enabled'] = False
    if args.no_checkpoints:
        config['checkpoints']['enabled'] = False

    # Generar dataset
    tasks, stats = generate_dataset_high_performance(
        config=config,
        mode=args.mode,
        fitness_subset=args.fitness_types,
        output_dir=args.output_dir,
        resume_from_checkpoint=not args.no_resume
    )

    if tasks is None:
        print("❌ Falló la generación del dataset")
        return

    print("\n" + "=" * 60)
    print("🎉 GENERACIÓN COMPLETADA!")
    print(f"⏱️ Tiempo total: {stats['performance']['total_time']/60:.1f} minutos")
    print(f"📊 Tareas por minuto: {stats['performance']['tasks_per_minute']:.1f}")
    print(f"✅ {len(tasks)} tareas generadas")
    print(f"✅ {len(stats['successful_fitness_types'])} funciones de fitness")
    print(f"🔄 Procesamiento paralelo: {'Sí' if stats['performance']['parallel_processing'] else 'No'}")
    print(f"💾 Checkpoints: {'Sí' if stats['performance']['checkpoints_used'] else 'No'}")
    print("🚀 Dataset de alto rendimiento listo para entrenamiento")

if __name__ == "__main__":
    main()
