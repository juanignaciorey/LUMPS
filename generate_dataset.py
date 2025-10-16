"""
Script consolidado para generar dataset de Fase 0 con todas las funciones de fitness
"""

import sys
import argparse
from pathlib import Path
import json
import time

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def load_config(config_file: str = None):
    """Cargar configuración desde archivo o usar defaults."""
    if config_file and Path(config_file).exists():
        import yaml
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)

    # Configuración por defecto
    return {
        'evolution': {
            'grid_size': (8, 8),
            'population_size': 20,
            'max_generations': 40,
            'num_cas_per_fitness': 15,
            'steps': 8
        },
        'output': {
            'dataset_dir': 'dataset_phase0',
            'num_examples_per_task': 3
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

def generate_dataset(config: dict, mode: str = 'full', fitness_subset: list = None, output_dir: str = None):
    """Generar dataset con configuración especificada."""
    print(f"🚀 Generando dataset de Fase 0 (modo: {mode})...")

    try:
        from src.cellular_automata.evolution import CAEvolver
        from src.cellular_automata.task_generator import ARCTaskGenerator

        # Ajustar configuración según modo
        if mode == 'quick':
            config['evolution']['max_generations'] = 20
            config['evolution']['population_size'] = 10
            config['evolution']['num_cas_per_fitness'] = 5
            config['evolution']['steps'] = 5
        elif mode == 'debug':
            config['evolution']['max_generations'] = 5
            config['evolution']['population_size'] = 5
            config['evolution']['num_cas_per_fitness'] = 2
            config['evolution']['steps'] = 3

        # Determinar funciones de fitness a usar
        if fitness_subset:
            fitness_types = fitness_subset
        else:
            fitness_types = config['fitness_types'].get(mode, config['fitness_types']['all'])

        print(f"📊 Funciones de fitness: {len(fitness_types)}")
        print(f"🔍 Funciones: {fitness_types}")

        all_tasks = []
        successful_fitness_types = []
        failed_fitness_types = []

        for i, fitness_type in enumerate(fitness_types):
            print(f"\n📊 [{i+1}/{len(fitness_types)}] Generando CAs para {fitness_type}...")

            try:
                # Evolucionar CAs
                evolver = CAEvolver(
                    grid_size=config['evolution']['grid_size'],
                    population_size=config['evolution']['population_size'],
                    max_generations=config['evolution']['max_generations'],
                    fitness_type=fitness_type,
                    seed=42 + i  # Diferente seed para cada función
                )

                results = evolver.evolve(
                    steps=config['evolution']['steps'],
                    verbose=False
                )
                print(f"   ✅ Fitness: {results['best_fitness']:.3f}")

                # Generar tareas
                generator = ARCTaskGenerator(
                    grid_size=config['evolution']['grid_size'],
                    num_examples_per_task=config['output']['num_examples_per_task'],
                    seed=42 + i
                )

                tasks_generated = 0
                for j in range(config['evolution']['num_cas_per_fitness']):
                    try:
                        task = generator.generate_task_from_ca(
                            results['best_ca'],
                            f"{fitness_type}_{j}"
                        )
                        if task:
                            all_tasks.append({
                                'task_id': f"{fitness_type}_{j}",
                                'fitness_type': fitness_type,
                                'task': task,
                                'fitness_score': results['best_fitness'],
                                'generation_config': {
                                    'grid_size': config['evolution']['grid_size'],
                                    'population_size': config['evolution']['population_size'],
                                    'max_generations': config['evolution']['max_generations'],
                                    'steps': config['evolution']['steps']
                                }
                            })
                            tasks_generated += 1
                    except Exception as e:
                        print(f"      ⚠️ Error generando tarea {j}: {e}")
                        continue

                print(f"   ✅ {tasks_generated} tareas generadas")
                successful_fitness_types.append(fitness_type)

            except Exception as e:
                print(f"   ❌ Error con {fitness_type}: {e}")
                failed_fitness_types.append(fitness_type)
                continue

        # Guardar dataset
        dataset_dir = Path(output_dir or config['output']['dataset_dir'])
        dataset_dir.mkdir(exist_ok=True)

        # Guardar todas las tareas
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
            }
        }

        with open(dataset_dir / "dataset_stats.json", 'w') as f:
            json.dump(stats, f, indent=2)

        print(f"\n✅ Dataset generado exitosamente!")
        print(f"   Total tareas: {len(all_tasks)}")
        print(f"   Funciones exitosas: {len(successful_fitness_types)}")
        print(f"   Funciones fallidas: {len(failed_fitness_types)}")
        print(f"   Directorio: {dataset_dir}")

        if failed_fitness_types:
            print(f"   ⚠️ Funciones fallidas: {failed_fitness_types}")

        # Mostrar estadísticas detalladas
        print(f"\n📊 Estadísticas por categoría:")

        for category, functions in stats['fitness_categories'].items():
            print(f"\n   {category.upper()}:")
            for func in functions:
                if func in tasks_by_fitness:
                    count = tasks_by_fitness[func]
                    avg_score = sum(fitness_scores[func]) / len(fitness_scores[func]) if fitness_scores[func] else 0
                    print(f"     {func:20}: {count:2d} tareas (fitness avg: {avg_score:.3f})")

        return all_tasks, stats

    except Exception as e:
        print(f"❌ Error generando dataset: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def main():
    """Función principal con argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(description='Generar dataset consolidado de Fase 0')
    parser.add_argument('--mode', choices=['full', 'quick', 'debug'], default='full',
                       help='Modo de generación (default: full)')
    parser.add_argument('--fitness-types', nargs='+',
                       help='Funciones de fitness específicas a usar')
    parser.add_argument('--output-dir', default='dataset_phase0',
                       help='Directorio de salida (default: dataset_phase0)')
    parser.add_argument('--config',
                       help='Archivo de configuración YAML')

    args = parser.parse_args()

    print("LUMPS - Generación de Dataset Consolidado")
    print("=" * 50)

    start_time = time.time()

    # Cargar configuración
    config = load_config(args.config)

    # Generar dataset
    tasks, stats = generate_dataset(
        config=config,
        mode=args.mode,
        fitness_subset=args.fitness_types,
        output_dir=args.output_dir
    )

    if tasks is None:
        print("❌ Falló la generación del dataset")
        return

    end_time = time.time()
    duration = end_time - start_time

    print("\n" + "=" * 50)
    print("🎉 GENERACIÓN COMPLETADA!")
    print(f"⏱️ Tiempo total: {duration/60:.1f} minutos")
    print(f"✅ {len(tasks)} tareas generadas")
    print(f"✅ {len(stats['successful_fitness_types'])} funciones de fitness")
    print("🚀 Dataset consolidado listo para entrenamiento")

if __name__ == "__main__":
    main()
