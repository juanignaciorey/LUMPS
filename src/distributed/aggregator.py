"""
Agregador de resultados para trabajo distribuido.

Combina resultados parciales de todos los batches en un dataset final
consolidado, validando integridad y generando estadísticas.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime
import hashlib

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

logger = logging.getLogger(__name__)


class ResultAggregator:
    """
    Agregador que combina resultados de batches en dataset final.

    Funcionalidades:
    - Combinar tareas de todos los batches completados
    - Validar integridad de datos
    - Generar estadísticas consolidadas
    - Detectar batches faltantes o corruptos
    - Crear dataset final en formato ARC
    """

    def __init__(self, drive_path: str):
        """
        Inicializar agregador.

        Args:
            drive_path: Ruta base en Google Drive
        """
        self.drive_path = Path(drive_path)
        self.manifests_dir = self.drive_path / "manifests"
        self.results_dir = self.drive_path / "results"
        self.final_dir = self.drive_path / "final"
        self.logs_dir = self.drive_path / "logs"

        # Crear directorio final si no existe
        self.final_dir.mkdir(parents=True, exist_ok=True)

        # Configurar logging
        self.logger = self._setup_logging()

        self.logger.info(f"Agregador inicializado: {drive_path}")

    def _setup_logging(self) -> logging.Logger:
        """Configurar logging para el agregador."""
        log_file = self.logs_dir / f"aggregator_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        logger = logging.getLogger("aggregator")
        logger.setLevel(logging.INFO)

        # Crear handler para archivo
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)

        # Crear handler para consola
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Formato
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger

    def load_manifest(self) -> Dict[str, Any]:
        """Cargar manifiesto de batches."""
        manifest_file = self.manifests_dir / "batch_manifest.json"

        if not manifest_file.exists():
            raise FileNotFoundError(f"Manifiesto no encontrado: {manifest_file}")

        with open(manifest_file, 'r') as f:
            return json.load(f)

    def get_completed_batches(self) -> List[Dict[str, Any]]:
        """
        Obtener lista de batches completados.

        Returns:
            Lista de batches completados con sus resultados
        """
        manifest = self.load_manifest()
        completed_batches = []

        for batch in manifest['batches']:
            if batch['status'] == 'completed':
                batch_result_dir = self.results_dir / batch['batch_id']

                if batch_result_dir.exists():
                    # Verificar que existen los archivos necesarios
                    tasks_file = batch_result_dir / "tasks.json"
                    stats_file = batch_result_dir / "stats.json"

                    if tasks_file.exists() and stats_file.exists():
                        try:
                            # Cargar tareas
                            with open(tasks_file, 'r') as f:
                                tasks = json.load(f)

                            # Cargar estadísticas
                            with open(stats_file, 'r') as f:
                                stats = json.load(f)

                            completed_batches.append({
                                'batch_info': batch,
                                'tasks': tasks,
                                'stats': stats,
                                'tasks_file': tasks_file,
                                'stats_file': stats_file
                            })

                        except Exception as e:
                            self.logger.error(f"Error cargando batch {batch['batch_id']}: {e}")
                    else:
                        self.logger.warning(f"Archivos faltantes para batch {batch['batch_id']}")
                else:
                    self.logger.warning(f"Directorio faltante para batch {batch['batch_id']}")

        self.logger.info(f"Encontrados {len(completed_batches)} batches completados")
        return completed_batches

    def validate_batch_integrity(self, batch_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validar integridad de un batch.

        Args:
            batch_data: Datos del batch a validar

        Returns:
            (es_válido, lista_de_errores)
        """
        errors = []
        batch_info = batch_data['batch_info']
        tasks = batch_data['tasks']
        stats = batch_data['stats']

        # Verificar que el número de tareas coincide
        expected_tasks = batch_info['batch_size']
        actual_tasks = len(tasks)

        if actual_tasks != expected_tasks:
            errors.append(f"Número de tareas incorrecto: esperado {expected_tasks}, encontrado {actual_tasks}")

        # Verificar que las tareas tienen el formato correcto
        for i, task in enumerate(tasks):
            if not isinstance(task, dict):
                errors.append(f"Tarea {i} no es un diccionario")
                continue

            required_fields = ['task_id', 'fitness_type', 'task', 'fitness_score']
            for field in required_fields:
                if field not in task:
                    errors.append(f"Tarea {i} falta campo '{field}'")

            # Verificar formato de tarea ARC
            if 'task' in task:
                arc_task = task['task']
                if not isinstance(arc_task, dict):
                    errors.append(f"Tarea {i}: campo 'task' no es un diccionario")
                elif 'train' not in arc_task:
                    errors.append(f"Tarea {i}: falta campo 'train' en tarea ARC")
                elif not isinstance(arc_task['train'], list):
                    errors.append(f"Tarea {i}: campo 'train' no es una lista")

        # Verificar estadísticas
        if not isinstance(stats, dict):
            errors.append("Estadísticas no son un diccionario")
        elif 'tasks_generated' not in stats:
            errors.append("Falta campo 'tasks_generated' en estadísticas")
        elif stats['tasks_generated'] != actual_tasks:
            errors.append(f"Estadísticas inconsistentes: {stats['tasks_generated']} vs {actual_tasks}")

        is_valid = len(errors) == 0
        return is_valid, errors

    def aggregate_tasks(self, completed_batches: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Agregar todas las tareas de los batches completados.

        Args:
            completed_batches: Lista de batches completados

        Returns:
            (lista_de_todas_las_tareas, estadísticas_de_agregación)
        """
        all_tasks = []
        aggregation_stats = {
            'total_batches_processed': len(completed_batches),
            'total_tasks': 0,
            'valid_batches': 0,
            'invalid_batches': 0,
            'fitness_type_distribution': {},
            'workers_used': set(),
            'batch_errors': []
        }

        self.logger.info(f"Agregando {len(completed_batches)} batches...")

        for batch_data in completed_batches:
            batch_id = batch_data['batch_info']['batch_id']

            # Validar integridad del batch
            is_valid, errors = self.validate_batch_integrity(batch_data)

            if is_valid:
                # Agregar tareas
                tasks = batch_data['tasks']
                all_tasks.extend(tasks)
                aggregation_stats['valid_batches'] += 1
                aggregation_stats['total_tasks'] += len(tasks)

                # Actualizar distribución por fitness type
                for task in tasks:
                    fitness_type = task['fitness_type']
                    if fitness_type not in aggregation_stats['fitness_type_distribution']:
                        aggregation_stats['fitness_type_distribution'][fitness_type] = 0
                    aggregation_stats['fitness_type_distribution'][fitness_type] += 1

                # Registrar worker usado
                worker_id = batch_data['stats'].get('worker_id', 'unknown')
                aggregation_stats['workers_used'].add(worker_id)

                self.logger.info(f"✅ Batch {batch_id}: {len(tasks)} tareas agregadas")

            else:
                aggregation_stats['invalid_batches'] += 1
                aggregation_stats['batch_errors'].append({
                    'batch_id': batch_id,
                    'errors': errors
                })

                self.logger.error(f"❌ Batch {batch_id} inválido: {errors}")

        # Convertir set a lista para JSON
        aggregation_stats['workers_used'] = list(aggregation_stats['workers_used'])

        self.logger.info(f"Agregación completada: {aggregation_stats['total_tasks']} tareas de {aggregation_stats['valid_batches']} batches válidos")

        return all_tasks, aggregation_stats

    def generate_final_stats(self, all_tasks: List[Dict[str, Any]],
                           aggregation_stats: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generar estadísticas finales del dataset.

        Args:
            all_tasks: Lista de todas las tareas
            aggregation_stats: Estadísticas de agregación

        Returns:
            Estadísticas finales consolidadas
        """
        # Calcular estadísticas de fitness scores
        fitness_scores = [task['fitness_score'] for task in all_tasks]

        # Calcular estadísticas de tareas ARC
        arc_stats = {
            'total_examples': 0,
            'examples_per_task': [],
            'grid_sizes': [],
            'num_states_used': []
        }

        for task in all_tasks:
            arc_task = task['task']
            if 'train' in arc_task:
                num_examples = len(arc_task['train'])
                arc_stats['total_examples'] += num_examples
                arc_stats['examples_per_task'].append(num_examples)

                # Analizar primer ejemplo para estadísticas de grid
                if arc_task['train']:
                    first_example = arc_task['train'][0]
                    if 'input' in first_example and 'output' in first_example:
                        input_grid = first_example['input']
                        output_grid = first_example['output']

                        # Tamaño de grid
                        if isinstance(input_grid, list) and len(input_grid) > 0:
                            grid_size = (len(input_grid), len(input_grid[0]) if input_grid[0] else 0)
                            arc_stats['grid_sizes'].append(grid_size)

                        # Número de estados únicos
                        all_values = []
                        for row in input_grid:
                            all_values.extend(row)
                        for row in output_grid:
                            all_values.extend(row)

                        unique_states = len(set(all_values))
                        arc_stats['num_states_used'].append(unique_states)

        # Estadísticas finales
        final_stats = {
            'dataset_info': {
                'total_tasks': len(all_tasks),
                'total_examples': arc_stats['total_examples'],
                'created_at': datetime.now().isoformat(),
                'aggregation_method': 'distributed_workers'
            },
            'fitness_analysis': {
                'fitness_type_distribution': aggregation_stats['fitness_type_distribution'],
                'fitness_scores': {
                    'min': min(fitness_scores) if fitness_scores else 0,
                    'max': max(fitness_scores) if fitness_scores else 0,
                    'mean': sum(fitness_scores) / len(fitness_scores) if fitness_scores else 0,
                    'count': len(fitness_scores)
                }
            },
            'arc_task_analysis': {
                'avg_examples_per_task': sum(arc_stats['examples_per_task']) / len(arc_stats['examples_per_task']) if arc_stats['examples_per_task'] else 0,
                'min_examples_per_task': min(arc_stats['examples_per_task']) if arc_stats['examples_per_task'] else 0,
                'max_examples_per_task': max(arc_stats['examples_per_task']) if arc_stats['examples_per_task'] else 0,
                'unique_grid_sizes': list(set(arc_stats['grid_sizes'])),
                'avg_states_per_task': sum(arc_stats['num_states_used']) / len(arc_stats['num_states_used']) if arc_stats['num_states_used'] else 0
            },
            'processing_info': {
                'total_batches_processed': aggregation_stats['total_batches_processed'],
                'valid_batches': aggregation_stats['valid_batches'],
                'invalid_batches': aggregation_stats['invalid_batches'],
                'workers_used': aggregation_stats['workers_used'],
                'batch_errors': aggregation_stats['batch_errors']
            },
            'data_integrity': {
                'checksum': self._calculate_checksum(all_tasks),
                'validation_passed': aggregation_stats['invalid_batches'] == 0
            }
        }

        return final_stats

    def _calculate_checksum(self, tasks: List[Dict[str, Any]]) -> str:
        """Calcular checksum de las tareas para verificación de integridad."""
        # Crear string representativo de todas las tareas
        task_strings = []
        for task in tasks:
            task_strings.append(f"{task['task_id']}:{task['fitness_type']}:{task['fitness_score']}")

        # Calcular hash
        content = "|".join(sorted(task_strings))
        return hashlib.md5(content.encode()).hexdigest()

    def save_final_dataset(self, all_tasks: List[Dict[str, Any]],
                          final_stats: Dict[str, Any]) -> Tuple[Path, Path]:
        """
        Guardar dataset final y estadísticas.

        Args:
            all_tasks: Lista de todas las tareas
            final_stats: Estadísticas finales

        Returns:
            (ruta_archivo_tareas, ruta_archivo_estadísticas)
        """
        # Guardar tareas
        tasks_file = self.final_dir / "all_tasks.json"
        with open(tasks_file, 'w') as f:
            json.dump(all_tasks, f, indent=2)

        # Guardar estadísticas
        stats_file = self.final_dir / "dataset_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(final_stats, f, indent=2)

        self.logger.info(f"Dataset final guardado:")
        self.logger.info(f"   Tareas: {tasks_file}")
        self.logger.info(f"   Estadísticas: {stats_file}")

        return tasks_file, stats_file

    def aggregate(self) -> Dict[str, Any]:
        """
        Ejecutar proceso completo de agregación.

        Returns:
            Estadísticas finales del proceso
        """
        self.logger.info("🚀 Iniciando agregación de resultados...")

        try:
            # Obtener batches completados
            completed_batches = self.get_completed_batches()

            if not completed_batches:
                self.logger.warning("No hay batches completados para agregar")
                return {'error': 'No hay batches completados'}

            # Agregar tareas
            all_tasks, aggregation_stats = self.aggregate_tasks(completed_batches)

            if not all_tasks:
                self.logger.warning("No se pudieron agregar tareas válidas")
                return {'error': 'No hay tareas válidas'}

            # Generar estadísticas finales
            final_stats = self.generate_final_stats(all_tasks, aggregation_stats)

            # Guardar dataset final
            tasks_file, stats_file = self.save_final_dataset(all_tasks, final_stats)

            self.logger.info("✅ Agregación completada exitosamente")
            self.logger.info(f"   Total tareas: {len(all_tasks)}")
            self.logger.info(f"   Batches válidos: {aggregation_stats['valid_batches']}")
            self.logger.info(f"   Batches inválidos: {aggregation_stats['invalid_batches']}")

            return {
                'success': True,
                'total_tasks': len(all_tasks),
                'tasks_file': str(tasks_file),
                'stats_file': str(stats_file),
                'final_stats': final_stats
            }

        except Exception as e:
            self.logger.error(f"❌ Error en agregación: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {'error': str(e)}

    def print_summary(self, result: Dict[str, Any]) -> None:
        """Imprimir resumen de la agregación."""
        if 'error' in result:
            print(f"❌ Error en agregación: {result['error']}")
            return

        if not result.get('success'):
            print("❌ Agregación falló")
            return

        final_stats = result['final_stats']

        print("\n" + "="*80)
        print("🎉 AGREGACIÓN COMPLETADA")
        print("="*80)

        print(f"📊 DATASET FINAL:")
        print(f"   Total tareas: {final_stats['dataset_info']['total_tasks']}")
        print(f"   Total ejemplos: {final_stats['dataset_info']['total_examples']}")
        print(f"   Archivo tareas: {result['tasks_file']}")
        print(f"   Archivo estadísticas: {result['stats_file']}")

        print(f"\n📈 ANÁLISIS DE FITNESS:")
        fitness_dist = final_stats['fitness_analysis']['fitness_type_distribution']
        for fitness_type, count in sorted(fitness_dist.items()):
            print(f"   {fitness_type}: {count} tareas")

        print(f"\n🎯 ANÁLISIS DE TAREAS ARC:")
        arc_stats = final_stats['arc_task_analysis']
        print(f"   Ejemplos promedio por tarea: {arc_stats['avg_examples_per_task']:.1f}")
        print(f"   Estados promedio por tarea: {arc_stats['avg_states_per_task']:.1f}")
        print(f"   Tamaños de grid únicos: {len(arc_stats['unique_grid_sizes'])}")

        print(f"\n👥 PROCESAMIENTO:")
        proc_info = final_stats['processing_info']
        print(f"   Batches procesados: {proc_info['total_batches_processed']}")
        print(f"   Batches válidos: {proc_info['valid_batches']}")
        print(f"   Batches inválidos: {proc_info['invalid_batches']}")
        print(f"   Workers utilizados: {len(proc_info['workers_used'])}")

        print(f"\n🔒 INTEGRIDAD:")
        integrity = final_stats['data_integrity']
        print(f"   Validación: {'✅ PASÓ' if integrity['validation_passed'] else '❌ FALLÓ'}")
        print(f"   Checksum: {integrity['checksum']}")

        print("="*80)


def main():
    """Función principal para agregar resultados desde línea de comandos."""
    import argparse

    parser = argparse.ArgumentParser(description='Agregar resultados de trabajo distribuido')
    parser.add_argument('--drive-path', required=True,
                       help='Ruta base en Google Drive')
    parser.add_argument('--verbose', action='store_true',
                       help='Mostrar logs detallados')

    args = parser.parse_args()

    # Configurar logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')

    # Ejecutar agregación
    aggregator = ResultAggregator(args.drive_path)
    result = aggregator.aggregate()
    aggregator.print_summary(result)


if __name__ == "__main__":
    main()
