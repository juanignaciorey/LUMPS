"""
Worker distribuido para Google Colab.

Maneja la lógica de claim de batches, procesamiento, checkpoints
y manejo de errores para workers individuales.
"""

import json
import time
import os
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from tqdm import tqdm
from .coordinator import BatchCoordinator
from src.cellular_automata.evolution import CAEvolver
from src.cellular_automata.task_generator import ARCTaskGenerator
from src.utils.logger import setup_logger


class DistributedWorker:
    """
    Worker que procesa batches de trabajo en Google Colab.

    Funcionalidades:
    - Claim batches disponibles
    - Procesar evolución de CAs y generación de tareas
    - Checkpoints locales para recuperación
    - Manejo de errores y reintentos
    - Logging detallado
    """

    def __init__(self, worker_id: str, drive_path: str,
                 local_checkpoint_dir: str = "/tmp/lumps_checkpoints"):
        """
        Inicializar worker distribuido.

        Args:
            worker_id: ID único del worker
            drive_path: Ruta base en Google Drive
            local_checkpoint_dir: Directorio local para checkpoints
        """
        self.worker_id = worker_id
        self.drive_path = Path(drive_path)
        self.local_checkpoint_dir = Path(local_checkpoint_dir)
        self.local_checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Inicializar coordinador
        self.coordinator = BatchCoordinator(str(drive_path))

        # Configurar logging
        self.logger = self._setup_logging()

        # Estadísticas del worker
        self.stats = {
            'batches_processed': 0,
            'tasks_generated': 0,
            'start_time': time.time(),
            'last_batch_time': None,
            'errors': 0
        }

        # Configuración de evolución
        self.evolution_config = {
            'grid_size': (12, 12),
            'population_size': 50,
            'max_generations': 200,
            'steps': 8,
            'mutation_rate': 0.1,
            'crossover_rate': 0.7,
            'elite_size': 10
        }

        # Configuración de generación de tareas
        self.task_config = {
            'num_examples_per_task': 3,
            'grid_size': (12, 12)
        }

        self.logger.info(f"Worker {worker_id} inicializado")
        self.logger.info(f"Drive path: {drive_path}")
        self.logger.info(f"Local checkpoints: {local_checkpoint_dir}")

    def _setup_logging(self) -> logging.Logger:
        """Configurar logging para el worker."""
        logs_dir = self.drive_path / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        log_file = logs_dir / f"worker_{self.worker_id}.log"

        logger = setup_logger(
            name=f"worker_{self.worker_id}",
            log_file=str(log_file),
            level="INFO"
        )

        return logger

    def run(self, max_batches: Optional[int] = None,
            sleep_seconds: int = 30) -> Dict[str, Any]:
        """
        Ejecutar worker en loop continuo.

        Args:
            max_batches: Número máximo de batches a procesar (None = infinito)
            sleep_seconds: Segundos a esperar entre intentos de claim

        Returns:
            Estadísticas finales del worker
        """
        self.logger.info(f"🚀 Iniciando worker {self.worker_id}")
        self.logger.info(f"Max batches: {max_batches or 'infinito'}")

        try:
            while True:
                if max_batches and self.stats['batches_processed'] >= max_batches:
                    self.logger.info(f"✅ Worker completado: {self.stats['batches_processed']} batches procesados")
                    break

                # Intentar claim un batch
                try:
                    self.logger.debug(f"🔍 Intentando claim batch...")
                    batch = self.coordinator.claim_batch(self.worker_id)

                    if batch:
                        self.logger.info(f"📦 Batch claimado: {batch['batch_id']} (fitness: {batch['fitness_type']})")
                        success = self._process_batch(batch)

                        if success:
                            self.stats['batches_processed'] += 1
                            self.stats['last_batch_time'] = time.time()
                            self.logger.info(f"✅ Batch {batch['batch_id']} completado exitosamente")
                        else:
                            self.stats['errors'] += 1
                            self.logger.error(f"❌ Batch {batch['batch_id']} falló")
                    else:
                        # No hay batches disponibles, mostrar estadísticas y esperar
                        self.logger.info(f"⏳ No hay batches disponibles")
                        self._show_progress_summary()
                        self.logger.info(f"⏳ Esperando {sleep_seconds}s antes de reintentar...")
                        time.sleep(sleep_seconds)

                except Exception as e:
                    self.logger.error(f"❌ Error obteniendo batch: {e}")
                    self.stats['errors'] += 1

                    # Si es error de manifiesto, intentar esperar a que se recupere
                    if "Manifiesto no encontrado" in str(e):
                        self.logger.info("🔄 Intentando esperar a que se recupere el manifiesto...")
                        manifest = self.coordinator.wait_for_manifest(max_wait_seconds=60, check_interval=5)
                        if manifest is None:
                            self.logger.warning("⚠️ Manifiesto no se recuperó, continuando con espera normal...")

                    self.logger.info(f"⏳ Esperando {sleep_seconds}s antes de reintentar...")
                    time.sleep(sleep_seconds)
                    continue

                # Mostrar estadísticas cada 10 batches
                if self.stats['batches_processed'] % 10 == 0:
                    self._log_stats()

        except KeyboardInterrupt:
            self.logger.info("🛑 Worker interrumpido por usuario")
        except Exception as e:
            self.logger.error(f"💥 Error fatal en worker: {e}")
            self.logger.error(traceback.format_exc())

        finally:
            self._log_final_stats()
            return self.stats

    def _process_batch(self, batch: Dict[str, Any]) -> bool:
        """
        Procesar un batch completo.

        Args:
            batch: Información del batch a procesar

        Returns:
            True si se procesó exitosamente
        """
        batch_id = batch['batch_id']
        fitness_type = batch['fitness_type']
        start_idx = batch['start_task_idx']
        end_idx = batch['end_task_idx']
        batch_size = batch['batch_size']

        try:
            # Verificar checkpoint local
            checkpoint_file = self.local_checkpoint_dir / f"{batch_id}_checkpoint.pkl"
            checkpoint_data = self._load_local_checkpoint(checkpoint_file)

            if checkpoint_data:
                self.logger.info(f"📂 Checkpoint encontrado para {batch_id}, continuando...")
                tasks = checkpoint_data.get('tasks', [])
                completed_tasks = len(tasks)
            else:
                tasks = []
                completed_tasks = 0

            # Procesar tareas restantes
            remaining_tasks = batch_size - completed_tasks

            if remaining_tasks > 0:
                self.logger.info(f"🔄 Procesando {remaining_tasks} tareas restantes para {batch_id}")

                # Evolucionar CA para esta fitness function
                ca_data = self._evolve_ca(fitness_type, batch_id)

                if not ca_data:
                    self.logger.error(f"❌ Falló evolución de CA para {fitness_type}")
                    return False

                # Generar tareas restantes
                new_tasks = self._generate_tasks(
                    ca_data, fitness_type, start_idx + completed_tasks,
                    remaining_tasks, batch_id
                )

                tasks.extend(new_tasks)

                # Guardar checkpoint local
                self._save_local_checkpoint(checkpoint_file, {
                    'tasks': tasks,
                    'batch_info': batch,
                    'timestamp': datetime.now().isoformat()
                })

            # Guardar resultados en Drive
            success = self._save_batch_results(batch_id, tasks)

            if success:
                # Eliminar checkpoint local
                if checkpoint_file.exists():
                    checkpoint_file.unlink()

                self.stats['tasks_generated'] += len(tasks)
                return True
            else:
                return False

        except Exception as e:
            self.logger.error(f"💥 Error procesando batch {batch_id}: {e}")
            self.logger.error(traceback.format_exc())
            return False

    def _evolve_ca(self, fitness_type: str, batch_id: str) -> Optional[Dict[str, Any]]:
        """
        Evolucionar CA para una fitness function específica.

        Args:
            fitness_type: Tipo de fitness function
            batch_id: ID del batch para logging

        Returns:
            Datos del CA evolucionado o None si falla
        """
        try:
            self.logger.info(f"🧬 Evolucionando CA para {fitness_type}...")

            evolver = CAEvolver(
                grid_size=self.evolution_config['grid_size'],
                population_size=self.evolution_config['population_size'],
                max_generations=self.evolution_config['max_generations'],
                fitness_type=fitness_type,
                seed=hash(f"{batch_id}_{fitness_type}") % 2**32
            )

            results = evolver.evolve(
                steps=self.evolution_config['steps'],
                verbose=False
            )

            ca_data = {
                'ca': results['best_ca'],
                'fitness': results['best_fitness'],
                'generations': results.get('generations', 0),
                'fitness_type': fitness_type,
                'timestamp': datetime.now().isoformat()
            }

            self.logger.info(f"✅ CA evolucionado: fitness={results['best_fitness']:.3f}")
            return ca_data

        except Exception as e:
            self.logger.error(f"❌ Error evolucionando CA para {fitness_type}: {e}")
            return None

    def _generate_tasks(self, ca_data: Dict[str, Any], fitness_type: str,
                       start_idx: int, num_tasks: int, batch_id: str) -> List[Dict[str, Any]]:
        """
        Generar tareas desde un CA evolucionado.

        Args:
            ca_data: Datos del CA evolucionado
            fitness_type: Tipo de fitness function
            start_idx: Índice de inicio para IDs de tareas
            num_tasks: Número de tareas a generar
            batch_id: ID del batch para logging

        Returns:
            Lista de tareas generadas
        """
        try:
            self.logger.info(f"🎯 Generando {num_tasks} tareas para {fitness_type}...")

            generator = ARCTaskGenerator(
                grid_size=self.task_config['grid_size'],
                num_examples_per_task=self.task_config['num_examples_per_task'],
                seed=hash(f"{batch_id}_{start_idx}") % 2**32
            )

            tasks = []

            with tqdm(total=num_tasks, desc=f"Generando tareas {batch_id}") as pbar:
                for i in range(num_tasks):
                    try:
                        task_id = f"{fitness_type}_{start_idx + i}"
                        task = generator.generate_task_from_ca(
                            ca_data['ca'],
                            task_id
                        )

                        if task:
                            task_data = {
                                'task_id': task_id,
                                'fitness_type': fitness_type,
                                'task': task,
                                'fitness_score': ca_data['fitness'],
                                'generation_config': self.evolution_config,
                                'ca_metadata': {
                                    'generations': ca_data['generations'],
                                    'timestamp': ca_data['timestamp']
                                }
                            }
                            tasks.append(task_data)
                        else:
                            self.logger.warning(f"⚠️ Tarea {task_id} no generada (inválida)")

                        pbar.update(1)

                    except Exception as e:
                        self.logger.warning(f"⚠️ Error generando tarea {i}: {e}")
                        continue

            self.logger.info(f"✅ Generadas {len(tasks)} tareas válidas de {num_tasks} intentadas")
            return tasks

        except Exception as e:
            self.logger.error(f"❌ Error generando tareas: {e}")
            return []

    def _save_batch_results(self, batch_id: str, tasks: List[Dict[str, Any]]) -> bool:
        """
        Guardar resultados del batch en Google Drive.

        Args:
            batch_id: ID del batch
            tasks: Lista de tareas generadas

        Returns:
            True si se guardó exitosamente
        """
        try:
            # Crear estadísticas del batch
            stats = {
                'batch_id': batch_id,
                'worker_id': self.worker_id,
                'tasks_generated': len(tasks),
                'completed_at': datetime.now().isoformat(),
                'fitness_types': list(set(task['fitness_type'] for task in tasks)),
                'avg_fitness_score': sum(task['fitness_score'] for task in tasks) / len(tasks) if tasks else 0
            }

            # Completar batch en coordinador
            success = self.coordinator.complete_batch(batch_id, self.worker_id, tasks, stats)

            if success:
                self.logger.info(f"💾 Resultados guardados para {batch_id}: {len(tasks)} tareas")
                return True
            else:
                self.logger.error(f"❌ Error guardando resultados para {batch_id}")
                return False

        except Exception as e:
            self.logger.error(f"❌ Error guardando resultados: {e}")
            return False

    def _load_local_checkpoint(self, checkpoint_file: Path) -> Optional[Dict[str, Any]]:
        """Cargar checkpoint local."""
        if not checkpoint_file.exists():
            return None

        try:
            import pickle
            with open(checkpoint_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            self.logger.warning(f"⚠️ Error cargando checkpoint {checkpoint_file}: {e}")
            return None

    def _save_local_checkpoint(self, checkpoint_file: Path, data: Dict[str, Any]) -> None:
        """Guardar checkpoint local."""
        try:
            import pickle
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            self.logger.warning(f"⚠️ Error guardando checkpoint {checkpoint_file}: {e}")

    def _log_stats(self) -> None:
        """Mostrar estadísticas del worker."""
        elapsed_time = time.time() - self.stats['start_time']
        tasks_per_hour = (self.stats['tasks_generated'] / elapsed_time) * 3600 if elapsed_time > 0 else 0

        self.logger.info(f"📊 Estadísticas Worker {self.worker_id}:")
        self.logger.info(f"   Batches procesados: {self.stats['batches_processed']}")
        self.logger.info(f"   Tareas generadas: {self.stats['tasks_generated']}")
        self.logger.info(f"   Errores: {self.stats['errors']}")
        self.logger.info(f"   Tiempo transcurrido: {elapsed_time/3600:.1f} horas")
        self.logger.info(f"   Tareas/hora: {tasks_per_hour:.1f}")

    def _show_progress_summary(self) -> None:
        """Mostrar resumen de progreso general."""
        try:
            stats = self.coordinator.get_progress_stats()

            self.logger.info(f"📊 Progreso General:")
            self.logger.info(f"   Total batches: {stats['total_batches']}")
            self.logger.info(f"   Disponibles: {stats['available']}")
            self.logger.info(f"   En progreso: {stats['locked']}")
            self.logger.info(f"   Completados: {stats['completed']}")
            self.logger.info(f"   Fallidos: {stats['failed']}")
            self.logger.info(f"   Workers activos: {stats['workers_active']}")
            self.logger.info(f"   Tareas generadas: {stats['total_tasks_generated']}")

            if stats['total_batches'] > 0:
                completion_pct = (stats['completed'] / stats['total_batches']) * 100
                self.logger.info(f"   Progreso: {completion_pct:.1f}%")

        except Exception as e:
            self.logger.warning(f"⚠️ Error obteniendo estadísticas: {e}")

    def _log_final_stats(self) -> None:
        """Mostrar estadísticas finales del worker."""
        elapsed_time = time.time() - self.stats['start_time']

        self.logger.info(f"🏁 Estadísticas Finales Worker {self.worker_id}:")
        self.logger.info(f"   Batches procesados: {self.stats['batches_processed']}")
        self.logger.info(f"   Tareas generadas: {self.stats['tasks_generated']}")
        self.logger.info(f"   Errores: {self.stats['errors']}")
        self.logger.info(f"   Tiempo total: {elapsed_time/3600:.1f} horas")

        if self.stats['batches_processed'] > 0:
            avg_time_per_batch = elapsed_time / self.stats['batches_processed']
            self.logger.info(f"   Tiempo promedio por batch: {avg_time_per_batch/60:.1f} minutos")

        if self.stats['tasks_generated'] > 0:
            tasks_per_hour = (self.stats['tasks_generated'] / elapsed_time) * 3600
            self.logger.info(f"   Tareas/hora: {tasks_per_hour:.1f}")

    def get_status(self) -> Dict[str, Any]:
        """
        Obtener estado actual del worker.

        Returns:
            Diccionario con estado del worker
        """
        return {
            'worker_id': self.worker_id,
            'stats': self.stats.copy(),
            'elapsed_time': time.time() - self.stats['start_time'],
            'coordinator_status': self.coordinator.get_worker_status(self.worker_id)
        }
