"""
Coordinador de trabajo distribuido para Google Colab.

Maneja el sistema de locks basado en archivos JSON en Google Drive
para coordinar múltiples workers trabajando en paralelo.
"""

import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class BatchCoordinator:
    """
    Coordinador que maneja locks y estados de batches en Google Drive.

    Proporciona funciones para:
    - Claim un batch disponible
    - Marcar batch como completado
    - Detectar y liberar locks expirados
    - Obtener estado general del trabajo
    """

    def __init__(self, drive_path: str, lock_timeout_hours: int = 4):
        """
        Inicializar coordinador.

        Args:
            drive_path: Ruta base en Google Drive donde están los archivos
            lock_timeout_hours: Horas después de las cuales un lock expira
        """
        self.drive_path = Path(drive_path)
        self.lock_timeout = timedelta(hours=lock_timeout_hours)

        # Crear directorios necesarios
        self.manifests_dir = self.drive_path / "manifests"
        self.locks_dir = self.drive_path / "locks"
        self.results_dir = self.drive_path / "results"
        self.logs_dir = self.drive_path / "logs"

        for dir_path in [self.manifests_dir, self.locks_dir, self.results_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        self.manifest_file = self.manifests_dir / "batch_manifest.json"

    def load_manifest(self) -> Dict:
        """Cargar manifiesto de batches."""
        if not self.manifest_file.exists():
            raise FileNotFoundError(f"Manifiesto no encontrado: {self.manifest_file}")

        with open(self.manifest_file, 'r') as f:
            return json.load(f)

    def save_manifest(self, manifest: Dict) -> None:
        """Guardar manifiesto de batches."""
        with open(self.manifest_file, 'w') as f:
            json.dump(manifest, f, indent=2)

    def cleanup_expired_locks(self) -> List[str]:
        """
        Limpiar locks expirados y marcar batches como disponibles.

        Returns:
            Lista de batch_ids que fueron liberados
        """
        manifest = self.load_manifest()
        now = datetime.now()
        freed_batches = []

        for batch in manifest['batches']:
            if batch['status'] == 'locked':
                lock_file = self.locks_dir / f"{batch['batch_id']}.lock"

                if lock_file.exists():
                    try:
                        with open(lock_file, 'r') as f:
                            lock_data = json.load(f)

                        locked_at = datetime.fromisoformat(lock_data['locked_at'])

                        if now - locked_at > self.lock_timeout:
                            # Lock expirado, liberar batch
                            batch['status'] = 'available'
                            batch['locked_by'] = None
                            batch['locked_at'] = None
                            batch['attempts'] = batch.get('attempts', 0) + 1

                            # Eliminar archivo de lock
                            lock_file.unlink()

                            freed_batches.append(batch['batch_id'])
                            logger.info(f"Lock expirado liberado: {batch['batch_id']}")

                    except Exception as e:
                        logger.error(f"Error procesando lock {lock_file}: {e}")
                        # Si hay error, liberar el batch de todas formas
                        batch['status'] = 'available'
                        batch['locked_by'] = None
                        batch['locked_at'] = None
                        freed_batches.append(batch['batch_id'])

        if freed_batches:
            self.save_manifest(manifest)
            logger.info(f"Liberados {len(freed_batches)} batches por locks expirados")

        return freed_batches

    def claim_batch(self, worker_id: str) -> Optional[Dict]:
        """
        Claim un batch disponible para procesamiento.

        Args:
            worker_id: ID único del worker

        Returns:
            Diccionario con información del batch claimado, o None si no hay disponibles
        """
        # Limpiar locks expirados primero
        self.cleanup_expired_locks()

        manifest = self.load_manifest()
        now = datetime.now().isoformat()

        # Buscar batch disponible
        for batch in manifest['batches']:
            if batch['status'] == 'available':
                # Verificar que no haya sido claimado por otro worker
                # (doble verificación para evitar race conditions)
                if batch.get('locked_by') is None:
                    # Crear lock
                    lock_data = {
                        'worker_id': worker_id,
                        'locked_at': now,
                        'expires_at': (datetime.now() + self.lock_timeout).isoformat(),
                        'status': 'in_progress'
                    }

                    lock_file = self.locks_dir / f"{batch['batch_id']}.lock"

                    try:
                        # Escribir lock file
                        with open(lock_file, 'w') as f:
                            json.dump(lock_data, f, indent=2)

                        # Actualizar manifiesto
                        batch['status'] = 'locked'
                        batch['locked_by'] = worker_id
                        batch['locked_at'] = now

                        self.save_manifest(manifest)

                        logger.info(f"Batch {batch['batch_id']} claimado por {worker_id}")
                        return batch

                    except Exception as e:
                        logger.error(f"Error creando lock para {batch['batch_id']}: {e}")
                        # Limpiar lock file si existe
                        if lock_file.exists():
                            lock_file.unlink()
                        continue

        logger.info("No hay batches disponibles")
        return None

    def complete_batch(self, batch_id: str, worker_id: str,
                      tasks: List[Dict], stats: Dict) -> bool:
        """
        Marcar batch como completado y guardar resultados.

        Args:
            batch_id: ID del batch completado
            worker_id: ID del worker que completó el batch
            tasks: Lista de tareas generadas
            stats: Estadísticas del batch

        Returns:
            True si se completó exitosamente
        """
        manifest = self.load_manifest()
        now = datetime.now().isoformat()

        # Buscar batch en manifiesto
        batch = None
        for b in manifest['batches']:
            if b['batch_id'] == batch_id:
                batch = b
                break

        if not batch:
            logger.error(f"Batch {batch_id} no encontrado en manifiesto")
            return False

        # Verificar que el worker tiene el lock
        if batch.get('locked_by') != worker_id:
            logger.error(f"Worker {worker_id} no tiene lock en batch {batch_id}")
            return False

        try:
            # Crear directorio de resultados
            batch_result_dir = self.results_dir / batch_id
            batch_result_dir.mkdir(exist_ok=True)

            # Guardar tareas
            tasks_file = batch_result_dir / "tasks.json"
            with open(tasks_file, 'w') as f:
                json.dump(tasks, f, indent=2)

            # Guardar estadísticas
            stats_file = batch_result_dir / "stats.json"
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2)

            # Actualizar manifiesto
            batch['status'] = 'completed'
            batch['completed_at'] = now
            batch['tasks_generated'] = len(tasks)

            # Eliminar lock file
            lock_file = self.locks_dir / f"{batch_id}.lock"
            if lock_file.exists():
                lock_file.unlink()

            self.save_manifest(manifest)

            logger.info(f"Batch {batch_id} completado por {worker_id}: {len(tasks)} tareas")
            return True

        except Exception as e:
            logger.error(f"Error completando batch {batch_id}: {e}")
            return False

    def fail_batch(self, batch_id: str, worker_id: str, error_msg: str) -> bool:
        """
        Marcar batch como fallido.

        Args:
            batch_id: ID del batch que falló
            worker_id: ID del worker que reportó el fallo
            error_msg: Mensaje de error

        Returns:
            True si se marcó exitosamente
        """
        manifest = self.load_manifest()

        # Buscar batch en manifiesto
        batch = None
        for b in manifest['batches']:
            if b['batch_id'] == batch_id:
                batch = b
                break

        if not batch:
            logger.error(f"Batch {batch_id} no encontrado en manifiesto")
            return False

        # Verificar que el worker tiene el lock
        if batch.get('locked_by') != worker_id:
            logger.error(f"Worker {worker_id} no tiene lock en batch {batch_id}")
            return False

        # Incrementar intentos
        attempts = batch.get('attempts', 0) + 1

        if attempts >= 3:
            # Marcar como fallido permanentemente
            batch['status'] = 'failed'
            batch['failed_at'] = datetime.now().isoformat()
            batch['error_msg'] = error_msg
            logger.warning(f"Batch {batch_id} marcado como fallido permanentemente")
        else:
            # Liberar para reintento
            batch['status'] = 'available'
            batch['locked_by'] = None
            batch['locked_at'] = None
            batch['attempts'] = attempts
            logger.info(f"Batch {batch_id} liberado para reintento (intento {attempts}/3)")

        # Eliminar lock file
        lock_file = self.locks_dir / f"{batch_id}.lock"
        if lock_file.exists():
            lock_file.unlink()

        self.save_manifest(manifest)
        return True

    def get_progress_stats(self) -> Dict:
        """
        Obtener estadísticas de progreso del trabajo.

        Returns:
            Diccionario con estadísticas de progreso
        """
        manifest = self.load_manifest()

        stats = {
            'total_batches': len(manifest['batches']),
            'available': 0,
            'locked': 0,
            'completed': 0,
            'failed': 0,
            'total_tasks_generated': 0,
            'workers_active': set(),
            'estimated_completion': None
        }

        for batch in manifest['batches']:
            status = batch['status']
            stats[status] += 1

            if status == 'completed':
                stats['total_tasks_generated'] += batch.get('tasks_generated', 0)
            elif status == 'locked':
                stats['workers_active'].add(batch.get('locked_by'))

        stats['workers_active'] = len(stats['workers_active'])

        # Calcular tiempo estimado de finalización
        if stats['available'] > 0 and stats['workers_active'] > 0:
            # Asumir 1.5 horas por batch en promedio
            remaining_time_hours = (stats['available'] / stats['workers_active']) * 1.5
            stats['estimated_completion_hours'] = remaining_time_hours

        return stats

    def get_worker_status(self, worker_id: str) -> Dict:
        """
        Obtener estado de un worker específico.

        Args:
            worker_id: ID del worker

        Returns:
            Diccionario con estado del worker
        """
        manifest = self.load_manifest()

        status = {
            'worker_id': worker_id,
            'current_batch': None,
            'batches_completed': 0,
            'batches_failed': 0,
            'total_tasks_generated': 0
        }

        for batch in manifest['batches']:
            if batch.get('locked_by') == worker_id and batch['status'] == 'locked':
                status['current_batch'] = batch['batch_id']
            elif batch.get('locked_by') == worker_id and batch['status'] == 'completed':
                status['batches_completed'] += 1
                status['total_tasks_generated'] += batch.get('tasks_generated', 0)
            elif batch.get('locked_by') == worker_id and batch['status'] == 'failed':
                status['batches_failed'] += 1

        return status
