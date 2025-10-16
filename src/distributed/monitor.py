"""
Monitor de progreso para trabajo distribuido.

Dashboard CLI para ver estado del trabajo distribuido en tiempo real,
mostrando progreso, workers activos, y alertas de problemas.
"""

import json
import time
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from .coordinator import BatchCoordinator


class ProgressMonitor:
    """
    Monitor que muestra progreso del trabajo distribuido.

    Funcionalidades:
    - Dashboard en tiempo real
    - Estadísticas de workers
    - Alertas de problemas
    - Estimaciones de tiempo
    - Historial de progreso
    """

    def __init__(self, drive_path: str, refresh_interval: int = 30):
        """
        Inicializar monitor.

        Args:
            drive_path: Ruta base en Google Drive
            refresh_interval: Intervalo de actualización en segundos
        """
        self.drive_path = Path(drive_path)
        self.refresh_interval = refresh_interval
        self.coordinator = BatchCoordinator(str(drive_path))

        # Configurar logging
        self.logger = self._setup_logging()

        # Historial de progreso
        self.progress_history = []

        self.logger.info(f"Monitor inicializado: {drive_path}")

    def _setup_logging(self) -> logging.Logger:
        """Configurar logging para el monitor."""
        logs_dir = self.drive_path / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        log_file = logs_dir / f"monitor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        logger = logging.getLogger("monitor")
        logger.setLevel(logging.INFO)

        # Solo handler para archivo (no consola para evitar interferir con dashboard)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)

        return logger

    def get_current_status(self) -> Dict[str, Any]:
        """
        Obtener estado actual del trabajo distribuido.

        Returns:
            Diccionario con estado completo
        """
        try:
            # Obtener estadísticas del coordinador
            progress_stats = self.coordinator.get_progress_stats()

            # Obtener información de workers activos
            active_workers = self._get_active_workers_info()

            # Obtener información de batches recientes
            recent_batches = self._get_recent_batches_info()

            # Calcular métricas adicionales
            metrics = self._calculate_metrics(progress_stats)

            status = {
                'timestamp': datetime.now().isoformat(),
                'progress': progress_stats,
                'active_workers': active_workers,
                'recent_batches': recent_batches,
                'metrics': metrics
            }

            return status

        except Exception as e:
            self.logger.error(f"Error obteniendo estado: {e}")
            return {'error': str(e)}

    def _get_active_workers_info(self) -> List[Dict[str, Any]]:
        """Obtener información de workers activos."""
        active_workers = []

        try:
            manifest = self.coordinator.load_manifest()

            # Obtener workers únicos que tienen locks
            worker_ids = set()
            for batch in manifest['batches']:
                if batch['status'] == 'locked' and batch.get('locked_by'):
                    worker_ids.add(batch['locked_by'])

            # Obtener información detallada de cada worker
            for worker_id in worker_ids:
                worker_status = self.coordinator.get_worker_status(worker_id)

                # Obtener batch actual
                current_batch = None
                for batch in manifest['batches']:
                    if batch.get('locked_by') == worker_id and batch['status'] == 'locked':
                        current_batch = batch
                        break

                # Obtener información de logs
                log_info = self._get_worker_log_info(worker_id)

                worker_info = {
                    'worker_id': worker_id,
                    'status': worker_status,
                    'current_batch': current_batch,
                    'log_info': log_info
                }

                active_workers.append(worker_info)

        except Exception as e:
            self.logger.error(f"Error obteniendo información de workers: {e}")

        return active_workers

    def _get_recent_batches_info(self) -> List[Dict[str, Any]]:
        """Obtener información de batches recientes."""
        recent_batches = []

        try:
            manifest = self.coordinator.load_manifest()

            # Obtener batches completados recientemente (últimas 24 horas)
            cutoff_time = datetime.now() - timedelta(hours=24)

            for batch in manifest['batches']:
                if batch['status'] == 'completed' and batch.get('completed_at'):
                    try:
                        completed_at = datetime.fromisoformat(batch['completed_at'])
                        if completed_at > cutoff_time:
                            recent_batches.append(batch)
                    except:
                        continue

            # Ordenar por fecha de completado (más recientes primero)
            recent_batches.sort(key=lambda x: x.get('completed_at', ''), reverse=True)

            # Limitar a los 10 más recientes
            recent_batches = recent_batches[:10]

        except Exception as e:
            self.logger.error(f"Error obteniendo batches recientes: {e}")

        return recent_batches

    def _get_worker_log_info(self, worker_id: str) -> Dict[str, Any]:
        """Obtener información de logs de un worker."""
        log_info = {
            'log_exists': False,
            'last_activity': None,
            'error_count': 0,
            'last_error': None
        }

        try:
            log_file = self.drive_path / "logs" / f"worker_{worker_id}.log"

            if log_file.exists():
                log_info['log_exists'] = True

                # Leer últimas líneas del log
                with open(log_file, 'r') as f:
                    lines = f.readlines()

                if lines:
                    # Última actividad
                    last_line = lines[-1].strip()
                    log_info['last_activity'] = last_line

                    # Contar errores en las últimas 100 líneas
                    recent_lines = lines[-100:] if len(lines) > 100 else lines
                    error_count = sum(1 for line in recent_lines if 'ERROR' in line or '❌' in line)
                    log_info['error_count'] = error_count

                    # Último error
                    for line in reversed(recent_lines):
                        if 'ERROR' in line or '❌' in line:
                            log_info['last_error'] = line.strip()
                            break

        except Exception as e:
            self.logger.error(f"Error leyendo logs de worker {worker_id}: {e}")

        return log_info

    def _calculate_metrics(self, progress_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Calcular métricas adicionales."""
        metrics = {}

        try:
            total_batches = progress_stats['total_batches']
            completed_batches = progress_stats['completed']

            # Progreso porcentual
            if total_batches > 0:
                progress_pct = (completed_batches / total_batches) * 100
                metrics['progress_percentage'] = progress_pct

            # Velocidad de procesamiento
            if len(self.progress_history) >= 2:
                # Calcular batches completados por hora
                recent_history = self.progress_history[-10:]  # Últimas 10 mediciones

                if len(recent_history) >= 2:
                    time_diff = (datetime.now() - datetime.fromisoformat(recent_history[0]['timestamp'])).total_seconds() / 3600
                    batches_diff = recent_history[-1]['progress']['completed'] - recent_history[0]['progress']['completed']

                    if time_diff > 0:
                        batches_per_hour = batches_diff / time_diff
                        metrics['batches_per_hour'] = batches_per_hour

                        # Tiempo estimado de finalización
                        remaining_batches = total_batches - completed_batches
                        if batches_per_hour > 0:
                            hours_remaining = remaining_batches / batches_per_hour
                            metrics['estimated_completion_hours'] = hours_remaining
                            metrics['estimated_completion_time'] = (datetime.now() + timedelta(hours=hours_remaining)).isoformat()

            # Eficiencia de workers
            active_workers = progress_stats['workers_active']
            if active_workers > 0 and completed_batches > 0:
                metrics['batches_per_worker'] = completed_batches / active_workers

        except Exception as e:
            self.logger.error(f"Error calculando métricas: {e}")

        return metrics

    def display_dashboard(self, status: Dict[str, Any]) -> None:
        """Mostrar dashboard en consola."""
        if 'error' in status:
            print(f"❌ Error: {status['error']}")
            return

        # Limpiar pantalla
        os.system('clear' if os.name == 'posix' else 'cls')

        progress = status['progress']
        active_workers = status['active_workers']
        recent_batches = status['recent_batches']
        metrics = status['metrics']

        print("🚀 LUMPS - MONITOR DE PROGRESO DISTRIBUIDO")
        print("=" * 80)
        print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        # Progreso general
        print("📊 PROGRESO GENERAL:")
        total_batches = progress['total_batches']
        completed = progress['completed']
        available = progress['available']
        locked = progress['locked']
        failed = progress['failed']

        progress_pct = metrics.get('progress_percentage', 0)

        print(f"   Total batches: {total_batches}")
        print(f"   Completados: {completed} ({progress_pct:.1f}%)")
        print(f"   Disponibles: {available}")
        print(f"   En progreso: {locked}")
        print(f"   Fallidos: {failed}")
        print(f"   Tareas generadas: {progress['total_tasks_generated']}")
        print()

        # Barra de progreso visual
        bar_length = 50
        filled_length = int(bar_length * progress_pct / 100)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        print(f"   Progreso: [{bar}] {progress_pct:.1f}%")
        print()

        # Workers activos
        print("👥 WORKERS ACTIVOS:")
        if active_workers:
            for worker in active_workers:
                worker_id = worker['worker_id']
                current_batch = worker['current_batch']
                log_info = worker['log_info']

                print(f"   🔄 {worker_id}")
                if current_batch:
                    print(f"      Batch actual: {current_batch['batch_id']} ({current_batch['fitness_type']})")
                    print(f"      Tamaño: {current_batch['batch_size']} tareas")

                if log_info['log_exists']:
                    print(f"      Última actividad: {log_info['last_activity'][:60]}...")
                    if log_info['error_count'] > 0:
                        print(f"      ⚠️ Errores recientes: {log_info['error_count']}")
        else:
            print("   No hay workers activos")
        print()

        # Métricas de rendimiento
        print("📈 RENDIMIENTO:")
        if 'batches_per_hour' in metrics:
            print(f"   Velocidad: {metrics['batches_per_hour']:.1f} batches/hora")

        if 'estimated_completion_hours' in metrics:
            hours = metrics['estimated_completion_hours']
            if hours < 24:
                print(f"   Tiempo estimado: {hours:.1f} horas")
            else:
                days = hours / 24
                print(f"   Tiempo estimado: {days:.1f} días")

        if 'batches_per_worker' in metrics:
            print(f"   Eficiencia: {metrics['batches_per_worker']:.1f} batches/worker")
        print()

        # Batches recientes
        print("🕒 BATCHES RECIENTES:")
        if recent_batches:
            for batch in recent_batches[:5]:  # Mostrar solo los 5 más recientes
                batch_id = batch['batch_id']
                fitness_type = batch['fitness_type']
                completed_at = batch.get('completed_at', 'N/A')
                tasks_generated = batch.get('tasks_generated', 'N/A')

                print(f"   ✅ {batch_id} ({fitness_type}) - {tasks_generated} tareas - {completed_at}")
        else:
            print("   No hay batches completados recientemente")
        print()

        # Alertas
        alerts = self._check_alerts(status)
        if alerts:
            print("⚠️ ALERTAS:")
            for alert in alerts:
                print(f"   {alert}")
            print()

        print("=" * 80)
        print(f"🔄 Actualizando cada {self.refresh_interval} segundos... (Ctrl+C para salir)")

    def _check_alerts(self, status: Dict[str, Any]) -> List[str]:
        """Verificar alertas y problemas."""
        alerts = []

        try:
            progress = status['progress']
            active_workers = status['active_workers']

            # Alerta si no hay workers activos pero hay trabajo pendiente
            if progress['workers_active'] == 0 and progress['available'] > 0:
                alerts.append("No hay workers activos pero hay trabajo pendiente")

            # Alerta si hay muchos batches fallidos
            if progress['failed'] > 0:
                failed_pct = (progress['failed'] / progress['total_batches']) * 100
                if failed_pct > 5:  # Más del 5% fallidos
                    alerts.append(f"Alto porcentaje de batches fallidos: {failed_pct:.1f}%")

            # Alerta si workers tienen muchos errores
            for worker in active_workers:
                log_info = worker['log_info']
                if log_info['error_count'] > 10:  # Más de 10 errores recientes
                    alerts.append(f"Worker {worker['worker_id']} tiene muchos errores: {log_info['error_count']}")

            # Alerta si el progreso está muy lento
            metrics = status['metrics']
            if 'batches_per_hour' in metrics and metrics['batches_per_hour'] < 0.5:
                alerts.append("Progreso muy lento: menos de 0.5 batches/hora")

        except Exception as e:
            self.logger.error(f"Error verificando alertas: {e}")

        return alerts

    def run_monitor(self, continuous: bool = True) -> None:
        """
        Ejecutar monitor en modo continuo o una sola vez.

        Args:
            continuous: Si True, ejecutar en loop continuo
        """
        try:
            if continuous:
                print("🚀 Iniciando monitor continuo...")
                print("Presiona Ctrl+C para detener")
                print()

                while True:
                    status = self.get_current_status()
                    self.display_dashboard(status)

                    # Guardar en historial
                    self.progress_history.append(status)
                    if len(self.progress_history) > 100:  # Mantener solo últimas 100 mediciones
                        self.progress_history = self.progress_history[-100:]

                    time.sleep(self.refresh_interval)
            else:
                status = self.get_current_status()
                self.display_dashboard(status)

        except KeyboardInterrupt:
            print("\n🛑 Monitor detenido por usuario")
        except Exception as e:
            print(f"\n💥 Error en monitor: {e}")
            self.logger.error(f"Error en monitor: {e}")

    def save_progress_report(self, output_file: Optional[str] = None) -> str:
        """
        Guardar reporte de progreso en archivo.

        Args:
            output_file: Archivo de salida (opcional)

        Returns:
            Ruta del archivo guardado
        """
        if output_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = self.drive_path / "logs" / f"progress_report_{timestamp}.json"

        status = self.get_current_status()

        report = {
            'generated_at': datetime.now().isoformat(),
            'current_status': status,
            'progress_history': self.progress_history[-50:]  # Últimas 50 mediciones
        }

        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)

        self.logger.info(f"Reporte de progreso guardado: {output_file}")
        return str(output_file)


def main():
    """Función principal para ejecutar monitor desde línea de comandos."""
    import argparse

    parser = argparse.ArgumentParser(description='Monitor de progreso para trabajo distribuido')
    parser.add_argument('--drive-path', required=True,
                       help='Ruta base en Google Drive')
    parser.add_argument('--refresh-interval', type=int, default=30,
                       help='Intervalo de actualización en segundos (default: 30)')
    parser.add_argument('--once', action='store_true',
                       help='Mostrar estado una sola vez (no continuo)')
    parser.add_argument('--save-report', action='store_true',
                       help='Guardar reporte de progreso')

    args = parser.parse_args()

    # Crear y ejecutar monitor
    monitor = ProgressMonitor(args.drive_path, args.refresh_interval)

    if args.save_report:
        report_file = monitor.save_progress_report()
        print(f"📄 Reporte guardado: {report_file}")

    monitor.run_monitor(continuous=not args.once)


if __name__ == "__main__":
    main()
