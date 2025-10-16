#!/usr/bin/env python3
"""
Script principal para generación distribuida de dataset en Google Colab.

Este script coordina múltiples workers trabajando en paralelo sobre
Google Drive compartido para generar el dataset de la Fase 0.

Uso:
    python generate_dataset_distributed.py --drive-path "/content/drive/MyDrive/LUMPS_Distributed" --worker-id colab_worker_1
"""

import sys
import argparse
import os
import socket
import time
from pathlib import Path
from typing import Optional
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.distributed.worker import DistributedWorker
from src.distributed.batch_generator import BatchGenerator
from src.distributed.coordinator import BatchCoordinator


def setup_logging(level: str = "INFO") -> None:
    """Configurar logging básico."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def generate_worker_id(prefix: str = "colab") -> str:
    """
    Generar ID único para el worker.

    Args:
        prefix: Prefijo para el ID

    Returns:
        ID único del worker
    """
    hostname = socket.gethostname()
    pid = os.getpid()
    timestamp = int(time.time())

    return f"{prefix}_{hostname}_{pid}_{timestamp}"


def check_drive_connection(drive_path: str) -> bool:
    """
    Verificar conexión a Google Drive.

    Args:
        drive_path: Ruta del Drive

    Returns:
        True si la conexión es válida
    """
    drive_path_obj = Path(drive_path)

    if not drive_path_obj.exists():
        print(f"❌ Error: La ruta del Drive no existe: {drive_path}")
        return False

    if not drive_path_obj.is_dir():
        print(f"❌ Error: La ruta del Drive no es un directorio: {drive_path}")
        return False

    # Verificar que podemos escribir en el directorio
    try:
        test_file = drive_path_obj / "test_write.tmp"
        test_file.write_text("test")
        test_file.unlink()
        print(f"✅ Conexión a Drive verificada: {drive_path}")
        return True
    except Exception as e:
        print(f"❌ Error: No se puede escribir en Drive: {e}")
        return False


def check_manifest_exists(drive_path: str) -> bool:
    """
    Verificar que existe el manifiesto de batches.

    Args:
        drive_path: Ruta del Drive

    Returns:
        True si el manifiesto existe
    """
    manifest_file = Path(drive_path) / "manifests" / "batch_manifest.json"

    if not manifest_file.exists():
        print(f"❌ Error: Manifiesto no encontrado: {manifest_file}")
        print("💡 Ejecuta primero: python -m src.distributed.batch_generator --create-manifest --drive-path <ruta>")
        return False

    print(f"✅ Manifiesto encontrado: {manifest_file}")
    return True


def show_worker_info(worker_id: str, drive_path: str) -> None:
    """Mostrar información del worker."""
    print("\n" + "="*80)
    print("🚀 LUMPS - WORKER DISTRIBUIDO")
    print("="*80)
    print(f"Worker ID: {worker_id}")
    print(f"Drive Path: {drive_path}")
    print(f"Hostname: {socket.gethostname()}")
    print(f"PID: {os.getpid()}")
    print(f"Python: {sys.version}")
    print("="*80)


def show_progress_summary(drive_path: str) -> None:
    """Mostrar resumen de progreso general."""
    try:
        coordinator = BatchCoordinator(drive_path)
        stats = coordinator.get_progress_stats()

        print(f"\n📊 RESUMEN DE PROGRESO:")
        print(f"   Total batches: {stats['total_batches']}")
        print(f"   Disponibles: {stats['available']}")
        print(f"   En progreso: {stats['locked']}")
        print(f"   Completados: {stats['completed']}")
        print(f"   Fallidos: {stats['failed']}")
        print(f"   Workers activos: {stats['workers_active']}")
        print(f"   Tareas generadas: {stats['total_tasks_generated']}")

        if stats.get('estimated_completion_hours'):
            print(f"   Tiempo estimado: {stats['estimated_completion_hours']:.1f} horas")

        completion_pct = (stats['completed'] / stats['total_batches']) * 100
        print(f"   Progreso: {completion_pct:.1f}%")

    except Exception as e:
        print(f"⚠️ Error obteniendo estadísticas: {e}")


def run_worker(worker_id: str, drive_path: str, max_batches: Optional[int] = None,
               sleep_seconds: int = 30, verbose: bool = False) -> None:
    """
    Ejecutar worker distribuido.

    Args:
        worker_id: ID único del worker
        drive_path: Ruta del Google Drive
        max_batches: Número máximo de batches a procesar
        sleep_seconds: Segundos a esperar entre intentos
        verbose: Mostrar logs detallados
    """
    # Configurar logging
    log_level = "DEBUG" if verbose else "INFO"
    setup_logging(log_level)

    # Verificaciones previas
    if not check_drive_connection(drive_path):
        return

    if not check_manifest_exists(drive_path):
        return

    # Mostrar información
    show_worker_info(worker_id, drive_path)
    show_progress_summary(drive_path)

    # Crear y ejecutar worker
    try:
        worker = DistributedWorker(worker_id, drive_path)

        print(f"\n🔄 Iniciando worker...")
        print(f"   Max batches: {max_batches or 'infinito'}")
        print(f"   Sleep entre intentos: {sleep_seconds}s")
        print(f"   Presiona Ctrl+C para detener")
        print("\n" + "-"*80)

        # Ejecutar worker
        final_stats = worker.run(max_batches=max_batches, sleep_seconds=sleep_seconds)

        print("\n" + "="*80)
        print("🏁 WORKER FINALIZADO")
        print("="*80)
        print(f"Batches procesados: {final_stats['batches_processed']}")
        print(f"Tareas generadas: {final_stats['tasks_generated']}")
        print(f"Errores: {final_stats['errors']}")

        elapsed_time = time.time() - final_stats['start_time']
        print(f"Tiempo total: {elapsed_time/3600:.1f} horas")

        if final_stats['batches_processed'] > 0:
            avg_time = elapsed_time / final_stats['batches_processed']
            print(f"Tiempo promedio por batch: {avg_time/60:.1f} minutos")

        print("="*80)

    except KeyboardInterrupt:
        print("\n🛑 Worker interrumpido por usuario")
    except Exception as e:
        print(f"\n💥 Error fatal en worker: {e}")
        import traceback
        traceback.print_exc()


def create_manifest(drive_path: str, batch_size: int = 5) -> None:
    """
    Crear manifiesto de batches.

    Args:
        drive_path: Ruta del Google Drive
        batch_size: Tamaño de cada batch
    """
    print("📋 Creando manifiesto de batches...")

    try:
        generator = BatchGenerator(drive_path, batch_size)
        manifest_file = generator.create_manifest()

        print(f"✅ Manifiesto creado: {manifest_file}")
        generator.print_summary()

    except Exception as e:
        print(f"❌ Error creando manifiesto: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Función principal."""
    parser = argparse.ArgumentParser(
        description='Worker distribuido para generación de dataset en Google Colab',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:

1. Crear manifiesto inicial:
   python generate_dataset_distributed.py --create-manifest --drive-path "/content/drive/MyDrive/LUMPS_Distributed"

2. Ejecutar worker:
   python generate_dataset_distributed.py --drive-path "/content/drive/MyDrive/LUMPS_Distributed" --worker-id colab_worker_1

3. Ejecutar worker con límite de batches:
   python generate_dataset_distributed.py --drive-path "/content/drive/MyDrive/LUMPS_Distributed" --max-batches 10

4. Ejecutar worker con logs detallados:
   python generate_dataset_distributed.py --drive-path "/content/drive/MyDrive/LUMPS_Distributed" --verbose
        """
    )

    # Argumentos principales
    parser.add_argument('--drive-path', required=True,
                       help='Ruta base en Google Drive (ej: /content/drive/MyDrive/LUMPS_Distributed)')
    parser.add_argument('--worker-id',
                       help='ID único del worker (se genera automáticamente si no se especifica)')

    # Opciones de ejecución
    parser.add_argument('--max-batches', type=int,
                       help='Número máximo de batches a procesar (default: infinito)')
    parser.add_argument('--sleep-seconds', type=int, default=30,
                       help='Segundos a esperar entre intentos de claim (default: 30)')
    parser.add_argument('--verbose', action='store_true',
                       help='Mostrar logs detallados')

    # Opciones de setup
    parser.add_argument('--create-manifest', action='store_true',
                       help='Crear manifiesto de batches')
    parser.add_argument('--batch-size', type=int, default=5,
                       help='Tamaño de cada batch para crear manifiesto (default: 5)')

    args = parser.parse_args()

    # Crear manifiesto si se solicita
    if args.create_manifest:
        create_manifest(args.drive_path, args.batch_size)
        return

    # Generar worker ID si no se especifica
    worker_id = args.worker_id or generate_worker_id()

    # Ejecutar worker
    run_worker(
        worker_id=worker_id,
        drive_path=args.drive_path,
        max_batches=args.max_batches,
        sleep_seconds=args.sleep_seconds,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()
