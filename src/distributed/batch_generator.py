"""
Generador de batches para trabajo distribuido.

Subdivide el trabajo de generación de dataset en batches pequeños
que pueden ser procesados por workers individuales en Google Colab.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)


class BatchGenerator:
    """
    Generador que subdivide el trabajo en batches pequeños.

    Divide cada fitness function en sub-batches de 50 tareas,
    creando un total de ~1,000 batches para distribuir entre workers.
    """

    def __init__(self, drive_path: str, batch_size: int = 50):
        """
        Inicializar generador de batches.

        Args:
            drive_path: Ruta base en Google Drive
            batch_size: Número de tareas por batch (default: 50 para 500k tareas)
        """
        self.drive_path = Path(drive_path)
        self.batch_size = batch_size

        # Configuración de fitness functions
        self.fitness_types = [
            # Funciones originales
            'expand', 'symmetry', 'count', 'topology', 'replicate',
            # Funciones cognitivas
            'pattern_match', 'transform_consistency', 'compression', 'analogy', 'objectness',
            # Funciones estructurales
            'rule_entropy', 'invariance', 'relational_distance', 'causal_score',
            # Funciones de coherencia
            'divergence_penalty', 'compositionality'
        ]

        # Configuración de evolución por fitness function
        self.tasks_per_fitness = 31250  # Total de tareas por fitness function (500k/16)
        self.estimated_time_per_task = 18  # minutos por tarea (estimado)

    def generate_manifest(self) -> Dict[str, Any]:
        """
        Generar manifiesto con todos los batches.

        Returns:
            Diccionario con manifiesto completo
        """
        batches = []
        batch_counter = 1

        for fitness_type in self.fitness_types:
            # Dividir tareas de esta fitness function en batches
            num_batches_for_fitness = (self.tasks_per_fitness + self.batch_size - 1) // self.batch_size

            for i in range(num_batches_for_fitness):
                start_idx = i * self.batch_size
                end_idx = min(start_idx + self.batch_size, self.tasks_per_fitness)
                actual_batch_size = end_idx - start_idx

                batch_id = f"batch_{batch_counter:04d}"

                batch = {
                    'batch_id': batch_id,
                    'fitness_type': fitness_type,
                    'start_task_idx': start_idx,
                    'end_task_idx': end_idx,
                    'batch_size': actual_batch_size,
                    'status': 'available',
                    'locked_by': None,
                    'locked_at': None,
                    'completed_at': None,
                    'attempts': 0,
                    'estimated_time_minutes': actual_batch_size * self.estimated_time_per_task,
                    'priority': self._get_fitness_priority(fitness_type)
                }

                batches.append(batch)
                batch_counter += 1

        manifest = {
            'created_at': self._get_timestamp(),
            'total_batches': len(batches),
            'total_tasks': sum(b['batch_size'] for b in batches),
            'fitness_types': self.fitness_types,
            'tasks_per_fitness': self.tasks_per_fitness,
            'batch_size': self.batch_size,
            'estimated_total_time_hours': sum(b['estimated_time_minutes'] for b in batches) / 60,
            'batches': batches
        }

        return manifest

    def _get_fitness_priority(self, fitness_type: str) -> int:
        """
        Obtener prioridad de una fitness function.

        Args:
            fitness_type: Tipo de fitness function

        Returns:
            Prioridad (1 = más alta, 5 = más baja)
        """
        # Priorizar funciones originales y cognitivas
        if fitness_type in ['expand', 'symmetry', 'count', 'topology', 'replicate']:
            return 1  # Funciones originales - alta prioridad
        elif fitness_type in ['pattern_match', 'transform_consistency', 'compression', 'analogy', 'objectness']:
            return 2  # Funciones cognitivas - alta prioridad
        elif fitness_type in ['rule_entropy', 'invariance', 'relational_distance', 'causal_score']:
            return 3  # Funciones estructurales - media prioridad
        else:
            return 4  # Funciones de coherencia - baja prioridad

    def _get_timestamp(self) -> str:
        """Obtener timestamp actual en formato ISO."""
        from datetime import datetime
        return datetime.now().isoformat()

    def save_manifest(self, manifest: Dict[str, Any]) -> Path:
        """
        Guardar manifiesto en Google Drive.

        Args:
            manifest: Manifiesto a guardar

        Returns:
            Ruta del archivo guardado
        """
        manifests_dir = self.drive_path / "manifests"
        manifests_dir.mkdir(parents=True, exist_ok=True)

        manifest_file = manifests_dir / "batch_manifest.json"

        with open(manifest_file, 'w') as f:
            json.dump(manifest, f, indent=2)

        logger.info(f"Manifiesto guardado: {manifest_file}")
        logger.info(f"Total batches: {manifest['total_batches']}")
        logger.info(f"Total tareas: {manifest['total_tasks']}")
        logger.info(f"Tiempo estimado: {manifest['estimated_total_time_hours']:.1f} horas")

        return manifest_file

    def create_manifest(self) -> Path:
        """
        Crear y guardar manifiesto completo.

        Returns:
            Ruta del archivo de manifiesto creado
        """
        logger.info("Generando manifiesto de batches...")

        manifest = self.generate_manifest()
        manifest_file = self.save_manifest(manifest)

        # Crear directorios necesarios
        self._create_directories()

        logger.info("Manifiesto creado exitosamente")
        return manifest_file

    def _create_directories(self) -> None:
        """Crear directorios necesarios en Google Drive."""
        directories = [
            "locks",
            "results",
            "logs",
            "final"
        ]

        for dir_name in directories:
            dir_path = self.drive_path / dir_name
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Directorio creado: {dir_path}")

    def get_batch_summary(self) -> Dict[str, Any]:
        """
        Obtener resumen de distribución de batches.

        Returns:
            Diccionario con resumen de batches por fitness type
        """
        manifest = self.generate_manifest()

        summary = {}
        for batch in manifest['batches']:
            fitness_type = batch['fitness_type']
            if fitness_type not in summary:
                summary[fitness_type] = {
                    'batches': 0,
                    'total_tasks': 0,
                    'estimated_time_hours': 0,
                    'priority': batch['priority']
                }

            summary[fitness_type]['batches'] += 1
            summary[fitness_type]['total_tasks'] += batch['batch_size']
            summary[fitness_type]['estimated_time_hours'] += batch['estimated_time_minutes'] / 60

        return summary

    def print_summary(self) -> None:
        """Imprimir resumen de distribución de batches."""
        summary = self.get_batch_summary()

        print("\n" + "="*80)
        print("RESUMEN DE DISTRIBUCIÓN DE BATCHES")
        print("="*80)

        total_batches = 0
        total_tasks = 0
        total_time = 0

        for fitness_type, stats in summary.items():
            print(f"\n{fitness_type.upper()}:")
            print(f"  Batches: {stats['batches']}")
            print(f"  Tareas: {stats['total_tasks']}")
            print(f"  Tiempo estimado: {stats['estimated_time_hours']:.1f} horas")
            print(f"  Prioridad: {stats['priority']}")

            total_batches += stats['batches']
            total_tasks += stats['total_tasks']
            total_time += stats['estimated_time_hours']

        print(f"\n{'='*80}")
        print(f"TOTALES:")
        print(f"  Batches: {total_batches}")
        print(f"  Tareas: {total_tasks}")
        print(f"  Tiempo estimado: {total_time:.1f} horas ({total_time/24:.1f} días)")
        print(f"  Con 5 workers: {total_time/5:.1f} horas por worker ({total_time/5/24:.1f} días)")
        print("="*80)


def main():
    """Función principal para crear manifiesto desde línea de comandos."""
    parser = argparse.ArgumentParser(description='Generar manifiesto de batches para trabajo distribuido')
    parser.add_argument('--drive-path', required=True,
                       help='Ruta base en Google Drive (ej: /content/drive/MyDrive/LUMPS_Distributed)')
    parser.add_argument('--batch-size', type=int, default=5,
                       help='Número de tareas por batch (default: 5)')
    parser.add_argument('--create-manifest', action='store_true',
                       help='Crear manifiesto de batches')
    parser.add_argument('--summary', action='store_true',
                       help='Mostrar resumen de distribución')

    args = parser.parse_args()

    # Configurar logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    generator = BatchGenerator(args.drive_path, args.batch_size)

    if args.create_manifest:
        manifest_file = generator.create_manifest()
        print(f"\n✅ Manifiesto creado: {manifest_file}")

    if args.summary:
        generator.print_summary()


if __name__ == "__main__":
    main()
