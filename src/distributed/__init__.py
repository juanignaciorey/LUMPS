"""
Sistema distribuido para generación de dataset en Google Colab.

Este módulo proporciona herramientas para coordinar múltiples workers
de Google Colab trabajando en paralelo sobre Google Drive compartido.
"""

from .coordinator import BatchCoordinator
from .batch_generator import BatchGenerator
from .worker import DistributedWorker
from .aggregator import ResultAggregator
from .monitor import ProgressMonitor

__all__ = [
    'BatchCoordinator',
    'BatchGenerator',
    'DistributedWorker',
    'ResultAggregator',
    'ProgressMonitor'
]

