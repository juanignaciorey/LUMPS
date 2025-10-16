# LUMPS: Meta-Learning System for ARC-AGI 2025

## Fase 0: Genesis - Computational Lumps Discovery

Este proyecto implementa la Fase 0 del sistema LUMPS, que evoluciona autómatas celulares para descubrir "computational lumps" y generar tareas sintéticas en formato ARC.

## Estructura del Proyecto

```
LUMPS/
├── src/
│   ├── cellular_automata/     # Evolución de autómatas celulares
│   ├── training/              # Arquitectura del modelo y entrenamiento
│   ├── diagnostics/           # Métricas de meta-aprendizaje
│   └── utils/                 # Utilidades y configuración
├── configs/                   # Archivos de configuración
├── data/                      # Datos generados y modelos
├── tests/                     # Tests unitarios
└── requirements.txt           # Dependencias
```

## Instalación

1. Clonar el repositorio:
```bash
git clone <repository-url>
cd LUMPS
```

2. Instalar dependencias:
```bash
pip install -r requirements.txt
```

3. Configurar Weights & Biases (opcional):
```bash
wandb login
```

## Uso Rápido

### Estudio Piloto
Para ejecutar un estudio piloto pequeño (100 CAs):
```python
from src.training.phase0_pipeline import Phase0Pipeline

pipeline = Phase0Pipeline(
    config_path="configs/phase0.yaml",
    output_dir="data",
    seed=42
)

results = pipeline.run_pilot_study(num_cas=100)
```

### Pipeline Completo
Para ejecutar la Fase 0 completa:
```python
from src.training.phase0_pipeline import Phase0Pipeline

pipeline = Phase0Pipeline(
    config_path="configs/phase0.yaml",
    output_dir="data",
    seed=42
)

results = pipeline.run_full_pipeline()
```

## Componentes Principales

### 1. Autómatas Celulares (`src/cellular_automata/`)
- **core.py**: Lógica base de autómatas celulares
- **evolution.py**: Motor evolutivo genético
- **fitness.py**: 5 funciones de fitness (expand, symmetry, count, topology, replicate)
- **task_generator.py**: Conversión CA → formato ARC

### 2. Modelo y Entrenamiento (`src/training/`)
- **model.py**: LUMPSTransformer con heads especializados
- **trainer.py**: Entrenamiento energy-based con aprendizaje contrastivo
- **data_loader.py**: Carga de datos con PyTorch
- **phase0_pipeline.py**: Orquestador completo

### 3. Diagnósticos (`src/diagnostics/`)
- **metrics.py**: Métricas de meta-aprendizaje
- **visualizer.py**: Visualización de resultados

## Configuración

El archivo `configs/phase0.yaml` contiene todos los hiperparámetros:

```yaml
evolution:
  grid_size: 15
  population_size: 100
  generations: 1000
  tasks_per_fitness: 10000

model:
  d_model: 512
  n_layers: 12
  n_heads: 8
  max_grid_size: 30

training:
  batch_size: 32
  learning_rate: 1e-4
  epochs: 50
  num_candidates: 30
```

## Criterios de Éxito Fase 0

- ✅ **Transfer gap < 25%**: Generalización entre distribuciones
- ✅ **Lump diversity > 500**: Diversidad de primitivas computacionales
- ✅ **Size generalization > 70%**: Generalización a diferentes tamaños
- ✅ **Primitive emergence > 60%**: Emergencia de primitivas cognitivas

## Tests

Ejecutar tests:
```bash
python -m pytest tests/ -v
```

## Logs y Monitoreo

- Los logs se guardan en `data/logs/`
- Métricas de entrenamiento en Weights & Biases
- Checkpoints del modelo en `data/checkpoints/`

## Próximos Pasos

Después de completar exitosamente la Fase 0, el sistema estará listo para:
- **Fase 1**: Entrenamiento en primitivas de conocimiento
- **Fase 2**: Composición de primitivas
- **Fase 3**: Inmersión en dataset ARC oficial
- **Fase 4**: Optimización en tiempo de prueba

## Referencias

- [PRD del Proyecto](project_lumps_prd.md)
- [ARC Prize](https://arcprize.org)
- [Computational Irreducibility](https://writings.stephenwolfram.com/2021/11/the-concept-of-the-ruliad/)
