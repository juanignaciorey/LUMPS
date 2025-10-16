# Changelog

Todos los cambios notables a este proyecto serán documentados en este archivo.

El formato está basado en [Keep a Changelog](https://keepachangelog.com/es-ES/1.0.0/),
y este proyecto adhiere a [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Sistema de logging persistente en `generate_dataset_high_performance.py`
- Sistema de checkpoints para recuperación automática de generación de datasets
- Funcionalidad de continuación desde último checkpoint
- Logging estructurado con timestamps y niveles de log
- Archivos de checkpoint en formato pickle y JSON para legibilidad
- Directorios automáticos para logs y checkpoints
- Opciones de línea de comandos para control de checkpoints
- Integración completa con `src/utils/logger.py`
- Creada regla de consolidación de scripts para prevenir duplicación de archivos
- Script consolidado `generate_dataset.py` con múltiples modos (full, quick, debug)
- Preparación del proyecto para commit inicial de Fase 0
- Recreado generate_dataset_high_performance.py con funcionalidades avanzadas

### Changed
- Mejorado `generate_dataset_high_performance.py` con sistema de logging y checkpoints
- Agregadas nuevas opciones de línea de comandos para control de checkpoints
- Modificada función `evolve_ca_parallel` para soportar logging estructurado
- Actualizada configuración para incluir directorios de logs y checkpoints
- Consolidados scripts redundantes de generación de dataset en uno solo
- Eliminados archivos duplicados: `generate_dataset_simple.py`, `generate_dataset_consolidated.py`
- Mejorado sistema de categorización de cambios en changelog
- Limpiado proyecto eliminando archivos innecesarios para Fase 0
- Eliminados archivos de Fase 1 y versiones alternativas de scripts
- Removido directorio `dataset_phase0_hp` y archivos de alto rendimiento
- Eliminados scripts de análisis y comparación no esenciales
- Recreado generate_dataset_high_performance.py con sistema completo de alto rendimiento

### Documentation
- Agregados ejemplos de uso detallados en `generate_dataset_high_performance.py`
- Documentadas características del sistema de logging y checkpoints
- Incluida información sobre archivos generados y estructura de directorios
- Reorganizada estructura de documentación en directorio docs/
- Consolidados archivos de estado en formato estándar
- Creado directorio docs/guias/ para documentación técnica
- Corregida nomenclatura de archivos a snake_case
- Eliminados archivos redundantes y consolidada información
- Eliminados archivos temporales de estado
- Estructura final cumple completamente con estándares de documentación
- Creada documentación de configuración de editor
- Agregada guía completa de uso de configuraciones automáticas
- Consolidados archivos de documentación duplicados en estructura coherente
- Eliminados archivos de estado duplicados de raíz
- Creado estado_fase0 consolidado con información actualizada
- Creado mejoras_implementadas con optimizaciones del sistema
- Creado resumen_ejecutivo con estado general del proyecto
- Aplicadas reglas de documentación: snake_case, ubicación en docs/, formato UTF-8

### Technical
- Implementado sistema de checkpoints con pickle para serialización de datos
- Agregado sistema de logging estructurado con niveles INFO, WARNING, ERROR
- Creadas funciones de gestión de checkpoints con carga/guardado automático
- Implementada detección automática de checkpoints existentes
- Agregado sistema de directorios automáticos para logs y checkpoints
- Mejorada función de verificación de GPU con logging integrado
- Creadas reglas para nomenclatura snake_case de archivos
- Implementadas reglas para codificación UTF-8 y formato de líneas
- Agregadas reglas para estructura obligatoria de archivos de estado
- Creadas reglas para integración automática con changelog
- Inicializado repositorio Git con configuración automática
- Implementado sistema completo de alto rendimiento con procesamiento paralelo
- Agregado sistema de checkpoints con recuperación automática
- Implementado logging persistente con timestamps y niveles estructurados
- Creado sistema de limpieza automática de checkpoints antiguos
- Agregado procesamiento paralelo con ProcessPoolExecutor
- Implementado sistema de reanudación desde checkpoints existentes
- Configurado .gitignore para excluir datasets generados del control de versiones
- Eliminados archivos de dataset del staging area de Git

## [0.1.0] - 2024-01-20

### Added
- Proyecto LUMPS inicial
- Sistema de autómatas celulares
- Pipeline de entrenamiento
- Generación de datasets
- Sistema de diagnóstico y métricas
