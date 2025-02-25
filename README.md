# Artemisa

Módulo de Python especializado en la extracción y procesamiento de información desde múltiples formatos de documentos (Excel, PowerPoint, Word y PDF).

## Descripción General

Artemisa proporciona una interfaz unificada para extraer y analizar contenido de diversos tipos de documentos, aprovechando diferentes modelos de inteligencia artificial tanto en la nube como localmente.

## Estructura del Proyecto

```
Artemisa/
├── test/           # Ejemplos y documentación de uso
│   └── ollamatest.py   # Guía de implementación con Ollama
└── ...
```

## Instalación via Pypi

`pip install Artemisa`

## Características Principales

### Integración con APIs de IA

El módulo integra múltiples proveedores de IA para el procesamiento y consulta de documentos:

#### Proveedores en la Nube

##### OpenAI

- Excelente compatibilidad con modelos no razonadores
- En desarrollo: Soporte expandido para más modelos de OpenAI

##### Deep Seek R1 (HuggingFace)

- Compatible con el modelo `DeepSeek-R1-Distill-Qwen-32B`
- Disponible mientras HuggingFace mantenga su API de inferencia gratuita

##### Google Gemini

- Alta compatibilidad con modelos no razonadores
- Pendiente: Validación con modelos razonadores

##### HuggingFace Client

- Soporte robusto para modelos de generación de texto
- Nota: Algunos modelos requieren suscripción Pro para acceso API

##### Anthropic

- Compatibilidad básica
- Estado: Pendiente de pruebas exhaustivas

### Procesamiento Local

#### Ollama

- Versión estable disponible
- Documentación detallada en `test/ollamatest.py`
- Incluye notas de optimización para consultas

#### Notas de Implementación

- ❌ Transformers: Descartado por ineficiencia para el caso de uso específico

## Guía de Inicio

Para comenzar con el procesamiento local usando Ollama, consulte la documentación y ejemplos en `test/ollamatest.py`. Los comentarios incluidos proporcionan información crucial para la optimización de consultas.

## Estado del Proyecto

El proyecto se encuentra en desarrollo activo, con énfasis en:

- Expansión de compatibilidad con modelos OpenAI
- Pruebas exhaustivas con la API de Anthropic
- Optimización de procesamiento local con Ollama
