# Artemisa

Módulo de Python para la extracción de información de documentos Excel, PowerPoint, Word y PDF.

## Estructura del proyecto

Todos los ejemplos y documentación de uso se encuentran en la carpeta `test`.

## Características principales

El módulo incluye integraciones con APIs de Inteligencia Artificial para procesar y realizar consultas sobre los documentos procesados. Adicionalmente, se implementará integración local mediante Transformers u Ollama.

### Uso de Providers

#### OpenAI

El módulo ofrece una compatibilidad muy buena con el proveedor de OpenAI y sus modelos no razonadores.
Aún se esta trabajando en aportar mayor compatibilidad entre más modelos de OpenAI

#### Deep Seek R1 con HuggingFace

El módulo ofrece una compatibilidad muy buena con el proveedor de HuggingFace con el modelo `DeepSeek-R1-Distill-Qwen-32B` (Solo estara disponible mientras HuggingFace provea su HF Inference API gratuita)

#### Gemini

El módulo ofrece compatibilidad muy buena con el proveedor de Google Gemini y sus modelos no razonadores. Aún no se ha probado la implementación con los modelos razonadores de Gemini.

#### HuggingFaceClient

El módulo ofrece compatibilidad muy buena con el proveedor de HuggingFace y sus modelos de generación de texto, tome en cuenta que hay modelos que necesitan ser usuario Pro para acceder desde la API.

#### Anthropic

El módulo ofrece una compatibilidad limitada debido a la falta de pruebas exhaustivas con la API de Anthropic.

### Uso local con Ollama

La primera versión estable del uso local con Ollama esta disponible y si quiere saber como se usa ve a la carpeta `test` y archivo `ollamatest.py`, toma en cuenta todos los comentarios que dejamos, para que puedas tener una ejecución optima de tus query's

#### Se descarto el uso de Transformers para uso local, no era eficiente ni eficaz para nuestro caso de uso.
