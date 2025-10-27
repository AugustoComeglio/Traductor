# Traductor

## Descripción
Este proyecto es un traductor automático de inglés a español basado en un modelo de red neuronal Seq2Seq con una arquitectura de Transformer. El modelo es capaz de recibir una oración en inglés y devolver su traducción al español.

## El proyecto está compuesto por:
- Un modelo de PyTorch entrenado para la tarea de traducción.
- Una API de FastAPI que expone el modelo para realizar traducciones.
- Una interfaz de usuario web creada con Streamlit para interactuar con el modelo de forma sencilla.

## Funcionalidades
_ Traducción de texto: La funcionalidad principal es la traducción de oraciones del inglés al español.
API REST: Permite la integración del modelo de traducción con otras aplicaciones a través de una API. El endpoint /model/translate recibe una oración en inglés y devuelve la traducción en formato JSON.
Interfaz de usuario: Facilita la interacción con el modelo a través de una interfaz web simple donde el usuario puede introducir texto en inglés y obtener la traducción.

## Tecnologías Utilizadas
- Python: El lenguaje de programación principal utilizado en el proyecto.
- PyTorch: Para la implementación y el entrenamiento del modelo de red neuronal.
- FastAPI: Para la creación de la API REST que sirve el modelo.
- Streamlit: Para la construcción de la interfaz de usuario web interactiva.
- Uvicorn: Como servidor para la API de FastAPI.

## Arquitectura del Modelo
El modelo de traducción se basa en la arquitectura Transformer, que consta de:

- Encoder: Procesa la oración de entrada en inglés y la convierte en una representación vectorial.
- Decoder: Utiliza la representación del encoder para generar la traducción en español.
- Multi-Head Attention: Permite al modelo prestar atención a diferentes partes de la oración de entrada al generar la traducción.
- Position-wise Feedforward Networks: Redes neuronales que se aplican a cada posición de la secuencia para procesar la información.

## Cómo ejecutar el proyecto
Para ejecutar el proyecto, sigue estos pasos:
## Inistalacion
```
pip install torch fastapi "uvicorn[standard]" streamlit requests
```

### Iniciar la API:
```
uvicorn ml_api:app --reload
```

### Iniciar la interfaz de usuario:
```
streamlit run ml_ui.py
```
