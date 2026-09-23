---
title: Reconocimiento de Actividades Humanas
emoji: 🏃
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Clasificación de actividades humanas en tiempo real con MediaPipe Pose + XGBoost desde la cámara del navegador.
---

# Proyecto-Final-APO-III

Reconocimiento de actividad humana en tiempo real usando MediaPipe Pose + XGBoost.

La cámara se captura desde el navegador del visitante; los frames se envían al backend Flask (`/predict_frame`), que calcula las features de pose, corre el modelo y devuelve el frame anotado con la actividad detectada.

**Integrantes:** Luis Manuel Rojas Correa, William Joseph Verdesoto, Juan Camilo Corrales Osvath.
