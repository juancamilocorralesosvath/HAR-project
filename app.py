import base64

from flask import Flask, render_template, Response, request, jsonify
import cv2
import mediapipe as mp
import joblib
import numpy as np
from collections import deque
import time

# Importar la NUEVA función de features y sus columnas
from utils import calculate_frame_features_v2, FEATURE_COLUMNS_V2

# -------------------------------------------------------
# Configuración general
# -------------------------------------------------------

app = Flask(__name__)

MODEL_PATH = "model/best_xgboost_v2.joblib"
SCALER_PATH = "model/scaler_v2.joblib"
LABEL_ENCODER_PATH = "model/label_encoder_v2.joblib"

DEBUG_PREDICTION = False  # ponlo en True si quieres ver prints de debug

# Cargar modelo, scaler y label encoder
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    le = joblib.load(LABEL_ENCODER_PATH)
    print("Modelo v2, Scaler v2 y LabelEncoder v2 cargados correctamente.")
    print("Clases:", list(le.classes_))
except FileNotFoundError as e:
    print(f"Error cargando archivos del modelo: {e}")
    print(
        "Asegúrate de que 'best_xgboost_v2.joblib', 'scaler_v2.joblib' y "
        "'label_encoder_v2.joblib' estén en la carpeta 'model'."
    )
    exit()

# MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
mp_drawing = mp.solutions.drawing_utils

# Ventana deslizante (debe coincidir con el entrenamiento)
WINDOW_SIZE = 30
FEATURE_COUNT = len(FEATURE_COLUMNS_V2)  # 7
feature_buffer = deque(maxlen=WINDOW_SIZE)

# Historial de predicciones crudas del modelo (IDs numéricos)
PREDICTION_HISTORY_SIZE = 30            # ~1 segundo si vas ~30 FPS
prediction_history = deque(maxlen=PREDICTION_HISTORY_SIZE)

# Cada cuánto tiempo se actualiza el texto en pantalla
UPDATE_INTERVAL = 0.5                   # segundos
last_update_time = 0.0

current_prediction = "Initializing..."

COLOR_MAP = {
    "sitting_down": (0, 255, 255),            # amarillo
    "sitting_still": (255, 0, 255),           # magenta
    "standing_still": (0, 255, 0),            # verde
    "standing_up": (255, 255, 0),             # cian
    "turning": (255, 0, 0),                   # azul
    "walking_away_from_camera": (0, 165, 255),  # naranja
    "walking_towards_camera": (0, 0, 255),    # rojo
}


# -------------------------------------------------------
# Rutas Flask
# -------------------------------------------------------

@app.route("/")
def index():
    """Página principal."""
    return render_template("index.html")


def process_frame(frame):
    """
    Procesa un frame capturado por el navegador del visitante:
    calcula features v2, arma la ventana, predice y anota el frame.

    Devuelve el frame anotado (BGR) listo para re-codificar.
    """
    global current_prediction, feature_buffer, prediction_history, last_update_time

    # --------- MediaPipe: detección de pose ----------
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results = pose.process(image_rgb)
    image_rgb.flags.writeable = True
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    frame_features = None

    if results.pose_landmarks:
        landmarks_current = results.pose_landmarks.landmark

        # NUEVA función de features (7 features, sin frame previo)
        frame_features = calculate_frame_features_v2(landmarks_current)

        # Dibujar landmarks para referencia
        mp_drawing.draw_landmarks(
            image_bgr, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
        )
    else:
        # Si perdemos la pose, reseteamos el buffer y las predicciones
        feature_buffer.clear()
        prediction_history.clear()
        current_prediction = "No Pose Detected"

    # --------- Lógica de predicción ----------
    if frame_features is not None:
        feature_buffer.append(frame_features)

        if len(feature_buffer) == WINDOW_SIZE:
            # Ventana completa → aplanar: (30, 7) → (210,)
            window_data = np.array(
                feature_buffer, dtype=np.float32).flatten()
            window_data = window_data.reshape(1, -1)  # (1, 210)

            # Escalar
            scaled_data = scaler.transform(window_data)

            if DEBUG_PREDICTION:
                print("\n--- Último frame (features v2) ---")
                for i, name in enumerate(FEATURE_COLUMNS_V2):
                    print(f"{name}: {frame_features[i]:.4f}")
                print("\n--- Primeros 5 valores escalados de la ventana ---")
                print(scaled_data[0, :5])

            # Predecir y suavizar
            try:
                prediction_encoded = int(model.predict(scaled_data)[0])

                # Guardar predicción cruda en el historial
                prediction_history.append(prediction_encoded)

                # ¿Toca actualizar el texto visible?
                now = time.time()
                if (
                    now - last_update_time >= UPDATE_INTERVAL
                    and len(prediction_history) >= 5  # mínimo para votar
                ):
                    values, counts = np.unique(
                        np.array(prediction_history), return_counts=True
                    )
                    dominant_encoded = int(values[np.argmax(counts)])

                    current_prediction = le.inverse_transform(
                        [dominant_encoded]
                    )[0]

                    last_update_time = now

                # Si aún no se cumple el intervalo, dejamos current_prediction como está

            except Exception as e:
                current_prediction = f"Prediction Error: {e}"
        else:
            # Todavía no hay ventana completa
            current_prediction = (
                f"Collecting frames... "
                f"({len(feature_buffer)}/{WINDOW_SIZE})"
            )

    # --------- Dibujar la predicción en el frame ----------
    color = COLOR_MAP.get(current_prediction, (255, 255, 255))

    cv2.putText(
        image_bgr,
        f"Activity: {current_prediction}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA,
    )

    return image_bgr, current_prediction


@app.route("/predict_frame", methods=["POST"])
def predict_frame():
    """
    Recibe un frame JPEG en base64 capturado por la cámara del navegador,
    lo procesa (MediaPipe + XGBoost) y devuelve el frame anotado + la
    predicción actual, también en base64.
    """
    data = request.get_json(silent=True)
    if not data or "image" not in data:
        return jsonify({"error": "Falta el campo 'image'"}), 400

    image_data = data["image"]
    if "," in image_data:
        # Quitar el prefijo "data:image/jpeg;base64,"
        image_data = image_data.split(",", 1)[1]

    try:
        image_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("No se pudo decodificar la imagen")
    except Exception as e:
        return jsonify({"error": f"Imagen inválida: {e}"}), 400

    annotated_frame, prediction = process_frame(frame)

    ret, buffer = cv2.imencode(".jpg", annotated_frame)
    if not ret:
        return jsonify({"error": "No se pudo codificar el frame anotado"}), 500

    annotated_b64 = base64.b64encode(buffer.tobytes()).decode("utf-8")

    return jsonify(
        {
            "prediction": prediction,
            "image": f"data:image/jpeg;base64,{annotated_b64}",
        }
    )


if __name__ == "__main__":
    # host='0.0.0.0' → accesible en la red local (solo para pruebas locales)
    app.run(debug=True, host="0.0.0.0", port=5000)
