import os
import warnings

warnings.filterwarnings("ignore")

# NumPy must be imported first to reduce version-mismatch issues
import numpy as np

from flask import Flask, Response, jsonify, render_template, request
from werkzeug.utils import secure_filename

DEPENDENCY_ERROR = None

try:
    import cv2
    import mediapipe as mp
    from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
except Exception as exc:
    cv2 = None
    mp = None
    Sequential = None
    DATA_DIR = "D:/temp/Shrush/sign-lang-fixed/data"
    ImageDataGenerator = None
    Conv2D = Dense = Dropout = Flatten = MaxPooling2D = None
    DEPENDENCY_ERROR = str(exc)

# ------------------- CONFIG -------------------
DATA_DIR = "data"
IMG_SIZE = 224
MODEL_PATH = "modelnet_model.h5"
LABELS_PATH = "labels.txt"
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "jpg", "jpeg", "png"}

LABELS = []
model = None
MODEL_ERROR = None

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ------------------- MEDIAPIPE HANDS -------------------
if mp is not None:
    mp_hands = mp.solutions.hands
    hands_detector = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
else:
    mp_hands = None
    hands_detector = None


def ensure_dependencies():
    if DEPENDENCY_ERROR is not None:
        raise RuntimeError(
            "Required ML dependencies could not be imported. "
            "Install the versions from requirements.txt in a compatible environment. "
            f"Original error: {DEPENDENCY_ERROR}"
        )


def get_dataset_labels():
    if not os.path.isdir(DATA_DIR):
        return []
    return sorted(
        entry
        for entry in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, entry))
    )


def save_labels(labels):
    with open(LABELS_PATH, "w", encoding="utf-8") as file:
        file.write("\n".join(labels))


def load_labels():
    if os.path.exists(LABELS_PATH):
        with open(LABELS_PATH, "r", encoding="utf-8") as file:
            return [line.strip() for line in file if line.strip()]
    return get_dataset_labels()


# ------------------- TRAIN MODEL -------------------
def build_and_train_model():
    global LABELS
    ensure_dependencies()

    if not os.path.isdir(DATA_DIR):
        raise RuntimeError(
            f"Dataset folder '{DATA_DIR}' was not found. Add class subfolders or place a trained model at '{MODEL_PATH}'."
        )

    datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)

    train_gen = datagen.flow_from_directory(
        DATA_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=32,
        class_mode="categorical",
        subset="training",
    )
    val_gen = datagen.flow_from_directory(
        DATA_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=32,
        class_mode="categorical",
        subset="validation",
    )

    LABELS = list(train_gen.class_indices.keys())
    if not LABELS:
        raise RuntimeError(f"No class folders were found inside '{DATA_DIR}'.")

    print("Detected classes:", LABELS)
    save_labels(LABELS)

    trained_model = Sequential(
        [
            Conv2D(3, (3, 3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 3)),
            MaxPooling2D(2, 2),
            Conv2D(3, (3, 3), activation="relu"),
            MaxPooling2D(2, 2),
            Conv2D(3, (3, 3), activation="relu"),
            MaxPooling2D(2, 2),
            Flatten(),
            Dense(128, activation="relu"),
            Dropout(0.5),
            Dense(len(LABELS), activation="softmax"),
        ]
    )
    trained_model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    trained_model.fit(train_gen, epochs=10, validation_data=val_gen)
    trained_model.save(MODEL_PATH)
    print(f"Model trained and saved to {MODEL_PATH}")
    return trained_model


def initialize_model():
    global model, LABELS, MODEL_ERROR

    ensure_dependencies()

    if model is not None:
        return model
    if MODEL_ERROR is not None:
        raise RuntimeError(MODEL_ERROR)

    try:
        LABELS = load_labels()

        if os.path.exists(MODEL_PATH):
            if not LABELS:
                raise RuntimeError(
                    f"Model file '{MODEL_PATH}' exists, but no labels were found. Add '{LABELS_PATH}' or restore the dataset folder."
                )
            model = load_model(MODEL_PATH)
            print("Loaded pre-trained model")
            return model

        print("Model not found, starting training...")
        model = build_and_train_model()
        return model
    except Exception as exc:
        MODEL_ERROR = str(exc)
        print(f"Model initialization failed: {MODEL_ERROR}")
        raise RuntimeError(MODEL_ERROR) from exc


# ------------------- HELPERS -------------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def detect_and_crop_hand(frame):
    ensure_dependencies()
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(img_rgb)
    if not results.multi_hand_landmarks:
        return None

    h, w, _ = frame.shape
    for hand_landmarks in results.multi_hand_landmarks:
        x_coords = [lm.x * w for lm in hand_landmarks.landmark]
        y_coords = [lm.y * h for lm in hand_landmarks.landmark]
        x_min, x_max = int(min(x_coords)) - 20, int(max(x_coords)) + 20
        y_min, y_max = int(min(y_coords)) - 20, int(max(y_coords)) + 20
        x_min, y_min = max(0, x_min), max(0, y_min)
        x_max, y_max = min(w, x_max), min(h, y_max)
        return frame[y_min:y_max, x_min:x_max]
    return None


def preprocess_frame(frame):
    ensure_dependencies()
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)


def predict_frame(frame):
    current_model = initialize_model()
    cropped = detect_and_crop_hand(frame)
    if cropped is None:
        return "No Hand Detected", 0.0
    processed = preprocess_frame(cropped)
    preds = current_model.predict(processed, verbose=0)
    class_index = int(np.argmax(preds))
    confidence = float(np.max(preds))
    return LABELS[class_index], confidence


def extract_frames_and_predict(video_path, step=5):
    ensure_dependencies()
    cap = cv2.VideoCapture(video_path)
    sequence = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % step == 0:
            label, _ = predict_frame(frame)
            if label != "No Hand Detected":
                sequence.append(label)
        frame_count += 1

    cap.release()

    collapsed = []
    for char in sequence:
        if not collapsed or char != collapsed[-1]:
            collapsed.append(char)
    return " ".join(collapsed)


# ------------------- FLASK ROUTES -------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "model_ready": model is not None,
            "model_file_exists": os.path.exists(MODEL_PATH),
            "dataset_exists": os.path.isdir(DATA_DIR),
            "labels_loaded": LABELS,
            "model_error": MODEL_ERROR,
            "dependency_error": DEPENDENCY_ERROR,
        }
    )


@app.route("/predict_image", methods=["POST"])
def predict_image():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "No file selected"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type"}), 400

    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Invalid image file"}), 400

    try:
        label, conf = predict_frame(img)
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503

    return jsonify({"prediction": label, "confidence": conf})


@app.route("/predict_video", methods=["POST"])
def predict_video():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "No file selected"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type"}), 400

    filename = secure_filename(file.filename)
    video_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(video_path)

    try:
        sequence = extract_frames_and_predict(video_path, step=5)
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503

    return jsonify({"prediction": sequence})


# ------------------- LIVE WEBCAM FEED -------------------
def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break

        label, conf = predict_frame(frame)
        display_text = f"{label} ({conf:.2f})" if label != "No Hand Detected" else label
        cv2.putText(
            frame,
            display_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )

    cap.release()


@app.route("/predict_live")
def predict_live():
    try:
        initialize_model()
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# ------------------- MAIN -------------------
if __name__ == "__main__":
    app.run(debug=True)
