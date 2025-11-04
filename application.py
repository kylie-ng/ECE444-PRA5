import os
import logging
import threading
import numpy
from typing import Optional
from flask import Flask, request, jsonify, render_template_string

# Flask app (Elastic Beanstalk Procfile expects "application:application")
application = Flask(__name__)

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Resolve artifact paths relative to this file; allow env overrides (empty env won't override)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.getenv("MODEL_PATH") or os.path.join(BASE_DIR, "basic_classifier.pkl")
VECTORIZER_PATH = os.getenv("VECTORIZER_PATH") or os.path.join(BASE_DIR, "count_vectorizer.pkl")

# Log resolved paths
logger.info("CWD: %s", os.getcwd())
logger.info("Resolved MODEL_PATH: %s", MODEL_PATH)
logger.info("Resolved VECTORIZER_PATH: %s", VECTORIZER_PATH)

# Global variables for loaded artifacts
_loaded_model: Optional[object] = None
_vectorizer: Optional[object] = None
_artifact_lock = threading.Lock()

# Artifact loading
def _load_artifacts_once() -> None:
    """Lazily load model and vectorizer once per process."""
    global _loaded_model, _vectorizer
    if _loaded_model is not None and _vectorizer is not None:
        return
    with _artifact_lock:
        if _loaded_model is None or _vectorizer is None:
            import pickle
            logger.info("Loading artifacts...")
            with open(MODEL_PATH, "rb") as mf:
                _loaded_model = pickle.load(mf)
            with open(VECTORIZER_PATH, "rb") as vf:
                _vectorizer = pickle.load(vf)
            logger.info("Artifacts loaded.")

# Inference function
def _predict_text(message: str) -> str:
    """Run inference and return the predicted class as a string label."""
    _load_artifacts_once()
    X = _vectorizer.transform([message])
    pred = _loaded_model.predict(X)
    # pred[0] could be a numpy scalar; normalize to native str
    val = pred[0]
    val_py = val.item() if hasattr(val, "item") else val
    return str(val_py)

# Eager load artifacts in a background thread at startup
def _eager_load_background():
    try:
        _load_artifacts_once()
    except Exception as e:
        # Log and continue; app remains healthy and will lazy-load on first request
        logger.warning("Background eager load failed: %s", e, exc_info=True)


# Non-blocking eager load at startup
threading.Thread(target=_eager_load_background, daemon=True).start()

DEMO_HTML = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Fake News Detection Demo</title>
    <style>
      body {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        background-color: #f5f6fa;
        margin: 0;
        padding: 0;
        display: flex;
        flex-direction: column;
        align-items: center;
        min-height: 100vh;
      }
      header {
        background-color: #2f3640;
        color: white;
        width: 100%;
        text-align: center;
        padding: 1.5rem 0;
      }
      h1 {
        margin: 0;
        font-size: 1.8rem;
      }
      main {
        flex: 1;
        width: 100%;
        max-width: 600px;
        background-color: white;
        margin-top: 2rem;
        border-radius: 10px;
        padding: 2rem;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
      }
      form {
        display: flex;
        flex-direction: column;
        gap: 1rem;
      }
      textarea {
        resize: vertical;
        min-height: 120px;
        font-size: 1rem;
        padding: 0.8rem;
        border-radius: 6px;
        border: 1px solid #ccc;
      }
      button {
        background-color: #0984e3;
        color: white;
        font-size: 1rem;
        padding: 0.8rem;
        border: none;
        border-radius: 6px;
        cursor: pointer;
        transition: background-color 0.3s;
      }
      button:hover {
        background-color: #005cbf;
      }
      .result {
        margin-top: 1.5rem;
        font-size: 1.1rem;
        text-align: center;
        padding: 1rem;
        border-radius: 8px;
      }
      .fake {
        background-color: #ffe3e3;
        color: #c0392b;
      }
      .real {
        background-color: #e3ffe7;
        color: #27ae60;
      }
      .error {
        background-color: #ffe9b3;
        color: #b35c00;
      }
      .info {
        margin-top: 1rem;
        font-size: 0.85rem;
        color: #555;
        text-align: center;
      }
    </style>
  </head>
  <body>
    <header>
      <h1>📰 Fake News Detection</h1>
    </header>
    <main>
      <form action="/predict-form" method="post">
        <label for="message"><strong>Enter a news headline or snippet:</strong></label>
        <textarea id="message" name="message" placeholder="Type or paste a news article snippet here..."></textarea>
        <button type="submit">Detect</button>
      </form>

      {% if prediction %}
        {% if prediction == '1' %}
          <div class="result fake"><strong>Result:</strong> 🚨 This looks like <b>Fake News</b>.</div>
        {% elif prediction == '0' %}
          <div class="result real"><strong>Result:</strong> ✅ This looks like <b>Real News</b>.</div>
        {% else %}
          <div class="result"><strong>Prediction:</strong> {{ prediction }}</div>
        {% endif %}
      {% endif %}

      {% if error %}
        <div class="result error"><strong>Error:</strong> {{ error }}</div>
      {% endif %}

      <div class="info">
        Model loaded: {{ '✅' if model_loaded else '❌' }}<br>
        Model path: {{ model_path }}
      </div>
    </main>
  </body>
</html>
"""

# Routes
@application.get("/")
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": bool(_loaded_model is not None and _vectorizer is not None),
        "model_path": MODEL_PATH,
        "vectorizer_path": VECTORIZER_PATH,
    }), 200

# Demo page rendering endpoint
@application.get("/demo")
def demo():
    return render_template_string(
        DEMO_HTML,
        model_loaded=bool(_loaded_model is not None and _vectorizer is not None),
        model_path=MODEL_PATH,
        prediction=None,
        error=None,
    )

# Form submission endpoint for demo page
@application.post("/predict-form")
def predict_form():
    message = (request.form.get("message") or "").strip()
    if not message:
        return render_template_string(
            DEMO_HTML,
            model_loaded=bool(_loaded_model is not None and _vectorizer is not None),
            model_path=MODEL_PATH,
            prediction=None,
            error="Field 'message' is required and must be non-empty.",
        ), 400
    try:
        label = _predict_text(message)
        return render_template_string(
            DEMO_HTML,
            model_loaded=True,
            model_path=MODEL_PATH,
            prediction=label,
            error=None,
        )
    except FileNotFoundError:
        return render_template_string(
            DEMO_HTML,
            model_loaded=False,
            model_path=MODEL_PATH,
            prediction=None,
            error="Model artifacts not found on server.",
        ), 503
    except Exception as e:
        logger.exception("Inference error: %s", e)
        return render_template_string(
            DEMO_HTML,
            model_loaded=bool(_loaded_model is not None and _vectorizer is not None),
            model_path=MODEL_PATH,
            prediction=None,
            error="Inference failed.",
        ), 500

# JSON API endpoint for predictions
@application.post("/predict")
def predict_json():
    data = request.get_json(silent=True) or {}
    message = str(data.get("message", "")).strip()
    if not message:
        return jsonify({"error": "Field 'message' is required and must be non-empty."}), 400
    try:
        label = _predict_text(message)
        return jsonify({"label": label}), 200
    except FileNotFoundError:
        return jsonify({"error": "Model artifacts not found on server."}), 503
    except Exception as e:
        logger.exception("Inference error: %s", e)
        return jsonify({"error": "Inference failed."}), 500


if __name__ == "__main__":
    # Local dev run; in EB, Gunicorn (from Procfile) will host the app
    port = int(os.getenv("PORT", "8000"))
    application.run(host="0.0.0.0", port=port, debug=False)
