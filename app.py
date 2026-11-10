"""
F1 ML Lab - Flask-приложение.

  GET  /                    страница приложения
  GET  /api/models          список моделей (какие есть, медленные, чего не хватает)
  POST /api/run             запуск пайплайна в фоне: {"quick": bool, "models": [...]}
  GET  /api/status          прогресс выполнения (state, percent, message, log)
  GET  /api/result          последний результат (из памяти или results/latest.json)
  GET  /api/predict/schema  значения по умолчанию и диапазоны признаков для формы what-if
  POST /api/predict         what-if: вероятность подиума от всех моделей для одного пилота
"""

import json
import os
import threading
import time
import traceback

from flask import Flask, jsonify, render_template, request

from ml.pipeline import CONTEXT_MODELS, apply_whatif, available_models, run_ml

app = Flask(__name__)

DATA_PATH = "data"          # папка с results.csv, races.csv, qualifying.csv, drivers.csv, constructors.csv
RESULTS_DIR = "results"
RESULT_FILE = os.path.join(RESULTS_DIR, "latest.json")
BUNDLE_FILE = os.path.join(RESULTS_DIR, "bundle.joblib")
PLOTS_DIR = os.path.join(app.static_folder, "plots")   # доступно по /static/plots/...

_lock = threading.Lock()
_state = {"state": "idle", "percent": 0, "message": "Ready", "log": [],
          "started": None, "finished": None, "error": None}
_result = {"data": None}
_bundle = {"data": None}


def _plots_to_urls(plots, version):
    """Пути к PNG -> URL для браузера. ?v=... сбрасывает кэш браузера после пересчёта."""
    if not plots:
        return None
    to_url = lambda p: f"/static/plots/{os.path.basename(p)}?v={version}"
    return {
        "models": {name: to_url(p) for name, p in plots["models"].items()},
        "summary": {key: to_url(p) for key, p in plots["summary"].items()},
    }


def _load_saved_result():
    if _result["data"] is None and os.path.exists(RESULT_FILE):
        try:
            with open(RESULT_FILE, encoding="utf-8") as f:
                _result["data"] = json.load(f)
        except (OSError, ValueError):       # файл повреждён или ещё пишется
            return None
    return _result["data"]


def _get_bundle():
    if _bundle["data"] is None and os.path.exists(BUNDLE_FILE):
        import joblib
        _bundle["data"] = joblib.load(BUNDLE_FILE)
    return _bundle["data"]


def _progress(percent, message):
    with _lock:
        _state["percent"] = percent
        _state["message"] = message
        _state["log"].append(f"[{time.strftime('%H:%M:%S')}] {percent:3d}%  {message}")
        _state["log"] = _state["log"][-60:]


def _job(quick, include):
    try:
        result = run_ml(DATA_PATH, save_model_path=BUNDLE_FILE, plots_dir=PLOTS_DIR,
                        quick=quick, include=include, progress=_progress)
        version = int(time.time())
        result["plots"] = _plots_to_urls(result.get("plots"), version)
        result["generated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        os.makedirs(RESULTS_DIR, exist_ok=True)
        tmp = RESULT_FILE + ".tmp"           # пишем во временный файл и атомарно подменяем
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, allow_nan=False)
        os.replace(tmp, RESULT_FILE)
        _result["data"] = result
        _bundle["data"] = None            # перечитать свежие модели
        with _lock:
            _state.update(state="done", percent=100, message="Done", finished=time.time())
    except Exception as exc:                      # noqa: BLE001
        traceback.print_exc()
        with _lock:
            _state.update(state="error", message=f"{type(exc).__name__}: {exc}",
                          error=str(exc), finished=time.time())


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/models")
def api_models():
    return jsonify(available_models())


@app.route("/api/run", methods=["POST"])
def api_run():
    body = request.get_json(silent=True) or {}
    quick = bool(body.get("quick", False))
    include = set(body["models"]) if body.get("models") else None
    with _lock:
        if _state["state"] == "running":
            return jsonify({"error": "Pipeline is already running"}), 409
        _state.update(state="running", percent=0, message="Starting", log=[],
                      started=time.time(), finished=None, error=None)
    threading.Thread(target=_job, args=(quick, include), daemon=True).start()
    return jsonify({"started": True})


@app.route("/api/status")
def api_status():
    with _lock:
        out = dict(_state)
    out["has_result"] = _load_saved_result() is not None
    out["elapsed"] = round((out["finished"] or time.time()) - out["started"], 1) \
        if out["started"] else 0
    return jsonify(out)


@app.route("/api/result")
def api_result():
    data = _load_saved_result()
    if data is None:
        return jsonify({"error": "No results yet. Run the pipeline first."}), 404
    return jsonify(data)


@app.route("/api/predict/schema")
def api_predict_schema():
    bundle = _get_bundle()
    if bundle is None:
        return jsonify({"error": "No trained models yet. Run the pipeline first."}), 404
    return jsonify({
        "features": bundle["features"],
        "defaults": bundle["defaults"],
        "ranges": bundle["ranges"],
        "models": list(bundle["models"]),
        "thresholds": bundle.get("thresholds", {}),
        "tuned_name": bundle.get("tuned_name"),
        "unsupported_models": sorted(CONTEXT_MODELS),
    })


@app.route("/api/predict", methods=["POST"])
def api_predict():
    bundle = _get_bundle()
    if bundle is None:
        return jsonify({"error": "No trained models yet. Run the pipeline first."}), 404
    body = request.get_json(silent=True) or {}
    try:
        probs, row = apply_whatif(bundle, body.get("features", {}), body.get("teammate_grid"))
    except (KeyError, ValueError, TypeError) as exc:
        return jsonify({"error": f"Bad input: {exc}"}), 400
    return jsonify({"probabilities": probs, "thresholds": bundle.get("thresholds", {}),
                    "used": {k: row[k] for k in bundle["features"]}})


if __name__ == "__main__":
    app.run(debug=False, threaded=True)