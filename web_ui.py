"""
Flask Web Application for IM-Milan: Best Time to Call Predictor.
Fetches seller and hotlead details via Redash PostgreSQL DWH and predicts
the optimal 15-minute calling sequence using IM-Milan Model 4.
"""

import os
import sys
import argparse
from flask import Flask, render_template, request, jsonify

from src.milan_engine import predict_im_milan, fetch_from_redash

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True) if request.is_json else request.form
        glid_raw = data.get("glid", 79959)
        try:
            glid = int(glid_raw)
        except (ValueError, TypeError):
            glid = 79959

        result = predict_im_milan(glid)
        return jsonify(result)

    except ValueError as ve:
        return jsonify({"status": "error", "message": str(ve)}), 400
    except Exception as e:
        return jsonify({"status": "error", "message": f"Server processing error: {str(e)}"}), 500


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "app": "IM-Milan", "version": "model4-q15-gap-2026-10"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run IM-Milan Flask Web UI")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    args = parser.parse_args()

    print(f"🚀 Starting IM-Milan Web UI on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)
