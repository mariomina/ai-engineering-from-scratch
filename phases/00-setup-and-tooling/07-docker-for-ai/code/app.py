# Phase 0 · Lesson 07 — Damer for AI: simple Flask API server.
# Used by Exercise 3 ("add flask, rebuild, run a simple API server on port
# 5000"). Serves one endpoint so the container is easy to test with curl.

from flask import Flask, jsonify

app = Flask(__name__)


@app.route("/")
def home():
    return jsonify({"message": "Hello from Docker!", "torch": "cpu"})


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)