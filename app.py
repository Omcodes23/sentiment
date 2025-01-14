from flask import Flask, request, jsonify
from transformers import pipeline

# Initialize the Flask app
app = Flask(__name__)

# Load pre-trained models for sentiment and emotion analysis
sentiment_analyzer = pipeline("sentiment-analysis")
emotion_analyzer = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)

@app.route("/sentiment", methods=["POST"])
def sentiment():
    try:
        # Extract the input message from the request
        data = request.get_json()
        msg = data.get("msg", "")

        if not msg:
            return jsonify({"error": "Message not provided."}), 400

        # Perform sentiment analysis
        result = sentiment_analyzer(msg)[0]

        # Map sentiment results to positive, negative, and neutral scores
        sentiment_scores = {
            "pos": result["score"] if result["label"] == "POSITIVE" else 0.0,
            "neg": result["score"] if result["label"] == "NEGATIVE" else 0.0,
            "neu": 1.0 - result["score"]
        }

        return jsonify(sentiment_scores)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/emotion", methods=["POST"])
def emotion():
    try:
        # Extract the input message from the request
        data = request.get_json()
        msg = data.get("msg", "")

        if not msg:
            return jsonify({"error": "Message not provided."}), 400

        # Perform emotion analysis
        results = emotion_analyzer(msg)[0]

        # Map emotion scores
        emotion_scores = {result["label"].lower(): result["score"] for result in results}

        return jsonify(emotion_scores)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)