from flask import Flask, request, jsonify
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import os

# Initialize the Flask app
app = Flask(__name__)

# Load pre-trained models for sentiment and emotion analysis
sentiment_model_name = "distilbert-base-uncased-finetuned-sst-2-english"
sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name)
sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)
sentiment_analyzer = pipeline("text-classification", model=sentiment_model, tokenizer=sentiment_tokenizer)

emotion_model_name = "j-hartmann/emotion-english-distilroberta-base"
emotion_model = AutoModelForSequenceClassification.from_pretrained(emotion_model_name)
emotion_tokenizer = AutoTokenizer.from_pretrained(emotion_model_name)
emotion_analyzer = pipeline("text-classification", model=emotion_model, tokenizer=emotion_tokenizer, return_all_scores=True)

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

        # Prepare the sentiment scores in the required format
        sentiment_scores = {
            "pos": result["score"] if result["label"] == "POSITIVE" else 0.0,
            "neg": result["score"] if result["label"] == "NEGATIVE" else 0.0,
            "neu": 1.0 - result["score"]  # Assuming the remaining score is neutral
        }

        # Return the response with sentiment in the exact requested format
        sentiment_response = {
            "sentiment": sentiment_scores
        }

        return jsonify(sentiment_response)
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

        # Prepare the emotion scores in the required format
        emotion_format = {key: f"{value:.2f}" for key, value in emotion_scores.items()}

        return jsonify({"emotion": emotion_format})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Use environment variable for port or default to 5000
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
