
"""Flask server for the emotion detection application."""

from flask import Flask, render_template, request
from EmotionDetection import emotion_detector

app = Flask(__name__)


@app.route("/")
def home():
    """Render the home page."""
    return render_template("index.html")


@app.route("/emotionDetector")
def detect_emotion():
    """Detect emotions in the supplied text."""
    text_to_analyze = request.args.get("textToAnalyze", "").strip()

    if not text_to_analyze:
        return "Please enter some text to analyze.", 400

    response = emotion_detector(text_to_analyze)

    if not response or response.get("dominant_emotion") is None:
        return (
            "Emotion detection service is unavailable. Please try again later.",
            503,
        )

    return (
        f"For the given statement, the system response is "
        f"'anger': {response['anger']}, "
        f"'disgust': {response['disgust']}, "
        f"'fear': {response['fear']}, "
        f"'joy': {response['joy']} and "
        f"'sadness': {response['sadness']}. "
        f"The dominant emotion is {response['dominant_emotion']}"
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)
