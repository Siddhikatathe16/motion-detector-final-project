
"""Emotion detection using the Watson NLP service."""

import requests

URL = (
    "https://sn-watson-emotion.labs.skills.network/"
    "v1/watson.runtime.nlp.v1/NlpService/EmotionPredict"
)

HEADERS = {
    "grpc-metadata-mm-model-id":
        "emotion_aggregated-workflow_lang_en_stock"
}

EMOTIONS = ("anger", "disgust", "fear", "joy", "sadness")


def emotion_detector(text_to_analyze):
    """Return emotion scores and the dominant emotion."""
    result = {emotion: None for emotion in EMOTIONS}
    result["dominant_emotion"] = None

    if not isinstance(text_to_analyze, str) or not text_to_analyze.strip():
        return result

    try:
        response = requests.post(
            URL,
            json={"raw_document": {"text": text_to_analyze}},
            headers=HEADERS,
            timeout=15,
        )

        if response.status_code == 400:
            return result

        if response.status_code != 200:
            return result

        data = response.json()
        scores = data["emotionPredictions"][0]["emotion"]

        for emotion in EMOTIONS:
            result[emotion] = float(scores[emotion])

        result["dominant_emotion"] = max(
            EMOTIONS, key=lambda emotion: result[emotion]
        )

        return result

    except (
        requests.RequestException,
        ValueError,
        KeyError,
        IndexError,
        TypeError,
    ):
        return result
