
import unittest
from unittest.mock import Mock, patch

from EmotionDetection import emotion_detector


class TestEmotionDetector(unittest.TestCase):

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_emotion_detector(self, mock_post):
        response = Mock()
        response.status_code = 200
        response.json.return_value = {
            "emotionPredictions": [{
                "emotion": {
                    "anger": 0.1,
                    "disgust": 0.05,
                    "fear": 0.1,
                    "joy": 0.7,
                    "sadness": 0.05
                }
            }]
        }
        mock_post.return_value = response

        result = emotion_detector("I am very happy today")

        self.assertEqual(result["dominant_emotion"], "joy")
        self.assertAlmostEqual(result["joy"], 0.7)

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_bad_request(self, mock_post):
        response = Mock()
        response.status_code = 400
        mock_post.return_value = response

        result = emotion_detector("test")

        self.assertIsNone(result["dominant_emotion"])

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_request_failure(self, mock_post):
        import requests
        mock_post.side_effect = requests.ConnectionError()

        result = emotion_detector("test")

        self.assertIsNone(result["dominant_emotion"])


if __name__ == "__main__":
    unittest.main()
