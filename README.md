# UNOGame
Play UNO with Chitti

To try the game, run  `python3 uno7.1_gemini.py`

## How it works
- This uses OpenCV to read the image from user's camera.
- Then it uploads the image to Gemini's 1.5 Flash model to understand the image.
- Based on the response, and the designated strategy, it figures out next card to play.
- Once complete, it used Google's Text-to-speech service to speak out the card which Chitti wants to play.
