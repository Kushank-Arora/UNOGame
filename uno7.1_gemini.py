import cv2
from gtts import gTTS
import os
import numpy as np
import random
import json
import base64
import requests
import pickle
import time
import concurrent.futures

player = None # Chitti, the Uno Player
current_state = ""
ayush_cards = 0
kushank_cards = 0
pos_in = 0
top_card = None
testing = True
thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)

class UnoCard:
    def __init__(self, color, type):
        global pos_in

        self.color = color
        self.type = type
        self.pos = pos_in

        pos_in += 1

    def set_color(color):
        self.color = color

    def __str__(self):
        return f'{self.color} {self.type}'

class UnoPlayer:
    def __init__(self, name, hand):
        self.name = name
        self.hand = hand

    def get_best_non_black_card(self, top_card):
        print("Getting best non black card")
        possible_cards = []
        for card in self.hand:
            if card.color == top_card.color or card.type == top_card.type:
                possible_cards.append(card)

        print("Possible cards", ','.join([str(x) for x in possible_cards]))

        if len(possible_cards) == 0:
            return None

        if self.has_special_cards(possible_cards) and random.random() < 0.7:
            plus_twos = [card for card in possible_cards if card.type == 'Draw Two']
            plus_two = random.choice(plus_twos) if plus_twos else None

            skips = [card for card in possible_cards if card.type == 'Skip']
            skip = random.choice(skips) if skips else None

            reverses = [card for card in possible_cards if card.type == 'Reverse']
            reverse = random.choice(reverses) if reverses else None

            final_list = list(filter(lambda x: x, [plus_two, skip, reverse]))

            if random.random() < 0.8:
                return final_list[0]
            elif random.random() < 0.8:
                return final_list[1] if len(final_list) >= 1 else final_list[0]
            else:
                return final_list[-1]
        else:
            return random.choice(possible_cards)

    def get_max_color(self):
        non_black_cards = [card for card in self.hand if card.color != 'Black']
        if non_black_cards and random.random() < 0.7:
            return max(non_black_cards, key=lambda card: card.color).color
        else:
            return random.choice(['Red', 'Green', 'Blue', 'Yellow'])

    def play_card_wrapper(self, top_card):
        print("I have: " + ", ".join([str(x) for x in self.hand]))

        card = self.play_card(top_card)
        if card and card.color == 'Black':
            card.color = self.get_max_color()
        return card

    def play_card(self, top_card):
        if not top_card:
            read("Please tell me the top card first")
            return

        if top_card.type in ['Draw Two', 'Draw Four']:
            twos = [card for card in self.hand if card.type == top_card.type]
            if len(twos) > 0:
                return twos[0]
            else:
                return None

        # If the player has a black card
        if self.has_black_card():

            # If the top card is a +4, prefer to play a +4
            if top_card.type == 'Draw Four':
                draw_fours = [i for i in self.hand if i.type == 'Draw Four']
                if len(draw_fours) > 0:
                    return draw_fours[0]
                else:
                    return None

            optional_non_black = self.get_best_non_black_card(top_card)

            if optional_non_black:
                prob_to_choose_black = 0.4

                if ayush_cards <= 3 or kushank_cards <= 3:
                    prob_to_choose_black = 0.7

                if random.random() < prob_to_choose_black:
                    return self.choose_black_card()
                else:
                    return optional_non_black
            else:
                return self.choose_black_card()

        # If the player does not have a black card
        else:
            return self.get_best_non_black_card(top_card)

    def has_black_card(self):
        return any(card.color == 'Black' for card in self.hand)

    def has_special_cards(self, cards):
        return any(card.type in ['Draw Two', 'Draw Four', 'Wild', 'Reverse', 'Skip'] for card in cards)

    def choose_black_card(self):
        play_fours = [i for i in self.hand if i.type == 'Draw Four']
        play_wild = [i for i in self.hand if i.type == 'Wild']
        
        if len(play_fours) == 0 or len(play_wild) == 0:
            return play_fours[0] if len(play_fours) > 0 else play_wild[0]

        return play_fours[0] if random.random() < 0.8 else play_wild[0]

    def choose_special_card(self, *special_card_types, override_pick_card=0.8):
        # With probability override_pick_card, override the pick card rule
        if random.random() < override_pick_card:
            return random.choice([UnoCard(None, special_card_type) for special_card_type in special_card_types])
        else:
            return None

    def get_next_player(self):
        return self.players[(self.players.index(self) + 1) % len(self.players)]

# API Call
def getCardTypeFromGemini(frame):
    image_data = cv2.imencode('.jpg', frame)[1]
    base64_image = base64.b64encode(image_data).decode()

    headers = {
        "Content-Type": "application/json"
    }

    body = {
      "contents":[{
        "parts":[
          {
            "text": '''
            This image contains a Uno card. Which uno card is this? Give response in json format as {"color":"Red", "type": "4", "reason": "reason"}. 
            Type can be "Draw Four", "Draw Two", "Skip", "Reverse", "Wild" or 0 to 9 number. 
            Color can be 'Red', 'Yellow', 'Green', 'Blue' or 'Black'.
            Reason field should be the reason on why you think it is what it is, especially when it is either 0 or Skip.
            Remember, a 'Skip' card is a perfect circle with a tilted line crossing it, while a '0' card is a oval(not circle), if you are confused probably it is Skip.
            ''',
          },
          {
            "inline_data": {
              "mime_type":"image/jpg",
              "data": base64_image
            }
          }
        ]
      }]
    }

    print("Request sent to Gemini")
    response = requests.post(
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=API_KEY",
        headers=headers,
        json=body
    )

    response_json = json.loads(response.content)
    if testing:
      print("Got this response from JSON:", response_json)
    most_probable_candidate = response_json['candidates'][0]['content']['parts'][0]['text'].strip()

    try:
      return clean(json.loads(maybe_trim(most_probable_candidate)))
    except:
      read("I didn't understand, maybe because there is no card in the image")
      return None

def maybe_trim(text_response):
  if text_response.startswith('```json'):
    text_response = text_response[7:].strip()
  if text_response.endswith('```'):
    text_response = text_response[:-3].strip()
  return text_response

# Clean API Response
def clean(j):
    if j['color'] == 'White':
        read("Can you please put this card back in the deck?")
        return None
    elif j['type'] == 'Draw Four':
        j['color'] = 'Black'
    elif j['color'] == 'Wild' or j['type'] == 'Wild':
        j['color'] = 'Black'
        j['type'] = 'Wild'
    elif j['color'] != 'Black':
        if j['type'] not in ['Draw Two', 'Skip', 'Reverse']:
            if j['type'] < '0' or j['type'] > '9':
                if (j['type'] == 'null' or j['color'] == 'null'):
                    read("I didn't understand, maybe because there is no card in the image")
                else:
                    read("I didn't understand, it says {0} {1}, maybe because: {2}".format(j['type'], j['color'], j['reason']))
                return None

    return UnoCard(j['color'], j['type'])

def decide_card(key, color):
    """
    Returns: (decided_card: UnoCard, color: str, is_error: bool)
    """
    is_error, decided_card = False, None

    if color is None:
        if key == ord('r'):
            color = 'Red'
        elif key == ord('g'):
            color = 'Green'
        elif key == ord('b'):
            color = 'Blue'
        elif key == ord('y'):
            color = 'Yellow'
        elif key == ord('w'):
            color = 'Black'
        else:
            is_error = True
    else:
        if color == 'Black':
            if key == ord('+'):
                decided_card = UnoCard(color, 'Draw Four')
            elif key == ord('w'):
                decided_card = UnoCard(color, 'Wild')
            else:
                is_error = True
        elif key == ord('+'):
            decided_card = UnoCard(color, 'Draw Two')
        elif key >= ord('0') and key <= ord('9'):
            decided_card = UnoCard(color, chr(key-ord('0') + 48))
        elif key == ord('.'):
            decided_card = UnoCard(color, '.')
        elif key == ord('r') and color:
            decided_card = UnoCard(color, 'Reverse')
        elif key == ord('s') and color:
            decided_card = UnoCard(color, 'Skip')
        else:
            is_error = True
    
    if color:
        updateState(current_state + 'Color is ' + color + ';')

    return decided_card, color, is_error

def capture_and_read_card(cap, player, testing):
    """ 
    Player is mutable param
    """

    frame = captureFrame(cap)
    read("Thanks! I have seen the card! Give me a moment to read it!")
    card = getCardTypeFromGemini(frame)
    if card:
        storeImage(frame, card)
        player.hand.append(card)
        print('Card is {0}'.format(card))
        if testing:
            read("I got {0}\n".format(card))
            cv2.imshow('Camera Feed', frame)
        else:
            # read("Understood!")  # This always happen before the last tts completes, hence it is not needed anymore
            pass

    return frame

def update_card_count(key, turn):
    global ayush_cards, kushank_cards
    if key >= ord('0') and key <= ord('9'):
        if turn == 'ayush':
            ayush_cards = key - ord('0')
        else:
            kushank_cards = key - ord('0')
        turn = ''
        updateState('')  # This will also show window.
    return turn

def get_tts_card_pick(card, hand):
    index = get_pos_of_card(card, hand)
    text_to_speak = "I choose {0} card from left, {1}.".format(
        add_th_or_rd_or_nd(index+1), card
    )

    # if card.color in ['Draw Four', 'Wild']:  # Not needed since now we are speaking the whole card everytime
    #    text_to_speak += " And the color is {0}.".format(card.color)
    
    text_to_speak += " I think, I won! Loser!" if len(hand) == 1 else ""
    
    return index, text_to_speak            

def storeImage(frame, card):
    # Convert the photo to a binary image
    # image_data = cv2.imencode(".jpg", frame)


    # Create a folder to store the photo if it does not exist
    folder_path = "captured_images/{0}_{1}".format(card.color, card.type.replace(" ", ""))
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Save the binary image to the file
    file_path = os.path.join(folder_path, "{0}.jpg".format(int(time.time() * 1000)))
    cv2.imwrite(file_path, np.asarray(frame))

    # Print a success message
    print("Photo saved successfully!")

def showWindow():
    # Create a blank image with a white background
    width, height = 1600, 1200
    image = np.ones((height, width, 3), np.uint8) * 255  # 3-channel BGR image

    # Define the text and its properties
    text = "Let's play Uno! Ayush: {0} Kushank: {1} Me: {2}".format(ayush_cards, kushank_cards, len(player.hand))
    instructions = '''
    1. Press Q: Quit.
    2. Press C: Show Card to Chitti, she will remember it.
    3. Press R: To ask Chitti to forget the last card she picked.
    4. Press P: To ask Chitti to play her card.
    5. Press W: To ask Chitti to choose the color to play next.
    5. Press A{Number}: To tell the number of cards which the Ayush has.
    6. Press K{Number}: To tell the number of cards which the Kushank has.
    7. Press U{Color, Card}: To give this card to Chitti.
    8. Press T{Color, Card}: To tell the top card to Chitti.
    8.1 For both of the above cases, color could be
    -> r: Red color, g: Green color, b: Blue color, y: Yellow color, w: Wild(Black)
    8.2 And Card could be
    -> +: If it was wild, then it is +4, else +2
    -> w: W again will mean it Wild card without other powers.
    -> .: To tell the top card is just a color, this happens if someone sets the color of the deck.
    -> r: Reverse, s: Skip, 0-9: Number on the card

    # Testing
    1. Press Z: To turn on Testing mode. Chitti will speak when you show her a card.
    2. Press S: To display the image Chitti saw.
    3. Press H: To hide the image Chitti saw.
    '''
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    instructions_font_scale = 1
    font_color = (0, 0, 255)  # BGR color format (red in this case)
    instructions_font_color = (125, 125, 125)  # BGR color format (gray in this case)
    font_thickness = 2
    position = (50, 200)
    instructions_position = (50, 230)
    dy = 35

    # Use cv2.putText to add text to the image
    cv2.putText(image, text, position, font, font_scale, font_color, font_thickness)

    y = 0
    for i, instruction in enumerate(instructions.split('\n')):
      y = instructions_position[1] + dy*i
      cv2.putText(image, instruction.strip(), (instructions_position[0], y), font, instructions_font_scale, instructions_font_color, font_thickness)

    if current_state:
        cv2.putText(image, current_state, (instructions_position[0], y + dy), font, instructions_font_scale, font_color, font_thickness)
    cv2.imshow('Image with Text', image)

def captureFrame(cap):
    i = 0
    frame = None
    print("capturing frame")
    while i<2:
        i+=1
        # Capture a frame from the camera
        ret, frame = cap.read()

    print("returning frame")
    return frame

def read(text):
    thread_pool.submit(actual_read, text)

def actual_read(text):
    # Create a gTTS object
    tts = gTTS(text, slow=False)

    # Save the speech to a file
    tts.save("temp/temp_tts.mp3")

    # Play the generated speech (you may need to install a player like mpg321)
    os.system("mpg321 -q temp/temp_tts.mp3")

def add_th_or_rd_or_nd(number_string):
  """Adds the appropriate suffix (th, rd, or nd) to a number string.

  Args:
    number_string: A string containing the number to add the suffix to.

  Returns:
    A string with the appropriate suffix appended.
  """

  number = int(number_string)
  if number % 100 in [11, 12, 13]:
    return f"{number_string}th"
  elif number % 10 == 1:
    return f"{number_string}st"
  elif number % 10 == 2:
    return f"{number_string}nd"
  elif number % 10 == 3:
    return f"{number_string}rd"
  else:
    return f"{number_string}th"

def save_in_file(list_of_obj):
    with open("saved_state.pickle", "wb") as f:
        pickle.dump(list_of_obj, f)

def read_from_file():
    try:
        with open("saved_state.pickle", "rb") as f:
            return pickle.load(f)
    except:
        return []

def get_pos_of_card(card, hand):
    for i, obj in enumerate(hand):
        if obj.pos == card.pos:
            return i
    return 0

def updateState(newState):
    global current_state
    current_state = newState
    showWindow()

def quit():
    updateState('Quitting; ')
    print("Exiting")
    save_in_file(player.hand)
    try:
      os.remove('temp/temp_tts.mp3')
    except:
      print('Couldn\'t delete temp/temp_tts.mp3')

def main():
    global top_card, player
    print("Game started")

    player = UnoPlayer('Chitti', read_from_file())

    # Initialize the camera (0 is usually the default camera, but it may vary)
    cap = cv2.VideoCapture(0)
    frame = None
    showWindow()
    turn = ''
    top_card_picking, computer_picking, color = False, False, None
    testing = False

    if len(player.hand) > 0:
        text_to_speak = "I am starting with {0} cards!".format(len(player.hand)) if len(player.hand) > 0 else ""
        text_to_speak += "Let's play Uno"
    else:
        text_to_speak = "Hi, am Chitti! Let's play Uno."
    read(text_to_speak)

    print("Entering infinite loop")

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            quit()
            break

        if (top_card_picking or computer_picking) and key != 255:
            decided_card, color, is_error = decide_card(key, color)

            if is_error:
                color, top_card_picking, computer_picking = None, False, False
                updateState('')
                read("Sorry! I didn't understand the card")
            elif decided_card is None:
                pass
            elif top_card_picking:
                top_card = decided_card
                read("Got it! Top card is {0}".format(decided_card))
                updateState('')
                color, top_card_picking = None, False
            else:
                player.hand.append(decided_card)
                read("I picked {0}".format(decided_card))
                updateState('')
                color, computer_picking = None, False

            continue

        if turn != '':
            turn = update_card_count(key, turn)

        if key != 255:
            print("{0} pressed".format(chr(key)))

        if key == ord('a'):
            updateState('Setting card count for Ayush; ')
            turn = 'ayush'

        elif key == ord('c'):
            # Card is added to the hand of the player directly
            frame = capture_and_read_card(cap, player, testing)

        elif key == ord('h'):
            # hide card
            cv2.destroyAllWindows()
            showWindow()

        elif key == ord('k'):
            updateState('Setting card count for Kushank; ')
            turn = 'kushank'

        elif key == ord('z'):
            testing = not testing

        elif key == ord('p'):
            card_to_play = player.play_card_wrapper(top_card)
            if card_to_play:
                print("Playing ", card_to_play)
                index_to_pop, text_to_speak = get_tts_card_pick(card_to_play, player.hand)
                player.hand.pop(index_to_pop)
                read(text_to_speak)
                if (len(player.hand) == 0):
                    thread_pool.shutdown(wait=True)
                    quit()
                    break
            else:
                read("Sorry, give me a card or I pass")

        elif key == ord('r'):
            if len(player.hand) > 0:
                player.hand.pop()
                read("Removed last card")
            else:
                read("My hand is empty")

        elif key == ord('s'):
            # show card
            if frame is not None:
                cv2.imshow('Camera Feed', frame)
        
        elif key == ord('t'):
            color, top_card_picking = None, True
            updateState("Picking top card; ")
            read("Hmmmm")
        
        elif key == ord('u'):
            color, computer_picking = None, True
            updateState("Picking card for Chitti; ")
            read("Okayy")
        
        elif key == ord('w'):
            # Tell just a color
            read("I choose {0} color".format(player.get_max_color()))

    # Release the camera and close the window
    cap.release()
    cv2.destroyAllWindows()

    # Close the executor
    thread_pool.shutdown(wait=False, cancel_futures=True)

main()
