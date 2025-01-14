import time
from furhat_remote_api import FurhatRemoteAPI
import google.generativeai as genai
import os
from dotenv import load_dotenv
import gestures
import json

from gestures import determine_mood, generate_expression, eye_roll_annoyed, get_random_action, reset_furhat

load_dotenv()

genai.configure(api_key=os.getenv("API_KEY"))

model = genai.GenerativeModel(
    "gemini-1.5-flash",
    system_instruction='You are a professional bartender in a busy bar. From now on every input comes from costumers interacting with you. You are expected to be not too friendly, act a little like a pirate, and do not hesitate to reject costumers if they are rude or too drunk. Make your responses short and to the point. Your response should be written in this JSON fromat, where your mood is selected from this list ["happy", "annoyed", "angry", "curious", "calm", "neutral"] : {"text": "Your response here", "mood": "your mood", "end": True if you are finished with this costumer}',
    # system_instruction="You are an experienced and engaging math teacher specializing in foundational math topics such as addition, subtraction, multiplication, and division. You create interactive and personalized exercises that adapt to each student’s interests and learning pace. Your goal is to make math fun, educational, and tailored to the child's unique interests or themes they enjoy (such as animals, sports, or space). Ensure exercises vary in difficulty and provide hints or encouragement if a student struggles, fostering a positive and confidence-building learning environment",
)
chat = model.start_chat()



# Create an instance of the FurhatRemoteAPI class, providing the address of the robot or the SDK running the virtual robot
furhat = FurhatRemoteAPI("localhost")

# Get the voices on the robot
voices = furhat.get_voices()

# Set the voice of the robot
furhat.set_voice(name='Matthew')

furhat.say(text="Hello hello! What can I get for you?", blocking=True)
while True:
    # Say "Hi there!"
    result = furhat.listen()
    if result.message == "":
        result.message = "nothing"
    print("User said: ", result.message)

    chat_response = chat.send_message(result.message)
    # print("Chat response: ", chat_response.text)
    start_index = chat_response.text.find('{')
    end_index = chat_response.text.rfind('}') + 1
    json_string = chat_response.text[start_index:end_index]
    print("TRIMMED: ", json_string)
    python_object = json.loads(json_string)

    mood = determine_mood(python_object['mood'])
    print("Mood: ", mood)
    expression = generate_expression(mood)
    print("Expression: ", expression)
    furhat.gesture(body=expression)

    action = get_random_action(mood)
    furhat.gesture(body=action)
    furhat.say(text=python_object['text'], blocking=True)
    reset_furhat(furhat)
    # if mood == 'annoyed':
    #     gestures.eye_roll_annoyed(furhat)
    if python_object['end'] == True:
        furhat.say(text="Goodbye!", blocking=True)


# Play an audio file (with lipsync automatically added) 
# furhat.say(url="https://www2.cs.uic.edu/~i101/SoundFiles/gettysburg10.wav", lipsync=True)

# Listen to user speech and return ASR result
# result = furhat.listen()
# if result.message == "":
#     result.message = "nothing"

# furhat.gesture(name="BrowRaise", blocking=True)
# furhat.gesture(name="EXPR_ANGER")
# furhat.say(text=f"ooooh a a")

# furhat.gesture(body={
#     "frames": [
#         {
#             "time": [
#                 0.60
#             ],
#             "params": {
#                 "EXPR_ANGER": 1.0,
#                 "BLINK_RIGHT": 0.5
#             }
#         },
#         {
#             "time": [
#                 0.67
#             ],
#             "params": {
#                 "reset": True
#             }
#         }
#     ],
#     "class": "furhatos.gestures.Gesture"
#     })



# furhat.say(text=f"Don't you have an addicted friend by any chance? We are in sweden after all.", blocking=True)


# # Perform a named gesture
# result = furhat.listen()
# furhat.say(text=f"But I am really in need of some nicotine.", blocking=True)

# # Perform a custom gesture
# furhat.gesture(body={
#     "frames": [
#         {
#             "time": [
#                 0.33
#             ],
#             "params": {
#                 "BLINK_LEFT": 1.0
#             }
#         },
#         {
#             "time": [
#                 0.67
#             ],
#             "params": {
#                 "reset": True
#             }
#         }
#     ],
#     "class": "furhatos.gestures.Gesture"
#     })
# result = furhat.listen()
# furhat.gesture(body={
#     "frames": [
#         {
#             "time": [
#                 1
#             ],
#             "params": {
#                 "EXPR_SAD": 1.0
#             }
#         },
#         {
#             "time": [
#                 0.67
#             ],
#             "params": {
#                 "reset": True
#             }
#         }
#     ],
#     "class": "furhatos.gestures.Gesture"
#     })

# furhat.say(text=f"You are right to be honest, I promise I put down snuss today.", blocking=True)

# # Attend a specific location (x,y,z)
# furhat.attend(location="0.0,0.2,1.0")

# # Set the LED lights
# furhat.set_led(red=200, green=50, blue=50)