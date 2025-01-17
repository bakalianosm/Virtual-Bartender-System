import time
from furhat_remote_api import FurhatRemoteAPI
import google.generativeai as genai
import os
from dotenv import load_dotenv
import gestures
import json
import threading

from gestures import determine_mood, generate_expression, eye_roll_annoyed, get_random_action, reset_furhat
from mood_recognition import emotion_detection

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

def run_furhat(lock):
    global global_mood_variable
    furhat.say(text="Hello hello! What can I get for you?", blocking=True)
    while True:
        local_mood_variable = global_mood_variable
        # Say "Hi there!"
        result = furhat.listen()
        if result.message == "":
            result.message = "nothing"
        print("User said: ", result.message)
        with lock:
            local_mood_variable = global_mood_variable
        print("Mood of user: ", local_mood_variable)
        chat_response = chat.send_message(result.message)
        # print("Chat response: ", chat_response.text)
        start_index = chat_response.text.find('{')
        end_index = chat_response.text.rfind('}') + 1
        json_string = chat_response.text[start_index:end_index]
        # print("TRIMMED: ", json_string)
        python_object = json.loads(json_string)

        mood = determine_mood(python_object['mood'])
        print("Mood of agent: ", mood)
        expression = generate_expression(mood)
        # print("Expression: ", expression)
        furhat.gesture(body=expression)

        action = get_random_action(mood)
        furhat.gesture(body=action)
        furhat.say(text=python_object['text'], blocking=True)
        reset_furhat(furhat)
        # if mood == 'annoyed':
        #     gestures.eye_roll_annoyed(furhat)
        if python_object['end'] == True:
            furhat.say(text="Goodbye!", blocking=True)
            break


#RUN two threaded application here

global_mood_variable = 0
lock = threading.Lock()

# Create and start the threads
furhat_thread = threading.Thread(target=run_furhat, args=(lock,))
emotion_thread = threading.Thread(target=emotion_detection, args=(lock,))

lock.acquire()
emotion_thread.start()
lock.release()
time.sleep(1)
lock.acquire()
lock.release()
furhat_thread.start()



# join and close threads
furhat_thread.join()
emotion_thread.join()