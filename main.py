from feat import Detector
import cv2
import torch
import os

import time
from furhat_remote_api import FurhatRemoteAPI
import google.generativeai as genai
import os
from dotenv import load_dotenv
import gestures
import json
import threading


from gestures import determine_mood, generate_expression, eye_roll_annoyed, get_random_action, reset_furhat
# from mood_recognition import emotion_detection
from emotion_nn import EmotionModel
from emotion_nn_resnet import EmotionClassifier

load_dotenv()

genai.configure(api_key=os.getenv("API_KEY"))

# Initializing a rude pirate bartender
model = genai.GenerativeModel(
    "gemini-1.5-flash",
    system_instruction='You are a professional bartender in a busy bar. From now on every input comes from costumers interacting with you. You are expected to be not too friendly, act a little like a pirate, and do not hesitate to reject costumers if they are rude or too drunk. Make your responses short and to the point. Your response should be written in this JSON fromat, where your mood is selected from this list ["happy", "annoyed", "angry", "curious", "calm", "neutral"] : {"text": "Your response here", "mood": "your mood", "end": True if you are finished with this costumer}',
)

# Initializeing a kind bartender
# model = genai.GenerativeModel(
#     "gemini-1.5-flash",
#     system_instruction=""
# )
chat = model.start_chat()



# Create an instance of the FurhatRemoteAPI class, providing the address of the robot or the SDK running the virtual robot
furhat = FurhatRemoteAPI("localhost")

# Get the voices on the robot
voices = furhat.get_voices()

# Set the voice of the robot
furhat.set_voice(name='Matthew')

# Threading variables
global_mood_variable = 0
lock = threading.Lock()
emotion_detection_started = threading.Event()

def emotion_detection():
    global global_mood_variable
    print("Starting emotion detection")
    # Load the PyTorch model
    state_dict_path = os.path.join(os.path.dirname(__file__), 'emotion_model.pth')
    model = EmotionModel()
    # model = EmotionClassifier()
    model.load_state_dict(torch.load(state_dict_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
    # model = torch.load(model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    model.eval()

    # Initialize PyFeat detector
    detector = Detector(device="cuda")
    print(detector.device)
    cap = cv2.VideoCapture(0)

    frame_count = 0
    with lock:
        emotion_detection_started.set()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        
        # Convert frame from BGR (OpenCV format) to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Detect facial features every 10 frames
        faces = detector.detect_faces(frame=rgb_frame)

        # Draw rectangles around detected faces
        for face in faces[0]:
            x1, y1, x2, y2, confidence = face
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        # Display the live feed
        cv2.imshow('Webcam', frame)


        # Preprocess the detected face and make a prediction
        if len(faces[0]) > 0:
            x1, y1, x2, y2, _ = faces[0][0]
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            face = rgb_frame[y1:y2, x1:x2]
            face = cv2.resize(face, (128, 128))
            face = face.transpose((2, 0, 1))  # Convert to CHW format
            face = torch.tensor(face).float().unsqueeze(0)  # Add batch dimension

            with torch.no_grad():
                prediction = model(face)
                # print(prediction)
                emotion = torch.argmax(prediction, dim=1).item()
                # print(f"Detected emotion: {emotion}")
                with lock:
                    global_mood_variable = emotion
        
        # frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def run_furhat():
    global global_mood_variable
    emotion_detection_started.wait()
    new_costumer = True
    while True:
        if new_costumer:
            furhat.say(text="Hello hello! What can I get for you?", blocking=True)
            new_costumer = False
        local_mood_variable = global_mood_variable
        # Say "Hi there!"
        result = furhat.listen()
        if result.message == "":
            result.message = "nothing"
        print("User said: ", result.message)
        if result.message.lower() == "end session":
            break
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
            time.sleep(0.2)
            furhat.say(text="Goodbye!", blocking=True)
            new_costumer = True
            time.sleep(2)            

#RUN two threaded application here

# Create and start the threads
furhat_thread = threading.Thread(target=run_furhat)
emotion_thread = threading.Thread(target=emotion_detection)

emotion_thread.start()
furhat_thread.start()



# join and close threads
furhat_thread.join()
emotion_thread.join()