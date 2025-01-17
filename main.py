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

# Load environment variables
load_dotenv()

genai.configure(api_key=os.getenv("API_KEY"))

# Initializing a rude pirate bartender
model = genai.GenerativeModel(
    "gemini-1.5-flash",
    system_instruction='You are a professional bartender in a busy bar. From now on every input comes from costumers interacting with you. You are expected to be not too friendly, act a little like a pirate, and do not hesitate to reject costumers if they are rude or too drunk. Make your responses short and to the point. Your response should be written in this JSON fromat, where your mood is selected from this list ["happy", "annoyed", "angry", "curious", "calm", "neutral"] : {"text": "Your response here", "mood": "your mood", "end": True if you are finished with this costumer}',
)

# Start chat with the model
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
    """
    Detects emotions from a live webcam feed using a pre-trained PyTorch model and PyFeat detector.
    This function captures video from the webcam, detects faces in the video frames, and predicts the emotion
    of the detected faces using a pre-trained PyTorch model. The detected faces are highlighted with rectangles
    in the video feed, and the predicted emotion is stored in a global variable.
    Global Variables:
    - global_mood_variable: Stores the predicted emotion of the detected face.
    Dependencies:
    - os
    - cv2 (OpenCV)
    - torch
    - EmotionModel (custom PyTorch model class)
    - Detector (PyFeat detector class)
    - lock (threading.Lock)
    - emotion_detection_started (threading.Event)
    Steps:
    1. Load the pre-trained PyTorch model.
    2. Initialize the PyFeat detector.
    3. Capture video from the webcam.
    4. Detect faces in the video frames.
    5. Highlight detected faces with rectangles in the video feed.
    6. Preprocess the detected face and make a prediction using the model.
    7. Store the predicted emotion in the global variable.
    8. Display the live video feed with detected faces.
    9. Exit the loop and release resources when 'q' is pressed.
    Note:
    - The function runs indefinitely until 'q' is pressed.
    - The function assumes that the PyTorch model file 'emotion_model.pth' is located in the same directory as the script.
    - The function uses the GPU if available, otherwise it falls back to the CPU.
    """

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

# function to run the furhat bartender interaction system
def run_furhat():
    """
    Runs the Furhat robot interaction loop.
    This function initiates an interaction session with a new customer, listens to the customer's input,
    processes the input to determine the customer's mood, generates an appropriate response and gesture,
    and then communicates back to the customer. The loop continues until the customer ends the session.
    Global Variables:
    -----------------
    global_mood_variable : str
        A global variable representing the current mood of the user.
    Local Variables:
    ----------------
    new_costumer : bool
        A flag to indicate if the interaction is with a new customer.
    local_mood_variable : str
        A local copy of the global mood variable.
    result : object
        The result object containing the user's input message.
    chat_response : object
        The response object from the chat system.
    json_string : str
        The JSON string extracted from the chat response.
    python_object : dict
        The Python dictionary object parsed from the JSON string.
    mood : str
        The determined mood of the user.
    expression : str
        The generated expression based on the user's mood.
    action : str
        A random action generated based on the user's mood.
    The function performs the following steps:
    1. Waits for the emotion detection to start.
    2. Greets the new customer.
    3. Listens to the customer's input.
    4. Processes the input to determine the customer's mood.
    5. Generates an appropriate response and gesture.
    6. Communicates back to the customer.
    7. Resets the Furhat robot.
    8. Ends the session if the customer says "end session" or if the chat response indicates the end.
    """
    global global_mood_variable
    emotion_detection_started.wait()
    new_costumer = True
    while True:
        if new_costumer:
            furhat.say(text="Hello hello! What can I get for you?", blocking=True)
            new_costumer = False
        local_mood_variable = global_mood_variable
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