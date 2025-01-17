from feat import Detector
import cv2
import torch
import os
from emotion_nn import EmotionModel
from emotion_nn_resnet import EmotionClassifier

def emotion_detection():
    # global global_mood_variable
    # Load the PyTorch model
    # state_dict_path = os.path.join(os.path.dirname(__file__), 'emotion_model.pth')
    # model = EmotionModel()
    # model.load_state_dict(torch.load(state_dict_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
    state_dict_path = os.path.join(os.path.dirname(__file__), 'resnet_emotion_classifier.pth')
    print("Loading model")
    # model = EmotionModel()
    model = EmotionClassifier(7)
    model.load_state_dict(torch.load(state_dict_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), weights_only=True))
    print("Model loaded")
    # model = torch.load(model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    model.eval()

    # Initialize PyFeat detector
    detector = Detector(device="cuda")
    print(detector.device)
    cap = cv2.VideoCapture(0)

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        
        # Convert frame from BGR (OpenCV format) to RGB
        # rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Detect facial features every 10 frames
        rgb_frame = frame
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
            # cv2.imshow('cut out face', face)
            face = cv2.resize(face, (128, 128))
            # cv2.imshow('after resize', face)
            face = face.transpose((2, 0, 1))  # Convert to CHW format
            face = torch.tensor(face).float().unsqueeze(0)  # Add batch dimension

             # Make the prediction
            with torch.no_grad():
                output = model(face)
                _, predicted = torch.max(output.data, 1)
                predicted_class = predicted.item()
                print(f"Detected emotion: {predicted_class}")

            # with torch.no_grad():
            #     prediction = model(face)
            #     # print(prediction)
            #     emotion = torch.argmax(prediction, dim=1).item()
            #     print(f"Detected emotion: {emotion}")
                # global_mood_variable = emotion
        
        # frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

emotion_detection()