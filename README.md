# Intelligent and Interactive Systems Project

## Overview
This repository contains the project for the Intelligent and Interactive Systems course. The project involves creating an interactive system using the Furhat robot, which can recognize user moods and respond accordingly.

## Files and Directories

### main.py
`main.py` is our merged solution that integrates all subsystems to create a cohesive interactive experience with the Furhat robot.

### mood_recognition.py
`mood_recognition.py` is our user perception subsystem. It is responsible for recognizing the mood of the user based on their input.

### interaction.py
`interaction.py` is our interaction subsystem. It handles the interaction logic between the user and the Furhat robot.

### gestures.py
`gestures.py` includes the possible gestures for our Furhat agent. It also defines the necessary functions to generate and execute these gestures.

### system_design
The `system_design` folder contains a diagram of the system design. This diagram provides a visual representation of how the different subsystems interact with each other.

![System Design Diagram](system_design/system_architecture.png)

### train_model
The `train_model` folder was used for training the model for the user perception subsystem. It contains scripts and data used during the training process.

## Getting Started
To get started with this project, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/your-repo.git
    ```
2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Run the main script:
    ```bash
    python main.py
    ```

## Usage
The Furhat robot will start interacting with users, recognizing their moods, and responding with appropriate gestures and dialogue.

## Acknowledgements
We would like to thank our course instructors and peers for their support and guidance throughout this project.