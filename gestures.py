from furhat_remote_api import FurhatRemoteAPI
from time import sleep
import random

def reset_furhat(furhat):
    furhat.gesture(body={
        "frames": [
            {"time": [0.2], "params": {
                "SMILE_CLOSED": 0.0, 
                "SMILE_OPEN": 0.0, 
                "EXPR_ANGER": 0.0, 
                "EXPR_SAD": 0.0,
                "BROW_DOWN_LEFT": 0.0, 
                "BROW_DOWN_RIGHT": 0.0,
                "BROW_UP_LEFT": 0.0, 
                "BROW_UP_RIGHT": 0.0,
                "BLINK_LEFT": 0.0, 
                "BLINK_RIGHT": 0.0,
                "EYE_SQUINT_LEFT": 0.0, 
                "EYE_SQUINT_RIGHT": 0.0
            }},
            {"time": [0.3], "params": {
                "NECK_TILT": 0.0, 
                "NECK_PAN": 0.0, 
                "NECK_ROLL": 0.0, 
                "GAZE_PAN": 0.0, 
                "GAZE_TILT": 0.0
            }},
            {"time": [0.5], "params": {
                "LOOK_LEFT": 0.0, 
                "LOOK_RIGHT": 0.0, 
                "LOOK_UP": 0.0, 
                "LOOK_DOWN": 0.0
            }}
        ],
        "class": "furhatos.gestures.Gesture"
    })

def determine_mood(llm_response):
    """Determines the bartender's mood based on LLM response."""
    keywords_to_moods = {
        "happy": ["glad", "happy", "pleased", "joyful"],
        "annoyed": ["annoyed", "frustrated", "bothered", "irritated"],
        "angry": ["angry", "mad", "furious"],
        "curious": ["interested", "curious", "wondering"],
        "calm": ["calm", "relaxed", "peaceful"],
        "excited": ["excited", "thrilled", "enthusiastic"]
    }
    
    
    for mood, keywords in keywords_to_moods.items():
        if any(keyword in llm_response.lower() for keyword in keywords):
            return mood

    return "neutral"

def generate_expression(mood):
    """Generates Furhat expressions based on the determined mood."""
    expressions = {
        "happy": {
            "frames": [{"time": [0.5], "params": {"SMILE_CLOSED": 1.0}}],
            "class": "furhatos.gestures.Gesture"
        },
        "annoyed": {
            "frames": [
                {"time": [0.5], "params": {"BROW_UP_LEFT": 1.0, "BROW_UP_RIGHT": 1.0}},
                {"time": [0.7], "params": {"GAZE_PAN": -1.0, "GAZE_TILT": 0.5}},
                {"time": [1.0], "params": {"reset": True}}
            ],
            "class": "furhatos.gestures.Gesture"
        },
        "angry": {
            "frames": [{"time": [0.0], "params": {"reset": True}}, {"time": [0.5], "params": {"BROW_FURROWED": 1.0, "MOUTH_TIGHT": 1.0}}, {"time": [1.0], "params": {"reset": True}}], 
            "class": "furhatos.gestures.Gesture"
        },
        "curious": {
            "frames": [{"time": [0.5], "params": {"BROW_UP_LEFT": 1.0, "BROW_FURROWED": 0.0, "MOUTH_TIGHT": 0.0}}],
            "class": "furhatos.gestures.Gesture"
        },
        "calm": {
            "frames": [{"time": [0.5], "params": {"SMILE_CLOSED": 0.5, "BROW_FURROWED": 0.0, "MOUTH_TIGHT": 0.0}}],
            "class": "furhatos.gestures.Gesture"
        },
        "neutral": {
            "frames": [{"time": [0.5], "params": {"SMILE_CLOSED": 0.2, "BROW_FURROWED": 0.0, "MOUTH_TIGHT": 0.0}}],
            "class": "furhatos.gestures.Gesture"
        }
    }
    return expressions.get(str.lower(mood), expressions["neutral"])


def get_actions(mood):
    actions = {"happy": [
            {
                "frames": [
                    {"time": [0.0], "params": {"reset": True}},
                    {"time": [0.3], "params": {"SMILE_CLOSED": 1.0}},
                    {"time": [0.5], "params": {"SMILE_CLOSED": 0.8}}
                ], 
                "class": "furhatos.gestures.Gesture"
            },
            {
                "frames": [
                    {"time": [0.0], "params": {"reset": True}},
                    {"time": [0.3], "params": {"SMILE_OPEN": 0.8}},
                    {"time": [0.5], "params": {"SMILE_OPEN": 1.0}},
                    {"time": [0.7], "params": {"SMILE_OPEN": 0.7}}
                ], 
                "class": "furhatos.gestures.Gesture"
            }
        ],
        "nod_head": [
            {
                "frames": [
                    {"time": [0.2], "params": {"NECK_TILT": 10.0}},
                    {"time": [0.5], "params": {"NECK_TILT": 5.0}},
                    {"time": [1.0], "params": {"reset": True}}
                ], 
                "class": "furhatos.gestures.Gesture"
            },
            {
                "frames": [
                    {"time": [0.2], "params": {"NECK_TILT": 12.0}},
                    {"time": [0.4], "params": {"NECK_TILT": 20.0}},
                    {"time": [0.6], "params": {"NECK_TILT": 8.0}},
                    {"time": [1.0], "params": {"reset": True}}
                ], 
                "class": "furhatos.gestures.Gesture"
            }
        ],
        "annoyed": [
            {
                "frames": [
                    {"time": [0.0], "params": {"reset": True}},
                    {"time": [0.2], "params": {"EXPR_ANGER": 1.0}},
                    {"time": [0.5], "params": {"BROW_DOWN_LEFT": 1.0, "BROW_DOWN_RIGHT": 1.0}},
                    {"time": [0.7], "params": {"EXPR_ANGER": 0.8}}
                ], 
                "class": "furhatos.gestures.Gesture"
            },
            {
                "frames": [
                    {"time": [0.0], "params": {"reset": True}},
                    {"time": [0.2], "params": {"EXPR_ANGER": 0.9}},
                    {"time": [0.4], "params": {"BROW_DOWN_LEFT": 1.0, "BROW_DOWN_RIGHT": 1.0}},
                    {"time": [0.6], "params": {"EXPR_ANGER": 0.7}}
                ], 
                "class": "furhatos.gestures.Gesture"
            }
        ],
        "curious": [
            {
                "frames": [
                    {"time": [0.0], "params": {"reset": True}},
                    {"time": [0.3], "params": {"NECK_TILT": 10.0}},
                    {"time": [0.6], "params": {"NECK_TILT": 15.0}},
                    {"time": [0.8], "params": {"NECK_TILT": 10.0}}
                ], 
                "class": "furhatos.gestures.Gesture"
            },
            {
                "frames": [
                    {"time": [0.0], "params": {"reset": True}},
                    {"time": [0.3], "params": {"NECK_TILT": 12.0}},
                    {"time": [0.5], "params": {"NECK_TILT": 8.0}},
                    {"time": [0.7], "params": {"NECK_TILT": 14.0}}
                ], 
                "class": "furhatos.gestures.Gesture"
            }
        ],
        "calm": [
            {
                "frames": [
                    {"time": [0.0], "params": {"reset": True}},
                    {"time": [0.5], "params": {"BLINK_LEFT": 0.5, "BLINK_RIGHT": 0.5}},
                    {"time": [1.0], "params": {"BLINK_LEFT": 0.4, "BLINK_RIGHT": 0.4}}
                ], 
                "class": "furhatos.gestures.Gesture"
            },
            {
                "frames": [
                    {"time": [0.0], "params": {"reset": True}},
                    {"time": [0.6], "params": {"BLINK_LEFT": 0.6, "BLINK_RIGHT": 0.6}},
                    {"time": [1.2], "params": {"BLINK_LEFT": 0.5, "BLINK_RIGHT": 0.5}}
                ], 
                "class": "furhatos.gestures.Gesture"
            }
        ],
        "excited": [
            {
                "frames": [
                    {"time": [0.0], "params": {"reset": True}},
                    {"time": [0.2], "params": {"EXPR_SAD": 0.0}},
                    {"time": [0.5], "params": {"SMILE_CLOSED": 1.0}},
                    {"time": [0.8], "params": {"EXPR_SAD": 0.0}}
                ], 
                "class": "furhatos.gestures.Gesture"
            },
            {
                "frames": [
                    {"time": [0.0], "params": {"reset": True}},
                    {"time": [0.3], "params": {"SMILE_OPEN": 0.8}},
                    {"time": [0.6], "params": {"EXPR_SAD": 0.0}},
                    {"time": [0.9], "params": {"EXPR_SAD": 0.0}},
                ], 
                "class": "furhatos.gestures.Gesture"
            }
        ],
        "neutral": [
            {
                "frames": [
                    {"time": [0.0], "params": {"reset": True}},
                    {"time": [0.5], "params": {"BLINK_LEFT": 0.5, "BLINK_RIGHT": 0.5}},
                    {"time": [0.7], "params": {"BLINK_LEFT": 0.6, "BLINK_RIGHT": 0.6}},
                ], 
                "class": "furhatos.gestures.Gesture"
            },
            {
                "frames": [
                    {"time": [0.0], "params": {"reset": True}},
                    {"time": [0.4], "params": {"BLINK_LEFT": 0.6, "BLINK_RIGHT": 0.6}},
                    {"time": [0.6], "params": {"BLINK_LEFT": 0.5, "BLINK_RIGHT": 0.5}},
                    {"time": [0.8], "params": {"BLINK_LEFT": 0.5, "BLINK_RIGHT": 0.5}},
                ], 
                "class": "furhatos.gestures.Gesture"
            }
        ]}
    return actions.get(str.lower(mood), actions["neutral"])

def get_random_action(mood):
    possible_actions = get_actions(mood)
    return random.choice(possible_actions)



def eye_roll_annoyed(furhat):
    furhat.gesture(body={
        "frames": [
            {
                "time": [0.2],
                "params": {
                    "BROW_UP_LEFT": 1.0,
                    "BROW_UP_RIGHT": 1.0
                }
            },
            {
                "time": [0.4],
                "params": {
                    "GAZE_TILT": -50.0
                }
            },
            {
                "time": [0.6],
                "params": {
                    "GAZE_PAN": -50.0
                }
            },
            {
                "time": [0.8],
                "params": {
                    "GAZE_TILT": 50.0
                }
            },
            {
                "time": [1.0],
                "params": {
                    "GAZE_PAN": 0.0,
                    "GAZE_TILT": 0.0,
                    "BROW_UP_LEFT": 0.0,
                    "BROW_UP_RIGHT": 0.0
                }
            }
        ],
        "class": "furhatos.gestures.Gesture"
    })