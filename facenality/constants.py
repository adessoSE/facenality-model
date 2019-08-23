# Image parameters
IMAGE_SIZE = 224

PATH_X = "../dataset/all-cropped/neutral/"
PATH_Y = "../dataset/all.json"

PATH_DATA_SET_TRAIN = "../dataset/train"
PATH_DATA_SET_TEST = "../dataset/test"

N_MIN_CONV_LAYERS = 1
N_MAX_CONV_LAYERS = 16

GRID_SEARCH_NEURONS = [1, 5, 10, 15, 20, 25, 30]

TRAITS = ["A-Warmth", "B-Reasoning", "C-Emotional-Stability", "E-Dominance", "F-Liveliness", "G-Rule-Consciousness", "H-Social-Boldness", "I-Sensitivity", "L-Vigilance", "M-Abstractedness", "N-Privateness", "O-Apprehension", "Q1-Openness-to-Change", "Q2-Self-Reliance", "Q3-Perfectionism", "Q4-Tension"]
TRAIT_TRESHHOLDS = ["3.9", "3.7", "3.6", "3.7", "3.4", "3.3", "3.3", "3.7", "2.8", "3.7", "3", "3.2", "4", "3.4", "3.3", "2.7"]

IMAGE_SIZE = 224
COLOR_CHANNELS = 3

HIDDEN_LAYERS = 5
BATCH_SIZE = 15
EPOCHS = 75