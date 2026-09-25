"""All game settings in one place. Change numbers here to tune the game."""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # paths work from any folder

# ---------------- Window / camera ----------------
WINDOW_NAME = "Ninja Fruit CV"
GAME_WIDTH = 1280            # the game is drawn at this size
GAME_HEIGHT = 720
CAMERA_INDEX = 0             # 0 = default webcam. Try 1, 2 if you have more cameras
# Camera capture settings. Run `python tools/diagnose.py` to find the best ones for your webcam.
CAMERA_BACKEND = "dshow"     # "dshow" or "msmf" (Windows), "any" = let OpenCV choose
CAMERA_FOURCC = "MJPG"       # MJPG = compressed -> most webcams give 30 fps at 720p.
                             # Without it (raw YUY2) many webcams only give 5-10 fps at 720p!
CAPTURE_WIDTH = 1280
CAPTURE_HEIGHT = 720
CAPTURE_FPS = 60             # we ask for 60; the camera gives what it can
CAMERA_EXPOSURE = None       # None = auto. On Windows/DirectShow try -6 or -7 (shorter
                             # exposure = less motion blur + higher fps, but darker image)
MIRROR = True                # flip camera like a mirror (feels natural)
BACKGROUND_DIM = 0.0         # 0.0 = normal camera, 0.5 = camera 50% darker
FULLSCREEN = False           # press F in game to toggle

# ---------------- Hand tracking ----------------
MODEL_PATH = os.path.join(BASE_DIR, "models", "hand_landmarker.task")
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/latest/hand_landmarker.task")
MAX_HANDS = 2
DETECT_WIDTH = 640           # MediaPipe runs on a smaller frame -> faster
MIN_DETECTION_CONFIDENCE = 0.6
MIN_TRACKING_CONFIDENCE = 0.5
TRACKING_ASYNC = True        # True = MediaPipe LIVE_STREAM mode: runs in its own thread,
                             # so the game loop never waits for the model
INDEX_FINGER_TIP = 8         # MediaPipe landmark ids: 8 = index tip, 6 = PIP joint, 5 = MCP knuckle
INDEX_FINGER_PIP = 6
INDEX_FINGER_MCP = 5
# Finger guard: with motion blur the model often "folds" the fingertip back towards
# the palm (the point jumps down). If the tip looks too short or bent, we rebuild it
# from the two more stable joints (5 and 6).
FINGER_GUARD = True
FINGER_MIN_RATIO = 0.65      # |tip-PIP| / |PIP-MCP| below this = tip is suspicious
FINGER_MAX_BEND_DEG = 55     # angle between the two finger parts above this = suspicious

# One Euro filter (smooths finger jitter but keeps fast moves responsive)
FILTER_MIN_CUTOFF = 1.5
FILTER_BETA = 0.02

# ---------------- Blade (the cut line) ----------------
BLADE_TRAIL_TIME = 0.2       # seconds of trail kept on screen
BLADE_MIN_SPEED = 900        # px/s - finger must move faster than this to cut
BLADE_LOST_TIMEOUT = 0.2     # seconds without the hand -> start a new stroke
BLADE_MAX_SPEED = 9000      # px/s - faster than this is not a real hand = tracking glitch
BLADE_CORE_COLOR = (255, 255, 255)   # BGR
BLADE_GLOW_COLOR = (255, 220, 120)   # light blue glow (BGR)
BLADE_WIDTH = 10
BLADE_SMOOTH_STEPS = 6       # curve points between two tracked points (only for drawing)

# ---------------- Debug ----------------
SHOW_TIMING = False          # press T in game: shows ms spent in each step
TRACK_LOG_DIR = os.path.join(BASE_DIR, "logs")   # press L in game: saves finger data to CSV

# ---------------- Fruits ----------------
SHAPES = ["circle", "square", "triangle"]
FRUIT_RADIUS_MIN = 40
FRUIT_RADIUS_MAX = 60
GRAVITY = 1400               # px/s^2
PEAK_HEIGHT_MIN = 0.12       # fruit peak (fraction of screen height from top)
PEAK_HEIGHT_MAX = 0.45
SPIN_MAX = 3.0               # rad/s

# Spawning (difficulty goes up slowly over time)
SPAWN_INTERVAL_START = 1.4   # seconds between waves at the start
SPAWN_INTERVAL_MIN = 0.6
SPAWN_INTERVAL_DECAY = 0.01  # interval shrinks by this per second of play
WAVE_SIZE_START = 1          # fruits per wave at the start
WAVE_SIZE_MAX = 5
WAVE_SIZE_GROW_EVERY = 20    # +1 fruit per wave every N seconds

# ---------------- Slicing effects ----------------
HALF_PUSH_SPEED = 220        # how fast the two halves fly apart (px/s)
PARTICLES_PER_SLICE = 18
PARTICLE_LIFE = 0.6

# ---------------- Score / combo ----------------
POINTS_PER_FRUIT = 1
COMBO_WINDOW = 0.35          # cuts closer than this (seconds) join the same combo
COMBO_MIN = 3                # combo bonus starts at 3 fruits
HIGHSCORE_FILE = os.path.join(BASE_DIR, "highscore.json")
