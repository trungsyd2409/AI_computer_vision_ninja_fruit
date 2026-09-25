"""All game settings in one place. Change numbers here to tune the game."""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # paths work from any folder

# ---------------- Window / camera ----------------
WINDOW_NAME = "Ninja Fruit CV"
GAME_WIDTH = 1280            # the game is drawn at this size
GAME_HEIGHT = 720
CAMERA_INDEX = 0             # 0 = default webcam. Try 1, 2 if you have more cameras
# Camera capture settings. Run `python tools/diagnose.py` to find the best ones for your webcam.
CAMERA_BACKEND = "msmf"      # "msmf" or "dshow" (Windows), "any" = let OpenCV choose.
                             # Measured on this PC (tools/diagnose.py): dshow = only 10 fps at 720p
                             # (it ignores MJPG and sends raw YUY2), msmf = 30 fps.
CAMERA_FOURCC = "MJPG"       # MJPG = compressed -> most webcams give 30 fps at 720p
CAPTURE_WIDTH = 1280
CAPTURE_HEIGHT = 720
CAPTURE_FPS = 60             # we ask for 60; the camera gives what it can
CAMERA_EXPOSURE = None       # None = auto. Try -7 or -8 (shorter exposure = less motion blur,
                             # but darker image). Some cameras ignore this with msmf.
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
SHAPES = ["circle", "oval", "square", "rectangle", "triangle", "diamond",
          "pentagon", "hexagon", "octagon", "star", "heart", "cross"]
FRUIT_RADIUS_MIN = 42        # size of a circle; other shapes get the same AREA
FRUIT_RADIUS_MAX = 62
GRAVITY = 1400               # px/s^2
PEAK_HEIGHT_MIN = 0.12       # fruit peak (fraction of screen height from top)
PEAK_HEIGHT_MAX = 0.45
SPIN_MAX = 3.0               # rad/s

# Neon colours (BGR). Inside = flat colour, border = bright line + glow
NEON_COLORS = [
    (200, 40, 255),   # hot pink
    (255, 230, 0),    # cyan
    (40, 255, 120),   # lime
    (40, 240, 255),   # yellow
    (20, 140, 255),   # orange
    (255, 60, 170),   # purple
    (255, 120, 40),   # electric blue
    (80, 40, 255),    # red
    (170, 255, 40),   # mint
]
FILL_BRIGHTNESS = 0.55       # inside colour = neon colour * this (darker -> border pops)
BORDER_WIDTH = 3             # sharp bright border
GLOW_WIDTH = 14              # width of the soft glow around the border
GLOW_BLUR = 3.0              # blur of the glow (bigger = softer, wider)

# Spawning (difficulty goes up slowly over time)
# Spawning in WAVES: throw a wave -> wait until all its fruits are cut or fell ->
# short rest -> next wave. Waves get bigger over time.
WAVE_REST = 0.8              # seconds of rest after a wave is cleared
WAVE_FIRST_DELAY = 1.0       # seconds before the very first wave
WAVE_STAGGER = 0.12          # max delay between 2 fruits of the same wave (0 = all together)
WAVE_SIZE_MIN = 1            # each wave has a random number of fruits in [MIN, MAX]
WAVE_SIZE_MAX = 5
# Where fruits come from (chance of each). Side fruits fly in from the edge in an arc.
SPAWN_SIDES = {"bottom": 0.6, "left": 0.2, "right": 0.2}
SIDE_START_HEIGHT = (0.45, 0.8)   # side fruits enter between 45% and 80% of screen height
MAX_FRUITS_ON_SCREEN = 16    # whole fruits; keeps FPS stable

# ---------------- Slicing ----------------
MAX_CUTS_PER_FRUIT = 3       # a fruit can be cut, its pieces cut again... up to 3 levels deep
PIECE_MIN_AREA = 700         # px^2 - smaller pieces cannot be cut again
PIECE_CUT_COOLDOWN = 0.15    # s - new pieces cannot be cut by the same swipe right away
HALF_PUSH_SPEED = 220        # how fast the pieces fly apart (px/s)
PARTICLES_PER_SLICE = 18
PARTICLE_LIFE = 0.6

# ---------------- Score / combo ----------------
POINTS_PER_FRUIT = 1         # first cut of a whole fruit
POINTS_PER_PIECE = 1         # every extra cut of a piece
COMBO_WINDOW = 0.35          # cuts closer than this (seconds) join the same combo
COMBO_MIN = 3                # combo bonus starts at 3 fruits
HIGHSCORE_FILE = os.path.join(BASE_DIR, "highscore.json")
