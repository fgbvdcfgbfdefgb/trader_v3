"""Central configuration for trader_v3."""
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------- markets
COINS = ["BTC", "ETH", "LTC"]
SYMBOLS = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "LTC": "LTCUSDT"}

# ---------------------------------------------------------------- trading env
START_BALANCE = 2000.0        # fake USD balance per episode (one day)
FEE = 0.001                   # 0.1% taker fee per fill (Binance spot)
BUY_FRAC = 0.25               # a BUY action spends 25% of current equity
MINUTES_PER_DAY = 1440        # one episode = one UTC day of 1-minute bars

# action space (discrete)
ACTIONS = ["HOLD",
           "BUY_BTC", "SELL_BTC",
           "BUY_ETH", "SELL_ETH",
           "BUY_LTC", "SELL_LTC"]
N_ACTIONS = len(ACTIONS)      # 7

# ---------------------------------------------------------------- data
EVAL_DAYS = 45                # most recent N days are held out for evaluation

# ---------------------------------------------------------------- DQN agent
HIDDEN_LAYERS = (256, 256)
GAMMA = 0.999
LEARNING_RATE = 3e-4
BATCH_SIZE = 128
BUFFER_CAPACITY = 300_000
WARMUP_STEPS = 500            # random/exploratory steps before learning starts
LEARN_EVERY = 2               # gradient step every N env steps
TARGET_TAU = 0.005            # Polyak averaging coefficient for target network
GRAD_CLIP = 10.0
EPS_START = 1.0
EPS_MIN = 0.03
EPS_DECAY_STEPS = 150_000     # linear decay of epsilon over these env steps

# ---------------------------------------------------------------- run / io
SAVE_BUFFER_EVERY = 20        # epochs between replay-buffer snapshots
EVAL_EVERY = 25               # epochs between greedy evaluation episodes
PROGRESS_PNG_EVERY = 5        # epochs between progress.png refreshes
LLM_TIMEOUT_S = 240
LLM_MAX_TOKENS = 320
LLM_TEMPERATURE = 0.2

# ---------------------------------------------------------------- paths
DATA_DIR = os.path.join(REPO_ROOT, "data")
CACHE_DIR = os.path.join(REPO_ROOT, "cache")
RUNS_DIR = os.path.join(REPO_ROOT, "runs")
LLM_DIR = os.path.join(REPO_ROOT, "llm")
MODEL_GGUF = os.path.join(LLM_DIR, "qwen3-8b-q4km.gguf")
MODEL_PARTS_DIR = os.path.join(LLM_DIR, "qwen3-8b-parts")
LLAMA_DIR = os.path.join(LLM_DIR, "llama-b11249")
LLAMA_SERVER_BIN = os.path.join(LLAMA_DIR, "llama-server")
LLM_CACHE_DIRNAME = "llm_cache"   # lives under <runs>/shared/
