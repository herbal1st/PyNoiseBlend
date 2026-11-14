"""Global runtime and hardware settings for the PyNoiseBlend engine."""

from typing import Tuple

# ------ Display & Resolution Settings ------
APP_TITLE: str = "PyNoiseBlend - Procedural Terrain Studio"  # title
SCREEN_WIDTH: int = 1760  # pixels
SCREEN_HEIGHT: int = 990  # pixels
FPS: int = 60  # hertz
BACKGROUND_COLOR: Tuple[int, int, int] = (255, 255, 255)  # rgb

# ------ Viewport & Stepping Modes ------
DEFAULT_VIEW_MODE: int = 1  # mode
DEFAULT_TILE_SIZE: int = 8  # pixels
MIN_TILE_SIZE: int = 8  # pixels
MAX_TILE_SIZE: int = 32  # pixels
WARP_FACTOR: float = 0.15  # fraction

# ------ Relief Net (Isoline) Settings ------
DEFAULT_NET_ENABLED: bool = False  # toggle
DEFAULT_NET_INTERVALS: float = 9.0  # intervals
DEFAULT_NET_BANDWIDTH: float = 0.005  # fraction
COLOR_NET_BACKGROUND: Tuple[int, int, int] = (10, 12, 18)  # rgb

# ------ Mathematical Elevation Budgets ------
CRUST_BUDGET: float = 0.4  # fraction
RELIEF_BUDGET: float = 0.6  # fraction
MIN_RELIEF_FACTOR: float = 0.05  # fraction

# ------ Wave Frequency Defaults ------
DEFAULT_BASE_CYCLES: float = 6.0  # cycles
DEFAULT_RELIEF_CYCLES: float = 5.0  # cycles
DEFAULT_CRUST_CYCLES: float = 1.5  # cycles

# ------ Starting Elevation Phase Offsets ------
BASE_START: float = 1.00  # height
RELIEF_START: float = 1.00  # height
CRUST_START: float = 1.00  # height

# ------ Frequency Tuning Rates & Clamps ------
CYCLE_ZOOM_RATE: float = 0.75  # rate
BASE_MIN_CYCLES: float = 1.0  # cycles
BASE_MAX_CYCLES: float = 100.0  # cycles
RELIEF_MIN_CYCLES: float = 0.5  # cycles
RELIEF_MAX_CYCLES: float = 30.0  # cycles
CRUST_MIN_CYCLES: float = 0.25  # cycles
CRUST_MAX_CYCLES: float = 20.0  # cycles

# ------ Noise Scale Tuning Rates & Clamps ------
NOISE_SCALE_ZOOM_RATE: float = 0.75  # rate
MOUSE_ZOOM_SENSITIVITY: float = 0.005  # rate/px
NOISE_MIN_SCALE: float = 10.0  # pixels
NOISE_MAX_SCALE: float = 20000.0  # pixels

# ------ Extended Noise Parameter Clamps & Step Rates ------
OCTAVE_MIN: int = 1  # octaves
OCTAVE_MAX: int = 6  # octaves
DECAY_MIN: float = 0.05  # fraction
DECAY_MAX: float = 1.00  # fraction
DECAY_STEP_RATE: float = 0.50  # rate/sec
LACUNARITY_MIN: float = 1.00  # multiplier
LACUNARITY_MAX: float = 4.00  # multiplier
LACUNARITY_STEP_RATE: float = 0.50  # rate/sec
SUB_SAMPLE_FACTORS: Tuple[int, ...] = (1, 2, 4, 8)  # factors
NOISE_TYPES: Tuple[str, ...] = ("SIMPLEX", "PERLIN", "VALUE")  # algorithms

# ------ Parameter Matrix Row Indices ------
ROW_PROFILE: int = 0  # index
ROW_SCALE: int = 1  # index
ROW_OCTAVES: int = 2  # index
ROW_PERSIST: int = 3  # index
ROW_LACUNARITY: int = 4  # index
ROW_SUBSAMPLE: int = 5  # index
ROW_SEED: int = 6  # index
ROW_WORLD_SEED: int = 7  # index
MATRIX_ROW_COUNT: int = 8  # rows

# ------ Interactive Scrubbing Throttling ------
SCRUB_SUB_SAMPLE_FACTOR: int = 4  # divider
SCRUB_SETTLE_DELAY: float = 0.12  # seconds
CHUNK_REFINE_BUDGET_PER_FRAME: int = 6  # chunks

# ------ Dynamic Resolution Fallback Latch ------
ENABLE_AUTO_RESOLUTION_FALLBACK: bool = True  # toggle
RESOLUTION_FALLBACK_DIVIDER: int = 2  # divider
RESOLUTION_FALLBACK_FPS_THRESHOLD: int = 25  # hertz
RESOLUTION_FALLBACK_TRIGGER_SECONDS: float = 2.0  # seconds

# ------ Camera Navigation Settings ------
TEST_PAN_SPEED: float = 400.0  # px/sec

# ------ Active Profile Selectors ------
ACTIVE_CHUNK_PROFILE: str = "PERFORMANT"  # profile
ACTIVE_WORLD_PROFILE: str = "SIMPLEX"  # profile

# ------ Interface Colors & Dimensions ------
COLOR_BG: Tuple[int, int, int] = (15, 15, 20)  # rgb
COLOR_PANEL_DIVIDER: Tuple[int, int, int] = (200, 205, 215)  # rgb
COLOR_GRID: Tuple[int, int, int] = (230, 233, 240)  # rgb
COLOR_WAVE: Tuple[int, int, int] = (20, 25, 35)  # rgb
COLOR_SILENCED_WAVE: Tuple[int, int, int] = (185, 190, 200)  # rgb
COLOR_SELECTION: Tuple[int, int, int] = (35, 105, 225)  # rgb
COLOR_TITLE: Tuple[int, int, int] = (25, 30, 40)  # rgb
COLOR_MUTED_TITLE: Tuple[int, int, int] = (130, 135, 145)  # rgb
COLOR_TITLE_CARD: Tuple[int, int, int, int] = (255, 255, 255, 230)  # rgba
COLOR_HUD_CARD_2D: Tuple[int, int, int, int] = (15, 15, 20, 210)  # rgba
COLOR_HUD_TEXT_ACTIVE: Tuple[int, int, int] = (255, 220, 80)  # rgb
COLOR_HUD_TEXT_NORMAL: Tuple[int, int, int] = (220, 220, 220)  # rgb
COLOR_HUD_TEXT_MUTED: Tuple[int, int, int] = (130, 135, 145)  # rgb
COLOR_INPUT_ACTIVE: Tuple[int, int, int] = (80, 200, 255)  # rgb

# ------ Oscilloscope Metrics ------
PANEL_COUNT: int = 4  # panels
PANEL_TOP_MARGIN: int = 36  # pixels
PANEL_BOTTOM_MARGIN: int = 14  # pixels
PANEL_COMPOSITE_BOTTOM_MARGIN: int = 64  # pixels
PANEL_TITLE_X: int = 20  # pixels
PANEL_TITLE_Y: int = 4  # pixels
PANEL_TITLE_TEXT_X: int = 8  # pixels
PANEL_TITLE_TEXT_Y: int = 3  # pixels
PANEL_DIVIDER_WIDTH: int = 1  # pixels
PANEL_SELECTION_WIDTH: int = 2  # pixels
COMPONENT_LINE_WIDTH: int = 2  # pixels
COMPOSITE_LINE_WIDTH: int = 3  # pixels
