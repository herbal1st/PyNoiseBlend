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
DEFAULT_TILE_SIZE: int = 4  # pixels
MIN_TILE_SIZE: int = 4  # pixels
MAX_TILE_SIZE: int = 16  # pixels
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

# ------ Interactive Scrubbing Throttling ------
SCRUB_SUB_SAMPLE_FACTOR: int = 4  # divider
SCRUB_SETTLE_DELAY: float = 0.15  # seconds
CHUNK_REFINE_BUDGET_PER_FRAME: int = 12  # chunks

# ------ Slope Gradient Deferral ------
ENABLE_SLOPE_GRADIENTS: bool = False  # toggle

# ------ Solar Lighting Studio Rates & Clamps ------
LIGHT_AZIMUTH_RATE: float = 90.0  # deg/sec
LIGHT_ALTITUDE_RATE: float = 45.0  # deg/sec
LIGHT_CONTRAST_MIN: float = 10.0  # multiplier
LIGHT_CONTRAST_MAX: float = 400.0  # multiplier
LIGHT_CONTRAST_RATE: float = 120.0  # rate/sec
LIGHT_AMBIENT_RATE: float = 0.50  # rate/sec
LIGHT_TIME_RATE: float = 0.25  # rate/sec
LIGHT_EXPOSURE_MIN: float = 0.50  # fraction
LIGHT_EXPOSURE_MAX: float = 1.00  # fraction
LIGHT_EXPOSURE_RATE: float = 0.25  # rate/sec

# ------ Camera Navigation Settings ------
TEST_PAN_SPEED: float = 400.0  # px/sec

# ------ Active Profile Selectors ------
ACTIVE_CHUNK_PROFILE: str = "PERFORMANT"  # profile
ACTIVE_WORLD_PROFILE: str = "SIMPLEX"  # profile
ACTIVE_LIGHTING_PROFILE: str = "DEFAULT"  # profile

# ------ Interface Colors ------
COLOR_BG: Tuple[int, int, int] = (15, 15, 20)  # rgb
COLOR_PANEL_DIVIDER: Tuple[int, int, int] = (200, 205, 215)  # rgb
COLOR_GRID: Tuple[int, int, int] = (230, 233, 240)  # rgb
COLOR_WAVE: Tuple[int, int, int] = (20, 25, 35)  # rgb
COLOR_SILENCED_WAVE: Tuple[int, int, int] = (185, 190, 200)  # rgb
COLOR_SELECTION: Tuple[int, int, int] = (35, 105, 225)  # rgb
COLOR_HUD_CARD_2D: Tuple[int, int, int, int] = (15, 15, 20, 210)  # rgba
COLOR_HUD_TEXT_ACTIVE: Tuple[int, int, int] = (255, 220, 80)  # rgb
COLOR_HUD_TEXT_NORMAL: Tuple[int, int, int] = (220, 220, 220)  # rgb
COLOR_HUD_TEXT_MUTED: Tuple[int, int, int] = (130, 135, 145)  # rgb
COLOR_INPUT_ACTIVE: Tuple[int, int, int] = (80, 200, 255)  # rgb
