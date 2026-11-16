```
 ____           __  __                              ____    ___                      __     
/\  _`\        /\ \/\ \          __                /\  _`\ /\_ \                    /\ \    
\ \ \L\ \__  __\ \ `\\ \    ___ /\_\    ____     __\ \ \L\ \//\ \      __    ___    \_\ \   
 \ \ ,__/\ \/\ \\ \ , ` \  / __`\/\ \  /',__\  /'__`\ \  _ <'\ \ \   /'__`\/' _ `\  /'_` \  
  \ \ \/\ \ \_\ \\ \ \`\ \/\ \L\ \ \ \/\__, `\/\  __/\ \ \L\ \\_\ \_/\  __//\ \/\ \/\ \L\ \ 
   \ \_\ \/`____ \\ \_\ \_\ \____/\ \_\/\____/\ \____\\ \____//\____\ \____\ \_\ \_\ \___,_\
    \/_/  `/___/> \\/_/\/_/\/___/  \/_/\/___/  \/____/ \/___/ \/____/\/____/\/_/\/_/\/__,_ /
             /\___/                                                                         
             \/__/                                                                          
===============================================================================
                 PYNOISEBLEND - SYSTEM SPECIFICATIONS & GUIDE
===============================================================================

[!] CROSS-REFERENCE NOTICE:
    - For hardware constants, window settings, and YAML profile keys,
      refer to MANUAL.txt.
    - For unfulfilled milestone tasks and the active session handoff docket,
      refer to ROADMAP.txt.


[1.0 SYSTEM OVERVIEW & ARCHITECTURAL PHILOSOPHY]
-------------------------------------------------------------------------------
Core Philosophy: Sovereign Compute, Matrix-Isolated, Zero-Dependency.
Architecture   : PyNoiseBlend is an endless 2D terrain generator and interactive
                 math studio built with vectorized NumPy array math and
                 hardware-accelerated SDL graphics via Pygame-ce. It runs
                 without heavyweight game engines or external C++ extensions.

The Paradigm Shift: From Square Grid Tiles to Continuous Heightfields
In traditional 2D games, worlds are often built out of square sprite tiles like
plastic building blocks (for example, 16x16 or 32x32 pixel squares). Creating
natural coastlines or winding ridges requires complex corner-stamping rules and
edge-transition art.

PyNoiseBlend treats terrain as a smooth, continuous elevation landscape where
every individual pixel holds an exact height between 0.0 (deep ocean trenches)
and 1.0 (highest mountain peaks):
                 h(x, y) in [0.0, 1.0]

Think of it like moving from a world built out of toy bricks to a smooth,
infinitely zoomable topographical map. Every pixel on screen is an exact
evaluation of procedural waves or noise, painted through smooth color gradient
ramps and crisp elevation contour lines.

Key Architectural Pillars:
- Pre-Baked Raster Chunks: Instead of recalculating terrain every frame, the
  world is divided into square patches called chunks (128x128 pixels). Each
  chunk calculates its noise once, converts it into a display-ready image in
  RAM, and stamps onto the screen effortlessly. When standing still or panning
  across loaded terrain, procedural CPU math drops to zero.
- Anchor-Locked Grid Snapping: Floating camera coordinates can cause neighboring
  chunks to round to slightly different screen pixels, leaving thin 1-pixel
  black cracks. The engine solves this by anchoring all visible chunks to one
  rounded top-left corner and stamping outward in exact whole-pixel strides.
- Agnostic ChunkProvider Architecture: Memory streaming (ChunkManager) is
  completely separated from terrain math via the ChunkProvider contract and a
  shared BaseChunkProvider—making it easy to swap worlds, raw noise, or new map
  layers without touching the streaming engine.
- Clean Controller Separation: Session data (StudioState), input event routing
  (InputHandler), wave math tuning (wave_tuner), matrix/world tuning
  (matrix_tuner), and UI table drawing (parameter_matrix) each live in focused
  modules rather than one monolithic file.
- Engine-Wide Live Session State Machine: Adjustments made to a noise preset in
  Mode 3 modify active working copies in memory. Tuning a noise preset in
  Mode 3 immediately updates any world layer using that preset in Mode 4, while
  your original YAML files on disk stay untouched.
- Interactive 2D Parameter Matrix Deck: Modes 3 and 4 include an on-screen
  spreadsheet-style control deck exposing all noise settings (Profile, Scale,
  Octaves, Decay, Lacunarity, Sub-sampling, and Seeds) for live tuning via
  mouse or keyboard.
- Two-Tier Interactive Scrubbing & Progressive Refinement: While dragging
  sliders or zooming scales continuously, viewports switch temporarily to a
  lightweight low-resolution preview grid for smooth 60 FPS feedback. Once you
  stop adjusting, full detail snaps back in, rebuilding chunks smoothly from
  the center of the screen outward under a strict per-frame budget.
- Discrete Single-Tap Input Separation: Step-based settings (Octaves, Profile,
  Sub-sampling, Seeds) advance strictly once per key tap or mouse click,
  preventing key-repeat floods from stalling the CPU.
- Click-to-Type Seed Prompt: Clicking any seed cell (or pressing ENTER) opens
  direct keyboard digit entry so you can type exact world seeds on the fly.
- Color-Before-Expand Surface Scaling: Coarse preview grids and tiled rasters
  are colored directly while small and stretched up by SDL hardware scaling,
  avoiding millions of redundant per-pixel color writes.
- Mathematical Elevation Budgeting (Zero-Clipping Invariant): Total terrain
  height is strictly divided into additive and multiplicative shares so
  mountain peaks never flatten against a 1.0 ceiling, and inland valleys stay
  naturally elevated above sea level.
- Triple-Engine Procedural Suite: Supports Simplex, Perlin, and Value Noise,
  letting you balance organic flowing ridges against raw generation speed.
- 1-Pixel Gradient Halo: Each chunk samples an invisible 1-pixel border ring
  around its edges before measuring slopes, guaranteeing smooth lighting and
  contour continuity across chunk borders.
- 1D Strata Color Look-Up Tables (LUTs): Color mapping is decoupled from math.
  Chunks pass heights through precomputed 1,024-slot color strips (like a
  fast paint-by-numbers index) in a single NumPy lookup (<0.1 ms).
- (N + 1) Fencepost Sub-Sampling: Evaluates coarse grids with one extra shared
  boundary point so neighboring chunks always agree on border heights across
  2x, 4x, and 8x downscaling dividers.
- Elliptical Streaming with Margin Clamping: Streams chunks in a screen-shaped
  ellipse with a safety buffer zone (hysteresis) so chunks do not flicker in
  and out of memory near edges, and corners never show gaps on widescreens.
- Screen-Center Anchored Proportional Zoom: Zooming multiplies all active
  layers together while mathematically shifting offsets so features stay pinned
  to the center of your screen.
- Frame-Rate Independent Delta Time: Camera flight speed, debounce timers, and
  zoom rates scale by frame delta time (dt) for consistent feel across PCs.
- Deferred Slope Gradients: In Mode 4, WorldChunkProvider skips slope-tilt
  math (np.gradient) and its extra 1-pixel border ring unless lighting or
  tiled contour lines actually need them—reclaiming ~25–30% chunk bake speed.
- Dynamic Resolution Fallback Latch: TerrainStudioApp monitors frame rates
  in Modes 3, 4, and 5; if FPS stays below a safety threshold for a short
  grace period, the engine automatically switches chunk generation to a
  coarser sub-sampled grid (without ever upscaling already-coarse profiles),
  steps Mode 5's lightmap down to a coarser fallback divider, and flags
  [FALLBACK: Nx] in the HUD.
- Single-Pass 3D Solar Hillshading (world/lighting/): Mode 5 computes 3D
  sunlight direction (compass azimuth + sun altitude), adjustable neutral
  exposure (letting sunlit ridges brighten above flat plains without clipping),
  and time-of-day sky tinting on a screen-relative lightmap grid sampled
  directly from visible chunks via np.searchsorted, then scales and
  multiplies it over the screen in one pygame.BLEND_MULT pass—allowing 60 FPS
  sun orbits, live scrubbing hillshading, and crisp 1:1 tile-aligned shading
  in Tile Mode with zero chunk re-baking.


[2.0 PROCEDURAL GENERATION & MATHEMATICAL WAVE SYNTHESIS]
-------------------------------------------------------------------------------
The Continuous Noise Surface:
Mathematical 2D noise engines (utils/noise/) create smooth, repeatable random
patterns—similar to natural rolling clouds or marble grain. Given any (x, y)
coordinate and an integer seed, they return smooth values in [-1.0, 1.0] (which
the studio shifts into [0.0, 1.0] for elevation mapping).

The Triple-Engine Procedural Suite:
PyNoiseBlend provides three distinct noise algorithms so you can trade visual
character for CPU speed:

1. Vectorized 2D Simplex Noise (utils/noise/simplex.py):
   - Type    : Gradient Noise (directional slopes on a triangle grid).
   - Visuals : Extremely organic, flowing, and free of boxy grid artifacts.
   - Use Case: High-detail base terrain, craggy ridges, and rolling hills where
               natural visual flow matters most.

2. Classical 2D Perlin Noise (utils/noise/perlin.py):
   - Type    : Gradient Noise (directional slopes on a square grid).
   - Visuals : Classic smooth hills with balanced, gentle curvature.
   - Use Case: Traditional undulating landscapes and soft dune fields.

3. Vectorized 2D Value Noise (utils/noise/value.py):
   - Type    : Scalar Lattice Noise (direct height blending between corners).
   - Visuals : Softer, pillowed landmasses and broad plateaus.
   - Mechanism: Assigns heights directly to grid corners using fast bitwise
               number scrambling (primes and XOR shifts), then blends smoothly.
   - Speed Win: By skipping slope vector math and lookup tables, Value Noise
               runs noticeably faster on low-power CPUs—ideal for broad
               continent foundations and macro relief layers.
   - Smooth Curve (C1 Continuity): Uses a 5th-degree S-curve
               u(t) = 6t^5 - 15t^4 + 10t^3 so slopes transition smoothly across
               grid cells with zero faint creases.

Fractal Brownian Motion (FBM) Coordinator (NoiseGenerator):
Layering multiple passes of noise (like adding fine rocky crags on top of big
hills) is handled in utils/noise/noise_generator.py:
- Profile Isolation  : NoiseGenerator clones its bound NoiseProfile on setup
                       and exposes sync_from_profile() so changes in the UI
                       reliably trigger seed and algorithm updates across both
                       Mode 3 and Mode 4 without cross-contaminating presets.
- Scale (Wavelength) : Width of terrain features in pixels. Updated in place via
                       set_scale() without rebuilding engine tables.
- Spatial Offsets    : Per-layer (x, y) shifts updated via set_offset(),
                       allowing hills, relief, and continents to slide
                       independently or stay pinned during zooms.
- Octaves            : Number of progressively finer detail passes stacked on
                       top of each other (clamped safely between 1 and 6).
- Decay (Persistence): How quickly each finer detail pass shrinks in height
                       (clamped between 0.05 and 1.00).
- Lacunarity         : How tightly each finer pass shrinks in width (clamped
                       between 1.00 and 4.00).
- Seed & Noise Type  : Updatable live at runtime; automatically rebuilds the
                       underlying permutation tables or hash seed.
- Sub-Sample Factor  : Coarse sampling divider (1x, 2x, 4x, 8x) using an
                       (N + 1) fencepost grid, with optional upscaling. Tiled
                       grid sampling aligns strictly to world tile multiples
                       and slices by sub-tile offsets for seamless borders.

Multi-Layer Terrain Math & Elevation Budgeting:
In real geography, mountain ranges rise atop broad tectonic plates, while
oceans and beaches stay flat and low. Simply multiplying noise layers together
causes a classic procedural bug: valleys inside high mountain ranges plunge all
the way down to sea level, punching unnatural holes through continents.

PyNoiseBlend solves this by dividing the [0.0, 1.0] elevation range into a
strict mathematical "Elevation Budget":
  CRUST_BUDGET + RELIEF_BUDGET = 1.0   (for example: 0.40 crust + 0.60 relief)

The Three-Layer Master Formula:
  Height = (CRUST_BUDGET * C) + (RELIEF_BUDGET * B * ReliefEnvelope)

In Everyday Terms:
- Base Detail (B): The local topography—individual hills, valleys, and ridges.
- Relief Multiplier (R): A mid-scale envelope deciding where local hills swell
  into tall mountain ranges versus where they flatten into gentle plains.
- Continental Crust (C): The broad tectonic foundation lifting whole
  continents above water or dropping ocean basins.
- Zero-Clipping Guarantee: Because Crust and Relief shares sum to 1.0, peaks
  never hit a flat ceiling, while inland valleys stay naturally elevated above
  sea level on high ground.

Two-Layer Formulas (Relief Silenced; Base + Crust Active):
To save ~33% calculation time on low-power hardware—or to craft cleaner terrain
shapes—the five 2-layer formulas silence the middle Relief layer (R) and pair
Base Detail (B) directly with Continental Crust (C). For example, Factored
Virtual Crust uses C to lift both the ground floor and mountain amplitude:
  Height = C * (CRUST_BUDGET + (RELIEF_BUDGET * B))
When C is low, terrain drops into flat ocean basins. As C rises, both valley
floors and mountain peaks lift together—preventing sea-level cuts using only
two noise layers.

Decoupled Formula & World Build Architecture:
World recipes in profiles/world.yaml (loaded by WorldProfileLoader in
world/chunks/world_profile.py) act purely as initial noise-engine setups.
Mode 4 shares the active formula directly with Modes 1 and 2, so you can swap
any of the 12 formulas live ([TAB]) or switch noise builds ([B]) independently,
and even rebind individual layer presets directly inside the parameter deck.


[3.0 SPATIAL CHUNKING, ANCHOR SNAPPING & STREAMING HYSTERESIS]
-------------------------------------------------------------------------------
The Infinite World Dilemma:
A mathematical noise function extends forever in every direction, while RAM
and displays are finite. The engine cannot pre-compute an entire planet at
once, so it streams local patches on demand as the camera moves.

What a Chunk Is (world/chunks/chunk.py):
A chunk is a localized square tile of the continuous elevation surface:
- Stored at integer chunk grid coordinates (cx, cy).
- Default dimension: 128x128 pixels (configured in profiles/chunk.yaml).
- Holds the float32 elevation grid h, slope gradients (gx, gy), and the baked
  SDL image surface ready for instant screen stamping.

1-Pixel Halo & Deferred Slope Calculation:
To measure slope steepness accurately at a chunk's edge, pixels need to peek
one pixel across the boundary. When slope gradients (ENABLE_SLOPE_GRADIENTS)
or tiled contour lines are active, the provider samples a 1-pixel padded
border ring around the chunk, calculates slopes or edge crossings across the
padded grid, and trims the halo away—preventing lighting or contour seams.
When flat Mode 4 exploration has gradients deferred, this extra halo and
gradient math are skipped completely for faster chunk baking.

Anchor-Locked Grid Projection:
Floating-point camera positions can cause neighboring chunks to round to
slightly different screen pixels, creating thin 1-pixel black cracks.
ChunkManager avoids this by anchoring the visible grid to one rounded top-left
corner:
  anchor_x = round( center_x + (min_cx - focus_cx) * chunk_size )
  anchor_y = round( center_y + (min_cy - focus_cy) * chunk_size )
Every visible chunk stamps relative to this anchor in exact whole-pixel
strides, completely eliminating sub-pixel seams.

The Agnostic Provider & Base Provider Architecture (world/chunks/):
ChunkManager is purely a spatial memory and coordinate streaming engine:
- It has no knowledge of noise formulas, palettes, or biomes.
- When a chunk coordinate enters the load zone, ChunkManager delegates
  creation to an injected ChunkProvider.
- BaseChunkProvider (world/chunks/providers/base_chunk_provider.py) unifies
  shared LUT storage, visual mode settings, surface baking, and tile-contour
  halo cropping for both WorldChunkProvider (composite worlds) and
  NoiseChunkProvider (raw noise).
- Providers expose set_color_lut(), rebake_chunk_surface(), and
  sample_viewport_field() so color changes instantly repaint cached chunks in
  RAM without recomputing noise.

Center-Outward Progressive Chunk Refinement (world/chunks/chunk_manager.py):
After live parameter edits end, rebuilding ~200 chunks in a single frame would
stall low-power CPUs. ChunkManager supports a per-frame cap (max_new_chunks):
- Missing chunks inside the load ellipse are sorted by distance from the camera
  center so the middle of the screen sharpens first and outer edges fill in
  smoothly over the next few frames.
- are_visible_chunks_loaded() lets the renderer retire the coarse preview
  underneath on the exact frame every visible chunk finishes baking.

Elliptical Streaming with Margin Clamping (world/chunks/chunk_manager.py):
Chunks load inside a screen-shaped ellipse centered on the camera:
- Corner Safety Clamp (load_margin >= 0.75): A square chunk's half-diagonal is
  ~0.7071 chunks. Clamping load_margin >= 0.75 guarantees every chunk touching a
  screen corner stays inside the load ellipse.
- Streaming Hysteresis (Buffer Zone): To keep chunks from flickering in and out
  of RAM when pacing back and forth near a border, the unload boundary is
  pushed farther out:
    eff_unload_margin = max(eff_load_margin + 0.25, unload_margin_chunks)

Additive Layer Seeding:
World builds combine the master world_seed with each noise layer's seed and
fixed layer offset:
  effective_seed = (world_seed + layer_seed + seed_offset) & 0xFFFFFFFF
Changing world_seed generates an entirely new planet while keeping hills,
macro relief, and continents uncorrelated—and preserves any manual spatial
offsets you dragged onto individual layers.


[4.0 SUB-SAMPLING, FENCEPOST RESAMPLING & INTERACTIVE SCRUBBING]
-------------------------------------------------------------------------------
The Low-Spec Hardware Dilemma:
Evaluating multi-octave Simplex noise for all 1,742,400 pixels of a 1760x990
display on a low-power CPU would drop real-time interactivity to a crawl.

The Per-Layer Sub-Sampling Solution (WorldChunkProvider):
Controlled per noise layer (1x, 2x, 4x, 8x via profiles/noise.yaml or live in
the Mode 3/4 matrix), with WorldProfile.sub_sample_factor and the Fallback
Latch acting as a minimum floor so dividers never multiply together:
- At 1x (Native): Evaluates all 16,384 points per 128x128 chunk.
- At 2x Downsampled: Evaluates 64x64 points (4,096 samples), cutting noise math
  by 75%.
- At 4x or 8x: Cuts math by 93%–98% before smoothly upscaling back to native
  chunk resolution.
- Smart Layer Transport: Fine local detail (Base) can run at 1x or 2x while
  broad macro layers (Relief at 4x, Crust at 8x) evaluate far fewer points and
  align cleanly onto the Base grid before the formula combines them.

The (N + 1) Fencepost Rule:
Dividing a span into N intervals requires N + 1 boundary points (just like a
4-section fence needs 5 posts). If a 128px chunk only sampled 128 // 8 = 16
points, the final sample at x=120 would leave an 8-pixel gap before the next
chunk at x=128. Evaluating (chunk_size // sub_factor) + 1 points places Chunk
0's rightmost sample on the exact same coordinate as Chunk 1's leftmost sample,
sealing seams cleanly.

Zero-Dependency Vectorized Bilinear Resampler (utils/resampler.py):
Smoothly stretches coarse float32 elevation and slope arrays in pure NumPy
using a cached two-step ResamplePlan (blending horizontally first, then
vertically, with zero-allocation bilinear_resample_into() support).

Interactive Scrubbing & Settle-Debounce Pipeline (studio/state.py):
To keep live zooming and parameter dragging fluid while preserving crisp
static quality:
- Active Scrubbing: Continuous tuning (Scale, Decay, Lacunarity) engages
  scrubbing mode and resets a short settle timer (SCRUB_SETTLE_DELAY). Plain
  camera panning (WASD or unmodified Left-Drag in Modes 3–4) never triggers
  scrubbing because cached chunks already pan at 60 FPS.
- Discrete Single-Tap Decoupling: Discrete settings (Octaves, Profile,
  Sub-sampling, Seeds) advance strictly one step per key tap or click, forcing
  one clean rebuild without flooding CPU caches.
- Mode 2 Coarse Mesh Preview: Mode 2 drops temporarily to an endpoint-aligned
  (N + 1) scrub mesh. In color view, it colors the small grid first and lets
  SDL smoothscale stretch it; in Relief Net view, it draws crisp 1px isolines
  using coarse-cell culling.
- Modes 3 & 4 Single-Pass Viewport Preview: Instead of rebuilding ~200 chunks
  on every drag pixel, BaseChunkRenderer samples one compact screen-sized grid
  via provider.sample_viewport_field() (skipping slope math).
- Instant Mouse Release & Seamless Progressive Settle: Releasing a mouse drag
  settles immediately (and keyboard tuning settles after SCRUB_SETTLE_DELAY).
  Mode 2 snaps straight back to full detail; Modes 3 and 4 flush stale chunks,
  keep the preview underneath as a camera-synced backdrop, and refill chunks
  from the center outward at CHUNK_REFINE_BUDGET_PER_FRAME per frame.

Dynamic Resolution Fallback Latch (Wired Safety Net):
To keep Modes 3, 4, and 5 responsive on low-power CPUs, TerrainStudioApp.run()
checks live frame rates every tick via StudioState.check_resolution_fallback().
If FPS stays below RESOLUTION_FALLBACK_FPS_THRESHOLD for
RESOLUTION_FALLBACK_TRIGGER_SECONDS, the latch engages and routes
RESOLUTION_FALLBACK_DIVIDER into ChunkManager.update_focus() via
sub_sample_override, while Mode 5's lightmap steps down to
LIGHTMAP_FALLBACK_DIVIDER. Both WorldChunkProvider and NoiseGenerator clamp
with max(configured_factor, override) so already-coarse profiles never
accidentally upscale. When active, the HUD appends a [FALLBACK: Nx] tag next
to the FPS readout.

Optional Slope Gradient Deferral (ENABLE_SLOPE_GRADIENTS):
Mode 4 skips calculating slope-tilt arrays (gx, gy) and two extra
bilinear_resample_2d() calls per chunk when ENABLE_SLOPE_GRADIENTS is False,
reclaiming ~25–30% chunk bake speed during flat exploration. Mode 5 enables
gradients on its own ChunkManager so hillshading stays independent.


[5.0 CONTINUOUS STRATA PALETTES & 1D TRANSFER LUTS]
-------------------------------------------------------------------------------
Decoupled Color Translation (profiles/strata.yaml):
Chunk generators know nothing about biomes or colors. They pass continuous
float32 heights through an injected 1,024-slot Look-Up Table (LUT), keeping
terrain math completely independent from visual styling.

Vectorized C-Speed Color Gathering:
Instead of slow branching if/else chains per pixel, NumPy paints an entire
elevation array in one fast lookup:
  indices = clip(h * 1023.0, 0, 1023).astype(int32)
  rgb_buffer = strata_lut[indices]

Soft Transition Bandwidths:
Between elevation keyframes (such as Shallow Water at 0.145 to Wet Sand at
0.150 and Dry Sand at 0.180), colors blend smoothly across the 1,024 slots.
Repeating a color across two thresholds creates flat plateau color bands.

Continuous Terrain Speed Multipliers:
Each StrataProfile also compiles a 1,024-slot float32 speed_lut alongside its
color ramp so future gameplay movement can smoothly slow down in water or
thick forests without hard tile edges.

Unified Built-In Diagnostic Suite:
StrataLoader bakes four scientific heatmaps before loading YAML biomes:
- GREYSCALE : Linear monochrome mathematical baseline.
- MAGMA     : Perceptually uniform ramp (black through violet to pale yellow).
- TURBO     : High-contrast rainbow palette exposing subtle wave creases.
- VIRIDIS   : Perceptually uniform blue-to-yellow scientific palette.

Dynamic Color Inversion (Key [I]):
Every palette—built-in heatmaps and YAML biomes alike—pre-bakes both its
normal color_lut and its reversed inverted_lut once at load time. Pressing
[I] swaps the active table instantaneously with zero runtime re-baking,
turning Greyscale into a negative view or flipping ocean/mountain polarity.


[6.0 THE UNIFIED TERRAIN STUDIO & INSPECTION SUITE]
-------------------------------------------------------------------------------
PyNoiseBlend includes a five-mode interactive workbench under studio/:

The Modular Studio Architecture (studio/):
- Session State Store (state.py): Coordinates shared formulas, palettes,
  camera coordinates, active mode flags, scrub timers, and the seed typing
  buffer.
- Wave & Matrix Tuning Controllers (wave_tuner.py, matrix_tuner.py):
  * wave_tuner.py isolates Mode 1/2 channel validity, focus cycling, wave
    offset shifting, and cycle frequency scaling with 2D center anchoring.
  * matrix_tuner.py isolates Mode 3/4 row/column navigation, world-layer
    offset shifting, parameter stepping, camera-anchored scale math, and
    noise/world resets.
- Input Router (input_handler.py): Pure Pygame event polling and continuous
  held-key handling dispatching actions to wave_tuner, matrix_tuner, and state.

Shared Base & Chunk Renderers (studio/renderers/):
- BaseRenderer (base_renderer.py): Provides shared monospace typography,
  auto-sizing HUD cards backed by a bounded translucent surface cache (avoiding
  per-frame alpha surface allocations), and the unified two-line studio dock.
- BaseChunkRenderer (base_chunk_renderer.py): Extends BaseRenderer for Modes
  3, 4, and 5, managing coarse live previews, culled 1px isoline previews,
  world-locked tile previews, progressive center-outward chunk refinement, and
  cached scrub heightfields for live Mode 5 preview lighting.
- Parameter Matrix Deck (parameter_matrix.py): Standalone UI renderer drawing
  the interactive multi-column parameter spreadsheet (also using a cached
  translucent backdrop) and registering cell and tab click hitboxes.

The 5 Studio Viewports (studio/renderers/):

1. Mode 1: 1D Wave Oscilloscope (renderer_1d.py):
   - Four vertically stacked signal plots: Base Detail D(x), Relief Multiplier
     M(x), Continental Crust C(x), and Composite Result (Panel 4 / [ALL
     ACTIVE]).
   - Dirty-Frame Surface Cache: Rebuilds the oscilloscope surface only when
     state changes (needs_rebuild_1d), reducing idle frames to a single blit.
   - Smart Layer Silencing: On 2-layer formulas, Panel 2 (Relief) is silenced
     with a flat zero baseline while Panel 3 (Crust) remains active.
   - Direct Panel Selection: Click any active Panel 1–4 (or cycle with
     UP/DOWN) to focus it; clicks on silenced Panel 2 are ignored. Selecting
     Panel 4 ([ALL ACTIVE]) scales or resets all active layers together.
   - Dual Hypsometric Fill / Vector Trace: Standard mode fills under Panel 4's
     curve with vertical LUT color ribbons; Net mode paints a multi-colored
     curve in one vectorized NumPy surfarray pass with zero Python line loops.

2. Mode 2: 2D Spatial Synthesizer (renderer_2d_synth.py):
   - Projects the three mathematical wave roles onto a 16:9 2D canvas:
       * Base Detail (B)      : Vertical stripes along X (screen-centered).
       * Relief Multiplier (R): Horizontal bands along Y (screen-centered;
                                silenced on 2-layer formulas).
       * Continental Crust (C): Concentric radial dome/pond centered at
                                (0.5 + crust_offset_x, 0.5 + crust_offset_y).
   - Dynamic Spatial Perspective (Key [P], Mode 2 only):
     Toggles how active waves map across 2D coordinates:
       * RADIAL Perspective (Default): Concentric radial island/pond dome.
       * WARPED Perspective: Shears Base coordinates horizontally, bending
         straight stripes into winding mountain ranges.

3. Mode 3: Raw Noise Inspector (renderer_noise.py):
   - Streams isolated noise builds from profiles/noise.yaml via ChunkManager.
   - Interactive Parameter Matrix Deck: Displays a single-column table exposing
     Profile, Scale, Octaves, Persist (Decay), Lacunarity, Sub-sampling, and
     Seed for the active raw noise engine.
   - Press [B / SHIFT + B] or click the PROFILE cell to cycle raw noise
     presets.
   - Click the SEED cell (or press ENTER on it) to type a seed directly.
   - Edits made to noise profiles in Mode 3 mutate working copies in memory and
     immediately reflect across any Mode 4 layers using that profile.

4. Mode 4: Composite World Explorer (renderer_world.py):
   - Streams infinite procedural worlds driven by the active formula from
     FORMULA_CATALOG and the active noise layer profiles.
   - Interactive 3-Column Parameter Matrix Deck:
     * Columns: BASE (D), RELIEF (M), and CRUST (C).
     * Middle column (RELIEF) automatically dims and shows [ Silenced ] when
       a 2-layer formula is selected.
     * Rows: PROFILE, SCALE, OCTAVES, PERSIST, LACUNARITY, SUB-SAMPLE, SEED,
       and a dedicated master WORLD SEED row at the base.
   - Profile Rebinding: Click PROFILE in any column to swap which noise preset
     drives that specific layer (e.g. swap Crust from Simplex to Value noise).
   - Proportional All-Layer Zoom: Right-dragging or holding Left/Right on
     the SCALE row without Shift scales Base, Relief, and Crust simultaneously
     around the camera center. Holding [SHIFT] scales only the focused column.
   - Click any SEED cell or the WORLD SEED cell (or press ENTER on it) to type
     a new seed directly.

5. Mode 5: Solar Lighting Studio (renderer_lighting.py):
   - Inherits your live Mode 4 world (MODE 4 (LIVE)) by default—or lets you
     compare the baseline world.yaml builds (SIMPLEX, PERLIN, PERFORMANT)—
     using its own gradient-enabled ChunkManager (compute_gradients=True) so
     Mode 4 continues to skip slope math when not in lighting view.
   - Evaluates hillshading on a crisp screen-relative lightmap divider
     (lightmap_sub_sample, dropping to LIGHTMAP_FALLBACK_DIVIDER only when
     the fallback latch engages). In Tile Mode ([SPACE]), switches to exact
     1:1 tile-centroid lighting with nearest-neighbor scaling so every tile is
     cleanly flat-shaded with zero stripes.
   - During interactive Scale scrubbing, computes gradients directly from the
     compact scrub preview heightfield and holds the cached lightmap while
     chunks refine, preventing unlit holes.
   - Solar Control Deck: Click, drag, or use arrow keys to tune Preset, World
     source, Sun Azimuth (0-360 deg), Sun Altitude (0-90 deg), Hillshade
     Contrast, Ambient Ratio, Orbit Speed (PAUSED to 4.0x), Time-of-Day,
     Exposure (neutral flat-ground brightness baseline), and unified Scale
     (scales all layers proportionally around camera center without leaving
     your selected world preset).


[7.0 GLOBAL TILE STEPPING & COARSE CENTROID EVALUATION]
-------------------------------------------------------------------------------
Pressing [SPACE] and [SHIFT + SPACE] cycles terrain through a power-of-two tile
quantization ring across all five modes:
    OFF (Smooth) -> 4px -> 8px -> 16px -> OFF

Coarse Centroid Grid Evaluation (Up to 98% Less Math):
Instead of evaluating every pixel and rounding afterward, tile mode evaluates
waves or noise strictly at the center point of each tile block (for example,
16 points per 128x128 chunk at 32px tiles instead of 16,384). In Mode 2 and
Mode 3/4 scrubbing previews, the tiny tile array is colored directly and
stretched in SDL via pygame.transform.scale(), keeping zooming and panning
fluid.

Global World-Space Tile Anchoring & Sub-Tile Halo Alignment:
In streaming Modes 3 and 4, tile coordinates snap strictly to global multiples
of tile_size:
    snapped = (floor(world_coord / tile_size) + 0.5) * tile_size
When chunks request a 1-pixel outer halo (such as wx - 1.0) for seamless edge
contours, tiled grid evaluation aligns blocks to global tile boundaries and
slices by the exact sub-tile remainder, keeping retro pixel blocks and chunk
seams locked cleanly to world coordinates during panning with zero jitter.


[8.0 SHARED TOPOGRAPHICAL RELIEF NET (ISOLINE ENGINE)]
-------------------------------------------------------------------------------
Pressing [N] toggles Topographical Relief Net mode across Modes 1 through 5:
- In 1D: Replaces solid vertical color fills with a multi-colored trace curve.
- In 2D, 3, 4, and 5: Switches raster terrain to a dark slate blueprint
  contour map drawn with elevation isoline bands (like a hiking map).

The Multi-Kernel Evaluation Engine (core/contour_engine.py):
- Single-Pass Band Kernel (Settled Smooth Mode): Measures distance to the
  nearest contour interval in one vectorized pass; draw_contour_net_to_surface
  reuses preallocated scratch buffers and writes only contour pixels in place.
- Coarse-Cell Culled Kernel (Smooth Live Scrubbing Mode): Inspects the 4
  corners of each 4x coarse grid cell first and skips the ~80–90% of cells that
  do not cross a contour line. Only active cells are interpolated at full
  screen resolution, keeping 1px contour lines razor sharp while scrubbing.
- Full-Res Discrete Tile Mode: Checks cardinal neighbor elevation crossings
  across a 1-pixel world halo so tiled chunks draw seamless step outlines
  without gaps along chunk borders. Toggling Net or changing palettes while
  tile stepping is active triggers a fast halo-aware chunk refresh so boundary
  contour lines remain unbroken across all tile sizes (8px, 16px, 32px).
- Tiled Edge Preview Kernel (draw_tile_contour_edges_to_surface): Computes
  directional vertical and horizontal tile-crossing edges directly from compact
  grids via bake_tile_contour_edges and stamps them through one shared helper
  for instant Mode 2 and Mode 3/4 scrubbing previews.


[9.0 FORMULA CATALOG & 2D PRESETS]
-------------------------------------------------------------------------------
The studio synchronizes mathematical formulas across Modes 1, 2, and 4:

MODE 1 & MODE 4 CATALOG (12 Composite Formulas):
-------------------------------------------------------------------------------
Three-Layer Formulas (Base + Relief + Crust Active):
    - Master 3-Layer Budgeting   : Additive tectonic crust + modulated relief.
    - Global Envelope Factor     : Overall relief scaling of crust and base.
    - Volcanic Island Override   : Heavy crust dome with dual-stage detail.
    - Inverse Crust Bypass       : Shoreline bypass preserving low relief.
    - Dynamic Lerp / Morph       : Smooth linear blend between crust and base.
    - Tectonic Subduction Belt   : Squared crust basin with ridge clamping.
    - Stepped Mesa Terraces      : Multi-tier terrace quantization with crust.

Two-Layer Formulas (Relief Multiplier Silenced; Base + Crust Active):
    - Factored Virtual Crust     : Continental crust drives floor and peaks.
    - Tibetan High Plateau       : High plateau floor with sine-wave valleys.
    - Razor Ridge / Granite Peaks: Squared base ridges sharpened by crust.
    - Inverted Canyonlands       : Inverted crust carving steep ravines.
    - Soft Natural Saturation    : Hyperbolic tangent organic compression.


MODE 2 CATALOG (15 Streamlined Presets):
-------------------------------------------------------------------------------
Mode 2 includes three isolated Solo geometry tracks plus the 12 composites:
    - Presets 1 to 3   : Solo Components (Vertical Base D(x), Horizontal
                         Relief M(y), and Radial Continental Crust C(r)).
    - Presets 4 to 8   : 3-Layer Composite Series.
    - Presets 9 to 13  : 2-Layer Series (Toggled via [P] between Radial/Warped).
    - Presets 14 to 15 : Advanced 3-Layer Hybrids.


[10.0 CONTROLS & KEYBOARD / MOUSE SHORTCUTS]
-------------------------------------------------------------------------------
Universal Rule: Unmodified drags and zoom inputs affect ALL active layers
together; holding [SHIFT] isolates ONLY the currently focused layer.

VIEWPORT & GENERAL NAVIGATION:
-------------------------------------------------------------------------------
    1 - 5               : Switch Viewport Mode:
                            [1] 1D Wave Oscilloscope
                            [2] 2D Spatial Synthesizer
                            [3] Raw Noise Inspector
                            [4] Endless World Explorer
                            [5] Solar Lighting Studio

    TAB / SHIFT + TAB   : Cycle active mathematical formula forward / backward
                          (Modes 1, 2, and 4; automatically dims silenced
                          layers in Mode 4 matrix).

    B / SHIFT + B       : Cycle active YAML Build / Noise Profile (Modes 3
                          and 4) or Solar Lighting Preset (Mode 5).

    WASD                : Pan camera smoothly across terrain (Modes 3, 4,
                          and 5).

    UP / DOWN           : Cycle active wave panel/channel focus (Modes 1 & 2)
                          or move parameter/solar deck row (Modes 3, 4, & 5).

    SPACE / SHIFT+SPACE : Step tile size coarser / finer (OFF -> 8 -> 16 -> 32).

    C / SHIFT + C       : Cycle active color palette forward / backward.

    I                   : Toggle palette color inversion on/off.
    P                   : Toggle spatial perspective (Mode 2 only: Radial/Warped).
    N                   : Toggle Topographical Relief Net / 1D Vector trace.

    R                   : Reset ALL parameters, wave cycles, noise profiles,
                          lighting settings, and camera coordinates to defaults.
    SHIFT + R           : Reset ONLY the focused layer back to defaults.

    ESCAPE / CLOSE      : Clean application shutdown (or cancels seed typing).


MATRIX DECK CONTROLS (MODES 3 & 4):
-------------------------------------------------------------------------------
    UP / DOWN           : Move row focus across parameters:
                          Profile -> Scale -> Octaves -> Persist ->
                          Lacunarity -> Sub-Sample -> Seed -> World Seed.

    SHIFT + UP / DOWN   : Move column focus between layers in Mode 4:
                          [ BASE (D) | RELIEF (M) | CRUST (C) ]
                          (Automatically skips silenced column on 2-layer
                          formulas).

    LEFT MOUSE CLICK    : - On Cell: Selects cell. On discrete cells (Profile,
                            Sub-sample, Octaves), steps forward (+1).
                          - On Seed / World Seed: Opens direct typing prompt.
                          - On Layer Tab: Focuses that layer column.

    RIGHT MOUSE CLICK   : - On discrete cells (Profile, Sub-sample, Octaves),
    (or SHIFT + LMB)        steps backward (-1).
                          - On Seed cells: Decrements seed by 1.

    ENTER               : Opens direct numeric typing prompt when focused on
                          a Seed or World Seed row.

    SEED TEXT ENTRY     : Type digits (0-9) when seed prompt is active:
                          [ENTER] commits seed; [BACKSPACE] deletes;
                          [ESCAPE] cancels without changes.

    RIGHT DRAG          : - On Scale: Zooms ALL active layers proportionally
                            around camera center. (With [SHIFT]: zooms only the
                            focused layer).
                          - On Persist / Lacunarity: Smooth continuous scrub.

    HOLD LEFT / RIGHT   : Keyboard mirror of Right-Drag:
                          - On Scale / Persist / Lacunarity: Continuous scrub.
                          - (With [SHIFT] on Scale: zooms focused layer only).

    TAP LEFT / RIGHT    : On discrete rows (Profile, Octaves, Sub-sample, Seed):
                          Steps parameter by exactly 1 single increment per tap.

    LEFT DRAG           : Pan camera at 60 FPS (Modes 3-4) or slide waves (1-2).
    SHIFT + LEFT DRAG   : Translate / slide spatial offset of ONLY the focused
                          layer independently.


SOLAR CONTROL DECK CONTROLS (MODE 5):
-------------------------------------------------------------------------------
    UP / DOWN           : Move row focus across Preset, World, Azimuth,
                          Altitude, Contrast, Ambient, Orbit Spd, Time/Day,
                          Exposure, and Scale.

    LEFT MOUSE CLICK    : - On Discrete Rows (Preset, World, Orbit Spd):
                            Steps forward (+1).
                          - On Continuous Rows (Azimuth, Altitude, Contrast,
                            Ambient, Time/Day, Exposure, Scale): Focuses row
                            for dragging.

    RIGHT MOUSE CLICK   : On Discrete Rows: Steps backward (-1).
    (or SHIFT + LMB)

    RIGHT DRAG /        : Smoothly scrubs Azimuth, Altitude, Contrast,
    HOLD LEFT / RIGHT     Ambient Ratio, Time-of-Day, Exposure, or unified
                          world Scale.

    TAP LEFT / RIGHT    : Steps discrete rows (Preset, World, Orbit Spd) by 1.

    LEFT DRAG           : Pan camera across lit terrain at 60 FPS.


[11.0 PYBIWIS 2.0 (OBJECT OCCUPANCY & OBSTACLE ROADMAP)]
-------------------------------------------------------------------------------
Repurposing 64-Bit Bitmasks:
In original PyNevo, PyBiwis stored 32x32 discrete tile collision. In PyNoiseBlend,
ground terrain does not collide with entities (it only modifies speed).

However, future world layers will spawn impassable procedural objects:
- Forest trees, boulders, stone ruins, or cliff spires.
- Using PyBiwis, an entire 128x128 chunk's obstacle occupancy grid (16,384
  pixels) is compressed into exactly 256 uint64 words (2 KB of RAM).
- Entity circle collisions query surrounding obstacle pixels at register
  speeds via bitwise shifts, delivering sub-pixel collision precision at
  negligible memory cost.


[12.0 COMPLETE MODULAR CODEBASE STRUCTURE]
-------------------------------------------------------------------------------

pynoiseblend/
│
├── config.py                     # Hardware, display, zoom, clamps & row indices
├── main.py                       # Root entry point; boots studio.app.run_app()
├── README.md                     # Master architecture & technical specification
├── MANUAL.txt                    # Configuration manual & field guide
│
├── profiles/                     # Data-Driven Recipe Library
│   ├── chunk.yaml                # Spatial chunk presets (PERFORMANT, BALANCED)
│   ├── lighting.yaml             # Solar lighting & day/night presets
│   ├── noise.yaml                # Raw noise presets (Simplex, Perlin, Value)
│   ├── strata.yaml               # Biome color ramps & terrain speed modifiers
│   └── world.yaml                # Noise-engine builds (seed, sub-sample, layers)
│
├── core/                         # Mathematical & Procedural Kernel (Pure NumPy)
│   ├── contour_engine.py         # Shared culled, in-place & tile isoline kernel
│   ├── formulas.py               # Master 12-formula catalog, presets & 2D specs
│   └── wave_synthesizer.py       # 1D/2D wave math, dynamic centers & meshes
│
├── world/                        # Spatial World & Streaming Engine
│   ├── chunks/                   # Spatial Chunking Sub-Package
│   │   ├── chunk.py              # Chunk container (elevation, gradients, surf)
│   │   ├── chunk_profile.py      # ChunkProfile data model & YAML loader
│   │   ├── chunk_provider.py     # Abstract ChunkProvider contract
│   │   ├── chunk_manager.py      # Agnostic streaming, center-outward queue
│   │   ├── world_profile.py      # WorldProfile data model & YAML loader
│   │   └── providers/            # Concrete Procedural Chunk Factories
│   │       ├── base_chunk_provider.py  # Shared LUT, isolines & SDL baking
│   │       ├── world_chunk_provider.py # Formula world factory & halos
│   │       └── noise_chunk_provider.py # Raw noise factory & tile stepping
│   │
│   ├── lighting/                 # Solar Lighting & Hillshading Subsystem
│   │   ├── ambient_palette.py    # Time-of-day RGB ambient keyframe resolver
│   │   ├── atmosphere_overlay.py # Single-pass lightmap BLEND_MULT manager
│   │   ├── height_shadow_engine.py # 3D solar dot-product & lightmap kernel
│   │   ├── lighting_profile.py   # LightingProfile data model & YAML loader
│   │   ├── time_clock.py         # Normalized orbit clock & pause controller
│   │   └── viewport_height_sampler.py # Direct chunk gradient strider
│   │
│   └── strata/                   # Continuous Strata Color Subsystem
│       ├── strata_keyframe.py    # StrataKeyframe data model
│       ├── strata_profile.py     # StrataProfile container & pre-baked LUTs
│       └── strata_loader.py      # YAML strata compiler, validator & heatmaps
│
├── studio/                       # Developer Inspection & Wave Math Preview Suite
│   ├── app.py                    # Master studio lifecycle manager & main loop
│   ├── state.py                  # Session state store, timers & seed modal
│   ├── input_handler.py          # Unified mouse/keyboard event polling & router
│   ├── matrix_tuner.py           # Modes 3/4 matrix stepping, scaling & resets
│   ├── wave_tuner.py             # Modes 1/2 channel focus, shifts & scaling
│   └── renderers/                # Modular Viewport Renderers
│       ├── base_renderer.py      # Shared typography, HUD cards & 2-line dock
│       ├── base_chunk_renderer.py# Chunk preview, scrub & progressive refine
│       ├── parameter_matrix.py   # Interactive parameter matrix deck UI widget
│       ├── renderer_1d.py        # Mode 1: 1D Wave Oscilloscope
│       ├── renderer_2d_synth.py  # Mode 2: 2D Spatial Synthesizer & Net
│       ├── renderer_lighting.py  # Mode 5: Solar Lighting & Hillshading Studio
│       ├── renderer_noise.py     # Mode 3: Raw Noise Explorer
│       └── renderer_world.py     # Mode 4: Procedural World Explorer
│
└── utils/                        # Generic Utilities, Resampling & Noise Engines
    ├── error_handler.py          # fail_fast() and global crash pause hook
    ├── resampler.py              # Zero-dependency cached 2D bilinear upsampler
    └── noise/                    # Procedural Noise Package
        ├── simplex.py            # Pure vectorized 2D Simplex noise engine
        ├── perlin.py             # Pure vectorized 2D Perlin noise engine
        ├── value.py              # Vectorized 2D Value noise with quintic ease
        └── noise_generator.py    # FBM coordinator, offsets & profile cloning

===============================================================================
[!] END OF SPECIFICATIONS | MAINTAIN ALIGNMENT WITH ESTABLISHED DATA FLOWS
===============================================================================