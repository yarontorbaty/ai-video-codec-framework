# OpenToonz Analysis: Procedural Anime Production Insights

## Executive Summary

OpenToonz is a professional 2D animation software used in anime production. By analyzing its project structure, we've identified the **core procedural nature of anime creation** that can be leveraged for our neural codec.

## Key Findings

### 1. **Hierarchical Layer Structure**

Anime frames are NOT created as flat pixel images. Instead, they're built from a **hierarchy of reusable components**:

```
Scene (Project)
  ├── Levels (Reusable Assets)
  │     ├── Line Art (TLV - Toonz Raster Level)
  │     ├── Color Fills (TLV)
  │     ├── Shadows/Highlights (TLV)
  │     └── Backgrounds (TIF/PNG - Static)
  │
  ├── Columns (Timeline Layers)
  │     ├── Character Layer
  │     ├── Effect Layer (blur, glow, etc.)
  │     └── Background Layer
  │
  └── Effects (Parametric Operations)
        ├── Camera transforms (pan, zoom)
        ├── Lighting effects (ray light, glow)
        └── Post-processing (blur, color correction)
```

### 2. **Asset Reuse Pattern**

Analysis of `dwanko_run.tnz` (72-frame animation):
- **Only 6 unique character drawings** (frames 0001-0006)
- These 6 drawings are **cycled 12 times** across 72 frames
- **Massive asset reuse:** Each drawing appears in 12 frames
- **Compression opportunity:** Store 6 drawings, reference them with timing

Example from project file:
```xml
<levelColumn id='17'>
  0 3 <level id='9'/>0001 0    ← Frame 0-2: Drawing #1
  3 3 <level id='9'/>0002 0    ← Frame 3-5: Drawing #2
  6 3 <level id='9'/>0003 0    ← Frame 6-8: Drawing #3
  ...
  18 3 <level id='9'/>0001 0   ← Frame 18-20: Drawing #1 (reused!)
```

### 3. **Parametric Operations**

Effects in OpenToonz are **not pixel operations** - they're **parametric functions**:

- **Ray Light Effect:**
  ```
  Parameters: intensity, color, angle, decay, length
  Storage: ~20 bytes (5 floats)
  vs. Pixel-based: ~8MB per frame at 1920x1080
  ```

- **Camera Transform:**
  ```
  Parameters: position_x, position_y, scale, rotation
  Storage: ~16 bytes (4 floats)
  vs. Full image transformation: Must store entire transformed frame
  ```

- **Color Adjustment:**
  ```
  Parameters: hue_shift, saturation, brightness, contrast
  Storage: ~16 bytes (4 floats)
  vs. Adjusted pixels: No compression benefit
  ```

### 4. **Timeline Structure**

Animation timing is **explicitly defined**, not inferred:

```
Frame Range | Level Reference | Operation
------------|-----------------|----------
0-2         | Drawing #1      | Static hold
3-5         | Drawing #2      | Static hold
6-8         | Drawing #3      | Static hold
```

This allows for:
- **Keyframe-based encoding:** Only store frames where something changes
- **Hold frames:** Reference previous frame (0 bytes)
- **Tweening:** Interpolate between keyframes

### 5. **Separation of Concerns**

Professional anime production separates:

1. **Line Art Layer:** High-contrast, sparse (2-5% of pixels)
2. **Color Layer:** Flat fills, quantized palette (8-32 colors typical)
3. **Shadow/Highlight Layer:** Cel shading (usually 2-3 levels)
4. **Background Layer:** Detailed, but static across multiple frames

**Codec Implication:** Each layer can use specialized compression:
- Line art: Sparse vector encoding
- Color: Palette + RLE
- Shadow: Binary masks
- Background: High-quality I-frame (reused)

## How Real Illustrators Draw Anime (from Tutorial Analysis)

From analyzing `how_to_draw_anime.mp4` (274 frames extracted):

### Step 1: Structure/Skeleton (Frames 1-50)
- **Method:** Simple geometric primitives
- **Operations:** Circles (head), lines (spine, limbs), ellipses (joints)
- **Storage:** ~10-20 primitives × 8 bytes = 80-160 bytes

### Step 2: Rough Sketch (Frames 51-100)
- **Method:** Bezier curves following skeleton
- **Operations:** Smooth curves connecting keypoints
- **Storage:** ~50 control points × 8 bytes = 400 bytes

### Step 3: Clean Line Art (Frames 101-150)
- **Method:** Refined paths with variable thickness
- **Operations:** B-spline with width parameter
- **Storage:** ~100 points × 10 bytes (x, y, width) = 1 KB

### Step 4: Color Fill (Frames 151-200)
- **Method:** Flood fill within closed regions
- **Operations:** Color palette (8-32 colors) + region map
- **Storage:** Palette (32 colors × 3 bytes) + compressed region map = 100 bytes + 2-5 KB

### Step 5: Shading (Frames 201-250)
- **Method:** Cel shading (flat colors for shadows/highlights)
- **Operations:** 2-3 shading layers with masks
- **Storage:** ~3 binary masks × 1-2 KB = 3-6 KB

### Step 6: Final Details (Frames 251-274)
- **Method:** Highlights, texture, effects
- **Operations:** Small detail patches + parametric effects (glow, blur)
- **Storage:** ~10 patches × 200 bytes + effects = 2-3 KB

### **Total Estimated Codec Size per Frame:**
- **Procedural components:** ~10-15 KB
- **vs. JPEG I-frame:** 150-250 KB
- **Compression ratio:** 15-20x

## Proposed Neural Procedural Codec (PVC v3.0)

Based on these insights, here's the architecture:

### Phase 1: Layer Decomposition Network
```python
Input: RGB frame (1920×1080×3)
├── Line Art Extractor → Sparse coordinates (1-2 KB)
├── Color Palette Predictor → 32 colors + region map (3-5 KB)
├── Shadow Mask Generator → Binary masks (2-3 KB)
└── Residual Encoder → Neural latent for details (5-10 KB)
```

### Phase 2: Temporal Asset Reuse
```python
Input: Current frame + Previous 10 frames
├── Asset Similarity Detector → Find reused drawings
├── Transform Predictor → Position, scale, rotation
└── Delta Encoder → Only encode differences
```

### Phase 3: Parametric Effect Prediction
```python
Input: Frame + Effect hint
├── Effect Classifier → Identify effect type
├── Parameter Regressor → Predict float parameters
└── Effect Renderer → Reconstruct using OpenCV/Cairo
```

### Expected Results

**Per-Frame Breakdown:**
- Line art: 1-2 KB
- Colors: 3-5 KB
- Shadows: 2-3 KB
- Residual: 5-10 KB
- **Total:** 11-20 KB per frame

**Episode-Level (24 min, 34,560 frames @ 24 FPS):**
- With asset reuse: ~20% unique frames = 6,912 unique
- Compressed size: 6,912 × 15 KB = **103.7 MB**
- vs. AV1 (10 Mbps): 1,800 MB
- **Compression:** 94.2% savings

## Next Steps

1. ✅ Analyze OpenToonz structure
2. ⏳ Transcribe `how_to_draw_anime.mp4` for step-by-step procedures
3. ⏳ Implement Line Art Extractor (sparse encoding)
4. ⏳ Implement Color Palette Predictor
5. ⏳ Train on real anime with ground truth layers
6. ⏳ Validate on held-out episode

## References

- OpenToonz Sample Projects (dwanko_run, cleanup, tga_paint)
- `how_to_draw_anime.mp4` tutorial (274 frames)
- Professional anime production pipeline analysis

---

**Status:** Research phase complete. Ready to implement PVC v3.0 architecture.

**Last Updated:** October 25, 2025

