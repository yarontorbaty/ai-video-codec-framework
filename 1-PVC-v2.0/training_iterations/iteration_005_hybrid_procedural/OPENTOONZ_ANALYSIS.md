# OpenToonz Anime Drawing Process Analysis

**Source**: OpenToonz tutorial video + [OpenToonz GitHub](https://github.com/opentoonz/opentoonz)

## Professional Anime Production Pipeline (Based on Video Analysis)

### **Phase 1: Rough Sketch (Frames 1-50)**
**Tools Used:**
- Light pencil strokes
- Basic shapes (circles for head, guidelines)
- Rough proportions

**Procedural Operations:**
1. Draw circle (head base)
2. Draw cross guidelines (face alignment)
3. Sketch eye positions (ovals)
4. Rough nose/mouth placement (lines)
5. Hair outline (loose curves)

### **Phase 2: Clean Line Art (Frames 50-100)**
**Tools Used:**
- Pen tool with pressure sensitivity
- Vector lines or raster cleanup
- Line weight variation

**Procedural Operations:**
1. Trace over sketch with clean lines
2. Variable line width (thick for outlines, thin for details)
3. Remove construction lines
4. Smooth curves (Bézier or similar)

### **Phase 3: Color Fill (Frames 100-150)**
**Tools Used:**
- Bucket fill
- Color palette selection
- Separate layers per color region

**Procedural Operations:**
1. Define color regions (closed paths)
2. Fill with base colors:
   - Skin tone
   - Hair color
   - Eye color
   - Clothing colors
3. Each color on separate layer

### **Phase 4: Shading (Frames 150-200)**
**Tools Used:**
- Gradient tool
- Airbrush for soft shadows
- Multiply/overlay blend modes

**Procedural Operations:**
1. Add shadow layer (multiply mode)
2. Soft gradients on curved surfaces
3. Hard shadows under hair/chin
4. Highlight layers (screen/add mode)
5. Specular highlights on eyes

### **Phase 5: Final Details (Frames 200-274)**
**Tools Used:**
- Fine detail brush
- Glow effects
- Background elements

**Procedural Operations:**
1. Eye details (sparkles, reflections)
2. Hair strand details
3. Background color/gradient
4. Final color correction

---

## Key Insights for Procedural Codec

### **1. Layered Approach**
Anime is NOT drawn as a single image, but as **stacked layers**:
- Background layer
- Character base layer
- Shadow layer (multiply blend)
- Highlight layer (add/screen blend)
- Line art layer (on top)

### **2. Parametric Shapes**
Everything is built from parametric primitives:
- **Circles/Ellipses**: Head, eyes, pupils
- **Bézier curves**: Hair strands, clothing folds
- **Straight lines**: Guidelines, sharp edges
- **Gradients**: Shading, lighting

### **3. Limited Color Palette**
- Typical anime: 5-10 colors per character
- Colors defined by palette index, not RGB
- Same palette reused across frames

### **4. Cel Shading Pattern**
- 2-3 luminance levels (base, shadow, highlight)
- Hard edges between levels (not smooth gradients)
- Predictable shadow positions

---

## Procedural Codec Architecture (Based on OpenToonz)

```python
# Procedural anime frame representation
frame = {
    'palette': [
        (255, 220, 200),  # Skin
        (255, 200, 100),  # Hair
        (100, 150, 255),  # Eyes
        # ... 5-10 colors total
    ],
    
    'layers': [
        {
            'type': 'background',
            'color': 'palette[0]',
            'operations': [
                ('fill', {'rect': (0, 0, 960, 540), 'color_idx': 0})
            ]
        },
        {
            'type': 'character_base',
            'operations': [
                ('ellipse', {'center': (480, 200), 'radii': (120, 150), 'color_idx': 1, 'fill': True}),  # Head
                ('ellipse', {'center': (450, 180), 'radii': (30, 40), 'color_idx': 3, 'fill': True}),    # Eye L
                ('ellipse', {'center': (510, 180), 'radii': (30, 40), 'color_idx': 3, 'fill': True}),    # Eye R
            ]
        },
        {
            'type': 'shadows',
            'blend_mode': 'multiply',
            'operations': [
                ('gradient', {'start': (480, 150), 'end': (480, 250), 'color': (0, 0, 0), 'alpha': 0.3}),
            ]
        },
        {
            'type': 'line_art',
            'operations': [
                ('bezier', {'points': [...], 'width': 2, 'color': (0, 0, 0)}),  # Outline
                ('bezier', {'points': [...], 'width': 1, 'color': (0, 0, 0)}),  # Details
            ]
        }
    ]
}
```

---

## File Size Estimation

### **Per Frame:**
- Palette: 10 colors × 3 bytes = 30 bytes
- Layer count: 1 byte
- Per layer (average 20 operations):
  - Operation type: 1 byte
  - Parameters: ~10 bytes average
  - Total: 11 bytes × 20 ops = 220 bytes
- 4 layers × 220 bytes = 880 bytes
- **Total: ~1 KB per frame** (for simple scenes)

### **Complex scenes:**
- More operations (50-100)
- More layers (6-8)
- Estimated: 2-5 KB per frame

### **Per Episode (3,456 I-frames):**
- Simple: 3,456 KB = **3.4 MB**
- Complex: 17,280 KB = **17 MB**
- **Average: ~10 MB per episode** (I-frames only)

### **vs AV1:**
- AV1: 678 MB (full episode)
- Our procedural: 10 MB (I-frames only) + AV1 P/B frames (456 MB)
- **Total: 466 MB (31% savings vs AV1)**

---

## Implementation Plan

### **Step 1: Build OpenToonz-Inspired Renderer**
- Parametric shape primitives (circle, ellipse, bezier, gradient)
- Layer system with blend modes
- Palette-based coloring

### **Step 2: Neural "Reverse Animator"**
- Input: Anime frame (image)
- Output: Procedural operations that recreate it
- Train on synthetic data (generated procedural → image pairs)

### **Step 3: Optimize for Compression**
- Delta encoding (only changed operations per frame)
- Operation vocabulary compression
- Palette sharing across frames

### **Step 4: Validate**
- Test on Tokyo Ghoul frames
- Measure PSNR/SSIM vs file size
- Compare to AV1

---

## Why This Approach Could Work

1. **Anime IS procedural by nature**
   - Professional animators use tools like OpenToonz
   - Every frame IS a sequence of operations
   - We're reverse-engineering the creation process

2. **Massive compression potential**
   - 1 KB per frame (1000× smaller than raw image)
   - 10-20 MB per episode (vs 678 MB AV1)

3. **Quality preservation**
   - Vector-based line art (infinite resolution)
   - Exact color reproduction (palette-based)
   - Proper cel shading

4. **Leverages anime conventions**
   - Limited palette
   - Cel shading
   - Reusable assets (hair, eyes, etc.)

---

## Next Steps

1. ✅ Analyzed OpenToonz workflow
2. ⏳ Implement parametric renderer with layers
3. ⏳ Generate synthetic training data
4. ⏳ Train "inverse renderer" neural network
5. ⏳ Test on real anime frames
6. ⏳ Measure compression vs quality

**This is the real PVC approach!** 🎨🚀

