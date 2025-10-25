# Iteration 005 v2.0: Layer-Based Hybrid Anime Codec

## Overview
Based on analysis of REAL anime frames, this codec decomposes anime into natural layers:
1. **Line Art Layer** (2.1% of pixels - sparse, vector-compressible)
2. **Color Palette** (16-32 colors)
3. **Color Map** (palette indices per pixel)
4. **Residual Details** (neural codec for soft gradients, lighting, textures)

## Key Insights from Real Anime Analysis

### From Bleach Frame Analysis:
- **Line art is sparse**: Only 2.1% of pixels are edges/outlines
- **Colors are quantizable**: 109K unique colors → 16 dominant colors (no quality loss)
- **Cel shading structure**: 3 luminance levels (shadows, mids, highlights)
- **Backgrounds are complex**: Gradients, blur effects, atmospheric lighting

### Compression Strategy:
```
Total: 1920×1080 = 2,073,600 pixels

Layer 1: Line Art (sparse)
  - Edge pixels: 2.1% = 43,545 pixels
  - Run-length encoding + coordinates
  - Estimated size: 1-2 KB

Layer 2: Color Palette
  - 16-32 colors × 3 bytes
  - Estimated size: 48-96 bytes

Layer 3: Color Map (palette indices)
  - 4-5 bits per pixel with RLE compression
  - Estimated size: 100-150 KB (uncompressed) → 20-40 KB (RLE)

Layer 4: Residual (soft details)
  - Neural codec for gradients, textures, lighting
  - 32-64 latent channels
  - Estimated size: 8-15 KB

Total estimated: ~30-60 KB per 1080p frame
Target: 40-50 KB (vs AV1 I-frame: 150-250 KB)
Compression: 3-6x better than AV1 I-frames!
```

## Architecture

### 1. Line Art Extractor
- Canny edge detection
- Thinning/skeletonization
- Run-length encoding
- Optional: vectorization for extreme compression

### 2. Color Palette Quantizer
- K-means clustering (K=16-32)
- Palette stored once per frame
- Each pixel mapped to palette index

### 3. Residual Neural Codec
- Encoder: Compress (original - line_art - palette_map)
- Decoder: Reconstruct soft details
- Small network (32-64 latent channels)

### 4. Reconstruction
```python
reconstructed_frame = (
    render_line_art(line_art_sparse) +
    apply_palette(color_map, palette) +
    decode_residual(residual_latent)
)
```

## Training Strategy

### Phase 1: Component Training (Separate)
1. Train line art extractor (supervised - use Canny as target)
2. Train palette quantizer (unsupervised - K-means)
3. Train residual codec (supervised - minimize reconstruction error)

### Phase 2: Joint Fine-tuning
- Fine-tune all components together
- End-to-end optimization
- Perceptual loss for visual quality

## Expected Results

### Compression:
- 1080p frame: ~40-50 KB
- AV1 I-frame: ~150-250 KB
- **3-6x better compression!**

### Quality:
- PSNR: 30-35 dB (competitive with AV1)
- SSIM: 0.95-0.98
- Perceptual quality: Excellent (preserves anime style)

### Advantages:
- Layer-based = interpretable
- Palette-based = authentic anime look
- Neural residual = captures complex details
- Fast encoding/decoding (parallel layers)

## Files Structure

```
iteration_005_v2_layered/
├── models/
│   ├── line_art_extractor.py
│   ├── palette_quantizer.py
│   ├── residual_codec.py
│   └── hybrid_layered_codec.py
├── training/
│   ├── train_line_art.py
│   ├── train_palette.py
│   ├── train_residual.py
│   └── train_joint.py
├── utils/
│   ├── layer_extraction.py
│   ├── compression.py
│   └── visualization.py
├── tests/
│   └── test_on_bleach.py
└── README.md
```

## Next Steps

1. ✅ Analyze real anime structure (DONE!)
2. ⏳ Implement line art extractor
3. ⏳ Implement palette quantizer
4. ⏳ Implement residual codec
5. ⏳ Train on real anime dataset (50K frames)
6. ⏳ Evaluate on Bleach/Frozen test videos
7. ⏳ Compare against AV1 I-frames

## References
- Real anime analysis: `/tmp/bleach_frame_18sec.png`
- Analysis visualization: `/tmp/anime_analysis.png`
- Key insight: Anime is naturally layered (line art + flat colors + soft details)

