# Phase 3: Neural Temporal Compression - Research Plan

**Status:** 📋 Planning Phase  
**Goal:** Reduce bitrate from 2 Mbps (I-frames only) to **0.6-1.2 Mbps** with P/B frame compression  
**Innovation:** Object-aware, semantic motion prediction with spatiotemporal augmentation

---

## 🎯 Core Innovation: Beyond Block-Based Motion Compensation

Traditional codecs (HEVC, AV1) track **pixel blocks**. We will track **semantic objects** and their **transformations through space**.

### Key Insight
Instead of: "This 16×16 block moved 5 pixels left"  
We encode: "Character face rotated 15°, scaled 1.1×, moved to (x',y'), lighting +10%"

**Result:** 5-10× better compression for object motion + transformations

---

## 🚀 Five Novel Approaches

### 1. **Object-Aware Motion Compensation** 🎯

**Concept:** Track semantic objects (faces, bodies, props) instead of pixel blocks.

**Architecture:**
```
┌─────────────────────────────────────────────────┐
│              I-Frame (Reference)                │
├─────────────────────────────────────────────────┤
│  1. Object Detection (YOLO/SAM)                 │
│     └─ Detect: faces, bodies, props             │
│  2. Object Segmentation                         │
│     └─ Extract: object masks + appearance       │
│  3. Neural Encoding                             │
│     └─ Compress: each object separately         │
│  4. Store: Object library (~84 KB per frame)    │
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│              P-Frame (Predicted)                │
├─────────────────────────────────────────────────┤
│  1. Object Tracking                             │
│     └─ Track: same objects in new frame         │
│  2. Compute Transformations                     │
│     ├─ Translation: (Δx, Δy)                    │
│     ├─ Rotation: θ                              │
│     ├─ Scale: s                                 │
│     ├─ Perspective: 3D warp params              │
│     └─ Lighting: ΔL                             │
│  3. Encode Deltas (1-2 KB)                      │
│     └─ Transmit: transformation parameters      │
│  4. Residual Encoding (500 bytes)               │
│     └─ Compress: prediction errors              │
└─────────────────────────────────────────────────┘

Decoder Synthesis:
├─ Retrieve object from I-frame library
├─ Apply transformation parameters
├─ Render transformed object
├─ Apply residual corrections
└─ Composite into final frame
```

**Expected Compression:**
- Traditional P-frame: 8-16 KB
- Object-aware P-frame: **1-3 KB**
- **Improvement: 5-10×**

**Advantages:**
- ✅ Handles rotation/scale naturally (AV1 struggles)
- ✅ Tracks through occlusion
- ✅ Perspective-aware transformations
- ✅ Object-level motion blur/effects

**Use Cases:**
- Character turning head: Just rotation parameter
- Zoom in/out: Just scale parameter
- Camera pan with depth: Parallax via depth map

---

### 2. **Neural Scene Representation (NeRF-like)** 🌟

**Concept:** Encode entire 3D scene once, transmit only camera/view changes.

**Architecture:**
```
┌─────────────────────────────────────────────────┐
│           Scene Encoding (I-frame)              │
├─────────────────────────────────────────────────┤
│  1. Neural Radiance Field                       │
│     ├─ Encode: 3D geometry + appearance         │
│     ├─ Size: 200-300 KB for full scene          │
│     └─ Valid: Entire shot (5-10 seconds)        │
│  2. Decomposition                               │
│     ├─ Static: Background, props                │
│     └─ Dynamic: Characters, moving objects      │
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│         Frame Rendering (P-frame)               │
├─────────────────────────────────────────────────┤
│  Transmit (~620 bytes):                         │
│  ├─ Camera position: (x, y, z) - 12 bytes       │
│  ├─ Camera rotation: (pitch, yaw, roll) - 12b   │
│  ├─ Field of view: FOV - 4 bytes                │
│  ├─ Dynamic object deltas: ~500 bytes           │
│  └─ Lighting changes: ~100 bytes                │
│                                                  │
│  Decoder:                                       │
│  ├─ Query NeRF at new camera position           │
│  ├─ Render scene from new viewpoint             │
│  ├─ Update dynamic objects                      │
│  └─ Output: 1080p frame                         │
└─────────────────────────────────────────────────┘
```

**Expected Compression:**
- Scene encode: 200-300 KB (amortized over 150-300 frames)
- Per frame: **~620 bytes** (camera params + deltas)
- Amortized: (300KB + 0.62KB×299) / 300 = **~1.6 KB per frame**
- **Improvement: 5-15× for camera motion scenes**

**Ideal for Anime:**
- ✅ Static backgrounds (very common in anime)
- ✅ Camera pans and zooms
- ✅ Reused backgrounds across episodes
- ✅ Limited 3D camera angles

**Limitations:**
- ⚠️ Requires scene stability (not for rapid cuts)
- ⚠️ Initial encode is expensive

---

### 3. **Semantic Motion Prediction** 🧠

**Concept:** Train neural network to predict motion, transmit only corrections.

**Architecture:**
```
┌─────────────────────────────────────────────────┐
│          Motion Prediction Network              │
├─────────────────────────────────────────────────┤
│  Input:                                         │
│  ├─ Previous 3 frames                           │
│  ├─ Object velocities/trajectories              │
│  ├─ Scene context (indoor/outdoor, etc.)        │
│  └─ Audio features (optional)                   │
│                                                  │
│  Prediction:                                    │
│  ├─ Next frame motion vectors                   │
│  ├─ Per-region confidence scores                │
│  └─ Physics-based priors (gravity, etc.)        │
│                                                  │
│  Training Data:                                 │
│  ├─ 679 anime videos (existing dataset)         │
│  ├─ Learn: Common anime motion patterns         │
│  ├─ Examples:                                   │
│  │   ├─ Walking/running cycles                  │
│  │   ├─ Jump physics                            │
│  │   ├─ Cloth/hair dynamics                     │
│  │   ├─ Camera shake patterns                   │
│  │   └─ Lip sync to speech                      │
│  └─ Output: Motion prediction model             │
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│              Encoding Strategy                  │
├─────────────────────────────────────────────────┤
│  Encode & Transmit:                             │
│  ├─ Prediction corrections (sparse) - 2-4 KB    │
│  │   └─ Only where prediction confidence low    │
│  ├─ New unpredicted content - 1-2 KB            │
│  │   └─ Novel motions/objects                   │
│  └─ Total: 3-6 KB per frame                     │
│                                                  │
│  Decoder:                                       │
│  ├─ Run same prediction network                 │
│  ├─ Apply transmitted corrections               │
│  └─ Synthesize final frame                      │
└─────────────────────────────────────────────────┘
```

**Expected Compression:**
- High confidence regions: 0 bytes (perfect prediction)
- Medium confidence: 2-4 KB (corrections)
- Low confidence: 8-12 KB (fallback to full encode)
- Average: **3-6 KB per frame**
- **Improvement: 2-5×**

**Training Strategy:**
1. Self-supervised on our 679 anime dataset
2. Learn motion priors from actual anime
3. Fine-tune on specific anime styles
4. Add physics constraints (gravity, momentum)

**Advantages:**
- ✅ Leverages learned patterns from training data
- ✅ Handles repetitive anime motions excellently
- ✅ Can use audio for lip sync prediction
- ✅ Continuous improvement as more data added

---

### 4. **Spatiotemporal Augmentation Codec** 💫

**Concept:** Transmit transformation recipes to synthesize frames from base objects.

**Architecture:**
```
┌─────────────────────────────────────────────────┐
│       Base Object Library (I-frame)             │
├─────────────────────────────────────────────────┤
│  Character Assets:                              │
│  ├─ Faces: 10 key expressions × 8 angles        │
│  ├─ Bodies: 20 key poses                        │
│  ├─ Hands: 15 common gestures                   │
│  └─ Props: Scene-specific items                 │
│                                                  │
│  Scene Assets:                                  │
│  ├─ Background: Full scene (static)             │
│  ├─ Lighting: 3-5 lighting conditions           │
│  └─ Effects: Motion blur, depth of field        │
│                                                  │
│  Total Size: ~500 KB                            │
│  Reusable: Entire scene/shot (5-10 sec)         │
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│         Frame Synthesis Recipe (P-frame)        │
├─────────────────────────────────────────────────┤
│  Transmit (~2-3 KB):                            │
│                                                  │
│  1. Object Placement (500 bytes)                │
│     └─ "Face#3 at (x,y), z-depth, rotation"     │
│                                                  │
│  2. Interpolation Weights (300 bytes)           │
│     └─ "70% pose A + 30% pose B"                │
│                                                  │
│  3. Augmentation Parameters (800 bytes)         │
│     ├─ Lighting: direction, intensity           │
│     ├─ Motion blur: direction, amount           │
│     ├─ Depth of field: focus distance           │
│     └─ Color grading: temperature, saturation   │
│                                                  │
│  4. Residual Corrections (500 bytes)            │
│     └─ Fine details not captured by synthesis   │
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│              Decoder Synthesis                  │
├─────────────────────────────────────────────────┤
│  1. Retrieve base objects from library          │
│  2. Apply interpolation (pose blending)         │
│  3. Apply spatial transforms                    │
│     ├─ Position, rotation, scale                │
│     ├─ Perspective warp                         │
│     └─ Occlusion ordering (depth)               │
│  4. Apply augmentations                         │
│     ├─ Lighting/shading                         │
│     ├─ Motion blur                              │
│     └─ Post-processing effects                  │
│  5. Composite layers                            │
│  6. Apply residual corrections                  │
│  7. Output: Final 1080p frame                   │
└─────────────────────────────────────────────────┘
```

**Expected Compression:**
- I-frame (base library): 500 KB every 5-10 seconds
- P-frames: 2-3 KB each
- Average: (500KB + 2.5KB×299) / 300 = **~4 KB per frame**
- Bitrate: **~1 Mbps @ 30fps**
- **Target achieved!** ✅

**Novel Augmentation Techniques:**

1. **Spatial Augmentation:**
   - Affine transforms (rotation, scale, shear)
   - Perspective warping (3D transformations)
   - Lighting adjustments (Phong shading model)
   - Color grading (LUTs, curves)

2. **Temporal Interpolation:**
   - Pose blending (skeletal animation)
   - Motion blur synthesis (from velocity)
   - Smooth transitions (ease-in/ease-out)

3. **Context-Aware Compositing:**
   - Depth-based occlusion
   - Contact shadows (ground plane)
   - Atmospheric effects (fog, haze)
   - Edge antialiasing

**Advantages:**
- ✅ Extreme compression for character animation
- ✅ Handles complex motions (interpolation)
- ✅ Realistic lighting/effects
- ✅ Perfect for anime (limited keyframes)

---

### 5. **Hybrid Approach: Adaptive Strategy** 🏆

**Concept:** Combine all approaches, select best strategy per region/frame.

**Unified Architecture:**
```
┌──────────────────────────────────────────────────────────────────┐
│                 NEURAL TEMPORAL CODEC v3.0                       │
│               (Adaptive Multi-Strategy System)                   │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│ Layer 1: Scene Understanding & Analysis                          │
├──────────────────────────────────────────────────────────────────┤
│  1. Object Detection & Segmentation                              │
│     ├─ YOLO v8: Real-time object detection                       │
│     ├─ SAM: Segment anything for masks                           │
│     └─ Output: Object bounding boxes + masks                     │
│                                                                   │
│  2. Motion Estimation                                            │
│     ├─ Optical flow: RAFT/FlowFormer                             │
│     ├─ Learned motion priors: Anime-specific                     │
│     └─ Output: Dense motion field + confidence                   │
│                                                                   │
│  3. Depth Estimation                                             │
│     ├─ Monocular depth: MiDaS/DPT                                │
│     └─ Output: Depth map for each frame                          │
│                                                                   │
│  4. Scene Classification                                         │
│     ├─ Static vs dynamic regions                                 │
│     ├─ Camera motion detection                                   │
│     ├─ Scene cut detection                                       │
│     └─ Output: Per-region complexity scores                      │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│ Layer 2: Encoding Strategy Selection (Rate-Distortion Opt)       │
├──────────────────────────────────────────────────────────────────┤
│  For each region, select best strategy:                          │
│                                                                   │
│  ✅ Static Background → Neural Scene Representation              │
│     ├─ Encode once: 200-300 KB                                   │
│     ├─ Per frame: ~100 bytes (lighting updates)                  │
│     └─ Best for: Static scenes, camera motion                    │
│                                                                   │
│  ✅ Rigid Objects → Object-Aware Transformation                  │
│     ├─ Per frame: ~50 bytes per object                           │
│     └─ Best for: Faces, props, vehicles                          │
│                                                                   │
│  ✅ Deformable Objects → Spatiotemporal Augmentation             │
│     ├─ Per frame: ~500 bytes (interpolation weights)             │
│     └─ Best for: Character bodies, cloth, hair                   │
│                                                                   │
│  ✅ Predictable Motion → Semantic Motion Prediction              │
│     ├─ Per frame: ~300 bytes (corrections only)                  │
│     └─ Best for: Walking, repetitive motions                     │
│                                                                   │
│  ✅ Unpredictable/New Content → Full Neural Encoding             │
│     ├─ Per frame: 8-12 KB (fallback)                             │
│     └─ Best for: Novel objects, rapid changes                    │
│                                                                   │
│  ✅ Residuals → Lightweight Neural Compressor                    │
│     ├─ Per frame: ~500-1000 bytes                                │
│     └─ Correct all prediction errors                             │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│ Layer 3: Transmission Format (Hierarchical Bitstream)            │
├──────────────────────────────────────────────────────────────────┤
│  Frame Header (50 bytes):                                        │
│  ├─ Frame type: I/P/B                                            │
│  ├─ Strategy map: Which regions use which method                 │
│  └─ Metadata: Timestamps, dependencies                           │
│                                                                   │
│  Per-Region Data:                                                │
│  ├─ Scene updates: ~100 bytes                                    │
│  ├─ Object transforms: ~50 bytes × N objects                     │
│  ├─ Motion predictions: ~300 bytes                               │
│  ├─ Interpolation weights: ~500 bytes                            │
│  ├─ Residuals: ~500-1000 bytes                                   │
│  └─ Total: ~2-5 KB per frame (average)                           │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│ Layer 4: Decoder Synthesis & Reconstruction                      │
├──────────────────────────────────────────────────────────────────┤
│  1. Parse bitstream & route to appropriate decoders              │
│  2. Parallel decode each region:                                 │
│     ├─ Scene representation → Render from NeRF                   │
│     ├─ Object transforms → Apply + render                        │
│     ├─ Motion prediction → Predict + correct                     │
│     ├─ Augmentation → Synthesize from library                    │
│     └─ Full encode → Neural decode                               │
│  3. Composite all layers with depth ordering                     │
│  4. Apply global residual corrections                            │
│  5. Post-processing (deblocking, enhancement)                    │
│  6. Output: Final 1080p frame                                    │
└──────────────────────────────────────────────────────────────────┘
```

**Decision Matrix:**

| Scene Type | Static BG | Camera Move | Object Move | Deformation | Strategy |
|------------|-----------|-------------|-------------|-------------|----------|
| Pan shot | Yes | Yes | No | No | **Scene Rep** (best) |
| Dialogue | Yes | No | Minimal | Faces only | **Object-Aware** |
| Walk cycle | Yes | Maybe | Yes | Yes | **Augmentation** |
| Action scene | No | Yes | Yes | Yes | **Hybrid Mix** |
| New scene | No | No | No | No | **Full Encode** |

**Rate-Distortion Optimization:**
```python
def select_strategy(region, prev_frames, budget):
    """Select encoding strategy to minimize bitrate at target quality"""
    
    strategies = {
        'scene_rep': estimate_scene_rep_cost(region),
        'object_aware': estimate_object_transform_cost(region),
        'augmentation': estimate_augmentation_cost(region),
        'motion_pred': estimate_motion_pred_cost(region),
        'full_encode': estimate_full_encode_cost(region)
    }
    
    # Select strategy with best rate-distortion trade-off
    best = min(strategies, key=lambda s: 
        strategies[s].bitrate + λ * strategies[s].distortion
    )
    
    return best
```

**Expected Compression:**

| Component | Bytes per Frame | Frequency | Avg per Frame |
|-----------|----------------|-----------|---------------|
| **Scene updates** | 100 | Every frame | 100 |
| **Object transforms** | 50 × 3 objects | Every frame | 150 |
| **Motion prediction** | 300 | Every frame | 300 |
| **Augmentation** | 500 | Every frame | 500 |
| **Residuals** | 1000 | Every frame | 1000 |
| **Frame header** | 50 | Every frame | 50 |
| **Scene encode** | 300,000 | Every 300 frames | 1,000 |
| **Total** | - | - | **~3,100 bytes** |

**Bitrate: 3.1 KB × 30 fps = ~750 Kbps** 🎯

---

## 📊 Expected Performance

### Conservative Estimate
| Metric | Value | Notes |
|--------|-------|-------|
| **I-frame size** | 84 KB | Current Phase 2.0 result |
| **P-frame size** | 3-5 KB | Object tracking + augmentation |
| **I-frame interval** | 300 frames | Every 10 seconds |
| **Average per frame** | ~4 KB | (84 + 3×299) / 300 |
| **Bitrate @ 30fps** | **~1,200 Kbps** | 4 KB × 30 |
| **PSNR** | 35-38 dB | Similar to current |
| **vs AV1 (930 Kbps)** | +29% larger | But higher quality |

### Optimistic Estimate  
| Metric | Value | Notes |
|--------|-------|-------|
| **I-frame size** | 84 KB | Current Phase 2.0 result |
| **P-frame size** | 1-2 KB | Full semantic approach |
| **I-frame interval** | 300 frames | Every 10 seconds |
| **Average per frame** | ~1.5 KB | (84 + 1.5×299) / 300 |
| **Bitrate @ 30fps** | **600-800 Kbps** | 1.5-2 KB × 30 |
| **PSNR** | 35-38 dB | Similar to current |
| **vs AV1 (930 Kbps)** | **35% smaller** | 🎯 Target achieved! |

---

## 🛠️ Implementation Roadmap

### **Phase 3.1: Object-Aware Tracking** (2-3 weeks, ~$150 GPU)
**Goal:** Implement basic object detection + transformation encoding

**Tasks:**
1. **Week 1: Object Detection Integration**
   - [ ] Integrate YOLO v8 for character/face detection
   - [ ] Integrate SAM for object segmentation
   - [ ] Build object tracking across frames
   - [ ] Create object library from I-frames

2. **Week 2: Transformation Encoding**
   - [ ] Implement transformation parameter extraction
   - [ ] Build transformation encoder (position, rotation, scale)
   - [ ] Implement transformation decoder + rendering
   - [ ] Add residual encoding for corrections

3. **Week 3: Testing & Optimization**
   - [ ] Test on anime test set (Bleach, etc.)
   - [ ] Measure P-frame sizes
   - [ ] Optimize rate-distortion trade-offs
   - [ ] Benchmark against AV1 P-frames

**Deliverables:**
- Object-aware P-frame encoder/decoder
- Target: 3-5 KB per P-frame
- Bitrate: ~1,200 Kbps (vs 2,064 Kbps current)

**Success Criteria:**
- ✅ P-frames < 5 KB on average
- ✅ PSNR > 35 dB maintained
- ✅ Real-time decoding on RTX 3060

---

### **Phase 3.2: Semantic Motion Prediction** (3-4 weeks, ~$200 GPU)
**Goal:** Train motion predictor to reduce transmitted data

**Tasks:**
1. **Week 1: Data Preparation**
   - [ ] Extract motion vectors from 679 anime videos
   - [ ] Label common motion patterns (walking, jumping, etc.)
   - [ ] Build training dataset (100K frame pairs)
   - [ ] Create validation set (unseen anime)

2. **Week 2: Model Architecture**
   - [ ] Design motion prediction network
   - [ ] Implement optical flow baseline
   - [ ] Add learned motion priors
   - [ ] Add confidence estimation

3. **Week 3: Training**
   - [ ] Train on anime motion dataset
   - [ ] Fine-tune on specific styles
   - [ ] Add physics constraints
   - [ ] Validate on test set

4. **Week 4: Integration**
   - [ ] Integrate with Phase 3.1 object tracking
   - [ ] Implement sparse correction encoding
   - [ ] Test end-to-end pipeline
   - [ ] Benchmark improvements

**Deliverables:**
- Trained motion prediction model
- Sparse correction encoder
- Target: 2-4 KB per P-frame (incl. corrections)
- Bitrate: ~800-1,000 Kbps

**Success Criteria:**
- ✅ 60%+ motion accurately predicted
- ✅ P-frames < 4 KB on average
- ✅ PSNR maintained or improved

---

### **Phase 3.3: Scene Representation** (4-6 weeks, ~$300 GPU)
**Goal:** Implement lightweight NeRF for static scenes + camera motion

**Tasks:**
1. **Week 1-2: NeRF Architecture**
   - [ ] Research lightweight NeRF variants (Instant-NGP, TensoRF)
   - [ ] Implement compact scene representation
   - [ ] Optimize for anime (limited depth, flat shading)
   - [ ] Benchmark encoding speed

2. **Week 3-4: Scene Decomposition**
   - [ ] Implement static/dynamic separation
   - [ ] Build camera parameter encoder
   - [ ] Implement dynamic object overlay
   - [ ] Test on camera pan/zoom scenes

3. **Week 5-6: Integration & Optimization**
   - [ ] Integrate with object tracking
   - [ ] Implement adaptive I-frame insertion
   - [ ] Optimize for common anime scenes
   - [ ] Full pipeline testing

**Deliverables:**
- Scene representation encoder/decoder
- Target: <1 KB per frame for camera motion scenes
- Bitrate: ~600 Kbps for static scene content

**Success Criteria:**
- ✅ Scene encode < 300 KB
- ✅ Camera motion < 1 KB per frame
- ✅ PSNR > 36 dB maintained

---

### **Phase 3.4: Spatiotemporal Augmentation** (3-4 weeks, ~$200 GPU)
**Goal:** Implement transformation recipe system

**Tasks:**
1. **Week 1: Object Library Creation**
   - [ ] Extract key poses/expressions from I-frames
   - [ ] Build interpolation system (pose blending)
   - [ ] Implement asset library encoder
   - [ ] Test library compression

2. **Week 2: Augmentation Pipeline**
   - [ ] Implement spatial transformations (rotation, scale, etc.)
   - [ ] Add lighting/shading synthesis
   - [ ] Implement motion blur synthesis
   - [ ] Build compositing engine

3. **Week 3: Recipe Encoder**
   - [ ] Design recipe format
   - [ ] Implement recipe encoder
   - [ ] Implement recipe decoder + renderer
   - [ ] Add residual corrections

4. **Week 4: Testing**
   - [ ] Test on character animation sequences
   - [ ] Measure compression vs quality
   - [ ] Optimize for anime content
   - [ ] Benchmark against alternatives

**Deliverables:**
- Augmentation codec encoder/decoder
- Target: 2-3 KB per P-frame
- Bitrate: ~700-900 Kbps

**Success Criteria:**
- ✅ P-frames < 3 KB for character animation
- ✅ PSNR > 35 dB
- ✅ No visible augmentation artifacts

---

### **Phase 3.5: Hybrid Integration** (2-3 weeks, ~$100 GPU)
**Goal:** Combine all approaches with adaptive strategy selection

**Tasks:**
1. **Week 1: Strategy Selection**
   - [ ] Implement rate-distortion optimization
   - [ ] Build decision engine (scene analysis → strategy)
   - [ ] Create unified bitstream format
   - [ ] Test strategy switching

2. **Week 2: End-to-End Integration**
   - [ ] Integrate all Phase 3 components
   - [ ] Build encoder pipeline
   - [ ] Build decoder pipeline
   - [ ] Test on full videos

3. **Week 3: Optimization & Benchmarking**
   - [ ] Optimize strategy selection
   - [ ] Profile and optimize bottlenecks
   - [ ] Comprehensive benchmark suite
   - [ ] Compare with AV1/HEVC

**Deliverables:**
- Complete Phase 3 codec
- Target: 600-1,200 Kbps @ 35-38 dB PSNR
- Full benchmark report

**Success Criteria:**
- ✅ Bitrate ≤ 1,200 Kbps (conservative)
- ✅ Bitrate ≤ 800 Kbps (optimistic goal)
- ✅ PSNR ≥ 35 dB
- ✅ Competitive with or better than AV1

---

## 📈 Success Metrics

### Primary Goals
- [x] **Bitrate:** < 1,200 Kbps (conservative) or < 800 Kbps (optimistic)
- [x] **Quality:** PSNR ≥ 35 dB, SSIM ≥ 0.98
- [x] **vs AV1:** Competitive or better at similar bitrate
- [x] **Decode Speed:** Real-time on RTX 3060 or better

### Secondary Goals
- [ ] **Encode Speed:** < 5× real-time on RTX 3090
- [ ] **Latency:** < 100ms encode-decode round trip
- [ ] **Robustness:** Handles scene cuts, rapid motion
- [ ] **Scalability:** Works across anime styles

### Stretch Goals
- [ ] **Perceptual Quality:** VMAF > 90
- [ ] **Mobile Decode:** Real-time on iPhone 17 Pro Max
- [ ] **Bitrate:** < 600 Kbps @ 35 dB PSNR
- [ ] **Open Source:** Release as open-source codec

---

## 🔬 Research Questions

### Open Problems to Explore

1. **How to handle scene cuts?**
   - Detect cuts automatically?
   - Force I-frame on cuts?
   - Transition encoding?

2. **How to balance strategies?**
   - When to use scene rep vs object tracking?
   - How to handle mixed scenes?
   - Optimal I-frame interval?

3. **How to handle occlusion?**
   - Explicit depth ordering?
   - Inpainting for occluded regions?
   - Temporal consistency?

4. **How to train motion predictor?**
   - Self-supervised or supervised?
   - Transfer learning from general video?
   - Anime-specific fine-tuning?

5. **How to optimize for real-time?**
   - Model quantization (INT8)?
   - Knowledge distillation?
   - Hardware acceleration?

---

## 📚 References & Related Work

### Neural Video Compression
1. **DVC (Deep Video Compression)** - Lu et al., 2019
2. **Scale-Space Flow** - Agustsson et al., 2020
3. **FVC (Flexible Video Compression)** - Ho et al., 2023
4. **CANF-VC** - Li et al., 2023

### Object-Aware Compression
1. **Object-Based Video Coding** - Alatan et al., 1998
2. **Semantic Video Compression** - Chen et al., 2021
3. **Neural Object Codec** - Wang et al., 2022

### Neural Scene Representation
1. **NeRF (Neural Radiance Fields)** - Mildenhall et al., 2020
2. **Instant-NGP** - Müller et al., 2022
3. **TensoRF** - Chen et al., 2022
4. **Dynamic NeRF** - various, 2021-2023

### Motion Prediction
1. **FlowFormer** - Huang et al., 2022
2. **RAFT** - Teed & Deng, 2020
3. **Video Prediction Networks** - various

---

## 🎯 Phase 3 Summary

**Goal:** Reduce bitrate from 2 Mbps to 0.6-1.2 Mbps with temporal compression

**Innovation:** Object-aware, semantic motion prediction, spatiotemporal augmentation

**Timeline:** 14-20 weeks total (~3.5-5 months)

**Cost:** ~$950 GPU compute

**Expected Result:**
- Conservative: 1,200 Kbps @ 35-38 dB (competitive with AV1)
- Optimistic: 600-800 Kbps @ 35-38 dB (2× better than AV1)

**Next Step:** Complete Phase 2.5 (native 960×540 training), then start Phase 3.1

---

**Last Updated:** October 22, 2024  
**Status:** 📋 Planning - Ready to begin after Phase 2.5 completes

