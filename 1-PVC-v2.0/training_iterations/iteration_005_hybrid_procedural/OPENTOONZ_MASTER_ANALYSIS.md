# OpenToonz Sample Projects - Master Analysis

Analyzed 0 projects.

## Key Insights for Procedural Codec

### Layer Structure
OpenToonz projects are organized in a hierarchical structure:
1. **Levels** - Reusable assets (drawings, backgrounds)
2. **Columns** - Timeline layers that reference levels
3. **Effects** - Transformations applied to layers

### Procedural Nature
- Anime frames are **composited from layers**
- Each layer is a **reference to a reusable asset** (not raw pixels)
- Effects are **parametric** (e.g., blur radius, color adjustment)
- Timeline is **frame-based** with explicit timing control

### Codec Implications
Our neural codec should:
1. **Decompose frames into layers** (line art, color, shadow)
2. **Identify reusable assets** across frames
3. **Predict parametric operations** (transforms, effects)
4. **Use temporal references** (level reuse = asset compression)
