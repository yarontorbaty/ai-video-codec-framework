# OpenToonz Project Analysis: dwanko_run.tnz

## Project Overview

- **Total Frames:** 72
- **OpenToonz Version:** 71.0
- **Total Levels (Assets):** 10
- **Total Columns (Layers):** 10

## Asset Reuse Analysis

- **Unique Assets:** 20
- **Total Asset Instances:** 56
- **Average Reuse per Asset:** 2.80x
- **Compression Opportunity:** 64.3%

### Top 10 Most Reused Assets

- **level_9** (Frame: 0001) - Used 4x
- **level_9** (Frame: 0002) - Used 4x
- **level_9** (Frame: 0003) - Used 4x
- **level_9** (Frame: 0004) - Used 4x
- **level_9** (Frame: 0005) - Used 4x
- **level_9** (Frame: 0006) - Used 4x
- **level_8** (Frame: 0001) - Used 4x
- **level_8** (Frame: 0002) - Used 4x
- **level_8** (Frame: 0003) - Used 4x
- **level_8** (Frame: 0004) - Used 4x

## Levels (Assets)

### Unknown (10 assets)

- **level_1** (ID: 1)
  - Path: `"$scenefolder\\BG\\01_sky.tif"`
- **level_2** (ID: 2)
  - Path: `"$scenefolder\\BG\\02_trees.tif"`
- **level_3** (ID: 3)
  - Path: `"$scenefolder\\BG\\03_ground.tif"`
- **level_4** (ID: 4)
  - Path: `"$scenefolder\\BG\\04_bush1.tif"`
- **level_5** (ID: 5)
  - Path: `"$scenefolder\\BG\\05_bush2.tif"`
  - ... and 5 more

## Procedural Operations

Total: 12

- **Effect:columnFx:** 10 instance(s)
- **Camera:Unknown:** 2 instance(s)

## Codec Implications

Based on this project's structure:

1. **Asset Storage:** Only 20 unique drawings need to be stored
2. **Temporal Compression:** 36 frames are references (0 bytes each)
3. **Parametric Effects:** 12 operations can be stored as parameters (~20-50 bytes each)
4. **Estimated Savings:** 64.3% from asset reuse alone
