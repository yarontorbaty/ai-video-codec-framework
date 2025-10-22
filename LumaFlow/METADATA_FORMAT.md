# LumaFlow Motion Metadata Format

## Overview

Each LumaFlow video recording produces two files:
- **`lumaflow_TIMESTAMP.mov`** - Video file with RGB (track 0) and depth (track 1)
- **`lumaflow_TIMESTAMP.json`** - Motion metadata sidecar file

The JSON file contains per-frame camera motion data captured from ARKit.

---

## JSON Structure

```json
{
  "version": "1.0",
  "format": "lumaflow_motion",
  "frame_count": 150,
  "frames": [
    {
      "frame": 0,
      "timestamp": 0.0,
      "tracking_state": "normal",
      "tracking_confidence": 1.0,
      "position": {
        "x": 0.0,
        "y": 0.0,
        "z": 0.0
      },
      "rotation": {
        "x": 0.0,
        "y": 0.0,
        "z": 0.0,
        "w": 1.0
      },
      "linear_velocity": {
        "x": 0.0,
        "y": 0.0,
        "z": 0.0
      },
      "angular_velocity": {
        "x": 0.0,
        "y": 0.0,
        "z": 0.0
      },
      "intrinsics": {
        "focal_length": {
          "x": 1456.2,
          "y": 1456.2
        },
        "principal_point": {
          "x": 960.0,
          "y": 540.0
        },
        "resolution": {
          "width": 1920,
          "height": 1080
        }
      },
      "transform": [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
      ]
    }
    // ... more frames
  ]
}
```

---

## Field Descriptions

### Top-Level Fields

| Field | Type | Description |
|-------|------|-------------|
| `version` | String | Metadata format version (current: "1.0") |
| `format` | String | Format identifier ("lumaflow_motion") |
| `frame_count` | Integer | Total number of frames |
| `frames` | Array | Array of per-frame metadata objects |

### Per-Frame Fields

#### Basic Info
| Field | Type | Units | Description |
|-------|------|-------|-------------|
| `frame` | Integer | - | Frame index (0-based) |
| `timestamp` | Float | seconds | Time since recording started |
| `tracking_state` | String | - | ARKit tracking state: "normal", "limited", "notAvailable" |
| `tracking_confidence` | Float | 0.0-1.0 | Tracking quality (1.0 = best) |

#### Camera Position (Translation)
| Field | Type | Units | Description |
|-------|------|-------|-------------|
| `position.x` | Float | meters | Camera X position in world space (right) |
| `position.y` | Float | meters | Camera Y position in world space (up) |
| `position.z` | Float | meters | Camera Z position in world space (backward) |

**Coordinate System:** Right-handed, ARKit world space
- **Origin:** Determined by ARKit at session start
- **X-axis:** Right
- **Y-axis:** Up (against gravity)
- **Z-axis:** Backward (camera looks toward -Z)

#### Camera Rotation (Orientation)
| Field | Type | Units | Description |
|-------|------|-------|-------------|
| `rotation.x` | Float | - | Quaternion X component |
| `rotation.y` | Float | - | Quaternion Y component |
| `rotation.z` | Float | - | Quaternion Z component |
| `rotation.w` | Float | - | Quaternion W component (scalar) |

**Format:** Unit quaternion (normalized to length 1)
- Represents rotation from world space to camera space
- Identity rotation (no rotation): `{x: 0, y: 0, z: 0, w: 1}`

#### Linear Velocity (Optional)
| Field | Type | Units | Description |
|-------|------|-------|-------------|
| `linear_velocity.x` | Float | m/s | Velocity in X direction |
| `linear_velocity.y` | Float | m/s | Velocity in Y direction |
| `linear_velocity.z` | Float | m/s | Velocity in Z direction |

**Calculation:** `(position_current - position_previous) / delta_time`
- Not available for first frame (no previous position)

#### Angular Velocity (Optional)
| Field | Type | Units | Description |
|-------|------|-------|-------------|
| `angular_velocity.x` | Float | rad/s | Rotation rate around X-axis |
| `angular_velocity.y` | Float | rad/s | Rotation rate around Y-axis |
| `angular_velocity.z` | Float | rad/s | Rotation rate around Z-axis |

**Calculation:** Derived from quaternion difference between frames
- Axis-angle representation scaled by time
- Not available for first frame

#### Camera Intrinsics
| Field | Type | Units | Description |
|-------|------|-------|-------------|
| `intrinsics.focal_length.x` | Float | pixels | Focal length in X direction (fx) |
| `intrinsics.focal_length.y` | Float | pixels | Focal length in Y direction (fy) |
| `intrinsics.principal_point.x` | Float | pixels | Optical center X coordinate (cx) |
| `intrinsics.principal_point.y` | Float | pixels | Optical center Y coordinate (cy) |
| `intrinsics.resolution.width` | Float | pixels | Native camera width |
| `intrinsics.resolution.height` | Float | pixels | Native camera height |

**Camera Projection Matrix:**
```
K = [fx  0   cx]
    [0   fy  cy]
    [0   0   1 ]
```

Used to project 3D points to 2D screen coordinates:
```python
x_2d = (X_3d * fx / Z_3d) + cx
y_2d = (Y_3d * fy / Z_3d) + cy
```

#### Transform Matrix
| Field | Type | Description |
|-------|------|-------------|
| `transform` | 4×4 Array | Camera transform matrix (column-major) |

**Format:** 4×4 transformation matrix (camera-to-world)
```
[R R R tx]
[R R R ty]  R = 3×3 rotation matrix
[R R R tz]  t = translation vector
[0 0 0 1 ]
```

- First 3 columns: Rotation matrix
- 4th column: Translation (position)
- Bottom row: [0, 0, 0, 1] (homogeneous coordinates)

---

## Usage Examples

### Python: Load Metadata

```python
import json
import numpy as np

def load_lumaflow_metadata(json_path):
    with open(json_path, 'r') as f:
        metadata = json.load(f)
    
    frame_count = metadata['frame_count']
    frames = metadata['frames']
    
    # Extract positions as numpy array
    positions = np.array([[f['position']['x'], 
                           f['position']['y'], 
                           f['position']['z']] for f in frames])
    
    # Extract rotations (quaternions)
    rotations = np.array([[f['rotation']['x'],
                           f['rotation']['y'],
                           f['rotation']['z'],
                           f['rotation']['w']] for f in frames])
    
    # Extract velocities (if available)
    velocities = []
    for f in frames:
        if 'linear_velocity' in f:
            velocities.append([f['linear_velocity']['x'],
                             f['linear_velocity']['y'],
                             f['linear_velocity']['z']])
        else:
            velocities.append([0, 0, 0])
    velocities = np.array(velocities)
    
    return {
        'positions': positions,
        'rotations': rotations,
        'velocities': velocities,
        'frame_count': frame_count
    }

# Usage
metadata = load_lumaflow_metadata('lumaflow_1234567890.json')
print(f"Total frames: {metadata['frame_count']}")
print(f"Camera trajectory: {metadata['positions'].shape}")
```

### Python: Compute Camera Path Length

```python
import numpy as np

def compute_path_length(positions):
    """Compute total distance traveled by camera"""
    deltas = np.diff(positions, axis=0)  # Frame-to-frame displacement
    distances = np.linalg.norm(deltas, axis=1)  # Euclidean distance
    total_distance = np.sum(distances)
    return total_distance

metadata = load_lumaflow_metadata('lumaflow_1234567890.json')
path_length = compute_path_length(metadata['positions'])
print(f"Camera traveled: {path_length:.2f} meters")
```

### Python: Quaternion to Rotation Matrix

```python
import numpy as np

def quat_to_matrix(q):
    """Convert quaternion [x, y, z, w] to 3×3 rotation matrix"""
    x, y, z, w = q
    return np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
    ])

# Convert first frame rotation to matrix
frame0_quat = metadata['rotations'][0]
R = quat_to_matrix(frame0_quat)
print("Rotation matrix:")
print(R)
```

### Python: Project 3D Point to 2D

```python
def project_point(point_3d, intrinsics):
    """Project 3D point to 2D pixel coordinates"""
    fx = intrinsics['focal_length']['x']
    fy = intrinsics['focal_length']['y']
    cx = intrinsics['principal_point']['x']
    cy = intrinsics['principal_point']['y']
    
    x, y, z = point_3d
    u = (x * fx / z) + cx
    v = (y * fy / z) + cy
    
    return (u, v)

# Example: Project a point 2 meters in front of camera
intrinsics = frames[0]['intrinsics']
pixel_coords = project_point([0, 0, 2.0], intrinsics)
print(f"Pixel coordinates: {pixel_coords}")
```

---

## Coordinate System Reference

### ARKit World Space
```
       Y (Up)
       |
       |
       +---- X (Right)
      /
     /
    Z (Backward)
```

- **Origin:** Set by ARKit when session starts
- **Gravity-aligned:** Y-axis points up against gravity
- **Right-handed:** X × Y = Z

### Camera Space
```
Camera looks toward -Z

    Y (Up)
    |
    |
    +---- X (Right)
   /
  /
 Z (Out of screen)
```

---

## File Pairing

Videos and metadata files share the same timestamp:
- **Video:** `lumaflow_1234567890.mov`
- **Metadata:** `lumaflow_1234567890.json`

Both files are saved to the app's Documents directory (accessible via Files app).

---

## Version History

### Version 1.0 (Current)
- Initial format with full 6DOF motion data
- Camera position, rotation, velocity
- Camera intrinsics
- ARKit tracking quality

---

## See Also

- [MOTION_DATA_GUIDE.md](MOTION_DATA_GUIDE.md) - Comprehensive motion data documentation
- [XCODE_SETUP.md](XCODE_SETUP.md) - iOS app setup guide
- [README.md](../README.md) - Main project documentation

