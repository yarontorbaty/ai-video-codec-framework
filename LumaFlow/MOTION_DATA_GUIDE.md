# 📐 Motion Data Capture Guide

## Overview

LumaFlow now captures **rich 6DOF (6 Degrees of Freedom) motion data** alongside RGB video and LiDAR depth. This motion data is critical for the generative codec to understand camera movement and produce temporally consistent reconstructions.

---

## 🎯 Captured Motion Data

### **1. Camera Position (3DOF Translation)**
```swift
cameraPosition: simd_float3  // (x, y, z) in meters
```
- **What it is:** The camera's 3D position in world space
- **Units:** Meters
- **Coordinate System:** ARKit world space (right-handed)
  - X: Right
  - Y: Up
  - Z: Backward (camera looks toward -Z)
- **Use Case:** Track where the camera is in 3D space over time

### **2. Camera Orientation (3DOF Rotation)**
```swift
cameraRotation: simd_quatf  // Quaternion (x, y, z, w)
```
- **What it is:** The camera's 3D orientation as a quaternion
- **Format:** Unit quaternion (normalized to length 1)
- **Use Case:** Track camera rotation (pan, tilt, roll)

### **3. Linear Velocity**
```swift
linearVelocity: simd_float3?  // (vx, vy, vz) in m/s
```
- **What it is:** Camera movement speed in each axis
- **Units:** Meters per second
- **Calculation:** `(position_current - position_last) / delta_time`
- **Use Case:** Motion prediction for next frame

### **4. Angular Velocity**
```swift
angularVelocity: simd_float3?  // (wx, wy, wz) in rad/s
```
- **What it is:** Camera rotation speed around each axis
- **Units:** Radians per second
- **Calculation:** Derived from quaternion difference between frames
- **Use Case:** Predict camera rotation in next frame

---

## 📷 Camera Intrinsics

### **5. Focal Length**
```swift
focalLength: CGPoint  // (fx, fy) in pixels
```
- **What it is:** The camera's focal length in pixel units
- **Use Case:** Project 3D points to 2D screen coordinates
- **Typical Values:** ~1400-1500 pixels for iPhone

### **6. Principal Point**
```swift
principalPoint: CGPoint  // (cx, cy) in pixels
```
- **What it is:** The optical center of the image
- **Use Case:** Center point for perspective projection
- **Typical Values:** Center of image (~960, 540 for 1920×1080)

### **7. Image Resolution**
```swift
imageResolution: CGSize  // Native camera resolution
```
- **What it is:** The actual resolution of the camera sensor
- **Use Case:** Convert between pixel and normalized coordinates

---

## 🎯 Tracking Quality

### **8. Tracking State**
```swift
trackingState: String  // "normal", "limited", "notAvailable"
```
- **normal:** ARKit is tracking well with high confidence
- **limited:** Tracking is degraded (low light, fast motion, featureless surfaces)
- **notAvailable:** Tracking has been lost

### **9. Tracking Confidence**
```swift
trackingConfidence: Float  // 0.0 to 1.0
```
- **1.0:** High confidence (normal tracking)
- **0.5:** Medium confidence (limited tracking)
- **0.0:** No tracking

---

## 🧠 How the Codec Uses This Data

### **1. Motion Compensation**
```
Frame N: Camera at (0, 0, 0), looking forward
Frame N+1: Camera at (0.1, 0, 0), looking forward
```
→ The codec knows pixels shifted right by ~10cm, not that the scene changed

### **2. Temporal Prediction**
```
Linear velocity: (0.5 m/s, 0, 0)
Current position: (1.0, 0, 0)
```
→ Predicted next position: (1.5, 0, 0) in 1 second

### **3. Optical Flow**
- Combine depth map + camera motion → compute optical flow
- Predict where each pixel will be in the next frame
- Encode only the residual (prediction error)

### **4. 3D Reconstruction**
```python
# Project 3D point to 2D pixel
def project(point_3d, focal_length, principal_point):
    x_2d = (point_3d.x * focal_length.x / point_3d.z) + principal_point.x
    y_2d = (point_3d.y * focal_length.y / point_3d.z) + principal_point.y
    return (x_2d, y_2d)
```

### **5. Generative Synthesis**
The LCM model can be conditioned on:
- Current frame latent
- Depth map
- Camera velocity (linear + angular)
- Tracking confidence

→ Generate next frame with correct motion blur and perspective shifts

---

## 📊 Data Format in Training

When loading iPhone captures for training, the motion data will be:

```python
{
    'rgb': torch.Tensor,           # (T, 3, H, W) - RGB video
    'depth': torch.Tensor,         # (T, 1, H, W) - Depth maps
    'position': torch.Tensor,      # (T, 3) - Camera position
    'rotation': torch.Tensor,      # (T, 4) - Quaternion
    'velocity': torch.Tensor,      # (T, 3) - Linear velocity
    'angular_vel': torch.Tensor,   # (T, 3) - Angular velocity
    'focal_length': torch.Tensor,  # (2,) - fx, fy
    'principal_point': torch.Tensor, # (2,) - cx, cy
    'tracking_confidence': torch.Tensor  # (T,) - Confidence per frame
}
```

---

## 🎬 Best Practices for Recording

### **For Clean Motion Data:**
1. **Move Smoothly** - Avoid jerky camera movements
2. **Well-Lit Scenes** - ARKit tracks better with good lighting
3. **Textured Surfaces** - Featureless walls reduce tracking quality
4. **Not Too Fast** - Rapid motion can cause tracking loss
5. **Avoid Reflections** - Glass/mirrors confuse ARKit

### **For Interesting Training Data:**
1. **Vary Motion Types:**
   - Forward/backward translation
   - Left/right panning
   - Up/down tilting
   - Rotation in place
   - Combined motions

2. **Vary Scene Types:**
   - Indoor rooms (structured geometry)
   - Outdoor scenes (natural features)
   - Close-ups (depth variation)
   - Wide shots (motion parallax)

---

## 🚀 Next Steps

The motion data is now being captured in every frame. To use it for training:

1. **Extend the data loader** (`generative_codec/data/iphone_loader.py`) to parse motion metadata
2. **Add motion conditioning** to the LCM model architecture
3. **Implement optical flow prediction** using depth + motion
4. **Train with motion-aware loss** that penalizes temporal inconsistency

---

## 📖 References

- [ARKit Documentation](https://developer.apple.com/documentation/arkit)
- [Camera Intrinsics & Extrinsics](https://en.wikipedia.org/wiki/Camera_matrix)
- [Quaternions for Rotation](https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation)
- [Optical Flow](https://en.wikipedia.org/wiki/Optical_flow)

