# LumaFlow Xcode Setup Guide

## ✅ Quick Checklist

### 1. Files Status
- ✅ **ContentView.swift** - Located in `LumaFlow/Views/`
- ✅ **LumaFlowApp.swift** - App entry point
- ✅ **CaptureMode.swift** - Capture modes enum
- ✅ **LiDARCaptureService.swift** - ARKit integration (syntax error fixed)
- ✅ **FileWriter.swift** - Local file saving
- ✅ **StreamingService.swift** - AWS streaming
- ✅ **OnDeviceEncoder.swift** - On-device encoding
- ✅ **Info.plist** - Permissions configured

### 2. Required Setup in Xcode

#### A. ARKit Framework (REQUIRED)
ARKit is **not** a capability - it's automatically available when you:
1. Import ARKit in your Swift files: `import ARKit`
2. Set the required device capabilities in Info.plist (already configured ✅)
3. Use a device with LiDAR (iPhone 12 Pro or later)

**No manual framework linking needed** - it's automatic in modern Xcode!

#### B. Background Modes (Recommended for streaming)
Open your project in Xcode and go to:
**Target → LumaFlow → Signing & Capabilities**

Add this capability by clicking **"+ Capability"**:
```
+ Capability → Background Modes
```
Then check:
- ☑️ Audio, AirPlay, and Picture in Picture
- ☑️ Background fetch

### 3. Device Requirements

The device must have:
- ✅ iPhone 12 Pro or later (LiDAR sensor required)
- ✅ iOS 17.0 or later
- ✅ Developer Mode enabled

### 4. Info.plist Permissions

Already configured ✅:
- `NSCameraUsageDescription` - Camera access
- `NSPhotoLibraryAddUsageDescription` - Photo library access
- `NSMicrophoneUsageDescription` - Microphone access
- `NSLocalNetworkUsageDescription` - Network access (just added)
- `UIRequiredDeviceCapabilities` - ARKit required

### 5. Build Settings to Verify

In **Build Settings**, verify:

1. **iOS Deployment Target**: 17.0 or later
2. **Swift Language Version**: Swift 5.9+
3. **Architectures**: arm64
4. **Build Active Architecture Only**: Yes (for Debug)

### 6. Signing & Team

1. Select your **Development Team** in Signing & Capabilities
2. Update **Bundle Identifier** if needed (e.g., `com.yourteam.LumaFlow`)
3. Ensure **Automatically manage signing** is checked

### 7. First Build Steps

1. **Connect your iPhone 12 Pro or later**
2. **Select your device** in Xcode toolbar (not Simulator)
3. **Enable Developer Mode** on iPhone:
   - Settings → Privacy & Security → Developer Mode → ON
   - Restart iPhone
4. **Trust your Mac** on the iPhone when prompted
5. **Build & Run** (Cmd+R)

### 8. Expected First Launch

On first launch, you'll see permission dialogs:
1. ✅ "LumaFlow would like to access the Camera" → **Allow**
2. ✅ "LumaFlow would like to access your Microphone" → **Allow**
3. ✅ "LumaFlow would like to access your Photos" → **Allow**
4. ✅ "LumaFlow would like to find and connect to devices on your local network" → **Allow** (for streaming)

### 9. Testing Modes

#### Mode 1: Save to File
- Should work out of the box
- Saves to Files app/Documents directory
- No network required

#### Mode 2: Stream to AWS
- Requires AWS server setup
- Update server URL in Settings
- Needs WiFi/5G connection

#### Mode 3: Encode on Device
- Experimental
- Uses placeholder encoder
- Should work but output is not optimized yet

### 10. Troubleshooting

#### "LiDAR not available"
- Verify device is iPhone 12 Pro or later
- Check ARKit capability is added
- Ensure camera permissions granted

#### "Build failed: Code signing"
- Select your Development Team
- Update Bundle Identifier to be unique
- Check provisioning profile

#### "App crashes on launch"
- Check all permissions in Info.plist are present
- Verify minimum iOS version (17.0)
- Check Console logs in Xcode

#### "Camera preview not showing"
- Grant camera permissions
- Check ARKit is supported on device
- Verify device has LiDAR sensor

### 11. Next Steps After Build

1. **Tap Record** → Should show permission dialogs
2. **Grant all permissions** → App should work
3. **Test "Save to File" mode** → Easiest to verify
4. **Check Files app** → Should see recordings
5. **Try all 3 capture modes** → Verify each works

## 📱 Quick Test Commands

### Check Xcode version:
```bash
xcodebuild -version
```

### List connected devices:
```bash
xcrun xctrace list devices
```

### Clean build:
```bash
cd /Users/yarontorbaty/Documents/Code/AiV1/LumaFlow
xcodebuild clean -project LumaFlow.xcodeproj -scheme LumaFlow
```

## 🎯 Summary

Your project is **ready to build**! Just:
1. Open `LumaFlow.xcodeproj` in Xcode
2. Select your Development Team in Signing & Capabilities
3. (Optional) Add **Background Modes** capability for streaming
4. Connect iPhone 12 Pro or later
5. Build & Run (Cmd+R)

**Note**: ARKit doesn't need to be manually added - it's automatically available!

The ContentView and all necessary files are in place. The only syntax error has been fixed.

## 🚀 You're All Set!

Once built, you'll have a working LiDAR capture app with 3 modes:
- 📁 Save to File - Works offline
- ☁️ Stream to AWS - Requires server setup
- 🔧 Encode on Device - Experimental codec

Happy capturing! 🎥✨

