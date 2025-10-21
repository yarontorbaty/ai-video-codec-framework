# LumaFlow - Deployment Guide for iPhone 14 Pro Max

## ✅ Your Device is Compatible!

iPhone 14 Pro Max specifications:
- ✅ LiDAR Scanner
- ✅ A16 Bionic chip
- ✅ iOS 16+
- ✅ ProMotion display (120Hz)
- ✅ 4K video recording

## Prerequisites

1. **Mac Computer** (for Xcode)
2. **Xcode 15.0+** (free from App Store)
3. **Apple ID** (free - no paid developer account needed for testing)
4. **USB-C to Lightning cable** (to connect iPhone)
5. **iPhone 14 Pro Max** with iOS 17.0+

## Step-by-Step Deployment

### Step 1: Install Xcode (if not already installed)

1. Open **App Store** on your Mac
2. Search for "Xcode"
3. Click **Get** / **Install** (it's ~15GB, may take 30-60 minutes)
4. Wait for installation to complete

### Step 2: Connect Your iPhone

1. Connect iPhone to Mac with USB cable
2. Unlock your iPhone
3. If prompted, tap **Trust This Computer**
4. Enter your iPhone passcode

### Step 3: Open the Project

1. Open Terminal
2. Navigate to the project:
   ```bash
   cd /Users/yarontorbaty/Documents/Code/AiV1
   open LumaFlow/LumaFlow.xcodeproj
   ```

Or:
1. Open Finder
2. Navigate to `/Users/yarontorbaty/Documents/Code/AiV1/LumaFlow`
3. Double-click `LumaFlow.xcodeproj`

### Step 4: Configure Code Signing

1. In Xcode, select **LumaFlow** in the project navigator (left sidebar)
2. Select the **LumaFlow** target
3. Go to **Signing & Capabilities** tab
4. Check **Automatically manage signing**
5. Select your **Team** (your Apple ID)
   - If you don't see your team, click "Add Account..." and sign in
6. Xcode will create a provisioning profile automatically

### Step 5: Select Your Device

1. In the toolbar at the top, click the device selector (next to the Play button)
2. Find your **iPhone 14 Pro Max** in the list
3. Select it

### Step 6: Enable Developer Mode on iPhone

**First time only:**
1. On your iPhone, go to **Settings** → **Privacy & Security**
2. Scroll down to **Developer Mode**
3. Toggle **Developer Mode** ON
4. Restart your iPhone when prompted
5. After restart, confirm you want to enable Developer Mode

### Step 7: Build and Run

1. In Xcode, click the **Play button** (▶️) or press **Cmd + R**
2. Xcode will:
   - Compile the code
   - Sign the app
   - Install it on your iPhone
   - Launch it automatically
3. First time: Wait 2-3 minutes for build

### Step 8: Grant Permissions

When the app launches on your iPhone:
1. **Camera Access** → Tap **Allow**
2. **Microphone Access** → Tap **Allow** (optional, for audio)
3. **ARKit Permissions** → Tap **Allow**
4. **Photo Library** → Tap **Allow** (for saving videos)

## 🎉 You're Ready!

The app should now be running on your iPhone 14 Pro Max!

## Troubleshooting

### Error: "Failed to create provisioning profile"

**Solution:**
1. Go to Xcode → Settings → Accounts
2. Click your Apple ID
3. Click **Download Manual Profiles**
4. Try building again

### Error: "Developer Mode Required"

**Solution:**
1. Settings → Privacy & Security → Developer Mode → ON
2. Restart iPhone
3. Try again

### Error: "Untrusted Developer"

**Solution:**
1. On iPhone: Settings → General → VPN & Device Management
2. Find your Apple ID / Developer App
3. Tap **Trust**
4. Launch app again

### Error: "Code signing certificate not found"

**Solution:**
1. Xcode → Settings → Accounts
2. Select your Apple ID
3. Click **Manage Certificates**
4. Click **+** → **Apple Development**
5. Try building again

### Build Errors in Code

If you see Swift compilation errors:
1. The code may have formatting issues from copy-paste
2. Try: Product → Clean Build Folder (Cmd + Shift + K)
3. Then rebuild (Cmd + R)

### iPhone Not Showing in Device List

**Solution:**
1. Unplug and replug iPhone
2. Trust computer again
3. Wait 30 seconds for Xcode to detect
4. If still not showing, restart Xcode

## First Use Tips

### Test Each Mode:

**Mode 1: Save to File** (Recommended First)
1. Select "Save to File"
2. Tap the red record button
3. Record 10-15 seconds of a scene with depth (objects at different distances)
4. Tap stop
5. Find video in Files app

**Mode 2: Stream to AWS** (Requires server setup)
1. Skip for now unless you have AWS server ready
2. See AWS setup guide in main README

**Mode 3: Encode on Device** (Experimental)
1. Select "Encode on Device"
2. Record 10-15 seconds
3. Note: Uses placeholder encoder, will be slow
4. See `.lfv` file in Files app

### Best Practices:

- 📱 **Hold iPhone steady** for best LiDAR capture
- 💡 **Good lighting** improves quality
- 🎯 **Objects 1-5 meters away** for optimal depth
- 🔋 **Keep iPhone charged** (encoding uses battery)
- 📊 **Check stats** to monitor FPS and file size

## Updating the App

To make changes and redeploy:
1. Edit code in Xcode
2. Press **Cmd + R** to rebuild and run
3. App updates automatically on device

## Uninstalling

To remove from iPhone:
1. Long-press the LumaFlow app icon
2. Tap **Remove App** → **Delete App**

## Next Steps

After testing:
1. Capture 5-10 clips of different scenes
2. Transfer to Mac (AirDrop or Files app)
3. Ready for training phase!

## Questions?

Common questions:
- **Do I need a paid developer account?** No! Free Apple ID works for testing
- **How long does build take?** 2-3 minutes first time, 30 seconds after
- **Can I use simulator?** No, LiDAR requires real device
- **Will it work on iPhone 13 Pro?** Yes! Any iPhone 12 Pro or later
- **Battery impact?** Moderate (~20%/hour while recording)

Enjoy capturing with LumaFlow! 🎬📱
