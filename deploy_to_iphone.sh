#!/bin/bash

echo "🎬 LumaFlow - iPhone Deployment Helper"
echo "======================================"
echo ""

# Check if Xcode is installed
if ! command -v xcodebuild &> /dev/null; then
    echo "❌ Xcode not found!"
    echo ""
    echo "Please install Xcode from the App Store:"
    echo "1. Open App Store"
    echo "2. Search for 'Xcode'"
    echo "3. Click Get/Install"
    echo ""
    exit 1
fi

echo "✅ Xcode found: $(xcodebuild -version | head -1)"
echo ""

# Navigate to project
cd /Users/yarontorbaty/Documents/Code/AiV1/LumaFlow

# Check if project exists
if [ ! -f "LumaFlow.xcodeproj/project.pbxproj" ]; then
    echo "❌ LumaFlow project not found!"
    echo "Make sure you're in the correct directory."
    exit 1
fi

echo "✅ LumaFlow project found"
echo ""

# Open in Xcode
echo "🚀 Opening project in Xcode..."
open LumaFlow.xcodeproj

echo ""
echo "📱 Next steps:"
echo ""
echo "1. Connect your iPhone 14 Pro Max to your Mac"
echo "2. Trust computer on iPhone (if prompted)"
echo "3. In Xcode:"
echo "   a. Select 'LumaFlow' target"
echo "   b. Go to 'Signing & Capabilities'"
echo "   c. Check 'Automatically manage signing'"
echo "   d. Select your Apple ID as Team"
echo "   e. Select your iPhone in device dropdown"
echo "   f. Click Play (▶️) or press Cmd+R"
echo ""
echo "4. On iPhone (first time only):"
echo "   • Settings → Privacy & Security → Developer Mode → ON"
echo "   • Restart iPhone"
echo ""
echo "5. Grant permissions when app launches:"
echo "   • Camera → Allow"
echo "   • Microphone → Allow"
echo "   • ARKit → Allow"
echo "   • Photos → Allow"
echo ""
echo "📖 Full guide: LumaFlow/DEPLOYMENT_GUIDE.md"
echo ""
echo "Happy filming! 🎥"

