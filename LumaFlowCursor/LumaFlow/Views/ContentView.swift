import SwiftUI

struct ContentView: View {
    @StateObject private var captureService = LiDARCaptureService()
    @State private var selectedMode: CaptureMode = .localSave
    @State private var isRecording = false
    @State private var showSettings = false
    
    var body: some View {
        ZStack {
            // Camera preview
            CameraPreviewView(captureService: captureService)
                .edgesIgnoringSafeArea(.all)
            
            // Depth visualization overlay
            if captureService.showDepthOverlay {
                DepthOverlayView(depthData: captureService.currentDepthData)
                    .opacity(0.6)
                    .edgesIgnoringSafeArea(.all)
            }
            
            // UI overlay
            VStack {
                // Top bar
                HStack {
                    Text("LumaFlow")
                        .font(.largeTitle)
                        .fontWeight(.bold)
                        .foregroundColor(.white)
                        .shadow(radius: 2)
                    
                    Spacer()
                    
                    Button(action: { showSettings.toggle() }) {
                        Image(systemName: "gearshape.fill")
                            .font(.title2)
                            .foregroundColor(.white)
                            .padding()
                            .background(Color.black.opacity(0.5))
                            .clipShape(Circle())
                    }
                }
                .padding()
                
                Spacer()
                
                // Mode selector
                if !isRecording {
                    ModeSelectionView(selectedMode: $selectedMode)
                        .padding()
                }
                
                // Recording controls
                VStack(spacing: 20) {
                    // Status indicators
                    if isRecording {
                        RecordingStatusView(
                            mode: selectedMode,
                            stats: captureService.recordingStats
                        )
                    }
                    
                    // Record button
                    Button(action: toggleRecording) {
                        ZStack {
                            Circle()
                                .fill(isRecording ? Color.red : Color.white)
                                .frame(width: 80, height: 80)
                            
                            Circle()
                                .stroke(Color.white, lineWidth: 4)
                                .frame(width: 90, height: 90)
                            
                            if isRecording {
                                RoundedRectangle(cornerRadius: 8)
                                    .fill(Color.white)
                                    .frame(width: 30, height: 30)
                            }
                        }
                    }
                    .padding(.bottom, 40)
                }
            }
        }
        .sheet(isPresented: $showSettings) {
            SettingsView(captureService: captureService)
        }
        .onAppear {
            captureService.setup()
        }
    }
    
    private func toggleRecording() {
        if isRecording {
            captureService.stopRecording()
            isRecording = false
        } else {
            captureService.startRecording(mode: selectedMode)
            isRecording = true
        }
    }
}

// MARK: - Supporting Views

struct ModeSelectionView: View {
    @Binding var selectedMode: CaptureMode
    
    var body: some View {
        VStack(spacing: 15) {
            ForEach(CaptureMode.allCases) { mode in
                Button(action: { selectedMode = mode }) {
                    HStack {
                        Image(systemName: mode.icon)
                            .font(.title2)
                            .frame(width: 40)
                        
                        VStack(alignment: .leading, spacing: 4) {
                            Text(mode.rawValue)
                                .font(.headline)
                            Text(mode.description)
                                .font(.caption)
                                .opacity(0.8)
                        }
                        
                        Spacer()
                        
                        if selectedMode == mode {
                            Image(systemName: "checkmark.circle.fill")
                                .foregroundColor(.green)
                        }
                    }
                    .padding()
                    .background(
                        RoundedRectangle(cornerRadius: 12)
                            .fill(selectedMode == mode ? Color.white.opacity(0.3) : Color.black.opacity(0.5))
                    )
                    .foregroundColor(.white)
                }
            }
        }
        .padding(.horizontal)
    }
}

struct RecordingStatusView: View {
    let mode: CaptureMode
    let stats: RecordingStats
    
    var body: some View {
        VStack(spacing: 12) {
            HStack(spacing: 20) {
                // Recording indicator
                HStack(spacing: 8) {
                    Circle()
                        .fill(Color.red)
                        .frame(width: 12, height: 12)
                    Text("REC")
                        .fontWeight(.bold)
                }
                
                // Duration
                Text(formatDuration(stats.duration))
                    .monospacedDigit()
                    .fontWeight(.medium)
                
                // Frame count
                Text("\(stats.frameCount) frames")
                    .font(.caption)
            }
            .padding(.horizontal, 20)
            .padding(.vertical, 10)
            .background(Color.black.opacity(0.7))
            .cornerRadius(20)
            
            // Mode-specific info
            HStack(spacing: 20) {
                VStack {
                    Text(formatSize(stats.totalBytes))
                        .font(.headline)
                    Text("Size")
                        .font(.caption2)
                }
                
                if mode == .cloudStream {
                    VStack {
                        Text("\(Int(stats.bitrateMbps))x")
                            .font(.headline)
                        Text("Bitrate")
                            .font(.caption2)
                    }
                    
                    VStack {
                        Image(systemName: stats.isStreaming ? "wifi" : "wifi.slash")
                            .font(.headline)
                        Text(stats.isStreaming ? "Live" : "Offline")
                            .font(.caption2)
                    }
                }
                
                VStack {
                    Text("\(Int(stats.fps)) fps")
                        .font(.headline)
                    Text("FPS")
                        .font(.caption2)
                }
            }
            .padding(.horizontal, 20)
            .padding(.vertical, 10)
            .background(Color.black.opacity(0.7))
            .cornerRadius(20)
            .foregroundColor(.white)
        }
    }
    
    private func formatDuration(_ seconds: Double) -> String {
        let mins = Int(seconds) / 60
        let secs = Int(seconds) % 60
        return String(format: "%02d:%02d", mins, secs)
    }
    
    private func formatSize(_ bytes: Int) -> String {
        let mb = Double(bytes) / 1_000_000
        if mb < 1 {
            return String(format: "%.1f KB", Double(bytes) / 1000)
        } else {
            return String(format: "%.1f MB", mb)
        }
    }
}

struct CameraPreviewView: UIViewRepresentable {
    let captureService: LiDARCaptureService
    
    func makeUIView(context: Context) -> UIView {
        return captureService.previewView
    }
    
    func updateUIView(_ uiView: UIView, context: Context) {}
}

struct DepthOverlayView: View {
    let depthData: Data?
    
    var body: some View {
        Color.clear // Depth visualization will be handled by Metal/Core Image
    }
}

struct SettingsView: View {
    @Environment(\.dismiss) var dismiss
    @ObservedObject var captureService: LiDARCaptureService
    
    var body: some View {
        NavigationView {
            Form {
                Section("Display") {
                    Toggle("Show Depth Overlay", isOn: $captureService.showDepthOverlay)
                    Toggle("Show Performance Stats", isOn: $captureService.showStats)
                }
                
                Section("Quality") {
                    Picker("Resolution", selection: $captureService.resolution) {
                        Text("1080p").tag(CaptureResolution.hd1080)
                        Text("4K").tag(CaptureResolution.uhd4k)
                    }
                    
                    Picker("Frame Rate", selection: $captureService.frameRate) {
                        Text("30 FPS").tag(30)
                        Text("60 FPS").tag(60)
                    }
                }
                
                Section("Streaming") {
                    TextField("Server URL", text: $captureService.serverURL)
                        .textContentType(.URL)
                        .autocapitalization(.none)
                }
                
                Section("About") {
                    HStack {
                        Text("Version")
                        Spacer()
                        Text("1.0.0")
                            .foregroundColor(.secondary)
                    }
                    
                    HStack {
                        Text("Codec")
                        Spacer()
                        Text("LumaFlow v1")
                            .foregroundColor(.secondary)
                    }
                }
            }
            .navigationTitle("Settings")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }
}

// MARK: - Preview

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}

