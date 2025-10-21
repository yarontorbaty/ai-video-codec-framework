import Foundation
import ARKit
import AVFoundation
import Combine
import SwiftUI

enum CaptureResolution {
    case hd1080
    case uhd4k
}

struct RecordingStats {
    var duration: Double = 0
    var frameCount: Int = 0
    var totalBytes: Int = 0
    var fps: Double = 0
    var bitrateMbps: Double = 0
    var isStreaming: Bool = false
}

class LiDARCaptureService: NSObject, ObservableObject {
    // MARK: - Published Properties
    @Published var showDepthOverlay = false
    @Published var showStats = false
    @Published var resolution: CaptureResolution = .hd1080
    @Published var frameRate: Int = 30
    @Published var serverURL: String = "rtmp://your-server.com:1935/live"
    @Published var recordingStats = RecordingStats()
    @Published var currentDepthData: Data?
    
    // MARK: - AR Session
    private var arSession: ARSession!
    private var arConfiguration: ARWorldTrackingConfiguration!
    
    // MARK: - Recording
    private var isRecording = false
    private var currentMode: CaptureMode?
    private var recordingStartTime: Date?
    private var frameBuffer: [CapturedFrame] = []
    
    // MARK: - Services
    private var streamingService: StreamingService?
    private var onDeviceEncoder: OnDeviceEncoder?
    private var fileWriter: FileWriter?
    
    // MARK: - UI
    let previewView: UIView
    private var previewLayer: CALayer?
    
    // MARK: - Performance
    private var lastFrameTime: TimeInterval = 0
    private var fpsCounter = 0
    private var fpsTimer: Timer?
    
    override init() {
        self.previewView = UIView()
        super.init()
        
        setupAR()
        setupFPSTimer()
    }
    
    // MARK: - Setup
    
    func setup() {
        // Request permissions
        ARSession.requestAuthorizationAnd Video Access()
    }
    
    private func setupAR() {
        arSession = ARSession()
        arSession.delegate = self
        
        arConfiguration = ARWorldTrackingConfiguration()
        
        // Enable scene depth (LiDAR)
        if ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) {
            arConfiguration.frameSemantics = [.sceneDepth, .smoothedSceneDepth]
        }
        
        // Set video format based on resolution
        updateVideoFormat()
    }
    
    private func updateVideoFormat() {
        let formats = ARWorldTrackingConfiguration.supportedVideoFormats
        
        // Filter by resolution and frame rate
        let targetWidth: Int
        switch resolution {
        case .hd1080:
            targetWidth = 1920
        case .uhd4k:
            targetWidth = 3840
        }
        
        if let format = formats.first(where: {
            $0.imageResolution.width == targetWidth &&
            $0.framesPerSecond == frameRate
        }) {
            arConfiguration.videoFormat = format
        }
    }
    
    private func setupFPSTimer() {
        fpsTimer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [weak self] _ in
            guard let self = self else { return }
            DispatchQueue.main.async {
                self.recordingStats.fps = Double(self.fpsCounter)
                self.fpsCounter = 0
            }
        }
    }
    
    // MARK: - Recording Control
    
    func startRecording(mode: CaptureMode) {
        guard !isRecording else { return }
        
        isRecording = true
        currentMode = mode
        recordingStartTime = Date()
        frameBuffer.removeAll()
        
        // Start AR session
        arSession.run(arConfiguration)
        
        // Initialize appropriate service
        switch mode {
        case .localSave:
            fileWriter = FileWriter()
            fileWriter?.startNewRecording()
            
        case .cloudStream:
            streamingService = StreamingService(serverURL: serverURL)
            streamingService?.connect { [weak self] success in
                DispatchQueue.main.async {
                    self?.recordingStats.isStreaming = success
                }
            }
            
        case .onDevice:
            onDeviceEncoder = OnDeviceEncoder()
            onDeviceEncoder?.startEncoding()
        }
        
        print("✅ Started recording in \(mode.rawValue) mode")
    }
    
    func stopRecording() {
        guard isRecording else { return }
        
        isRecording = false
        arSession.pause()
        
        // Finalize based on mode
        switch currentMode {
        case .localSave:
            fileWriter?.finalize { [weak self] url in
                print("✅ Saved to: \(url)")
                self?.showSaveConfirmation(url: url)
            }
            
        case .cloudStream:
            streamingService?.disconnect()
            print("✅ Stream ended")
            
        case .onDevice:
            onDeviceEncoder?.finalize { [weak self] url in
                print("✅ Encoded to: \(url)")
                self?.showSaveConfirmation(url: url)
            }
            
        case .none:
            break
        }
        
        currentMode = nil
        recordingStartTime = nil
    }
    
    private func showSaveConfirmation(url: URL) {
        // Show save confirmation UI
    }
    
    // MARK: - Frame Processing
    
    private func processFrame(_ frame: ARFrame) {
        guard isRecording else { return }
        
        fpsCounter += 1
        
        // Update stats
        if let startTime = recordingStartTime {
            recordingStats.duration = Date().timeIntervalSince(startTime)
            recordingStats.frameCount += 1
        }
        
        // Extract RGB image
        let pixelBuffer = frame.capturedImage
        
        // Extract depth data
        var depthData: CVPixelBuffer?
        if let sceneDepth = frame.sceneDepth {
            depthData = sceneDepth.depthMap
            currentDepthData = convertDepthToData(sceneDepth.depthMap)
        }
        
        // Create captured frame
        let capturedFrame = CapturedFrame(
            timestamp: frame.timestamp,
            rgbBuffer: pixelBuffer,
            depthBuffer: depthData,
            cameraTransform: frame.camera.transform
        )
        
        // Process based on mode
        switch currentMode {
        case .localSave:
            fileWriter?.writeFrame(capturedFrame)
            recordingStats.totalBytes = fileWriter?.currentSize ?? 0
            
        case .cloudStream:
            streamingService?.sendFrame(capturedFrame) { [weak self] bytesSent in
                DispatchQueue.main.async {
                    self?.recordingStats.totalBytes += bytesSent
                    if let duration = self?.recordingStats.duration, duration > 0 {
                        self?.recordingStats.bitrateMbps = Double(self?.recordingStats.totalBytes ?? 0) * 8 / duration / 1_000_000
                    }
                }
            }
            
        case .onDevice:
            onDeviceEncoder?.encodeFrame(capturedFrame) { [weak self] compressedSize in
                DispatchQueue.main.async {
                    self?.recordingStats.totalBytes += compressedSize
                }
            }
            
        case .none:
            break
        }
    }
    
    private func convertDepthToData(_ depthMap: CVPixelBuffer) -> Data? {
        // Convert depth buffer to Data for visualization
        CVPixelBufferLockBaseAddress(depthMap, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(depthMap, .readOnly) }
        
        let width = CVPixelBufferGetWidth(depthMap)
        let height = CVPixelBufferGetHeight(depthMap)
        guard let baseAddress = CVPixelBufferGetBaseAddress(depthMap) else { return nil }
        
        let bufferSize = width * height * MemoryLayout<Float32>.size
        return Data(bytes: baseAddress, count: bufferSize)
    }
}

// MARK: - ARSessionDelegate

extension LiDARCaptureService: ARSessionDelegate {
    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        processFrame(frame)
        
        // Update preview (simplified - would use Metal for better performance)
        DispatchQueue.main.async { [weak self] in
            // Update preview layer with current frame
        }
    }
    
    func session(_ session: ARSession, didFailWithError error: Error) {
        print("❌ AR Session failed: \(error.localizedDescription)")
        stopRecording()
    }
}

// MARK: - Captured Frame

struct CapturedFrame {
    let timestamp: TimeInterval
    let rgbBuffer: CVPixelBuffer
    let depthBuffer: CVPixelBuffer?
    let cameraTransform: simd_float4x4
}

