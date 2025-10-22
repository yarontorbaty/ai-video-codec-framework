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
    
    // Motion tracking for velocity calculation
    private var lastCameraPosition: simd_float3?
    private var lastCameraRotation: simd_quatf?
    private var lastFrameTimestamp: TimeInterval?
    
    // MARK: - Services
    private var streamingService: StreamingService?
    private var onDeviceEncoder: OnDeviceEncoder?
    private var fileWriter: FileWriter?
    
    // MARK: - UI
    let previewView: UIView
    private var previewLayer: AVCaptureVideoPreviewLayer?
    private var imageView: UIImageView?
    
    // MARK: - Performance
    private var lastFrameTime: TimeInterval = 0
    private var fpsCounter = 0
    private var fpsTimer: Timer?
    
    override init() {
        self.previewView = UIView()
        super.init()
        
        // Setup image view for camera preview
        let imageView = UIImageView(frame: previewView.bounds)
        imageView.contentMode = .scaleAspectFill
        imageView.autoresizingMask = [.flexibleWidth, .flexibleHeight]
        
        // Fix camera orientation (ARKit captures in landscape right)
        imageView.transform = CGAffineTransform(rotationAngle: .pi / 2)
        
        previewView.addSubview(imageView)
        self.imageView = imageView
        
        setupAR()
        setupFPSTimer()
    }
    
    // MARK: - Setup
    
    func setup() {
        // Request permissions
        AVCaptureDevice.requestAccess(for: .video) { granted in
            print("Camera access: \(granted)")
        }
        
        // Start AR session for preview (even when not recording)
        arSession.run(arConfiguration)
        print("✅ ARKit session started")
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
            Int($0.imageResolution.width) == targetWidth &&
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
        
        // Extract camera transform and pose
        let transform = frame.camera.transform
        let position = simd_float3(transform.columns.3.x, transform.columns.3.y, transform.columns.3.z)
        let rotation = simd_quatf(transform)
        
        // Calculate velocities from frame-to-frame motion
        var linearVelocity: simd_float3? = nil
        var angularVelocity: simd_float3? = nil
        
        if let lastPos = lastCameraPosition,
           let lastRot = lastCameraRotation,
           let lastTime = lastFrameTimestamp {
            
            let deltaTime = Float(frame.timestamp - lastTime)
            if deltaTime > 0 {
                // Linear velocity (m/s)
                linearVelocity = (position - lastPos) / deltaTime
                
                // Angular velocity (rad/s) - calculate from quaternion difference
                let deltaRotation = rotation * lastRot.inverse
                let angle = 2.0 * acos(min(1.0, abs(deltaRotation.vector.w)))
                let axis = normalize(simd_float3(deltaRotation.vector.x, deltaRotation.vector.y, deltaRotation.vector.z))
                angularVelocity = axis * (angle / deltaTime)
            }
        }
        
        // Store for next frame
        lastCameraPosition = position
        lastCameraRotation = rotation
        lastFrameTimestamp = frame.timestamp
        
        // Extract camera intrinsics
        let intrinsics = frame.camera.intrinsics
        let focalLength = CGPoint(x: CGFloat(intrinsics[0, 0]), y: CGFloat(intrinsics[1, 1]))
        let principalPoint = CGPoint(x: CGFloat(intrinsics[2, 0]), y: CGFloat(intrinsics[2, 1]))
        let imageResolution = frame.camera.imageResolution
        
        // Extract tracking state
        let trackingStateString: String
        switch frame.camera.trackingState {
        case .normal:
            trackingStateString = "normal"
        case .limited:
            trackingStateString = "limited"
        case .notAvailable:
            trackingStateString = "notAvailable"
        }
        
        // Calculate tracking confidence (simplified - based on tracking state)
        let confidence: Float
        switch frame.camera.trackingState {
        case .normal:
            confidence = 1.0
        case .limited:
            confidence = 0.5
        case .notAvailable:
            confidence = 0.0
        }
        
        // Create captured frame with all motion data
        let capturedFrame = CapturedFrame(
            timestamp: frame.timestamp,
            rgbBuffer: pixelBuffer,
            depthBuffer: depthData,
            cameraTransform: transform,
            cameraPosition: position,
            cameraRotation: rotation,
            linearVelocity: linearVelocity,
            angularVelocity: angularVelocity,
            focalLength: focalLength,
            principalPoint: principalPoint,
            imageResolution: imageResolution,
            trackingState: trackingStateString,
            trackingConfidence: confidence
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
            guard let self = self, let imageView = self.imageView else { return }
            let ciImage = CIImage(cvPixelBuffer: frame.capturedImage)
            let context = CIContext()
            if let cgImage = context.createCGImage(ciImage, from: ciImage.extent) {
                imageView.image = UIImage(cgImage: cgImage)
            }
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
    
    // Camera pose (position + orientation in 3D space)
    let cameraTransform: simd_float4x4
    
    // Motion data (for temporal prediction)
    let cameraPosition: simd_float3       // (x, y, z) in meters
    let cameraRotation: simd_quatf        // Orientation as quaternion
    let linearVelocity: simd_float3?      // m/s in each axis
    let angularVelocity: simd_float3?     // rad/s around each axis
    
    // Camera intrinsics (for 3D reconstruction)
    let focalLength: CGPoint              // (fx, fy) in pixels
    let principalPoint: CGPoint           // (cx, cy) in pixels
    let imageResolution: CGSize           // Native camera resolution
    
    // Tracking quality
    let trackingState: String             // "normal", "limited", "notAvailable"
    let trackingConfidence: Float         // 0.0 to 1.0
}

