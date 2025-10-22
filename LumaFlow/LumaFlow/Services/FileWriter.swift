import Foundation
import AVFoundation
import Photos

// MARK: - Mode 1: File Writer (Save to Local Storage)

class FileWriter {
    private var videoWriter: AVAssetWriter?
    private var videoInput: AVAssetWriterInput?
    private var depthInput: AVAssetWriterInput?
    private var depthAdaptor: AVAssetWriterInputPixelBufferAdaptor?
    private var metadataInput: AVAssetWriterInput?
    
    private var currentURL: URL?
    var currentSize: Int = 0
    
    private var firstFrameTimestamp: TimeInterval?
    private var frameCount: Int = 0
    
    func startNewRecording() {
        // Create unique filename
        let timestamp = Date().timeIntervalSince1970
        let filename = "lumaflow_\(Int(timestamp)).mov"
        
        // Save to Documents directory (accessible via Files app)
        let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let url = documentsPath.appendingPathComponent(filename)
        currentURL = url
        
        // Reset timing
        firstFrameTimestamp = nil
        frameCount = 0
        
        print("📁 Saving to Documents: \(url.path)")
        
        // Setup asset writer
        guard let writer = try? AVAssetWriter(outputURL: url, fileType: .mov) else {
            print("❌ Failed to create asset writer")
            return
        }
        
        videoWriter = writer
        
        // Video settings (HEVC for efficiency)
        let videoSettings: [String: Any] = [
            AVVideoCodecKey: AVVideoCodecType.hevc,
            AVVideoWidthKey: 1920,
            AVVideoHeightKey: 1080,
            AVVideoCompressionPropertiesKey: [
                AVVideoAverageBitRateKey: 10_000_000 // 10 Mbps
                // HEVC profile is auto-selected based on content
            ]
        ]
        
        videoInput = AVAssetWriterInput(mediaType: .video, outputSettings: videoSettings)
        videoInput?.expectsMediaDataInRealTime = true
        
        if let videoInput = videoInput, writer.canAdd(videoInput) {
            writer.add(videoInput)
        }
        
        // Depth track (grayscale HEVC video - smaller resolution)
        let depthSettings: [String: Any] = [
            AVVideoCodecKey: AVVideoCodecType.hevc,
            AVVideoWidthKey: 256,
            AVVideoHeightKey: 192,
            AVVideoCompressionPropertiesKey: [
                AVVideoAverageBitRateKey: 2_000_000 // 2 Mbps for depth
            ]
        ]
        
        depthInput = AVAssetWriterInput(mediaType: .video, outputSettings: depthSettings)
        depthInput?.expectsMediaDataInRealTime = true
        
        if let depthInput = depthInput, writer.canAdd(depthInput) {
            writer.add(depthInput)
            
            // Create pixel buffer adaptor for depth input
            let sourcePixelBufferAttributes: [String: Any] = [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
                kCVPixelBufferWidthKey as String: 256,
                kCVPixelBufferHeightKey as String: 192
            ]
            depthAdaptor = AVAssetWriterInputPixelBufferAdaptor(
                assetWriterInput: depthInput,
                sourcePixelBufferAttributes: sourcePixelBufferAttributes
            )
        }
        
        // Metadata track (camera transforms, timestamps)
        metadataInput = AVAssetWriterInput(mediaType: .metadata, outputSettings: nil)
        if let metadataInput = metadataInput, writer.canAdd(metadataInput) {
            writer.add(metadataInput)
        }
        
        writer.startWriting()
        writer.startSession(atSourceTime: .zero)
        
        print("✅ Started writing to: \(url.lastPathComponent)")
    }
    
    func writeFrame(_ frame: CapturedFrame) {
        guard let videoWriter = videoWriter,
              let videoInput = videoInput,
              videoInput.isReadyForMoreMediaData else {
            print("⚠️ Video input not ready or nil")
            return
        }
        
        // Check writer status
        if videoWriter.status == .failed {
            print("❌ Video writer failed: \(String(describing: videoWriter.error))")
            return
        }
        
        // Normalize timestamp to start from zero
        if firstFrameTimestamp == nil {
            firstFrameTimestamp = frame.timestamp
        }
        let relativeTimestamp = frame.timestamp - (firstFrameTimestamp ?? 0)
        frameCount += 1
        
        // Create sample buffer from pixel buffer
        let presentationTime = CMTime(seconds: relativeTimestamp, preferredTimescale: 600)
        
        var timingInfo = CMSampleTimingInfo(
            duration: CMTime(seconds: 1.0/30.0, preferredTimescale: 600),
            presentationTimeStamp: presentationTime,
            decodeTimeStamp: .invalid
        )
        
        var videoFormatDescription: CMFormatDescription?
        CMVideoFormatDescriptionCreateForImageBuffer(
            allocator: kCFAllocatorDefault,
            imageBuffer: frame.rgbBuffer,
            formatDescriptionOut: &videoFormatDescription
        )
        
        var sampleBuffer: CMSampleBuffer?
        CMSampleBufferCreateReadyWithImageBuffer(
            allocator: kCFAllocatorDefault,
            imageBuffer: frame.rgbBuffer,
            formatDescription: videoFormatDescription!,
            sampleTiming: &timingInfo,
            sampleBufferOut: &sampleBuffer
        )
        
        if let sampleBuffer = sampleBuffer {
            videoInput.append(sampleBuffer)
            print("✅ Wrote RGB frame")
            
            // Update size
            if let url = currentURL,
               let attributes = try? FileManager.default.attributesOfItem(atPath: url.path) {
                currentSize = attributes[.size] as? Int ?? 0
            }
        } else {
            print("❌ Failed to create RGB sample buffer")
        }
        
        // Write depth data as separate video track
        if let depthBuffer = frame.depthBuffer,
           let depthAdaptor = depthAdaptor,
           let depthInput = depthInput,
           depthInput.isReadyForMoreMediaData {
            
            // Convert Float32 depth to grayscale format for HEVC encoding
            let convertedDepth = convertDepthToGrayscale(depthBuffer)
            
            if let convertedDepth = convertedDepth {
                // Use same normalized timestamp
                let relativeTimestamp = frame.timestamp - (firstFrameTimestamp ?? 0)
                let presentationTime = CMTime(seconds: relativeTimestamp, preferredTimescale: 600)
                
                // Append converted depth pixel buffer
                let success = depthAdaptor.append(convertedDepth, withPresentationTime: presentationTime)
                if success {
                    print("✅ Wrote depth frame")
                } else {
                    print("❌ Failed to write depth frame")
                }
            } else {
                print("❌ Failed to convert depth format")
            }
        } else if frame.depthBuffer == nil {
            print("⚠️ No depth buffer in frame")
        }
    }
    
    private func convertDepthToGrayscale(_ depthBuffer: CVPixelBuffer) -> CVPixelBuffer? {
        // Lock the depth buffer for reading
        CVPixelBufferLockBaseAddress(depthBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(depthBuffer, .readOnly) }
        
        let width = CVPixelBufferGetWidth(depthBuffer)
        let height = CVPixelBufferGetHeight(depthBuffer)
        
        // Create output grayscale buffer
        var grayscaleBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            256,  // Match our encoder settings
            192,
            kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,  // Standard video format
            nil,
            &grayscaleBuffer
        )
        
        guard status == kCVReturnSuccess, let outputBuffer = grayscaleBuffer else {
            print("❌ Failed to create grayscale buffer")
            return nil
        }
        
        CVPixelBufferLockBaseAddress(outputBuffer, [])
        defer { CVPixelBufferUnlockBaseAddress(outputBuffer, []) }
        
        // Get depth data pointer (Float32)
        guard let depthData = CVPixelBufferGetBaseAddress(depthBuffer) else {
            return nil
        }
        let depthPointer = depthData.assumingMemoryBound(to: Float32.self)
        
        // Get output Y plane
        guard let outputData = CVPixelBufferGetBaseAddressOfPlane(outputBuffer, 0) else {
            return nil
        }
        let outputPointer = outputData.assumingMemoryBound(to: UInt8.self)
        let outputBytesPerRow = CVPixelBufferGetBytesPerRowOfPlane(outputBuffer, 0)
        
        // Convert depth values (0-5 meters) to grayscale (0-255)
        // with bilinear downsampling
        let scaleX = Float(width) / 256.0
        let scaleY = Float(height) / 192.0
        
        for y in 0..<192 {
            for x in 0..<256 {
                let srcX = Int(Float(x) * scaleX)
                let srcY = Int(Float(y) * scaleY)
                let srcIndex = srcY * width + srcX
                
                if srcIndex < width * height {
                    let depthMeters = depthPointer[srcIndex]
                    
                    // Normalize depth: 0m = 255 (white/close), 5m = 0 (black/far)
                    let normalized = max(0.0, min(1.0, depthMeters / 5.0))
                    let grayscale = UInt8((1.0 - normalized) * 255.0)
                    
                    let dstIndex = y * outputBytesPerRow + x
                    outputPointer[dstIndex] = grayscale
                }
            }
        }
        
        // CRITICAL FIX: Initialize UV plane to 128 (neutral gray, no color)
        // Without this, uninitialized memory creates green/purple artifacts!
        guard let uvData = CVPixelBufferGetBaseAddressOfPlane(outputBuffer, 1) else {
            return outputBuffer
        }
        let uvPointer = uvData.assumingMemoryBound(to: UInt8.self)
        let uvBytesPerRow = CVPixelBufferGetBytesPerRowOfPlane(outputBuffer, 1)
        let uvHeight = 192 / 2  // UV plane is half height in 420 format
        
        // Fill UV plane with 128 (neutral - no color tint)
        for y in 0..<uvHeight {
            for x in 0..<uvBytesPerRow {
                uvPointer[y * uvBytesPerRow + x] = 128
            }
        }
        
        return outputBuffer
    }
    
    func finalize(completion: @escaping (URL) -> Void) {
        videoInput?.markAsFinished()
        depthInput?.markAsFinished()
        metadataInput?.markAsFinished()
        
        videoWriter?.finishWriting { [weak self] in
            guard let url = self?.currentURL else { return }
            
            // Move to Photos library
            self?.saveToPhotosLibrary(url: url) { savedURL in
                completion(savedURL ?? url)
            }
        }
    }
    
    private func saveToPhotosLibrary(url: URL, completion: @escaping (URL?) -> Void) {
        // Check authorization status
        let status = PHPhotoLibrary.authorizationStatus(for: .addOnly)
        
        if status == .authorized || status == .limited {
            performSave(url: url, completion: completion)
        } else if status == .notDetermined {
            PHPhotoLibrary.requestAuthorization(for: .addOnly) { newStatus in
                if newStatus == .authorized || newStatus == .limited {
                    self.performSave(url: url, completion: completion)
                } else {
                    print("⚠️ Photo library access denied")
                    completion(url)
                }
            }
        } else {
            print("⚠️ Photo library access denied - check Settings")
            completion(url)
        }
    }
    
    private func performSave(url: URL, completion: @escaping (URL?) -> Void) {
        PHPhotoLibrary.shared().performChanges({
            PHAssetChangeRequest.creationRequestForAssetFromVideo(atFileURL: url)
        }) { success, error in
            if success {
                print("✅ Saved to Photos")
            } else if let error = error {
                print("❌ Failed to save to Photos: \(error.localizedDescription)")
            }
            completion(url)
        }
    }
}

