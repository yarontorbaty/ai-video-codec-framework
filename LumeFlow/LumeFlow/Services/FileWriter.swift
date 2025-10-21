import Foundation
import AVFoundation

// MARK: - Mode 1: File Writer (Save to Local Storage)

class FileWriter {
    private var videoWriter: AVAssetWriter?
    private var videoInput: AVAssetWriterInput?
    private var depthInput: AVAssetWriterInput?
    private var metadataInput: AVAssetWriterInput?
    
    private var currentURL: URL?
    var currentSize: Int = 0
    
    func startNewRecording() {
        // Create unique filename
        let timestamp = Date().timeIntervalSince1970
        let filename = "lumaflow_\(Int(timestamp)).mov"
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(filename)
        currentURL = url
        
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
                AVVideoAverageBitRateKey: 10_000_000, // 10 Mbps
                AVVideoProfileLevelKey: AVVideoProfileLevelH264HighAutoLevel,
                AVVideoH264EntropyModeKey: AVVideoH264EntropyModeCABAC
            ]
        ]
        
        videoInput = AVAssetWriterInput(mediaType: .video, outputSettings: videoSettings)
        videoInput?.expectsMediaDataInRealTime = true
        
        if let videoInput = videoInput, writer.canAdd(videoInput) {
            writer.add(videoInput)
        }
        
        // Depth track (stored as auxiliary data)
        let depthSettings: [String: Any] = [
            AVVideoCodecKey: AVVideoCodecType.hevc,
            AVVideoWidthKey: 256,
            AVVideoHeightKey: 192
        ]
        
        depthInput = AVAssetWriterInput(mediaType: .depthData, outputSettings: depthSettings)
        depthInput?.expectsMediaDataInRealTime = true
        
        if let depthInput = depthInput, writer.canAdd(depthInput) {
            writer.add(depthInput)
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
            return
        }
        
        // Create sample buffer from pixel buffer
        let presentationTime = CMTime(seconds: frame.timestamp, preferredTimescale: 600)
        
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
            
            // Update size
            if let url = currentURL,
               let attributes = try? FileManager.default.attributesOfItem(atPath: url.path) {
                currentSize = attributes[.size] as? Int ?? 0
            }
        }
        
        // Write depth data
        if let depthBuffer = frame.depthBuffer,
           let depthInput = depthInput,
           depthInput.isReadyForMoreMediaData {
            
            var depthFormatDescription: CMFormatDescription?
            CMVideoFormatDescriptionCreateForImageBuffer(
                allocator: kCFAllocatorDefault,
                imageBuffer: depthBuffer,
                formatDescriptionOut: &depthFormatDescription
            )
            
            var depthSampleBuffer: CMSampleBuffer?
            CMSampleBufferCreateReadyWithImageBuffer(
                allocator: kCFAllocatorDefault,
                imageBuffer: depthBuffer,
                formatDescription: depthFormatDescription!,
                sampleTiming: &timingInfo,
                sampleBufferOut: &depthSampleBuffer
            )
            
            if let depthSampleBuffer = depthSampleBuffer {
                depthInput.append(depthSampleBuffer)
            }
        }
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
        // Save to Photos (requires PhotoKit)
        // For now, keep in temp directory
        completion(url)
    }
}

