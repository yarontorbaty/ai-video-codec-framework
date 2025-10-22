import Foundation
import CoreML
import Accelerate

// MARK: - Mode 3: On-Device Encoder (LumaFlow Codec)

class OnDeviceEncoder {
    private var encodedFrames: [EncodedFrame] = []
    private var outputURL: URL?
    private var isEncoding = false
    
    // Simplified LCM-based encoder (placeholder for full implementation)
    // In production, this would use CoreML models
    
    struct EncodedFrame {
        let timestamp: TimeInterval
        let iFrame: Bool
        let latentData: Data
        let depthData: Data
        let motionVectors: Data?
        let compressedSize: Int
    }
    
    func startEncoding() {
        isEncoding = true
        encodedFrames.removeAll()
        
        let timestamp = Date().timeIntervalSince1970
        let filename = "lumaflow_encoded_\(Int(timestamp)).lfv" // LumaFlow Video format
        outputURL = FileManager.default.temporaryDirectory.appendingPathComponent(filename)
        
        print("✅ On-device encoding started")
    }
    
    func encodeFrame(_ frame: CapturedFrame, completion: @escaping (Int) -> Void) {
        guard isEncoding else {
            completion(0)
            return
        }
        
        // Determine if this should be an I-frame (every 30 frames)
        let frameNumber = encodedFrames.count
        let isIFrame = frameNumber % 30 == 0
        
        if isIFrame {
            encodeIFrame(frame, completion: completion)
        } else {
            encodePFrame(frame, completion: completion)
        }
    }
    
    // MARK: - I-Frame Encoding
    
    private func encodeIFrame(_ frame: CapturedFrame, completion: @escaping (Int) -> Void) {
        // 1. Downscale RGB to 64x64
        let downscaled = downscaleImage(frame.rgbBuffer, to: CGSize(width: 64, height: 64))
        
        // 2. Encode with LCM (placeholder - in production, use CoreML)
        let latentData = encodeLCMLatent(downscaled)
        
        // 3. Compress depth data
        let depthData = compressDepth(frame.depthBuffer)
        
        // 4. Store frame
        let encodedFrame = EncodedFrame(
            timestamp: frame.timestamp,
            iFrame: true,
            latentData: latentData,
            depthData: depthData,
            motionVectors: nil,
            compressedSize: latentData.count + depthData.count
        )
        
        encodedFrames.append(encodedFrame)
        
        print("🖼️ I-frame encoded: \(encodedFrame.compressedSize) bytes")
        completion(encodedFrame.compressedSize)
    }
    
    // MARK: - P-Frame Encoding
    
    private func encodePFrame(_ frame: CapturedFrame, completion: @escaping (Int) -> Void) {
        guard let previousFrame = encodedFrames.last else {
            // No previous frame, encode as I-frame
            encodeIFrame(frame, completion: completion)
            return
        }
        
        // 1. Estimate motion vectors (simplified)
        let motionVectors = estimateMotion(current: frame.rgbBuffer, previous: frame.rgbBuffer)
        
        // 2. Compute residual
        let residual = computeResidual(current: frame.rgbBuffer, previous: frame.rgbBuffer, motion: motionVectors)
        
        // 3. Compress residual
        let compressedResidual = compressResidual(residual)
        
        // 4. Compress depth delta
        let depthDelta = compressDepthDelta(frame.depthBuffer)
        
        // 5. Store frame
        let encodedFrame = EncodedFrame(
            timestamp: frame.timestamp,
            iFrame: false,
            latentData: compressedResidual,
            depthData: depthDelta,
            motionVectors: motionVectors,
            compressedSize: compressedResidual.count + depthDelta.count + motionVectors.count
        )
        
        encodedFrames.append(encodedFrame)
        
        print("📹 P-frame encoded: \(encodedFrame.compressedSize) bytes")
        completion(encodedFrame.compressedSize)
    }
    
    // MARK: - Helper Methods (Placeholders)
    
    private func downscaleImage(_ pixelBuffer: CVPixelBuffer, to size: CGSize) -> CVPixelBuffer {
        // Use Accelerate framework for fast downscaling
        // Placeholder: return original buffer
        return pixelBuffer
    }
    
    private func encodeLCMLatent(_ pixelBuffer: CVPixelBuffer) -> Data {
        // In production: Use CoreML LCM encoder
        // Placeholder: Return compressed representation (~4KB)
        let placeholderSize = 4 * 1024
        return Data(count: placeholderSize)
    }
    
    private func compressDepth(_ depthBuffer: CVPixelBuffer?) -> Data {
        guard let depthBuffer = depthBuffer else {
            return Data()
        }
        
        CVPixelBufferLockBaseAddress(depthBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(depthBuffer, .readOnly) }
        
        guard let baseAddress = CVPixelBufferGetBaseAddress(depthBuffer) else {
            return Data()
        }
        
        let width = CVPixelBufferGetWidth(depthBuffer)
        let height = CVPixelBufferGetHeight(depthBuffer)
        let bufferSize = width * height * MemoryLayout<Float32>.size
        
        let depthData = Data(bytes: baseAddress, count: bufferSize)
        
        // Compress with LZFSE
        if let compressed = try? (depthData as NSData).compressed(using: .lzfse) as Data {
            return compressed
        }
        
        return depthData
    }
    
    private func estimateMotion(current: CVPixelBuffer, previous: CVPixelBuffer) -> Data {
        // Simplified motion estimation
        // In production: Use optical flow or block matching
        // Placeholder: ~1KB of motion vectors
        return Data(count: 1024)
    }
    
    private func computeResidual(current: CVPixelBuffer, previous: CVPixelBuffer, motion: Data) -> CVPixelBuffer {
        // Compute difference after motion compensation
        // Placeholder: return current frame
        return current
    }
    
    private func compressResidual(_ residual: CVPixelBuffer) -> Data {
        // Compress residual frame
        // Placeholder: ~2KB
        return Data(count: 2 * 1024)
    }
    
    private func compressDepthDelta(_ depthBuffer: CVPixelBuffer?) -> Data {
        // Compress depth delta from previous frame
        // Placeholder: ~500 bytes
        return Data(count: 500)
    }
    
    // MARK: - Finalization
    
    func finalize(completion: @escaping (URL) -> Void) {
        guard let outputURL = outputURL else {
            return
        }
        
        isEncoding = false
        
        // Write all encoded frames to file
        var fileData = Data()
        
        // Header
        let header = LumaFlowHeader(
            version: 1,
            frameCount: UInt32(encodedFrames.count),
            width: 1920,
            height: 1080,
            frameRate: 30
        )
        
        fileData.append(header.toData())
        
        // Frames
        for frame in encodedFrames {
            fileData.append(frame.toData())
        }
        
        // Write to file
        try? fileData.write(to: outputURL)
        
        print("✅ Encoded \(encodedFrames.count) frames to: \(outputURL.lastPathComponent)")
        print("📊 Total size: \(fileData.count / 1024) KB")
        print("📊 Avg compression: \((Double(encodedFrames.count) * 1920 * 1080 * 3) / Double(fileData.count))x")
        
        completion(outputURL)
    }
}

// MARK: - Data Structures

struct LumaFlowHeader {
    let version: UInt32
    let frameCount: UInt32
    let width: UInt32
    let height: UInt32
    let frameRate: UInt32
    
    func toData() -> Data {
        var data = Data()
        data.append(contentsOf: withUnsafeBytes(of: version) { Data($0) })
        data.append(contentsOf: withUnsafeBytes(of: frameCount) { Data($0) })
        data.append(contentsOf: withUnsafeBytes(of: width) { Data($0) })
        data.append(contentsOf: withUnsafeBytes(of: height) { Data($0) })
        data.append(contentsOf: withUnsafeBytes(of: frameRate) { Data($0) })
        return data
    }
}

extension OnDeviceEncoder.EncodedFrame {
    func toData() -> Data {
        var data = Data()
        
        // Frame metadata
        data.append(contentsOf: withUnsafeBytes(of: timestamp) { Data($0) })
        data.append(contentsOf: withUnsafeBytes(of: iFrame) { Data($0) })
        
        // Latent data (length + data)
        data.append(contentsOf: withUnsafeBytes(of: UInt32(latentData.count)) { Data($0) })
        data.append(latentData)
        
        // Depth data (length + data)
        data.append(contentsOf: withUnsafeBytes(of: UInt32(depthData.count)) { Data($0) })
        data.append(depthData)
        
        // Motion vectors (length + data)
        if let motionVectors = motionVectors {
            data.append(contentsOf: withUnsafeBytes(of: UInt32(motionVectors.count)) { Data($0) })
            data.append(motionVectors)
        } else {
            data.append(contentsOf: withUnsafeBytes(of: UInt32(0)) { Data($0) })
        }
        
        return data
    }
}

