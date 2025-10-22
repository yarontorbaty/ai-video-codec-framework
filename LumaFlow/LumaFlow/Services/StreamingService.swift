import Foundation
import AVFoundation
import VideoToolbox
import Network

// MARK: - Mode 2: Streaming Service (HEVC + SRT to AWS)

class StreamingService {
    private let serverURL: String
    private var connection: NWConnection?
    private var isConnected = false
    
    // HEVC encoder
    private var videoEncoder: AVAssetWriter?
    private var compressionSession: VTCompressionSession?
    
    // Stats
    private var totalBytesSent: Int = 0
    private var frameQueue: [(CapturedFrame, (Int) -> Void)] = []
    
    init(serverURL: String) {
        self.serverURL = serverURL
        setupEncoder()
    }
    
    // MARK: - Connection
    
    func connect(completion: @escaping (Bool) -> Void) {
        // Parse server URL (rtmp://server:port/path)
        guard let url = URL(string: serverURL),
              let host = url.host,
              let port = url.port else {
            completion(false)
            return
        }
        
        // Create TCP connection for SRT
        let tcpPort = NWEndpoint.Port(integerLiteral: UInt16(port))
        let endpoint = NWEndpoint.hostPort(host: NWEndpoint.Host(host), port: tcpPort)
        
        connection = NWConnection(to: endpoint, using: .tcp)
        
        connection?.stateUpdateHandler = { [weak self] state in
            switch state {
            case .ready:
                print("✅ Connected to streaming server")
                self?.isConnected = true
                completion(true)
                
            case .failed(let error):
                print("❌ Connection failed: \(error)")
                self?.isConnected = false
                completion(false)
                
            default:
                break
            }
        }
        
        connection?.start(queue: .global())
    }
    
    func disconnect() {
        connection?.cancel()
        isConnected = false
        print("✅ Disconnected from streaming server")
    }
    
    // MARK: - Encoding
    
    private func setupEncoder() {
        var session: VTCompressionSession?
        
        let status = VTCompressionSessionCreate(
            allocator: kCFAllocatorDefault,
            width: 1920,
            height: 1080,
            codecType: kCMVideoCodecType_HEVC,
            encoderSpecification: nil,
            imageBufferAttributes: nil,
            compressedDataAllocator: nil,
            outputCallback: nil,
            refcon: nil,
            compressionSessionOut: &session
        )
        
        guard status == noErr, let session = session else {
            print("❌ Failed to create compression session")
            return
        }
        
        // Set encoding properties
        VTSessionSetProperty(session, key: kVTCompressionPropertyKey_RealTime, value: kCFBooleanTrue)
        VTSessionSetProperty(session, key: kVTCompressionPropertyKey_ProfileLevel, value: kVTProfileLevel_HEVC_Main_AutoLevel)
        VTSessionSetProperty(session, key: kVTCompressionPropertyKey_AverageBitRate, value: 5_000_000 as CFNumber) // 5 Mbps
        VTSessionSetProperty(session, key: kVTCompressionPropertyKey_ExpectedFrameRate, value: 30 as CFNumber)
        
        VTCompressionSessionPrepareToEncodeFrames(session)
        
        compressionSession = session
        print("✅ Encoder ready (HEVC @ 5 Mbps)")
    }
    
    // MARK: - Frame Transmission
    
    func sendFrame(_ frame: CapturedFrame, completion: @escaping (Int) -> Void) {
        guard isConnected else {
            completion(0)
            return
        }
        
        frameQueue.append((frame, completion))
        processNextFrame()
    }
    
    private func processNextFrame() {
        guard !frameQueue.isEmpty,
              let compressionSession = compressionSession else {
            return
        }
        
        let (frame, completion) = frameQueue.removeFirst()
        
        // Encode frame
        let presentationTime = CMTime(seconds: frame.timestamp, preferredTimescale: 600)
        
        let encodeCallback: VTCompressionOutputCallback = { refcon, sourceFrameRefcon, status, infoFlags, sampleBuffer in
            guard status == noErr, let sampleBuffer = sampleBuffer else {
                return
            }
            
            // Get the StreamingService instance from context
            let context = Unmanaged<StreamingServiceContext>.fromOpaque(refcon!).takeUnretainedValue()
            
            // Get compressed data
            guard let dataBuffer = CMSampleBufferGetDataBuffer(sampleBuffer) else {
                context.completion(0)
                context.service.processNextFrame()
                return
            }
            
            var length: Int = 0
            var dataPointer: UnsafeMutablePointer<Int8>?
            CMBlockBufferGetDataPointer(dataBuffer, atOffset: 0, lengthAtOffsetOut: nil, totalLengthOut: &length, dataPointerOut: &dataPointer)
            
            guard let dataPointer = dataPointer else {
                context.completion(0)
                context.service.processNextFrame()
                return
            }
            
            // Send over network
            let data = Data(bytes: dataPointer, count: length)
            context.service.sendData(data) { bytesSent in
                context.completion(bytesSent)
                context.service.processNextFrame()
            }
            
            // Also send depth data (compressed)
            if let depthBuffer = context.frame.depthBuffer {
                context.service.sendDepthData(depthBuffer)
            }
        }
        
        let context = StreamingServiceContext(service: self, frame: frame, completion: completion)
        let contextPtr = Unmanaged.passRetained(context).toOpaque()
        
        VTCompressionSessionEncodeFrame(
            compressionSession,
            imageBuffer: frame.rgbBuffer,
            presentationTimeStamp: presentationTime,
            duration: .invalid,
            frameProperties: nil,
            sourceFrameRefcon: contextPtr,
            infoFlagsOut: nil
        )
    }
    
    private func sendData(_ data: Data, completion: @escaping (Int) -> Void) {
        connection?.send(content: data, completion: .contentProcessed { [weak self] error in
            if let error = error {
                print("❌ Send error: \(error)")
                completion(0)
            } else {
                self?.totalBytesSent += data.count
                completion(data.count)
            }
        })
    }
    
    private func sendDepthData(_ depthBuffer: CVPixelBuffer) {
        // Compress depth data (simple run-length encoding or zlib)
        CVPixelBufferLockBaseAddress(depthBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(depthBuffer, .readOnly) }
        
        guard let baseAddress = CVPixelBufferGetBaseAddress(depthBuffer) else { return }
        
        let width = CVPixelBufferGetWidth(depthBuffer)
        let height = CVPixelBufferGetHeight(depthBuffer)
        let bufferSize = width * height * MemoryLayout<Float32>.size
        
        let depthData = Data(bytes: baseAddress, count: bufferSize)
        
        // Compress depth data
        if let compressed = try? (depthData as NSData).compressed(using: .lzfse) as Data {
            sendData(compressed) { _ in }
        }
    }
}

// Helper context for passing to VideoToolbox callback
class StreamingServiceContext {
    let service: StreamingService
    let frame: CapturedFrame
    let completion: (Int) -> Void
    
    init(service: StreamingService, frame: CapturedFrame, completion: @escaping (Int) -> Void) {
        self.service = service
        self.frame = frame
        self.completion = completion
    }
}

