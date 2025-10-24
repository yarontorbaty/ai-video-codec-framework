# NCI Codec - Neurally Compressed Image Format

A neural codec for high-quality image compression using deep learning.

## Quick Start

### Installation

```bash
pip install torch torchvision opencv-python numpy scikit-image pillow
```

### Usage

**Encode an image:**
```bash
python nci_codec.py encode input.jpg output.nci
```

**Decode an image:**
```bash
python nci_codec.py decode input.nci output.jpg
```

**Encode with metadata preservation:**
```bash
python nci_codec.py encode photo.jpg photo.nci --preserve-metadata
```

**Decode and add metrics to EXIF:**
```bash
python nci_codec.py decode photo.nci photo.jpg --add-metrics
```

## Features

- ✅ **High compression:** 88-92% smaller than JPEG at similar quality
- ✅ **Near-lossless quality:** 50+ dB PSNR, 0.995+ SSIM
- ✅ **Metadata preservation:** Retains EXIF data from original images
- ✅ **Quality metrics:** Outputs PSNR and SSIM values
- ✅ **Universal format:** Works with any image format (JPEG, PNG, etc.)
- ✅ **.nci extension:** Clean, semantic file format

## Performance

Tested on real photos:

| Image | Original (JPEG) | Compressed (.nci) | Ratio | PSNR | SSIM |
|-------|----------------|-------------------|-------|------|------|
| Portrait (768×1024) | 163 KB | 19 KB | **88.5% smaller** | 51.87 dB | 0.9967 |
| Landscape (768×1024) | 155 KB | 13 KB | **91.8% smaller** | 52.72 dB | 0.9973 |

## Requirements

- Python 3.7+
- PyTorch
- OpenCV (cv2)
- NumPy
- scikit-image (optional, for SSIM)
- Pillow (PIL)

## Model

The codec requires the trained model file `tier1_final_model.pth` in the same directory.

**Download:** 
```bash
wget https://ai-codec-v3-artifacts-580473065386.s3.us-east-1.amazonaws.com/pvc/hybrid/tier1_final_model.pth
```

Or the model is already included in this directory (58 MB).

## File Format

### .nci (Neurally Compressed Image)

The `.nci` file contains:
- Compressed neural latent representation (INT8 quantized + GZIP)
- Original image dimensions
- EXIF metadata (if preserved)
- PSNR and SSIM quality metrics
- Procedural encoding parameters

### Format Details:
- **Compression:** GZIP level 9
- **Latent quantization:** INT8 (8-bit)
- **Version:** 1.0
- **Typical size:** 10-20 KB per image (depending on content)

## Advanced Options

```bash
# Specify model path
python nci_codec.py encode input.jpg output.nci --model /path/to/model.pth

# Force CPU (even if CUDA available)
python nci_codec.py encode input.jpg output.nci --device cpu

# Force GPU
python nci_codec.py encode input.jpg output.nci --device cuda
```

## Output Format

### Encoding Output:
```
Encoded: photo.jpg -> photo.nci
Original size: 163.03 KB
Compressed size: 18.79 KB
Compression ratio: 11.5%
PSNR: 51.87 dB
SSIM: 0.9967
```

### Decoding Output:
```
Decoded: photo.nci -> photo.jpg
PSNR: 51.87 dB
SSIM: 0.9967
```

## Known Limitations

1. **Tiling artifacts:** Images are processed in tiles, which may cause minor visible seams at tile boundaries. This will be addressed in future model versions.

2. **Resolution:** Model trained on 960×540 frames. Larger images are tiled. For best results, images should be multiples of 32 pixels in each dimension.

3. **Content type:** Optimized for:
   - Animation (anime, Disney, Pixar)
   - Photos with smooth gradients
   - Natural lighting
   
   Less optimal for:
   - Highly detailed textures
   - Noisy images
   - Text-heavy images

## Examples

### Batch encode all images in a folder:
```bash
for img in *.jpg; do
    python nci_codec.py encode "$img" "${img%.jpg}.nci" --preserve-metadata
done
```

### Batch decode all .nci files:
```bash
for nci in *.nci; do
    python nci_codec.py decode "$nci" "${nci%.nci}_decoded.jpg"
done
```

## License

Research project - See main repository for license details.

## Citation

If you use this codec in your research, please cite:

```
PVC v2.0 - Procedural Video Codec
Neural I-frame compression for animation and natural images
October 2025
```

## Contact

For questions or issues, see the main PVC v2.0 repository.

