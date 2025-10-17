# Dashboard Parity and Media Downloads - FIXED

## Issues Fixed

### 1. Public Dashboard Not Matching Admin Dashboard

**Problem**: The public dashboard (`index.html` + `app.js`) had a simpler experiments table compared to the admin dashboard, missing important columns like:
- Runtime
- Code Evolution (LLM badge)
- Version
- GitHub commit
- Media downloads

**Solution**: Updated `dashboard/app.js` to match the admin dashboard table structure, showing all the same columns except the "Rerun" button (admin-only).

### 2. Missing Download Links for Reconstructed Videos and Decoder Code

**Problem**: Neither the public nor admin dashboard showed direct download links for:
- Reconstructed video files (the decompressed output)
- Decoder Python code

These were only accessible via the blog detail page, making them hard to find.

**Solution**: Added a new "Media" column to both dashboards that parses the experiment JSON data and displays download buttons for:
- **Video**: Reconstructed/decompressed video output
- **Decoder**: Python decoder code file

## Changes Made

### Files Updated

1. **`/Users/yarontorbaty/Documents/Code/AiV1/dashboard/app.js`**
   - Added Runtime column with progress display
   - Added Code Evolution column (✨ LLM badge)
   - Added Version column (v1, v2, etc.)
   - Added GitHub commit column
   - Added Media column with video and decoder download links
   - Parses `experiments` JSON field to extract `video_url` and `decoder_s3_key`

2. **`/Users/yarontorbaty/Documents/Code/AiV1/dashboard/admin.js`**
   - Added Media column with video and decoder download links
   - Same parsing logic as public dashboard

## Table Structure

### Public Dashboard (index.html via app.js)

| Column | Description |
|--------|-------------|
| Experiment ID | Unique experiment identifier |
| Time | When experiment ran |
| Status | completed/running/failed |
| Bitrate | Mbps (lower is better) |
| PSNR | Quality metric (higher is better) |
| Quality | excellent/good/acceptable/poor |
| Runtime | Elapsed vs estimated time |
| Phase | Current phase (design/deploy/validate/execute/etc.) |
| Code | ✨ LLM badge if code was generated |
| Ver | Version number (v1, v2, etc.) |
| Git | GitHub commit hash or 📦 Local |
| **Media** | 🎬 Video + 💾 Decoder download buttons |
| Actions | View Details button |

### Admin Dashboard (admin.html via admin.js)

Same as public dashboard, PLUS:
- Bug Analysis column
- Human Intervention column  
- **Rerun button** in Actions column (admin-only)

## Media Column Details

The Media column displays:

### When Video is Available
```html
<a href="{presigned_s3_url}" target="_blank">
  <i class="fas fa-video"></i> Video
</a>
```

### When Decoder is Available
```html
<a href="https://ai-video-codec-videos-580473065386.s3.amazonaws.com/{decoder_key}" target="_blank">
  <i class="fas fa-code"></i> Decoder
</a>
```

### Visual Styling
- Video button: Pink/magenta (#ec4899)
- Decoder button: Blue (#0ea5e9)
- Stacked vertically for easy access
- Only shown for successful experiments with uploads

## Data Flow

The download links are populated from experiment data:

```javascript
// Parse experiments JSON
const experimentsData = JSON.parse(exp.experiments);
const videoUrl = experimentsData[0].video_url;
const decoderKey = experimentsData[0].decoder_s3_key;
```

These fields are set in `procedural_experiment_runner.py` during the analysis phase:
- `video_url`: Presigned S3 URL for reconstructed video (7-day expiry)
- `decoder_s3_key`: S3 key for decoder Python file

## Testing

### Public Dashboard
1. Navigate to `https://<your-domain>/index.html`
2. Check experiments table has all columns including Media
3. Click Video button → should open video in new tab
4. Click Decoder button → should download Python file
5. Verify NO "Rerun" button is present

### Admin Dashboard
1. Navigate to `https://<your-domain>/admin.html`
2. Login with credentials
3. Check experiments table has Media column
4. Verify "Rerun" button IS present in Actions
5. Test video and decoder downloads

## Deployment

To deploy these changes:

```bash
# Copy updated files to S3 bucket
aws s3 cp dashboard/app.js s3://your-dashboard-bucket/app.js
aws s3 cp dashboard/admin.js s3://your-dashboard-bucket/admin.js

# Or if using CloudFront
aws s3 sync dashboard/ s3://your-dashboard-bucket/ --exclude "*.md"
aws cloudfront create-invalidation --distribution-id YOUR_DIST_ID --paths "/*"
```

## Comparison: Before vs After

### Before
- Public dashboard: 8 columns
- Admin dashboard: 14 columns
- No direct media downloads
- Different layouts between public/admin

### After
- Public dashboard: 13 columns (matches admin minus Rerun)
- Admin dashboard: 14 columns (includes Rerun)
- Direct download links for video and decoder
- Consistent layout between public/admin

## Related Files

- `/Users/yarontorbaty/Documents/Code/AiV1/dashboard/app.js` - Public dashboard logic
- `/Users/yarontorbaty/Documents/Code/AiV1/dashboard/admin.js` - Admin dashboard logic
- `/Users/yarontorbaty/Documents/Code/AiV1/dashboard/index.html` - Public dashboard HTML
- `/Users/yarontorbaty/Documents/Code/AiV1/dashboard/admin.html` - Admin dashboard HTML
- `/Users/yarontorbaty/Documents/Code/AiV1/src/agents/procedural_experiment_runner.py` - Generates video_url and decoder_s3_key


