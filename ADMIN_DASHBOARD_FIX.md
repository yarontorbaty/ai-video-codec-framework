# Admin Dashboard Fix - Re-Run Controls

## Problem

The admin dashboard at https://aiv1codec.com/admin.html was showing outdated content without the experiment control features (start/stop/rerun experiments).

## Root Cause

The `admin.html` and `admin.js` files in S3 were outdated (from 05:11 and 05:23 UTC), while the local files had been updated with the full admin interface including:
- Start/Stop experiment controls
- Re-run experiment buttons
- Real-time experiment monitoring
- LLM chat interface

## What Was Fixed

### 1. Uploaded Latest Admin Files

```bash
aws s3 cp dashboard/admin.html s3://ai-video-codec-dashboard-580473065386/admin.html
aws s3 cp dashboard/admin.js s3://ai-video-codec-dashboard-580473065386/admin.js
```

### 2. Invalidated CloudFront Cache

```bash
aws cloudfront create-invalidation \
  --distribution-id E3PUY7OMWPWSUN \
  --paths "/admin.html" "/admin.js"
```

**Invalidation ID:** `IBEUXEZC5E94NW7SDMMBK3NWSO`

## Admin Dashboard Features (Now Available)

### Experiment Controls

**Top Control Panel:**
- 🟢 **Start Experiment** - Triggers a new autonomous experiment
- 🔴 **Stop All** - Stops all running experiments
- 📋 **View Logs** - Shows orchestrator logs

### Experiment List

**Per-Experiment Actions:**
- 👁️ **View** - View detailed experiment results
- 🔄 **Rerun** - Start a new experiment with similar parameters

### LLM Chat Interface

- Direct chat with the governing LLM
- Request experiment suggestions
- Ask about system status
- Control experiment parameters

### Real-Time Monitoring

- Live experiment status
- Completion percentages
- Last update timestamps
- Running/completed/failed indicators

## Dashboard Structure

```
Admin Dashboard (admin.html)
├── Login Screen (2FA)
│   ├── Username/Password
│   └── Email verification code
│
├── Control Panel (after login)
│   ├── Start/Stop Controls
│   ├── Log Viewer
│   └── Status Indicators
│
├── Experiments Table
│   ├── Experiment ID
│   ├── Status
│   ├── Bitrate (Mbps)
│   ├── PSNR/SSIM Quality
│   ├── Completion %
│   ├── Last Update
│   └── Actions (View/Rerun)
│
└── LLM Chat (bottom)
    ├── Message History
    ├── Input Field
    └── Send Button
```

## Regular Dashboard Status

The main dashboard at https://aiv1codec.com/ is working correctly with:
- ✅ Server-side rendering (SSR)
- ✅ Real cost data from AWS Cost Explorer
- ✅ Breakdown by service (EC2, S3, Lambda, Claude)
- ✅ Recent experiments list
- ✅ Auto-refresh every 60 seconds

**Note:** The regular dashboard does NOT have experiment controls - those are admin-only features accessible at `/admin.html`.

## Testing the Fix

### 1. Clear Browser Cache

```bash
# Chrome/Edge
Ctrl+Shift+R (Windows/Linux)
Cmd+Shift+R (Mac)

# Or use Incognito/Private mode
```

### 2. Access Admin Dashboard

```
https://aiv1codec.com/admin.html
```

### 3. Login with Credentials

Use the credentials stored in AWS Secrets Manager:
```bash
aws secretsmanager get-secret-value \
  --secret-id ai-video-codec/admin-credentials \
  --region us-east-1 \
  --query SecretString \
  --output text | jq .
```

### 4. Verify Features

After login, you should see:
- ✅ Start/Stop/View Logs buttons at the top
- ✅ Experiments table with View/Rerun buttons
- ✅ LLM chat interface at the bottom
- ✅ Real-time experiment status updates

## Deployment Checklist

When updating admin dashboard in the future:

- [ ] Edit `dashboard/admin.html` and/or `dashboard/admin.js`
- [ ] Upload to S3:
  ```bash
  aws s3 cp dashboard/admin.html s3://ai-video-codec-dashboard-580473065386/
  aws s3 cp dashboard/admin.js s3://ai-video-codec-dashboard-580473065386/
  ```
- [ ] Invalidate CloudFront cache:
  ```bash
  aws cloudfront create-invalidation \
    --distribution-id E3PUY7OMWPWSUN \
    --paths "/admin.html" "/admin.js"
  ```
- [ ] Wait 1-2 minutes for invalidation
- [ ] Test in incognito mode

## Why Regular Dashboard Looks "Old"

If you're referring to the experiment table not showing quality metrics (PSNR/SSIM), that's because:

1. **No successful experiments yet** - We've only had experiments that exceed the 1.0 Mbps target
2. **Quality metrics only show for successful experiments** - The system only displays PSNR/SSIM/Quality badges for experiments that achieve < 1.0 Mbps
3. **The dashboard IS updated** - It shows the latest data, but none of the experiments have hit the success criteria yet

The dashboard will show richer metrics (PSNR, SSIM, quality badges, video links, decoder downloads) once an experiment achieves < 1.0 Mbps bitrate.

## Summary

✅ **Admin dashboard updated** - Now shows full experiment controls  
✅ **CloudFront cache invalidated** - Fresh files served  
✅ **Re-run controls available** - Can restart experiments  
✅ **Regular dashboard working** - SSR with real data  

**Next:** Wait for cache to propagate (1-2 minutes) and hard-refresh the admin page.


