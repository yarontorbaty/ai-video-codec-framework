# 📊 PSNR/SSIM Columns - Verification Guide

**Date:** 2025-10-17  
**Status:** Columns deployed but need browser refresh + new experiment data

---

## 🎯 Current Situation

### ✅ **Good News:**
1. PSNR and Quality columns ARE deployed to S3
2. Code is correct and verified
3. CloudFront cache has been invalidated
4. Orchestrator is running with quality verification

### ⚠️ **Why You Don't See Them:**

**Reason 1: Browser Cache**
Your browser cached the old JavaScript file before the columns were added.

**Reason 2: No Quality Data Yet**
All existing experiments (proc_exp_1760701917 and earlier) were run **before** quality verification was added, so they show "—" in PSNR/Quality columns.

---

## 🔧 **Fix - Step by Step:**

### **Step 1: Clear Browser Cache**

#### **Option A: Hard Refresh (Recommended)**
- **Chrome/Edge**: `Ctrl+Shift+R` (Windows) or `Cmd+Shift+R` (Mac)
- **Firefox**: `Ctrl+F5` (Windows) or `Cmd+Shift+R` (Mac)  
- **Safari**: Hold `Shift` + click refresh button

#### **Option B: Open Incognito/Private Mode**
- Bypasses all cache
- Guaranteed to load latest files

#### **Option C: Clear All Cache** (Nuclear option)
1. Open browser DevTools (`F12`)
2. Go to Network tab
3. Check "Disable cache"
4. Refresh page

---

### **Step 2: Verify Columns Are Visible**

After refreshing, you should see these **column headers** in the admin dashboard:

```
| Experiment ID | Time | Status | Tests | Bitrate | PSNR | Quality | Runtime | Phase | ... |
|---------------|------|--------|-------|---------|------|---------|---------|-------|-----|
```

**New Columns:**
- **PSNR** (with chart icon)
- **Quality** (with eye icon)

---

### **Step 3: Check Existing Experiments**

For old experiments (before quality verification), you'll see:

```
| proc_exp_1760702563 | ... | 2.66 Mbps | — | — | ... |
                                         ↑    ↑
                                       PSNR Quality
                                      (no data)
```

**This is expected!** Old experiments don't have quality data.

---

### **Step 4: Wait for New Experiment**

The orchestrator is currently running a new experiment:
- Status: Design phase (starting)
- Will include quality verification
- Should complete in ~30-60 minutes

**When it completes, you'll see:**

```
| proc_exp_XXXXXX | ... | 2.66 Mbps | 32.5 dB | ✅ GOOD | ... |
                                      ↑          ↑
                                    PSNR    Quality
                                  (with data!)
```

---

## 🔍 **How to Verify Columns Are Loaded:**

### **Method 1: Check Headers**
Look for the new column headers:
- `<i class="fas fa-chart-line"></i> PSNR`
- `<i class="fas fa-eye"></i> Quality`

### **Method 2: Inspect Network Tab**
1. Open DevTools (`F12`)
2. Go to Network tab
3. Find `admin.js` request
4. Check response size: should be **~48 KB** (not ~40 KB like old version)
5. Search in response for "PSNR" - should find 2 matches

### **Method 3: Check Console**
Open browser console and run:
```javascript
document.body.innerHTML.includes('PSNR')
// Should return: true
```

---

## 📊 **What Each Column Shows:**

### **PSNR Column:**
```
32.5 dB
Good
```
- Top: PSNR value in decibels
- Bottom: Quality label (Excellent/Good/Acceptable/Poor)
- Color: Green (>= 30), Yellow (>= 25), Red (< 25)

### **Quality Column:**
```
  ✅
 GOOD
SSIM: 0.920
```
- Top: Emoji (🏆 ✅ ⚠️ ❌)
- Middle: Quality status
- Bottom: SSIM value

---

## 🎯 **Expected Timeline:**

### **Now:**
- ✅ Columns deployed
- ⚠️ Need browser refresh
- ⚠️ Old experiments show "—"

### **In 30-60 minutes:**
- ✅ New experiment completes
- ✅ PSNR/SSIM data available
- ✅ Columns show real values

### **Going forward:**
- ✅ Every new experiment will have quality data
- ✅ Dashboard shows quality at a glance
- ✅ Easy to identify successful experiments

---

## 🚨 **If You Still Don't See Columns:**

### **Check 1: Verify URL**
Make sure you're on: `https://aiv1codec.com/admin.html`
(Not a local file or different URL)

### **Check 2: Check Browser Console for Errors**
1. Open DevTools (`F12`)
2. Go to Console tab
3. Look for JavaScript errors
4. Share any errors you see

### **Check 3: Verify S3 File**
```bash
# Download and check
aws s3 cp s3://ai-video-codec-dashboard-580473065386/admin.js /tmp/check.js
grep "PSNR" /tmp/check.js
# Should show 2 matches
```

### **Check 4: Try Different Browser**
Open in a completely different browser to rule out browser-specific issues.

---

## ✅ **Confirmation:**

**After hard refresh, you should see:**

**Admin Dashboard Table Headers:**
```
| Experiment ID | Time | Status | Tests Run | Best Bitrate | PSNR | Quality | Runtime | Phase | ... |
```

**Old Experiments (no data):**
```
| proc_exp_1760702563 | ... | 2.66 Mbps | — | — | 2.5m | ... |
```

**New Experiments (with data):**
```
| proc_exp_1760703XXX | ... | 2.66 Mbps | 32.5 dB (Good) | ✅ GOOD SSIM: 0.920 | 2.5m | ... |
```

---

## 📝 **Current Status:**

```
✅ Code deployed to S3
✅ CloudFront cache invalidated
✅ Lambda updated with quality data
✅ Orchestrator running with quality verification
⚠️ Browser may have old cached JS
⚠️ Existing experiments have no quality data (expected)
🔄 New experiment in progress (will have quality data)
```

---

**Hard refresh your browser and the columns should appear!** 🚀

Then wait ~30-60 min for the new experiment to complete, and you'll see actual PSNR/SSIM values in those columns.

