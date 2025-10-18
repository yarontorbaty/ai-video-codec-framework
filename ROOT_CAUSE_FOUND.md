# ROOT CAUSE: Auto-Restarting v1 Workers

**Discovery Time:** 18:52 UTC, October 17, 2025

---

## 🔴 CRITICAL FINDING

### The Problem: Zombie v1 Workers Keep Restarting

**What We Found:**
1. Stopped 3 old v1 workers at 18:36 UTC
2. By 18:42 UTC (6 minutes later) they had **automatically restarted**
3. New v1 worker processes appeared with fresh PIDs (e.g., PID 7085)
4. These restarted workers immediately resumed consuming SQS messages

### Evidence:
```
# First kill at 18:36
OLD PID: 8415, 8407, 1897

# Check at 18:42 - NEW PROCESS
NEW PID: 7085 (started at 18:42)
Process: /usr/bin/python3 /opt/ai-video-codec/workers/training_worker.py
```

---

## 🎯 Root Cause Analysis

### Why They Restart

The v1 workers are managed by one of:
1. **systemd service** - Most likely
2. **cron job** - Checking every N minutes
3. **supervisor/init.d** - Process manager
4. **User script in rc.local** - Startup script

### Impact

**Every time we kill the old workers, they restart within minutes and:**
- Resume consuming SQS messages
- Block v2 worker from receiving anything
- Make all our tests fail

This explains why:
- Messages keep showing as "in-flight"
- v2 worker never sees them
- Queue appears empty to v2 worker
- Tests keep failing even after "fixing" the issue

---

## 🛠️ Solution Required

### Must Do (in order):

**1. Find the Auto-Restart Mechanism**
```bash
# Check systemd
systemctl list-units | grep codec
systemctl list-units | grep training

# Check cron
crontab -l
crontab -l -u ec2-user

# Check supervisor
supervisorctl status

# Check rc.local / init scripts
cat /etc/rc.local
ls /etc/init.d/
```

**2. Disable the Service**
```bash
# If systemd:
systemctl stop training-worker
systemctl disable training-worker

# If cron:
crontab -r  # or edit to remove

# If supervisor:
supervisorctl stop training-worker
```

**3. Kill All Instances**
```bash
pkill -9 -f training_worker.py
```

**4. Verify No Restart**
```bash
# Wait 5 minutes
sleep 300
ps aux | grep training_worker
# Should show NOTHING
```

**5. THEN Test v2 Worker**
Only after confirming old workers stay dead.

---

## 🚨 Why This Wasn't Obvious

1. **Systemd restarts are silent** - No logs visible
2. **PIDs change** - Looks like different processes
3. **Restart delay** - 6-minute gap makes it seem unrelated
4. **Multiple instances** - All 3 instances doing this independently

---

## 📊 Timeline of Discovery

| Time | Event |
|------|-------|
| 18:02 | Found 3 old v1 workers running |
| 18:36 | Killed all 3 workers (PIDs 8415, 8407, 1897) |
| 18:36-18:42 | Sent test messages, all consumed |
| 18:42 | NEW worker appeared (PID 7085) |
| 18:50 | Killed again, sent new test |
| 18:52 | **DISCOVERED AUTO-RESTART** |

---

## ✅ Current Actions

Running commands to:
1. Find what's managing the workers
2. Disable the auto-restart
3. Kill all instances permanently

---

## 🎯 Next Steps

1. Wait for current command results
2. Identify the restart mechanism
3. Disable it permanently
4. Kill all workers one final time
5. Verify they stay dead for 5+ minutes
6. THEN test v2 worker

**This is the ACTUAL root cause.**

