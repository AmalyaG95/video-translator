# Disk Cleanup Guide

## Current Situation
- **Total Disk**: 125 GB
- **Used**: 86 GB (73% full)
- **Free**: 33 GB
- **Error**: "no space left on device" preventing Docker builds

## Quick Cleanup (Run the script)
```bash
./cleanup_disk.sh
```

This will free approximately **15-20 GB** safely.

## Manual Cleanup Options

### 1. Chrome Cache (4.2 GB) - SAFE
```bash
rm -rf ~/.var/app/com.google.Chrome/cache/*
```
Chrome will rebuild this cache automatically.

### 2. Gradle Cache (6.8 GB) - SAFE
```bash
rm -rf ~/.gradle/caches/*
```
Gradle will rebuild caches when you build projects.

### 3. System Journal Logs (2.0 GB) - SAFE
```bash
sudo journalctl --vacuum-time=3d
```
Keeps only last 3 days of logs.

### 4. System Log Files (2.6 GB) - SAFE
```bash
sudo find /var/log -type f -name "*.log" -mtime +7 -delete
sudo find /var/log -type f -name "*.gz" -delete
```
Removes old log files.

### 5. APT Package Cache - SAFE
```bash
sudo apt-get clean
sudo apt-get autoclean
```

### 6. Docker Cleanup - SAFE
```bash
docker system prune -a
```
Removes unused images, containers, and build cache.

### 7. npm Cache (452 MB) - SAFE
```bash
npm cache clean --force
```

### 8. Temporary Files - SAFE
```bash
rm -rf /tmp/*
rm -rf ~/.cache/*
```

## Large Directories to Review Manually

### Android SDK/Studio (~10.4 GB total)
- `~/.android`: 3.7 GB
- `~/Android`: 3.4 GB  
- `~/android-studio`: 3.3 GB

**Action**: Only clean if you don't need Android development:
```bash
# If you don't need Android development:
rm -rf ~/.android
rm -rf ~/Android
rm -rf ~/android-studio
```

### Chrome Config (3.8 GB)
- `~/.var/app/com.google.Chrome/config`: 3.8 GB

**Action**: Review what's in here. This might contain important data. You can:
- Clear browsing data from Chrome settings
- Remove specific cache directories manually

### Desktop Files (2.1 GB)
- `~/Desktop`: 2.1 GB

**Action**: Review and move/delete large files you don't need.

### Downloads (519 MB)
- `~/Downloads`: 519 MB

**Action**: Review and delete files you no longer need.

## Expected Results

After running the cleanup script:
- **Freed**: ~15-20 GB
- **New Free Space**: ~48-53 GB
- **Usage**: ~55-60% (down from 73%)

This should resolve the "no space left on device" error and allow Docker builds to complete.

## Additional Tips

1. **Monitor disk usage**:
   ```bash
   du -sh ~/* | sort -hr | head -20
   ```

2. **Find large files**:
   ```bash
   find ~ -type f -size +100M -exec ls -lh {} \; | awk '{print $5, $9}' | sort -hr
   ```

3. **Check what's using space in a directory**:
   ```bash
   du -sh ~/.var/app/* | sort -hr
   ```


