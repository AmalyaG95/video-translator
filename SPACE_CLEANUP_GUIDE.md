# Space Cleanup Guide for Docker Deployment

## Current Situation
- **Free space**: 6.5GB
- **Required**: 10-15GB for Docker build
- **Needed**: ~8-10GB more

## Your Largest Cache Directories

| Directory | Size | Safe to Clean? | Priority |
|-----------|------|----------------|----------|
| `.cache` | 7.3GB | ✅ Yes | **HIGH** |
| `.npm` | 5.1GB | ✅ Yes | **HIGH** |
| `.gradle` | 6.8GB | ⚠️ If not developing Android | Medium |
| `.local` | 11GB | ⚠️ Partial (pip cache only) | Low |
| `.var` | 8.2GB | ⚠️ Check contents first | Low |

## Quick Fix - Clean Safe Caches

### Option 1: Automated Script (Recommended)
```bash
cd /home/amalya/Desktop/translate-v
./clean-home-cache.sh
```

This will:
- Show you what can be cleaned
- Ask permission for each directory
- Free ~12-19GB safely

### Option 2: Manual Cleanup

#### 1. Clean System Cache (7.3GB)
```bash
rm -rf ~/.cache/*
```
**Safe**: This regenerates automatically

#### 2. Clean npm Cache (5.1GB)
```bash
rm -rf ~/.npm/*
npm cache clean --force
```
**Safe**: Packages re-download when needed

#### 3. Clean Gradle Cache (6.8GB) - Only if not developing Android
```bash
rm -rf ~/.gradle/caches/*
```
**Warning**: Only do this if you're not actively developing Android apps

#### 4. Clean pip Cache (if exists in .local)
```bash
pip cache purge
```
**Safe**: Packages re-download when needed

## Expected Results

After cleaning `.cache` + `.npm`:
- **Freed**: ~12GB
- **New free space**: ~18.5GB ✅
- **Status**: Ready for Docker deployment!

## Verification

After cleanup, verify:
```bash
# Check free space
df -h /home/amalya

# Should show ~18GB+ free
```

## What These Caches Contain

- **`.cache`**: Browser cache, app caches, temporary files
- **`.npm`**: Node.js package cache (downloaded npm packages)
- **`.gradle`**: Android/Gradle build cache (compiled dependencies)
- **`.local/share/pip`**: Python pip package cache

All of these will regenerate as needed when you use the applications.

## After Cleanup

Once you have 15GB+ free space:
```bash
cd /home/amalya/Desktop/translate-v
./deploy.sh up --build
```

## If Still Not Enough

1. **Check Flatpak apps** (`.var` - 8.2GB):
   ```bash
   flatpak list
   # Remove unused flatpaks
   flatpak uninstall --unused
   ```

2. **Check pip packages in .local**:
   ```bash
   # See what's installed
   pip list --user
   # Only remove if you know it's safe
   ```

3. **Check Android Studio/Android** (if not needed):
   - Can free ~7GB if Android development not needed
   - But be careful - this is software installation

## Quick Command Summary

```bash
# Clean everything safe at once (interactive)
./clean-home-cache.sh

# Or manual one-liners:
rm -rf ~/.cache/* ~/.npm/*
npm cache clean --force

# Then check space
df -h /home/amalya
```


