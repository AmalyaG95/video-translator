# Where is the uploads folder located?

## Important: Electron file picker shows HOST filesystem, not container filesystem

When you open the file picker in the Electron app, it shows your **host computer's filesystem**, not the Docker container's filesystem.

## Location of uploads folder:

**Host path (use this in Electron file picker):**
```
/home/amalya/Desktop/translate-v/uploads
```

**Container path (for reference only - not accessible from file picker):**
```
/app/uploads
```

## How to navigate to it in Electron file picker:

1. Open the file picker in Electron app
2. Click on "Other Locations" or navigate manually
3. Go to: `/home/amalya/Desktop/translate-v/uploads`
4. You'll see all your test videos there, including:
   - `English - Daily routine (A1-A2).mp4`
   - All other `.mp4` files you've added

## Quick navigation tips:

- **From Home**: `Home` → `Desktop` → `translate-v` → `uploads`
- **Direct path**: Type or paste `/home/amalya/Desktop/translate-v/uploads` in the file picker's path bar
- **Bookmark**: You can bookmark this location in the file picker for quick access

## Your test video is here:

```
/home/amalya/Desktop/translate-v/uploads/English - Daily routine (A1-A2).mp4
```

This file is immediately accessible - no restart needed!


