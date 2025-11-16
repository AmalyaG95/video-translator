# How to Add Videos for Testing in Electron App

## Method 1: Using the UI (Recommended)

The Electron app supports two ways to upload videos through the UI:

### Option A: Drag and Drop
1. Open the Electron app
2. Navigate to the upload screen
3. **Drag and drop** a video file (`.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`) onto the upload area
4. The file will be automatically uploaded and processed

### Option B: Click to Browse
1. Open the Electron app
2. Navigate to the upload screen
3. **Click** on the upload area (the dashed box that says "Drag and drop your video file here, or click to browse")
4. A file picker dialog will open
5. Select your video file
6. The file will be automatically uploaded and processed

## Method 2: Direct File Copy (For Testing)

If you want to add test videos directly to the backend without using the UI:

### Step 1: Copy videos to the uploads directory

The uploads directory is mounted as a volume at `./uploads` in the project root.

```bash
# From the project root directory
cp /path/to/your/test-video.mp4 ./uploads/
```

### Step 2: Verify the file is accessible

```bash
# Check if the file exists in the container
docker-compose -f docker-compose.v2.yml exec nestjs-api ls -lh /app/uploads/
```

### Step 3: Use the file in testing

The file will be accessible at `/app/uploads/test-video.mp4` inside the container, or you can reference it by its filename when testing the upload API directly.

## Method 3: Using Docker Copy Command

You can also copy files directly into a running container:

```bash
# Copy a video file into the NestJS container
docker cp /path/to/your/test-video.mp4 translate-v_nestjs-api_1:/app/uploads/test-video.mp4

# Or copy into the Python ML container
docker cp /path/to/your/test-video.mp4 translate-v_python-ml-v2_1:/app/uploads/test-video.mp4
```

## Supported Video Formats

The app supports the following video formats:
- `.mp4` (MPEG-4)
- `.avi` (Audio Video Interleave)
- `.mov` (QuickTime)
- `.mkv` (Matroska)
- `.webm` (WebM)

## File Size Limits

- Default maximum file size: **100 GB** (configurable)
- If you encounter "file too large" errors, check the `maxSize` prop in the `FileUpload` component

## Troubleshooting

### File not appearing in uploads directory
- Make sure the `./uploads` directory exists in the project root
- Check that the volume mount is correct in `docker-compose.v2.yml`
- Restart the containers: `docker-compose -f docker-compose.v2.yml restart nestjs-api`

### File picker not opening in Electron
- Make sure the Electron app is running with proper permissions
- Check that the `select-video-file` IPC handler is working (check DevTools console)
- Verify the preload script is loaded correctly

### Videos not processing
- Check that the file format is supported
- Verify the file is not corrupted
- Check the backend logs: `docker-compose -f docker-compose.v2.yml logs nestjs-api`
- Check the Python ML service logs: `docker-compose -f docker-compose.v2.yml logs python-ml-v2`

## Notes

- Videos uploaded through the UI are automatically copied to `/app/uploads` in the backend
- The backend generates a unique filename based on timestamp and session ID
- Original filenames are preserved in metadata
- Videos are processed asynchronously - check the processing screen for progress


