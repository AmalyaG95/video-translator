# Testing Guide - Package Upgrades & Advanced Segmentation

## ✅ Fixed Issues

1. **Import path fixed**: `src.logging` → `src.app_logging` in `verify_imports.py`

## 🐳 Testing in Docker (Recommended)

Since this is a Docker-based project, test everything in Docker:

### 1. Build the Docker Image

```bash
cd backend-python-ml-v2
docker build -t video-translation-ml-v2 .
```

This will:
- Install all new packages (librosa, rapidfuzz, polyglot, nltk, etc.)
- Set up the Python environment correctly
- Download NLTK data on first use
- Be ready for NLLB model downloads

### 2. Test Imports in Docker

```bash
# Run import verification inside Docker
docker run --rm video-translation-ml-v2 python3 verify_imports.py
```

### 3. Check New Packages

```bash
# Verify new packages are installed
docker run --rm video-translation-ml-v2 pip list | grep -E "librosa|rapidfuzz|polyglot|nltk|soundfile|resampy"
```

Expected output:
```
librosa         0.10.x
nltk            3.8.x
polyglot        16.7.x
pycld2          0.41.x
rapidfuzz        3.9.x
resampy          0.4.x
soundfile        0.12.x
```

### 4. Test with Docker Compose

If you have docker-compose set up:

```bash
# Start services
docker-compose up -d python-ml

# Check logs
docker-compose logs -f python-ml

# Test language detection
# (Use your API endpoint to test)
```

## 🔧 Local Testing (Optional)

If you want to test locally without Docker, create a virtual environment:

```bash
cd backend-python-ml-v2

# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run verification
python3 verify_imports.py
```

## ✅ Verification Checklist

After building Docker image, verify:

- [ ] All imports work (`verify_imports.py` passes)
- [ ] New packages are installed (librosa, rapidfuzz, polyglot, nltk)
- [ ] NLLB models can be loaded (will download on first use)
- [ ] Polyglot language detection works
- [ ] NLTK tokenization works (downloads punkt on first use)
- [ ] Configuration loads correctly

## 🚀 Next Steps After Testing

1. **Test with a real video**:
   - Upload a video through your API
   - Monitor logs for:
     - NLLB model loading
     - Duplicate detection and merging
     - NLTK tokenization
     - Prosodic analysis
     - Polyglot language detection

2. **Monitor first run**:
   - First NLLB model download (can take 5-10 minutes)
   - NLTK punkt tokenizer download (automatic)
   - Polyglot language data (automatic)

3. **Verify improvements**:
   - Check for no duplicate segments
   - Better sentence boundaries
   - More natural breaks
   - Better translation quality

## 📝 Expected First Run Behavior

When you first run with the new setup:

1. **NLLB Model Download** (if using NLLB):
   ```
   Loading NLLB model: facebook/nllb-200-1.3b
   Downloading model files... (this takes time)
   ```

2. **NLTK Data Download** (automatic):
   ```
   Downloading punkt tokenizer...
   ```

3. **Polyglot Language Data** (automatic):
   ```
   Downloading language detection data...
   ```

All downloads are automatic and cached for future use.

## 🐛 Troubleshooting

### Import Errors
- **Fix**: Use Docker (recommended) or virtual environment
- **Verify**: Run `verify_imports.py` in Docker

### Missing Packages
- **Fix**: Rebuild Docker image
- **Check**: `docker run --rm video-translation-ml-v2 pip list`

### NLLB Model Download Fails
- **Check**: Internet connection
- **Verify**: HuggingFace access
- **Fallback**: System will use Helsinki-NLP

### NLTK/Polyglot Downloads Fail
- **Check**: Internet connection
- **Note**: Downloads are automatic on first use
- **Fallback**: System uses regex/langdetect











