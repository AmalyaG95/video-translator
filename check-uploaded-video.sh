#!/bin/bash
# Quick script to check uploaded videos

echo "=== Checking videos in uploads directory ==="
echo ""
echo "Host directory (./uploads):"
ls -lh ./uploads/ | tail -10

echo ""
echo "NestJS container (/app/uploads):"
docker-compose -f docker-compose.v2.yml exec -T nestjs-api ls -lh /app/uploads/ | tail -10

echo ""
echo "Python ML container (/app/uploads):"
docker-compose -f docker-compose.v2.yml exec -T python-ml-v2 ls -lh /app/uploads/ | tail -10

echo ""
echo "=== To copy a new test video ==="
echo "cp /path/to/your/video.mp4 ./uploads/"
echo ""
echo "The file will be immediately visible - NO RESTART NEEDED!"


