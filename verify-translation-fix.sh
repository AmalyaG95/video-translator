#!/bin/bash
set -e

echo "🔍 Verifying Translation Fix..."
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Check if services are running
echo -e "${BLUE}1. Checking Docker services...${NC}"
if ! docker-compose ps | grep -q "Up"; then
    echo -e "${RED}❌ Services are not running. Start them with: docker-compose up -d${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Services are running${NC}"
echo ""

# Check Python ML service logs for the fix
echo -e "${BLUE}2. Checking Python ML service for translation validation logic...${NC}"
if docker-compose logs python-ml 2>&1 | grep -q "Translation validation"; then
    echo -e "${GREEN}✅ Translation validation logic is present${NC}"
else
    echo -e "${YELLOW}⚠️  Translation validation logs not found (may appear during actual translation)${NC}"
fi

# Check for the specific fix: translation_failed flag
echo -e "${BLUE}3. Verifying fix in code...${NC}"
if grep -q "translation_failed.*=.*True" backend-python-ml/src/pipeline/compliant_pipeline.py; then
    echo -e "${GREEN}✅ Fix found: translation_failed flag is set when translation returns empty${NC}"
else
    echo -e "${RED}❌ Fix not found: translation_failed flag missing${NC}"
    exit 1
fi

if grep -q "translated_count = sum" backend-python-ml/src/pipeline/compliant_pipeline.py && grep -q "translated_text.*tts_path" backend-python-ml/src/pipeline/compliant_pipeline.py; then
    echo -e "${GREEN}✅ Fix found: Validation checks for both translated_text AND tts_path${NC}"
else
    echo -e "${RED}❌ Fix not found: Validation logic missing${NC}"
    exit 1
fi

if grep -q "if translated_count == 0" backend-python-ml/src/pipeline/compliant_pipeline.py; then
    echo -e "${GREEN}✅ Fix found: Pipeline fails early if translated_count is 0${NC}"
else
    echo -e "${RED}❌ Fix not found: Early failure check missing${NC}"
    exit 1
fi
echo ""

# Check that the Docker image was rebuilt
echo -e "${BLUE}4. Checking Docker image build time...${NC}"
IMAGE_DATE=$(docker inspect translate-v_python-ml:latest --format='{{.Created}}' 2>/dev/null || echo "not found")
if [ "$IMAGE_DATE" != "not found" ]; then
    echo -e "${GREEN}✅ Python ML image exists${NC}"
    echo "   Image created: $IMAGE_DATE"
else
    echo -e "${RED}❌ Python ML image not found${NC}"
    exit 1
fi
echo ""

# Summary
echo -e "${GREEN}✅ Verification Summary:${NC}"
echo "   ✓ Services are running"
echo "   ✓ Code contains translation_failed flag"
echo "   ✓ Code validates both translated_text AND tts_path"
echo "   ✓ Pipeline fails early if no segments translated"
echo "   ✓ Docker image is built"
echo ""
echo -e "${BLUE}💡 To test actual translation:${NC}"
echo "   1. Upload a video through the frontend"
echo "   2. Check logs: docker-compose logs -f python-ml | grep -E 'Translation validation|translation_failed|segments fully translated'"
echo "   3. Look for: '📊 Translation validation: X/Y segments fully translated'"
echo ""

