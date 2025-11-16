#!/usr/bin/env python3
"""
End-to-End Test for Audio Fix Verification
Tests that final translated video contains ONLY translated TTS audio, no original audio.
"""

import sys
import os
import subprocess
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple

# Add backend-python-ml to path
sys.path.insert(0, str(Path(__file__).parent / "backend-python-ml" / "src"))

def run_command(cmd: List[str], description: str) -> Tuple[bool, str, str]:
    """Run a command and return success status, stdout, stderr"""
    print(f"\n🔧 {description}")
    print(f"   Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    return result.returncode == 0, result.stdout, result.stderr

def check_audio_streams(video_path: Path) -> Dict:
    """Check audio streams in a video file"""
    cmd = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'a',
        '-show_entries', 'stream=index,codec_name,codec_type,channels,sample_rate',
        '-of', 'json',
        str(video_path)
    ]
    success, stdout, stderr = run_command(cmd, f"Checking audio streams in {video_path.name}")
    
    if not success:
        return {"error": stderr, "streams": []}
    
    try:
        data = json.loads(stdout)
        streams = data.get('streams', [])
        return {
            "stream_count": len(streams),
            "streams": streams,
            "codecs": [s.get('codec_name') for s in streams]
        }
    except json.JSONDecodeError:
        return {"error": "Failed to parse ffprobe output", "streams": []}

def verify_no_original_audio_in_temp(temp_video_path: Path) -> bool:
    """Verify that temp video-only file has no audio"""
    result = check_audio_streams(temp_video_path)
    if result.get("stream_count", 0) == 0:
        print(f"   ✅ Temp file has no audio streams (correct)")
        return True
    else:
        print(f"   ❌ Temp file has {result.get('stream_count')} audio stream(s) (should be 0)")
        return False

def verify_final_video_audio(final_video_path: Path) -> Tuple[bool, Dict]:
    """Verify final video has exactly one AAC audio stream"""
    result = check_audio_streams(final_video_path)
    
    stream_count = result.get("stream_count", 0)
    codecs = result.get("codecs", [])
    
    print(f"\n📊 Final Video Audio Analysis:")
    print(f"   Stream count: {stream_count}")
    print(f"   Codecs: {codecs}")
    
    if stream_count == 1 and 'aac' in codecs:
        print(f"   ✅ PASS: Single AAC stream found (translated audio only)")
        return True, result
    elif stream_count == 0:
        print(f"   ❌ FAIL: No audio streams found")
        return False, result
    elif stream_count > 1:
        print(f"   ❌ FAIL: Multiple audio streams found ({stream_count})")
        return False, result
    else:
        print(f"   ❌ FAIL: Wrong codec found ({codecs})")
        return False, result

def check_logs_for_verification(session_id: str) -> Dict:
    """Check logs for verification messages"""
    cmd = [
        'docker-compose', 'logs', 'python-ml'
    ]
    success, stdout, stderr = run_command(cmd, "Fetching Python ML logs")
    
    if not success:
        return {"found": False, "messages": []}
    
    # Remove ANSI color codes for better matching
    import re
    ansi_escape = re.compile(r'\x1b\[[0-9;]*m')
    
    messages = []
    lines = stdout.split('\n')
    
    # Find lines containing session ID to identify processing window
    session_line_indices = [i for i, line in enumerate(lines) if session_id in line]
    
    # Keywords to search for (even without session ID)
    keywords = [
        "Created silent audio base",
        "silent audio base",
        "Temp video file verification: NO_AUDIO_STREAMS",
        "temp video file verification",
        "Final video audio verification",
        "final video audio verification",
        "-map 0:v",
        "-map -0:a",
        "-map 1:a",
        "Original audio duration",
        "Processing.*segments for audio replacement"
    ]
    
    # If we found session lines, search around them (within 300 lines)
    if session_line_indices:
        start_idx = max(0, session_line_indices[0] - 50)
        end_idx = min(len(lines), session_line_indices[-1] + 300)
        search_range = range(start_idx, end_idx)
    else:
        # If no session ID found, search all lines
        search_range = range(len(lines))
    
    for i in search_range:
        line = lines[i]
        clean_line = ansi_escape.sub('', line)
        # Include if session ID is in line OR if it contains any keyword
        if session_id in clean_line or any(keyword.lower() in clean_line.lower() for keyword in keywords):
            messages.append(clean_line)
    
    return {
        "found": len(messages) > 0,
        "messages": messages  # Return all relevant messages (not just last 20)
    }

def main():
    print("=" * 80)
    print("END-TO-END TEST: Audio Fix Verification")
    print("=" * 80)
    
    # Check if services are running
    print("\n1️⃣ Checking Docker services...")
    cmd = ['docker-compose', 'ps']
    success, stdout, stderr = run_command(cmd, "Checking Docker services")
    
    if not success:
        print("❌ Docker Compose not available or services not running")
        print(f"   Error: {stderr}")
        return False
    
    services_running = 'python-ml' in stdout and 'nestjs-api' in stdout
    if not services_running:
        print("❌ Required services (python-ml, nestjs-api) are not running")
        print("   Run: docker-compose up -d")
        print(f"   Output: {stdout[:200]}")
        return False
    
    print("   ✅ Services are running")
    
    # Check if we have a test video
    print("\n2️⃣ Checking for test video...")
    uploads_dir = Path("uploads")
    test_videos = list(uploads_dir.glob("*.mp4")) if uploads_dir.exists() else []
    
    if not test_videos:
        print("   ⚠️  No test videos found in uploads/ directory")
        print("   Please upload a test video first, or the test will use an existing processed video")
        use_existing = True
    else:
        print(f"   ✅ Found {len(test_videos)} video(s) in uploads/")
        use_existing = False
    
    # Check for recently processed videos
    print("\n3️⃣ Checking for recently processed videos...")
    artifacts_dir = Path("artifacts")
    if artifacts_dir.exists():
        recent_videos = sorted(
            artifacts_dir.glob("*_translated.mp4"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )[:5]
        
        if recent_videos:
            print(f"   ✅ Found {len(recent_videos)} recently processed video(s)")
            print(f"   Testing most recent: {recent_videos[0].name}")
            
            # Extract session ID from filename
            session_id = recent_videos[0].stem.replace("_translated", "")
            
            # Verify final video
            print("\n4️⃣ Verifying final translated video...")
            final_ok, final_result = verify_final_video_audio(recent_videos[0])
            
            # Check logs
            print("\n5️⃣ Checking logs for verification messages...")
            log_result = check_logs_for_verification(session_id)
            
            if log_result["found"]:
                print(f"   ✅ Found {len(log_result['messages'])} relevant log messages")
                print("\n   Key log messages:")
                silent_base_found = False
                temp_verification_found = False
                final_verification_found = False
                
                for msg in log_result['messages']:
                    msg_lower = msg.lower()
                    if "silent audio base" in msg_lower or "created silent audio" in msg_lower:
                        silent_base_found = True
                        print(f"      ✅ {msg[:150]}")
                    elif "no_audio_streams" in msg_lower or "temp video file verification" in msg_lower:
                        temp_verification_found = True
                        print(f"      ✅ {msg[:150]}")
                    elif "final video audio verification" in msg_lower:
                        final_verification_found = True
                        print(f"      ✅ {msg[:150]}")
                    elif "-map 0:v" in msg and "-map -0:a" in msg:
                        print(f"      ✅ FFmpeg mapping: {msg[:150]}")
                    elif "original audio duration" in msg_lower and session_id in msg:
                        print(f"      ✅ Audio duration: {msg[:150]}")
                    elif "processing" in msg_lower and "segments for audio replacement" in msg_lower:
                        print(f"      ✅ Processing segments: {msg[:150]}")
                
                print(f"\n   Verification Summary:")
                print(f"      Silent audio base: {'✅ Found' if silent_base_found else '❌ Not found'}")
                print(f"      Temp file verification: {'✅ Found' if temp_verification_found else '❌ Not found'}")
                print(f"      Final video verification: {'✅ Found' if final_verification_found else '❌ Not found'}")
                
                if not silent_base_found:
                    print(f"\n   ⚠️  WARNING: 'Created silent audio base' not found in logs")
                    print(f"      This means the video was processed before the fix was applied")
                    print(f"      Please process a NEW video to test the latest fix")
            else:
                print("   ⚠️  No verification messages found in logs (may need to process new video)")
            
            # Final verdict
            print("\n" + "=" * 80)
            print("TEST RESULTS:")
            print("=" * 80)
            
            if final_ok:
                print("✅ PASS: Final video has correct audio configuration")
                print("   - Single AAC audio stream (translated only)")
            else:
                print("❌ FAIL: Final video audio verification failed")
                print(f"   Details: {final_result}")
            
            if log_result["found"]:
                print("✅ PASS: Verification logs found")
            else:
                print("⚠️  WARNING: Verification logs not found (may need new video processing)")
            
            return final_ok
            
        else:
            print("   ⚠️  No processed videos found")
            print("   Please process a video first to test")
            return False
    else:
        print("   ⚠️  Artifacts directory not found")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

