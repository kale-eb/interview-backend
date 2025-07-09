#!/usr/bin/env python3
"""
Test script for the video analysis service
"""

import requests
import base64
import json
import time

def test_video_analysis():
    """Test the video analysis service"""
    
    # Service URL
    base_url = "http://localhost:8002"
    
    # Test health endpoint
    print("Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            print("✅ Health check passed")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Health check failed: {str(e)}")
        return
    
    # Test video analysis endpoint
    print("\nTesting video analysis endpoint...")
    
    # Create a simple test video (you would replace this with actual video data)
    # For now, we'll just test the endpoint structure
    test_data = {
        "session_id": "test_session_001",
        "user_id": "test_user_001",
        "video_data": base64.b64encode(b"test_video_data").decode('utf-8'),
        "video_format": "mp4"
    }
    
    try:
        response = requests.post(
            f"{base_url}/analyze_video",
            json=test_data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Video analysis endpoint working")
            print(f"   Session ID: {result.get('session_id')}")
            print(f"   Total Frames: {result.get('total_frames')}")
            print(f"   Processing Time: {result.get('processing_time', 0):.2f}s")
        else:
            print(f"❌ Video analysis failed: {response.status_code}")
            print(f"   Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Video analysis test failed: {str(e)}")

def test_with_real_video(video_file_path):
    """Test with a real video file"""
    
    base_url = "http://localhost:8002"
    
    print(f"Testing with real video: {video_file_path}")
    
    try:
        # Read video file
        with open(video_file_path, 'rb') as f:
            video_data = f.read()
        
        # Encode to base64
        video_base64 = base64.b64encode(video_data).decode('utf-8')
        
        # Prepare request
        test_data = {
            "session_id": f"real_test_{int(time.time())}",
            "user_id": "test_user_001",
            "video_data": video_base64,
            "video_format": "mp4"
        }
        
        print("Sending video for analysis...")
        start_time = time.time()
        
        response = requests.post(
            f"{base_url}/analyze_video",
            json=test_data,
            timeout=300  # 5 minutes timeout for video processing
        )
        
        processing_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Real video analysis successful!")
            print(f"   Session ID: {result.get('session_id')}")
            print(f"   Total Frames: {result.get('total_frames')}")
            print(f"   Processing Time: {result.get('processing_time', 0):.2f}s")
            print(f"   Overall Rating: {result.get('final_rating', {}).get('overall_rating', 0):.3f}")
            
            # Save result to file
            output_file = f"test_result_{result.get('session_id')}.json"
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"   Results saved to: {output_file}")
            
        else:
            print(f"❌ Real video analysis failed: {response.status_code}")
            print(f"   Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Real video test failed: {str(e)}")

if __name__ == "__main__":
    print("🎬 Video Analysis Service Test")
    print("=" * 40)
    
    # Test basic functionality
    test_video_analysis()
    
    # Uncomment to test with a real video file
    # test_with_real_video("path/to/your/video.mp4")
    
    print("\n✅ Test completed!") 