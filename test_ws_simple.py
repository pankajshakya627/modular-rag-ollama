
import asyncio
import json
import websockets
import sys
import urllib.request
import time

def check_http():
    url = "http://127.0.0.1:8000/docs"
    print(f"Checking API availability at {url}...")
    try:
        with urllib.request.urlopen(url) as response:
            if response.status == 200:
                print("✅ API is accessible via HTTP.")
                return True
    except Exception as e:
        print(f"❌ API HTTP check failed: {e}")
        return False
    return False

async def test_websocket():
    # Only proceed if HTTP works
    if not check_http():
        print("⚠️  Aborting WebSocket test because API is not reachable.")
        print("Please ensure the server is running with: python -m src.main --mode api --host 127.0.0.1")
        return

    uri = "ws://127.0.0.1:8000/ws/query"
    try:
        print(f"Connecting to {uri}...")
        async with websockets.connect(uri) as websocket:
            print("✅ Connected to WebSocket!")
            
            query = {"query": "What is Modular RAG?"}
            print(f"Sending query: {query}")
            await websocket.send(json.dumps(query))
            
            print("\nStream:")
            while True:
                response = await websocket.recv()
                data = json.loads(response)
                
                # Pretty print update
                if data.get("type") == "status":
                    stage = data.get("stage", "unknown")
                    details = data.get("details", {})
                    print(f"Status [{stage}]: {details}")
                    
                elif data.get("type") == "result":
                    print(f"\n✅ Final Result:")
                    print(f"Answer: {data.get('answer')[:100]}...")
                    print(f"Confidence: {data.get('confidence')}")
                    print(f"Sources: {len(data.get('sources', []))}")
                    break
                    
                elif data.get("type") == "error":
                    print(f"❌ Server Error: {data.get('message')}")
                    break
                    
    except ConnectionRefusedError:
        print("❌ Could not connect (Connection Refused).")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(test_websocket())
    except KeyboardInterrupt:
        print("\nTest cancelled.")
