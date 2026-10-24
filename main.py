import os
import uvicorn

# Root entrypoint compatible with Render default startup: python main.py
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting SkinNova AI server on port {port}...")
    uvicorn.run("backend.main:app", host="0.0.0.0", port=port, workers=1)
