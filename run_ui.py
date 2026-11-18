from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import os

UI_DIR = Path(__file__).parent / "ui"
os.chdir(UI_DIR)
address = ("127.0.0.1", 8080)
print("PathSnap UI: http://localhost:8080 (Ctrl+C to stop)")
ThreadingHTTPServer(address, SimpleHTTPRequestHandler).serve_forever()
