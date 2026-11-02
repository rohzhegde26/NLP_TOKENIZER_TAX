"""Launch the SPARSE-INDIC Interactive Comparative Tokenizer Demo.

Starts a lightweight local HTTP server and automatically opens the demo in the default browser.
"""
from __future__ import annotations

import http.server
import os
import socket
import socketserver
import sys
import webbrowser
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = REPO_ROOT / "demo"


def find_free_port(start_port: int = 8000, max_port: int = 8090) -> int:
    for port in range(start_port, max_port + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return start_port


def main() -> None:
    os.chdir(str(DEMO_DIR))
    port = find_free_port(8000)
    url = f"http://127.0.0.1:{port}/index.html"

    handler = http.server.SimpleHTTPRequestHandler
    # Add proper MIME types
    handler.extensions_map.update({
        ".json": "application/json",
        ".png": "image/png",
        ".svg": "image/svg+xml",
        ".css": "text/css",
        ".js": "application/javascript",
    })

    print("=" * 75)
    print("  SPARSE-INDIC: Interactive Comparative Tokenizer Playground")
    print(f"  Local Web Server: {url}")
    print(f"  Serving Directory: {DEMO_DIR}")
    print("=" * 75)
    print("Press Ctrl+C to terminate the local server.\n")

    webbrowser.open(url)

    with socketserver.TCPServer(("127.0.0.1", port), handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down demo server.")


if __name__ == "__main__":
    main()
