"""Local HTTP Server for ClimateGuard AI Chennai Heat Hazard Dashboard."""

import json
import os
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Optional

from src.dashboard.report_loader import HeatHazardReportLoader, DEFAULT_REPORT_PATH

STATIC_DIR = Path(__file__).resolve().parent / "static"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8080


class DashboardHTTPRequestHandler(SimpleHTTPRequestHandler):
    """HTTP Request Handler serving the Dashboard UI and read-only API."""

    report_path: Path = DEFAULT_REPORT_PATH

    def __init__(self, *args, directory=None, **kwargs):
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def do_GET(self):
        """Handle GET requests for static assets and API endpoints."""
        url_path = self.path.split("?")[0]

        if url_path == "/api/report":
            self._serve_api_report()
        elif url_path == "/api/health":
            self._serve_json(200, {"status": "HEALTHY", "service": "ClimateGuard AI Heat Hazard Dashboard"})
        elif url_path in ("/", "/index.html"):
            index_file = STATIC_DIR / "index.html"
            if index_file.exists():
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                self.end_headers()
                with open(index_file, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.send_error(404, "Dashboard UI index.html not found.")
        else:
            # Let standard handler serve static assets if needed
            super().do_GET()

    def _serve_api_report(self):
        """Load and return the forecast report via HeatHazardReportLoader."""
        loader = HeatHazardReportLoader(report_path=self.report_path)
        result = loader.load_report()
        status_code = 200 if result.get("status") == "SUCCESS" else 400
        self._serve_json(status_code, result)

    def _serve_json(self, status_code: int, data: dict):
        """Helper to send JSON response."""
        encoded = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format, *args):
        """Silence standard logging during unit tests unless verbose."""
        if os.environ.get("DASHBOARD_VERBOSE_LOGS") == "1":
            super().log_message(format, *args)


def create_server(host: str = DEFAULT_HOST,
                  port: int = DEFAULT_PORT,
                  report_path: Optional[Path] = None) -> HTTPServer:
    """Factory creating an HTTPServer instance configured for the dashboard."""
    handler_class = DashboardHTTPRequestHandler
    if report_path:
        handler_class.report_path = Path(report_path)
    else:
        handler_class.report_path = DEFAULT_REPORT_PATH

    return HTTPServer((host, port), handler_class)


def run_dashboard_server(host: str = DEFAULT_HOST,
                         port: int = DEFAULT_PORT,
                         report_path: Optional[Path] = None):
    """Run dashboard server indefinitely."""
    server = create_server(host=host, port=port, report_path=report_path)
    print(f"============================================================")
    print(f"ClimateGuard AI — Chennai Heat Hazard Dashboard Server")
    print(f"URL: http://{host}:{port}/")
    print(f"Reading Report: {server.RequestHandlerClass.report_path}")
    print(f"Press Ctrl+C to terminate the local server.")
    print(f"============================================================")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down ClimateGuard AI Dashboard Server...")
    finally:
        server.server_close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ClimateGuard AI Heat Hazard Dashboard")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Host interface (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port to listen on (default: 8080)")
    parser.add_argument("--report", type=str, default=None, help="Custom report path")
    args = parser.parse_args()

    custom_report = Path(args.report) if args.report else None
    run_dashboard_server(host=args.host, port=args.port, report_path=custom_report)
