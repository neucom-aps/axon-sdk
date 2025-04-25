import http.server
import socketserver
import webbrowser
import threading
import json

PORT = 8080

graph_data = {
    "nodes": [
        {"id": "input", "label": "input", "width": 144, "height": 100, "color": "#ffffff"},
        {"id": "first", "label": "first", "width": 160, "height": 100, "color": "#ffffff"},
        {"id": "last", "label": "last", "width": 108, "height": 100, "color": "#ffffff"},
        {"id": "acc", "label": "acc", "width": 168, "height": 100, "color": "#ffffff"},
        {"id": "recall", "label": "recall", "width": 144, "height": 100, "color": "#ffffff"},
        {"id": "output", "label": "output", "width": 121, "height": 100, "color": "#ffffff"},
    ],
    "edges": [
        {"source": "input", "target": "first", "label": "we; Tsyn", "color": "#000000"},
        {"source": "input", "target": "last", "label": "0.5we; Tsyn", "color": "#000000"},
        {"source": "first", "target": "first", "label": "wi; Tsyn", "color": "#000000"},
        {"source": "first", "target": "acc", "label": "wacc; Tsyn+Tmin", "color": "#FF0830"},
        {"source": "last", "target": "acc", "label": "-wacc; Tsyn", "color": "#FF0830"},
        {"source": "recall", "target": "acc", "label": "wacc; Tsyn", "color": "#FF0830"},
        {"source": "recall", "target": "output", "label": "we; 2Tsyn+Tneu", "color": "#000000"},
        {"source": "acc", "target": "output", "label": "we; Tsyn", "color": "#000000"},
    ],
    "groups": [
        {}
        # {
        #     "id": "best_players",
        #     "label": "Best Players",
        #     "nodes": ["kspacey", "swilliams"]
        # }
    ]
    
}

class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/graph_data':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(graph_data).encode())
        else:
            super().do_GET()

def open_browser():
    webbrowser.open(f'http://localhost:{PORT}')

def start_server():
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"Serving HTTP on localhost port {PORT}...")
        httpd.serve_forever()

if __name__ == "__main__":
    # Open the browser in a new thread
    threading.Thread(target=open_browser).start()
    # Start the server
    start_server()
