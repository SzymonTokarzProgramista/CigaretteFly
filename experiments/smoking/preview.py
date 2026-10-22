"""Local browser preview; keeps only the newest rendered frame."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock, Thread

PAGE = b'''<!doctype html><meta charset="utf-8"><title>CigaretteFly live</title>
<style>body{background:#181818;color:white;font-family:sans-serif;text-align:center}
img{max-width:100%;height:auto}p{color:#ccc}</style>
<h2>CigaretteFly: podglad symulacji</h2>
<p>Obraz aktualizuje sie podczas obliczen. Symulacja moze dzialac wolniej niz czas rzeczywisty.</p>
<img id="frame" alt="Oczekiwanie na pierwsza klatke...">
<script>const img=document.getElementById('frame');
function refresh(){img.src='/frame.jpg?t='+Date.now()}
img.onload=img.onerror=()=>setTimeout(refresh,100);refresh();</script>'''


class BrowserPreview:
    def __init__(self, port=8765):
        self._frame = None
        self._lock = Lock()
        preview = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == '/':
                    status, kind, body = 200, 'text/html; charset=utf-8', PAGE
                elif self.path.split('?')[0] == '/frame.jpg':
                    with preview._lock:
                        body = preview._frame
                    status, kind = (200, 'image/jpeg') if body else (503, 'text/plain')
                    body = body or b'Waiting for simulation frame'
                else:
                    status, kind, body = 404, 'text/plain', b'Not found'
                self.send_response(status)
                self.send_header('Content-Type', kind)
                self.send_header('Cache-Control', 'no-store')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                try:
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
        self.server.daemon_threads = True
        self.thread = Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f'http://127.0.0.1:{self.server.server_port}/'

    def update(self, frame):
        import cv2
        ok, encoded = cv2.imencode('.jpg', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                                  [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            with self._lock:
                self._frame = encoded.tobytes()

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
