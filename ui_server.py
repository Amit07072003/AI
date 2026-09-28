import asyncio
import threading
import json
import os
import shutil
import builtins
import aiohttp
from aiohttp import web

PORT = int(os.environ.get("PORT", 8001))
WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")

app_state = {
    "emotion": "neutral",
    "speaking": False,
    "last_text": "Waiting for commands...",
    "thoughts": [],
    "chat_history": [],
    "needs_permission": False,
    "is_listening": False,
    "mic_muted": False,
    "target_rot_x": 0.0,
    "target_rot_y": 0.0,
    "detected_faces_count": 0,
    "active_speaker_label": "STANDBY"
}

input_queue = []
_original_print = builtins.print
_active_ws_clients = set()
_server_loop = None

async def ws_handler(request):
    """Full-Duplex WebSocket Endpoint for Sub-10ms UI synchronization and input."""
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    
    _active_ws_clients.add(ws)
    # Send current state immediately upon connection
    try:
        await ws.send_str(json.dumps({"type": "state", "data": app_state}))
    except Exception:
        pass

    try:
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                try:
                    payload = json.loads(msg.data)
                    msg_type = payload.get("type", "input")
                    if msg_type == "input":
                        user_input = payload.get("text", "").strip()
                        source = payload.get("source", "text").strip().lower()
                        if user_input:
                            input_queue.append({"text": user_input, "source": source})
                    elif msg_type == "command":
                        cmd = payload.get("command", "").strip()
                        if cmd:
                            input_queue.append({"text": cmd, "source": "command"})
                except Exception as e:
                    pass
            elif msg.type == aiohttp.WSMsgType.ERROR:
                pass
    finally:
        _active_ws_clients.discard(ws)
    return ws

async def api_state_handler(request):
    """HTTP GET fallback for state."""
    return web.json_response(app_state, headers={
        'Access-Control-Allow-Origin': '*',
        'Cache-Control': 'no-cache, no-store, must-revalidate'
    })

async def api_input_handler(request):
    """HTTP POST fallback for user input."""
    try:
        data = await request.json()
        user_input = data.get('text', '').strip()
        source = data.get('source', 'text').strip().lower()
        if user_input:
            input_queue.append({"text": user_input, "source": source})
    except Exception:
        pass
    return web.json_response({"status": "ok"}, headers={'Access-Control-Allow-Origin': '*'})

async def video_feed_handler(request):
    """HTTP Multipart MJPEG Live Camera Perception Stream."""
    from vision import get_latest_jpeg
    response = web.StreamResponse(
        status=200,
        reason='OK',
        headers={
            'Content-Type': 'multipart/x-mixed-replace; boundary=frame',
            'Cache-Control': 'no-cache, no-store, must-revalidate',
            'Access-Control-Allow-Origin': '*'
        }
    )
    await response.prepare(request)
    try:
        while True:
            frame = get_latest_jpeg()
            if frame:
                await response.write(b'--frame\r\n')
                await response.write(b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            await asyncio.sleep(0.04)
    except Exception:
        pass
    return response

async def index_handler(request):
    index_file = os.path.join(WEB_DIR, "index.html")
    if os.path.exists(index_file):
        return web.FileResponse(index_file)
    return web.Response(text="Monica Dashboard Web UI not found.", status=404)

def broadcast_ws_state():
    """Broadcasts state update instantly over active WebSockets."""
    if not _active_ws_clients or not _server_loop or not _server_loop.is_running():
        return

    try:
        msg = json.dumps({"type": "state", "data": app_state})
        async def _broadcast():
            dead_clients = set()
            for ws in list(_active_ws_clients):
                if ws.closed:
                    dead_clients.add(ws)
                    continue
                try:
                    await ws.send_str(msg)
                except Exception:
                    dead_clients.add(ws)
            _active_ws_clients.difference_update(dead_clients)

        asyncio.run_coroutine_threadsafe(_broadcast(), _server_loop)
    except Exception:
        pass

def start_ui_server():
    global _server_loop
    if not os.path.exists(WEB_DIR):
        os.makedirs(WEB_DIR)
        
    glb_src = os.path.join(os.path.dirname(WEB_DIR), "charlotte_downs.glb")
    glb_dest = os.path.join(WEB_DIR, "charlotte_downs.glb")
    if os.path.exists(glb_src) and not os.path.exists(glb_dest):
        shutil.copy(glb_src, glb_dest)

    app = web.Application()
    app.router.add_get('/', index_handler)
    app.router.add_get('/ws', ws_handler)
    app.router.add_get('/api/state', api_state_handler)
    app.router.add_post('/api/input', api_input_handler)
    app.router.add_get('/api/video_feed', video_feed_handler)
    
    # Static assets serving
    app.router.add_static('/', path=WEB_DIR, name='static', show_index=True)

    def _run():
        global _server_loop
        import sys
        if sys.platform == 'win32':
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        _server_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_server_loop)
        runner = web.AppRunner(app)
        _server_loop.run_until_complete(runner.setup())
        site = web.TCPSite(runner, '0.0.0.0', PORT)
        _server_loop.run_until_complete(site.start())
        _original_print(f"[UI SERVER] High-Speed WebSocket & Web UI ACTIVE at http://localhost:{PORT}")
        _server_loop.run_forever()

    server_thread = threading.Thread(target=_run, daemon=True)
    server_thread.start()

def update_ui_state(emotion=None, text=None, speaking=None, needs_permission=None, is_listening=None, mic_muted=None, target_rot_x=None, target_rot_y=None, detected_faces_count=None, active_speaker_label=None):
    if emotion is not None: app_state["emotion"] = emotion
    if text is not None: app_state["last_text"] = text
    if speaking is not None: app_state["speaking"] = speaking
    if needs_permission is not None: app_state["needs_permission"] = needs_permission
    if is_listening is not None: app_state["is_listening"] = is_listening
    if mic_muted is not None: app_state["mic_muted"] = mic_muted
    if target_rot_x is not None: app_state["target_rot_x"] = target_rot_x
    if target_rot_y is not None: app_state["target_rot_y"] = target_rot_y
    if detected_faces_count is not None: app_state["detected_faces_count"] = detected_faces_count
    if active_speaker_label is not None: app_state["active_speaker_label"] = active_speaker_label
    broadcast_ws_state()

def log_thought(thought_text):
    _original_print(thought_text)
    app_state["thoughts"].append(thought_text)
    if len(app_state["thoughts"]) > 20: 
        app_state["thoughts"].pop(0)
    broadcast_ws_state()

def log_chat(sender, message, timed_out=False):
    app_state["chat_history"].append({"sender": sender, "message": message, "timed_out": bool(timed_out)})
    if len(app_state["chat_history"]) > 50: 
        app_state["chat_history"].pop(0)
    broadcast_ws_state()

def has_input():
    return len(input_queue) > 0

def get_input():
    if not input_queue:
        return "", "text"
    item = input_queue.pop(0)
    if isinstance(item, dict):
        return item.get("text", ""), item.get("source", "text")
    return str(item), "text"
