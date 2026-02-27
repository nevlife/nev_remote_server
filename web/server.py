"""
FastAPI web server for NEV Remote Server.

Endpoints:
  GET  /                      → index.html
  GET  /api/state             → current state snapshot (JSON)
  POST /api/cmd_mode          → send mode change (browser fallback)
  POST /api/estop             → send e-stop (browser, always allowed)
  WS   /ws                    → real-time state push
  POST /api/webrtc/offer      → WebRTC SDP offer/answer
"""
import asyncio
import json
import logging
from pathlib import Path

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from aiortc import RTCPeerConnection, RTCSessionDescription

from .video_relay import video_relay

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / 'static'

# 활성 PeerConnection 추적 (shutdown 시 정리용)
_pcs: set[RTCPeerConnection] = set()


def create_app(state, proto):
    app = FastAPI(title='NEV Remote Server', docs_url=None, redoc_url=None)

    app.mount('/static', StaticFiles(directory=str(STATIC_DIR)), name='static')

    # ------------------------------------------------------------------
    # Pages
    # ------------------------------------------------------------------

    @app.get('/', response_class=HTMLResponse)
    async def index():
        return (STATIC_DIR / 'index.html').read_text()

    # ------------------------------------------------------------------
    # REST
    # ------------------------------------------------------------------

    class CmdModeReq(BaseModel):
        mode: int

    class EStopReq(BaseModel):
        active: bool

    @app.get('/api/state')
    async def get_state():
        return json.loads(state.to_json())

    @app.post('/api/cmd_mode')
    async def set_cmd_mode(req: CmdModeReq):
        if req.mode not in (-1, 0, 1, 2):
            return {'ok': False, 'error': f'invalid mode: {req.mode}'}
        state.control.mode = req.mode
        proto.send_cmd_mode(req.mode)
        logger.info(f'Mode → {req.mode} (browser)')
        return {'ok': True, 'mode': req.mode, 'station_connected': state.station_connected}

    @app.post('/api/estop')
    async def set_estop(req: EStopReq):
        state.control.estop = req.active
        proto.send_estop(req.active)
        logger.info(f'E-stop → {req.active} (browser)')
        return {'ok': True, 'active': req.active}

    # ------------------------------------------------------------------
    # WebSocket — 브라우저 상태 스트림
    # ------------------------------------------------------------------

    @app.websocket('/ws')
    async def ws_endpoint(ws: WebSocket):
        await ws.accept()
        queue: asyncio.Queue = asyncio.Queue(maxsize=20)
        state.add_subscriber(queue)
        logger.info(f'Browser WebSocket connected: {ws.client}')

        try:
            await ws.send_text(state.to_json())
            while True:
                try:
                    data = await asyncio.wait_for(queue.get(), timeout=5.0)
                    await ws.send_text(data)
                except asyncio.TimeoutError:
                    await ws.send_text(state.to_json())
        except (WebSocketDisconnect, Exception):
            pass
        finally:
            state.remove_subscriber(queue)
            logger.info(f'Browser WebSocket disconnected: {ws.client}')

    # ------------------------------------------------------------------
    # WebRTC — H.265 디코딩 후 H.264로 브라우저에 전달
    # ------------------------------------------------------------------

    @app.post('/api/webrtc/offer')
    async def webrtc_offer(request: Request):
        params = await request.json()
        offer = RTCSessionDescription(sdp=params['sdp'], type=params['type'])

        pc = RTCPeerConnection()
        _pcs.add(pc)

        track = video_relay.create_track()
        pc.addTrack(track)

        @pc.on('connectionstatechange')
        async def on_connectionstatechange():
            logger.info(f'WebRTC [{pc.connectionState}]')
            if pc.connectionState in ('failed', 'closed', 'disconnected'):
                video_relay.remove_track(track)
                _pcs.discard(pc)
                await pc.close()

        await pc.setRemoteDescription(offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        return {'sdp': pc.localDescription.sdp, 'type': pc.localDescription.type}

    return app


async def shutdown_webrtc() -> None:
    """서버 종료 시 모든 WebRTC 연결 정리."""
    for pc in list(_pcs):
        await pc.close()
    _pcs.clear()
