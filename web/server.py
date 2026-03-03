import asyncio
import json
import logging
from pathlib import Path

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .video_relay import video_relay


class CmdModeReq(BaseModel):
    mode: int

class EStopReq(BaseModel):
    active: bool

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / 'static'


def create_app(state, proto):
    app = FastAPI(title='NEV Remote Server', docs_url=None, redoc_url=None)

    app.mount('/static', StaticFiles(directory=str(STATIC_DIR)), name='static')

    @app.get('/', response_class=HTMLResponse)
    async def index():
        return (STATIC_DIR / 'index.html').read_text()

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

    @app.websocket('/ws/video')
    async def ws_video(ws: WebSocket):
        await ws.accept()
        queue: asyncio.Queue = asyncio.Queue(maxsize=5)
        video_relay.add_subscriber(queue)
        logger.info(f'Video WebSocket connected: {ws.client}')

        try:
            while True:
                try:
                    jpeg = await asyncio.wait_for(queue.get(), timeout=10.0)
                    await ws.send_bytes(jpeg)
                except asyncio.TimeoutError:
                    continue
        except (WebSocketDisconnect, Exception):
            pass
        finally:
            video_relay.remove_subscriber(queue)
            logger.info(f'Video WebSocket disconnected: {ws.client}')

    return app
