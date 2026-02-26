"""
WebRTC video relay for NEV GCS.

Flow:
  Vehicle  ──WebSocket /ws/vehicle──▶  FrameBuffer
  Browser  ──POST /api/webrtc/offer──▶  RTCPeerConnection (aiortc)
                                               │ CameraVideoTrack
                                               ▼
                                        <video> element
"""
import asyncio
import logging
import threading

import av
import cv2
import numpy as np
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import VideoStreamTrack

logger = logging.getLogger(__name__)

# ── Frame buffer ──────────────────────────────────────────────────────────────

class FrameBuffer:
    """Thread-safe store for the latest JPEG frame from the vehicle."""

    def __init__(self):
        self._data: bytes | None = None
        self._lock = threading.Lock()

    def update(self, jpeg_bytes: bytes) -> None:
        with self._lock:
            self._data = jpeg_bytes

    def get_latest(self) -> bytes | None:
        with self._lock:
            return self._data


# ── aiortc video track ────────────────────────────────────────────────────────

_BLANK_RGB = np.zeros((480, 640, 3), dtype=np.uint8)


class CameraVideoTrack(VideoStreamTrack):
    """Pulls JPEG frames from FrameBuffer and delivers them as WebRTC video."""

    kind = "video"

    def __init__(self, buffer: FrameBuffer):
        super().__init__()
        self._buffer = buffer
        self._last_jpeg: bytes | None = None
        self._last_rgb:  np.ndarray | None = None

    async def recv(self) -> av.VideoFrame:
        pts, time_base = await self.next_timestamp()

        jpeg = self._buffer.get_latest()
        rgb  = self._get_rgb(jpeg)

        # av.VideoFrame은 매번 새로 생성 (pts 공유 방지)
        frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
        frame.pts = pts
        frame.time_base = time_base
        return frame

    def _get_rgb(self, jpeg: bytes | None) -> np.ndarray:
        if jpeg is None:
            return _BLANK_RGB

        # 같은 JPEG 객체면 디코딩 생략
        if jpeg is self._last_jpeg and self._last_rgb is not None:
            return self._last_rgb

        arr = np.frombuffer(jpeg, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is not None:
            self._last_rgb  = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            self._last_jpeg = jpeg
            return self._last_rgb

        return _BLANK_RGB


# ── Module-level state ────────────────────────────────────────────────────────

frame_buffer = FrameBuffer()
_pcs: set[RTCPeerConnection] = set()


# ── Public API called by server.py ────────────────────────────────────────────

async def handle_vehicle_frame(data: bytes) -> None:
    """Receive a raw JPEG frame from the vehicle WebSocket."""
    frame_buffer.update(data)


async def handle_webrtc_offer(sdp: str, type_: str) -> dict:
    """
    Exchange SDP with the browser.

    Browser sends an offer → we add a video track and return the answer.
    ICE gathering completes before we return (up to 3 s timeout).
    """
    pc = RTCPeerConnection()
    _pcs.add(pc)

    @pc.on("connectionstatechange")
    async def _on_state():
        logger.info(f"WebRTC state: {pc.connectionState}")
        if pc.connectionState in ("failed", "closed"):
            _pcs.discard(pc)
            await pc.close()

    pc.addTrack(CameraVideoTrack(frame_buffer))

    await pc.setRemoteDescription(RTCSessionDescription(sdp=sdp, type=type_))
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    # Wait for ICE candidates to be gathered
    ice_done = asyncio.Event()

    @pc.on("icegatheringstatechange")
    def _on_ice():
        if pc.iceGatheringState == "complete":
            ice_done.set()

    if pc.iceGatheringState != "complete":
        try:
            await asyncio.wait_for(ice_done.wait(), timeout=3.0)
        except asyncio.TimeoutError:
            logger.warning("ICE gathering timed out — returning partial candidates")

    logger.info(f"WebRTC offer handled, active peers: {len(_pcs)}")
    return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}


async def cleanup() -> None:
    """Close all active PeerConnections (call on shutdown)."""
    for pc in list(_pcs):
        await pc.close()
    _pcs.clear()
