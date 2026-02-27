import asyncio
import concurrent.futures
import logging

import av
from aiortc import VideoStreamTrack
from av import VideoFrame

logger = logging.getLogger(__name__)


class WebRTCVideoTrack(VideoStreamTrack):
    """WebRTC 피어 연결마다 생성되는 비디오 트랙.

    broadcast_async()에서 디코딩된 VideoFrame을 받아 aiortc에 공급한다.
    aiortc가 이 프레임을 H.264로 인코딩해서 브라우저로 전송한다.
    """

    kind = 'video'

    def __init__(self):
        super().__init__()
        self._queue: asyncio.Queue[VideoFrame] = asyncio.Queue(maxsize=5)

    def push_frame(self, frame: VideoFrame) -> None:
        """이벤트 루프에서 호출 — QueueFull이면 가장 오래된 프레임 드롭."""
        try:
            self._queue.put_nowait(frame)
        except asyncio.QueueFull:
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(frame)
            except Exception:
                pass

    async def recv(self) -> VideoFrame:
        """aiortc 내부에서 호출 — 다음 프레임 반환."""
        frame = await self._queue.get()
        pts, time_base = await self.next_timestamp()
        out = frame.reformat(format='yuv420p')
        out.pts = pts
        out.time_base = time_base
        return out


class VideoRelay:
    """차량에서 수신한 H.265 NAL 유닛을 디코딩해 WebRTC 트랙들로 팬아웃."""

    def __init__(self):
        self._tracks: set[WebRTCVideoTrack] = set()
        self._codec: av.CodecContext | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._executor: concurrent.futures.ThreadPoolExecutor | None = None

    def init(self, loop: asyncio.AbstractEventLoop) -> None:
        """서버 시작 시 한 번 호출."""
        self._loop = loop
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix='h265dec'
        )
        try:
            self._codec = av.CodecContext.create('hevc', 'r')
            logger.info('H.265 디코더 초기화 완료')
        except Exception as e:
            logger.error(f'H.265 디코더 초기화 실패: {e}')

    # ── 트랙 관리 ──────────────────────────────────────────────────────────────

    def create_track(self) -> WebRTCVideoTrack:
        track = WebRTCVideoTrack()
        self._tracks.add(track)
        logger.info(f'WebRTC 트랙 생성 (활성: {len(self._tracks)})')
        return track

    def remove_track(self, track: WebRTCVideoTrack) -> None:
        self._tracks.discard(track)
        logger.info(f'WebRTC 트랙 제거 (활성: {len(self._tracks)})')

    # ── 디코딩 + 팬아웃 ────────────────────────────────────────────────────────

    def _decode_sync(self, data: bytes) -> list[VideoFrame]:
        """executor 스레드에서 실행되는 동기 H.265 디코딩."""
        if self._codec is None:
            return []
        try:
            packet = av.Packet(data)
            return self._codec.decode(packet)
        except Exception as e:
            logger.debug(f'H.265 디코딩 오류: {e}')
            return []

    async def broadcast_async(self, data: bytes) -> None:
        """Zenoh 카메라 콜백에서 run_coroutine_threadsafe로 호출."""
        if not data or not self._tracks or self._loop is None:
            return
        frames = await self._loop.run_in_executor(
            self._executor, self._decode_sync, data
        )
        for frame in frames:
            for track in list(self._tracks):
                track.push_frame(frame)

    async def cleanup(self) -> None:
        self._tracks.clear()
        if self._executor:
            self._executor.shutdown(wait=False)


video_relay = VideoRelay()
