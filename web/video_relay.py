import asyncio
import concurrent.futures
import logging
import time

import av
import cv2

logger = logging.getLogger(__name__)

JPEG_QUALITY = 80


class VideoRelay:

    def __init__(self):
        self._codec:         av.CodecContext | None = None
        self._loop:          asyncio.AbstractEventLoop | None = None
        self._executor:      concurrent.futures.ThreadPoolExecutor | None = None
        self._subscribers:   list[asyncio.Queue] = []
        self._on_decode_ms:  callable | None = None

    def init(self, loop: asyncio.AbstractEventLoop, on_decode_ms=None) -> None:
        self._loop         = loop
        self._on_decode_ms = on_decode_ms
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix='h265dec'
        )
        try:
            self._codec = av.CodecContext.create('hevc', 'r')
            logger.info('H.265 decoder initialized')
        except Exception as e:
            logger.error(f'Failed to initialize H.265 decoder: {e}')

    def add_subscriber(self, q: asyncio.Queue) -> None:
        self._subscribers.append(q)

    def remove_subscriber(self, q: asyncio.Queue) -> None:
        try:
            self._subscribers.remove(q)
        except ValueError:
            pass

    def _decode_to_jpegs(self, data: bytes) -> tuple[list[bytes], float]:
        if self._codec is None:
            return [], 0.0
        t0 = time.perf_counter()
        try:
            frames = self._codec.decode(av.Packet(data))
            result = []
            for frame in frames:
                bgr = frame.to_ndarray(format='bgr24')
                ok, buf = cv2.imencode(
                    '.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
                )
                if ok:
                    result.append(bytes(buf))
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return result, elapsed_ms
        except Exception as e:
            logger.debug(f'H.265 decode/JPEG encode error: {e}')
            return [], 0.0

    async def broadcast_async(self, data: bytes) -> None:
        if not data or not self._subscribers or self._loop is None:
            return
        jpegs, elapsed_ms = await self._loop.run_in_executor(
            self._executor, self._decode_to_jpegs, data
        )
        if self._on_decode_ms and elapsed_ms > 0:
            self._loop.call_soon_threadsafe(self._on_decode_ms, elapsed_ms)
        for jpeg in jpegs:
            for q in list(self._subscribers):
                try:
                    q.put_nowait(jpeg)
                except asyncio.QueueFull:
                    try:
                        q.get_nowait()
                        q.put_nowait(jpeg)
                    except Exception:
                        pass

    async def cleanup(self) -> None:
        self._subscribers.clear()
        if self._executor:
            self._executor.shutdown(wait=False)


video_relay = VideoRelay()
