"""
Station bridge (서버 측).

nev/station/* 구독 → 스테이션 연결 추적 + nev/gcs/* 중계 → 차량

토픽:
  nev/station/heartbeat         ← 스테이션 keepalive (5 Hz)
  nev/station/teleop            ← 조이스틱 명령  → nev/gcs/teleop
  nev/station/estop             ← E-Stop 버튼   → nev/gcs/estop
  nev/station/cmd_mode          ← 모드 변경      → nev/gcs/cmd_mode
  nev/station/joystick_connected ← 조이스틱 연결 여부
"""
import asyncio
import json
import logging
import time

logger = logging.getLogger(__name__)


class StationBridge:
    def __init__(self, state, loop: asyncio.AbstractEventLoop, vehicle_proto):
        self._state        = state
        self._loop         = loop
        self._proto        = vehicle_proto
        self._subs: list   = []

    def start(self, session) -> None:
        self._subs = [
            session.declare_subscriber('nev/station/heartbeat',
                                       self._on_heartbeat),
            session.declare_subscriber('nev/station/teleop',
                                       self._on_teleop),
            session.declare_subscriber('nev/station/estop',
                                       self._on_estop),
            session.declare_subscriber('nev/station/cmd_mode',
                                       self._on_cmd_mode),
            session.declare_subscriber('nev/station/joystick_connected',
                                       self._on_joystick_connected),
        ]
        logger.info('StationBridge started')

    def stop(self) -> None:
        for sub in self._subs:
            sub.undeclare()

    # ── Zenoh callbacks (zenoh background thread) ─────────────────────────────

    def _on_heartbeat(self, sample):
        self._state.station_last_recv = time.monotonic()
        if not self._state.station_connected:
            self._loop.call_soon_threadsafe(
                self._state.update_station_connected, True
            )

    def _on_teleop(self, sample):
        try:
            data = json.loads(bytes(sample.payload))
            if self._state.station_connected:
                lx = float(data.get('linear_x', 0.0))
                az = float(data.get('angular_z', 0.0))
                self._proto.send_teleop(lx, az)
                self._loop.call_soon_threadsafe(self._update_control, lx, az)
        except Exception as e:
            logger.warning(f'station teleop parse error: {e}')

    def _on_estop(self, sample):
        try:
            data   = json.loads(bytes(sample.payload))
            active = bool(data.get('active', False))
            self._proto.send_estop(active)
            self._loop.call_soon_threadsafe(self._update_estop, active)
            logger.info(f'Station e-stop → {active}')
        except Exception as e:
            logger.warning(f'station estop parse error: {e}')

    def _on_cmd_mode(self, sample):
        try:
            data = json.loads(bytes(sample.payload))
            mode = int(data.get('mode', -1))
            self._proto.send_cmd_mode(mode)
            self._loop.call_soon_threadsafe(self._update_mode, mode)
            logger.info(f'Station cmd_mode → {mode}')
        except Exception as e:
            logger.warning(f'station cmd_mode parse error: {e}')

    def _on_joystick_connected(self, sample):
        try:
            data = json.loads(bytes(sample.payload))
            val  = bool(data.get('connected', False))
            self._loop.call_soon_threadsafe(
                self._state.update_joystick_connected, val
            )
        except Exception as e:
            logger.warning(f'station joystick_connected parse error: {e}')

    # ── asyncio thread state updates ──────────────────────────────────────────

    def _update_control(self, lx: float, az: float):
        self._state.control.linear_x  = lx
        self._state.control.angular_z = az

    def _update_estop(self, active: bool):
        self._state.control.estop = active
        self._state._broadcast_sync()

    def _update_mode(self, mode: int):
        self._state.control.mode = mode
        self._state._broadcast_sync()
