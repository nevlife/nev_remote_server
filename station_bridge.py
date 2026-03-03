import asyncio
import json
import logging
import math
import time

logger = logging.getLogger(__name__)

class StationBridge:
    def __init__(self, state, loop: asyncio.AbstractEventLoop, vehicle_proto, cfg: dict = None): # type: ignore
        self._state        = state
        self._loop         = loop
        self._proto        = vehicle_proto
        self._subs: list   = []
        self._wheelbase    = (cfg or {}).get('wheelbase', 0.650)

    def start(self, session) -> None:
        self._subs = [
            session.declare_subscriber('nev/station/heartbeat',         self._on_heartbeat),
            session.declare_subscriber('nev/station/teleop',            self._on_teleop),
            session.declare_subscriber('nev/station/estop',             self._on_estop),
            session.declare_subscriber('nev/station/cmd_mode',          self._on_cmd_mode),
            session.declare_subscriber('nev/station/joystick_connected',self._on_joystick_connected),
        ]
        logger.info('StationBridge started')

    def stop(self) -> None:
        for sub in self._subs:
            sub.undeclare()

    def _on_heartbeat(self, sample):
        self._loop.call_soon_threadsafe(self._recv_heartbeat)

    def _recv_heartbeat(self):
        self._state.station_last_recv = time.monotonic()
        if not self._state.station_connected:
            self._state.update_station_connected(True)

    def _on_teleop(self, sample):
        try:
            data = json.loads(bytes(sample.payload))
            if self._state.station_connected:
                lx    = float(data.get('linear_x',     0.0))
                steer = float(data.get('steer_angle',  0.0))
                rs    = float(data.get('raw_speed',    0.0))
                rt    = float(data.get('raw_steer',    0.0))
                if abs(steer) < 1e-6:
                    az = 0.0
                elif abs(lx) < 0.05:
                    az = steer
                else:
                    az = lx * math.tan(steer) / self._wheelbase
                self._proto.send_teleop(lx, az)
                steer_deg = math.degrees(steer)
                self._loop.call_soon_threadsafe(self._update_control, lx, az, steer_deg, rs, rt)
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

    def _update_control(self, lx: float, az: float, steer_deg: float = 0.0,
                        rs: float = 0.0, rt: float = 0.0):
        self._state.control.linear_x        = lx
        self._state.control.angular_z       = az
        self._state.control.steer_angle_deg = steer_deg
        self._state.control.raw_speed       = rs
        self._state.control.raw_steer       = rt
        self._state._broadcast_sync()

    def _update_estop(self, active: bool):
        self._state.control.estop = active

    def _update_mode(self, mode: int):
        self._state.control.mode = mode
