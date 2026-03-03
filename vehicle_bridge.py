import asyncio
import json
import logging
import struct
import time

import zenoh

from state import SharedState
from web.video_relay import video_relay

_CAMERA_HEADER_BYTES = 10    # struct layout: double(ts 8B) + uint16(encode_ms 2B)
_DISCONNECT_TIMEOUT  = 3.0   # sec — vehicle disconnect threshold
_BW_CALC_INTERVAL    = 1.0   # sec — bandwidth calculation interval
_SEQ_MAX             = 65536 # sequence number max (uint16)

logger = logging.getLogger(__name__)


class VehicleProtocol:

    def __init__(self, state: SharedState, loop: asyncio.AbstractEventLoop):
        self.state   = state
        self._loop   = loop
        self._pubs:  dict = {}
        self._subs:  list = []
        self._seq    = 0
        self._cam_bytes  = 0
        self._tele_bytes = 0
        self._bw_ts      = time.time()

    def start(self, session: zenoh.Session) -> None:
        for key in ('nev/gcs/heartbeat', 'nev/gcs/teleop',
                    'nev/gcs/estop', 'nev/gcs/cmd_mode'):
            self._pubs[key] = session.declare_publisher(key)

        self._subs = [
            session.declare_subscriber('nev/vehicle/mux',         self._on_mux),
            session.declare_subscriber('nev/vehicle/twist',        self._on_twist),
            session.declare_subscriber('nev/vehicle/network',      self._on_network),
            session.declare_subscriber('nev/vehicle/hunter',       self._on_hunter),
            session.declare_subscriber('nev/vehicle/estop',        self._on_estop),
            session.declare_subscriber('nev/vehicle/cpu',          self._on_cpu),
            session.declare_subscriber('nev/vehicle/mem',          self._on_mem),
            session.declare_subscriber('nev/vehicle/gpu',          self._on_gpu),
            session.declare_subscriber('nev/vehicle/disk',         self._on_disk),
            session.declare_subscriber('nev/vehicle/net',          self._on_net),
            session.declare_subscriber('nev/vehicle/camera',       self._on_camera),
            session.declare_subscriber('nev/vehicle/hb_ack',       self._on_hb_ack),
            session.declare_subscriber('nev/vehicle/video_stats',  self._on_video_stats),
        ]
        logger.info('VehicleProtocol started')

    def stop(self) -> None:
        for sub in self._subs:
            sub.undeclare()
        for pub in self._pubs.values():
            pub.undeclare()

    def _call(self, fn, *args):
        self._loop.call_soon_threadsafe(fn, *args)

    def _on_mux(self, sample):
        raw = bytes(sample.payload)
        self._tele_bytes += len(raw)
        data = json.loads(raw)
        ts = data.pop('ts', None)
        tele_delay_ms = (time.time() - ts) * 1000.0 if ts else None
        def _update():
            self.state.update_packet('mux', data)
            self.state.update_remote_enabled(data.get('remote_enabled', False))
            if tele_delay_ms is not None:
                self.state.network.tele_delay_ms = tele_delay_ms
        self._call(_update)

    def _on_twist(self, sample):
        raw = bytes(sample.payload)
        self._tele_bytes += len(raw)
        data = json.loads(raw)
        ts = data.pop('ts', None)
        tele_delay_ms = (time.time() - ts) * 1000.0 if ts else None
        def _update():
            self.state.update_packet('twist', data)
            if tele_delay_ms is not None:
                self.state.network.tele_delay_ms = tele_delay_ms
        self._call(_update)
        self._call(self.state._broadcast_sync)

    def _on_network(self, sample):
        raw = bytes(sample.payload)
        self._tele_bytes += len(raw)
        data = json.loads(raw)
        self._call(self.state.update_packet, 'network', data)

    def _on_hunter(self, sample):
        raw = bytes(sample.payload)
        self._tele_bytes += len(raw)
        data = json.loads(raw)
        ts = data.pop('ts', None)
        tele_delay_ms = (time.time() - ts) * 1000.0 if ts else None
        def _update():
            self.state.update_packet('hunter', data)
            if tele_delay_ms is not None:
                self.state.network.tele_delay_ms = tele_delay_ms
        self._call(_update)

    def _on_estop(self, sample):
        raw = bytes(sample.payload)
        self._tele_bytes += len(raw)
        data = json.loads(raw)
        ts = data.pop('ts', None)
        tele_delay_ms = (time.time() - ts) * 1000.0 if ts else None
        def _update():
            self.state.update_packet('estop', data)
            if tele_delay_ms is not None:
                self.state.network.tele_delay_ms = tele_delay_ms
        self._call(_update)

    def _on_cpu(self, sample):
        raw = bytes(sample.payload)
        self._tele_bytes += len(raw)
        self._call(self.state.update_packet, 'resources', json.loads(raw))

    def _on_mem(self, sample):
        raw = bytes(sample.payload)
        self._tele_bytes += len(raw)
        self._call(self.state.update_packet, 'resources', json.loads(raw))

    def _on_gpu(self, sample):
        raw = bytes(sample.payload)
        self._tele_bytes += len(raw)
        gpus = json.loads(raw)
        def _update():
            for g in gpus:
                self.state.update_gpu(g['idx'], {
                    'gpu_usage':     g['gpu_usage'],
                    'gpu_mem_used':  g['gpu_mem_used'],
                    'gpu_mem_total': g['gpu_mem_total'],
                    'gpu_temp':      g['gpu_temp'],
                    'gpu_power':     g['gpu_power'],
                })
        self._call(_update)

    def _on_disk(self, sample):
        raw = bytes(sample.payload)
        self._tele_bytes += len(raw)
        data = json.loads(raw)
        def _update():
            for p in data.get('partitions', []):
                self.state.update_disk_partition(p['idx'], {
                    'mountpoint':  p['mountpoint'],
                    'total_bytes': p['total_bytes'],
                    'used_bytes':  p['used_bytes'],
                    'percent':     p['percent'],
                    'accessible':  p['accessible'],
                })
        self._call(_update)

    def _on_camera(self, sample):
        raw = bytes(sample.payload)
        if len(raw) <= _CAMERA_HEADER_BYTES:
            return
        ts, encode_ms = struct.unpack_from('dH', raw, 0)
        nal = raw[_CAMERA_HEADER_BYTES:]
        self._cam_bytes += len(nal)
        video_delay_ms = (time.time() - ts) * 1000.0
        def _update_delay():
            self.state.network.video_net_delay = video_delay_ms
        self._call(_update_delay)
        asyncio.run_coroutine_threadsafe(
            video_relay.broadcast_async(nal), self._loop
        )

    def _on_video_stats(self, sample):
        raw = bytes(sample.payload)
        data = json.loads(raw)
        def _update():
            self.state.network.bw_video_tx  = data.get('bw_mbps', 0.0)
            self.state.network.encode_delay = data.get('encode_ms', 0.0)
        self._call(_update)

    def _on_hb_ack(self, sample):
        raw = bytes(sample.payload)
        self._tele_bytes += len(raw)
        data = json.loads(raw)
        ts = data.get('ts', 0.0)
        if ts > 0:
            rtt_ms = max(0.0, (time.time() - ts) * 1000.0)
            def _update():
                self.state.network.ht_rtt = rtt_ms
            self._call(_update)

    def _on_net(self, sample):
        raw = bytes(sample.payload)
        self._tele_bytes += len(raw)
        data = json.loads(raw)
        def _update():
            self.state.update_packet('resources', {
                'net_total_ifaces':  data['net_total_ifaces'],
                'net_active_ifaces': data['net_active_ifaces'],
                'net_down_ifaces':   data['net_down_ifaces'],
            })
            for iface in data.get('interfaces', []):
                self.state.update_net_interface(iface['idx'], {
                    'name':       iface['name'],
                    'is_up':      iface['is_up'],
                    'speed_mbps': iface['speed_mbps'],
                    'in_bps':     iface['in_bps'],
                    'out_bps':    iface['out_bps'],
                })
        self._call(_update)

    def calc_bandwidth(self):
        now = time.time()
        dt  = now - self._bw_ts
        if dt >= _BW_CALC_INTERVAL:
            cam_mbps  = round(self._cam_bytes  * 8 / (dt * 1e6), 3)
            tele_mbps = round(self._tele_bytes * 8 / (dt * 1e6), 3)
            self._cam_bytes  = 0
            self._tele_bytes = 0
            self._bw_ts      = now
            def _update():
                self.state.network.bw_video_rx  = cam_mbps
                self.state.network.bw_telemetry = tele_mbps
            self._call(_update)

    def _next_seq(self) -> int:
        s = self._seq
        self._seq = (self._seq + 1) % _SEQ_MAX
        return s

    def _zput(self, key: str, data: dict) -> None:
        try:
            self._pubs[key].put(json.dumps(data))
        except Exception as e:
            logger.warning(f'zenoh put [{key}]: {e}')

    def send_heartbeat(self):
        self._zput('nev/gcs/heartbeat', {'ts': time.time(), 'seq': self._next_seq()})

    def send_teleop(self, linear_x: float, angular_z: float):
        self._zput('nev/gcs/teleop', {
            'linear_x':  round(linear_x, 3),
            'angular_z': round(angular_z, 3),
            'seq':       self._next_seq(),
        })

    def send_estop(self, activate: bool):
        self._zput('nev/gcs/estop', {'active': activate, 'seq': self._next_seq()})

    def send_cmd_mode(self, mode: int):
        self._zput('nev/gcs/cmd_mode', {'mode': mode, 'seq': self._next_seq()})


async def run_send_loop(state: SharedState, proto: VehicleProtocol, cfg: dict):
    hb_interval        = 1.0 / cfg.get('heartbeat_rate',   5.0)
    push_interval      = cfg.get('state_push_interval', 0.05)  # 20 Hz
    station_timeout    = cfg.get('station_timeout', 2.0)
    disconnect_timeout = _DISCONNECT_TIMEOUT

    last_hb   = 0.0
    last_push = 0.0
    _veh_disconnected = False

    while True:
        now = time.monotonic()

        if state.last_vehicle_recv > 0:
            age = now - state.last_vehicle_recv
            if age > disconnect_timeout and not _veh_disconnected:
                _veh_disconnected = True
                logger.warning('Vehicle disconnected')
            elif age < 1.0 and _veh_disconnected:
                _veh_disconnected = False
                logger.info('Vehicle reconnected')

        if state.station_connected and state.station_last_recv > 0:
            if now - state.station_last_recv > station_timeout:
                logger.warning('Station heartbeat timeout — marking disconnected')
                state.update_station_connected(False)

        proto.calc_bandwidth()

        if now - last_hb >= hb_interval:
            proto.send_heartbeat()
            last_hb = now

        if now - last_push >= push_interval:
            state._validate()
            state._broadcast_sync()
            last_push = now

        await asyncio.sleep(0.01)
