import asyncio
import json
import logging
import time

import zenoh

from state import SharedState
from web.video_relay import video_relay

logger = logging.getLogger(__name__)


class VehicleProtocol:
    """
    Zenoh vehicle bridge (서버 측).

    - nev/vehicle/* 구독 → SharedState 업데이트
    - nev/gcs/heartbeat 발행 (서버가 차량에 keepalive 전송)
    - nev/gcs/teleop / estop / cmd_mode 발행 (StationBridge가 호출)
    """

    def __init__(self, state: SharedState, loop: asyncio.AbstractEventLoop):
        self.state   = state
        self._loop   = loop
        self._pubs:  dict = {}
        self._subs:  list = []
        self._seq    = 0
        self._cam_bytes  = 0
        self._tele_bytes = 0
        self._bw_ts      = time.time()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self, session: zenoh.Session) -> None:
        # Publishers (server → vehicle)
        for key in ('nev/gcs/heartbeat', 'nev/gcs/teleop',
                    'nev/gcs/estop', 'nev/gcs/cmd_mode'):
            self._pubs[key] = session.declare_publisher(key)

        # Subscribers (vehicle → server)
        self._subs = [
            session.declare_subscriber('nev/vehicle/mux',     self._on_mux),
            session.declare_subscriber('nev/vehicle/twist',   self._on_twist),
            session.declare_subscriber('nev/vehicle/network', self._on_network),
            session.declare_subscriber('nev/vehicle/hunter',  self._on_hunter),
            session.declare_subscriber('nev/vehicle/estop',   self._on_estop),
            session.declare_subscriber('nev/vehicle/cpu',     self._on_cpu),
            session.declare_subscriber('nev/vehicle/mem',     self._on_mem),
            session.declare_subscriber('nev/vehicle/gpu',     self._on_gpu),
            session.declare_subscriber('nev/vehicle/disk',    self._on_disk),
            session.declare_subscriber('nev/vehicle/net',     self._on_net),
            session.declare_subscriber('nev/vehicle/camera',  self._on_camera),
            session.declare_subscriber('nev/vehicle/hb_ack',  self._on_hb_ack),
        ]
        logger.info('VehicleProtocol started')

    def stop(self) -> None:
        for sub in self._subs:
            sub.undeclare()
        for pub in self._pubs.values():
            pub.undeclare()

    # ── Thread-safe asyncio bridge ────────────────────────────────────────────

    def _call(self, fn, *args):
        self._loop.call_soon_threadsafe(fn, *args)

    def _call_fn(self, fn):
        self._loop.call_soon_threadsafe(fn)

    # ── Vehicle → server subscribers ──────────────────────────────────────────

    def _count_tele(self, n: int):
        self._tele_bytes += n

    def _on_mux(self, sample):
        raw = bytes(sample.payload)
        self._tele_bytes += len(raw)
        data = json.loads(raw)
        def _update():
            self.state.update_packet('mux', data)
            self.state.update_remote_enabled(data.get('remote_enabled', False))
        self._call_fn(_update)

    def _on_twist(self, sample):
        raw = bytes(sample.payload)
        self._tele_bytes += len(raw)
        self._call(self.state.update_packet, 'twist', json.loads(raw))

    def _on_network(self, sample):
        raw = bytes(sample.payload)
        self._tele_bytes += len(raw)
        self._call(self.state.update_packet, 'network', json.loads(raw))

    def _on_hunter(self, sample):
        raw = bytes(sample.payload)
        self._tele_bytes += len(raw)
        self._call(self.state.update_packet, 'hunter', json.loads(raw))

    def _on_estop(self, sample):
        raw = bytes(sample.payload)
        self._tele_bytes += len(raw)
        self._call(self.state.update_packet, 'estop', json.loads(raw))

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
        self._call_fn(_update)

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
        self._call_fn(_update)

    def _on_camera(self, sample):
        # H.265 NAL 유닛을 수신하여 video_relay로 전달 (서버에서 디코딩)
        h265_data = bytes(sample.payload)
        self._cam_bytes += len(h265_data)
        asyncio.run_coroutine_threadsafe(
            video_relay.broadcast_async(h265_data), self._loop
        )

    def _on_hb_ack(self, sample):
        raw = bytes(sample.payload)
        self._tele_bytes += len(raw)
        data = json.loads(raw)
        ts = data.get('ts', 0.0)
        if ts > 0:
            rtt_ms = max(0.0, (time.time() - ts) * 1000.0)
            def _update():
                self.state.network.rtt_ms = rtt_ms
            self._call_fn(_update)

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
        self._call_fn(_update)

    def calc_bandwidth(self):
        now = time.time()
        dt  = now - self._bw_ts
        if dt >= 1.0:
            cam_mbps  = round(self._cam_bytes  * 8 / (dt * 1e6), 3)
            tele_mbps = round(self._tele_bytes * 8 / (dt * 1e6), 3)
            self._cam_bytes  = 0
            self._tele_bytes = 0
            self._bw_ts      = now
            def _update():
                self.state.network.bw_camera_mbps    = cam_mbps
                self.state.network.bw_telemetry_mbps = tele_mbps
            self._call_fn(_update)

    # ── Server → vehicle publishers ───────────────────────────────────────────

    def _next_seq(self) -> int:
        s = self._seq
        self._seq = (self._seq + 1) % 65536
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


# ── Send loop (asyncio) ───────────────────────────────────────────────────────

async def run_send_loop(state: SharedState, proto: VehicleProtocol, cfg: dict):
    """
    서버 전용 send loop.
    - heartbeat: 차량에 주기적으로 keepalive 전송 (서버 책임)
    - teleop은 StationBridge가 nev/station/teleop 수신 시 즉시 중계
    - state broadcast: UI WebSocket 클라이언트에 주기적 푸시
    """
    hb_interval        = 1.0 / cfg.get('heartbeat_rate',   5.0)
    push_interval      = cfg.get('state_push_interval', 0.05)  # 20 Hz
    station_timeout    = cfg.get('station_timeout', 2.0)
    disconnect_timeout = 3.0

    last_hb   = 0.0
    last_push = 0.0
    _veh_disconnected = False

    while True:
        now = time.monotonic()

        # 차량 연결 모니터링
        if state.last_vehicle_recv > 0:
            age = now - state.last_vehicle_recv
            if age > disconnect_timeout and not _veh_disconnected:
                _veh_disconnected = True
                logger.warning('Vehicle disconnected')
            elif age < 1.0 and _veh_disconnected:
                _veh_disconnected = False
                logger.info('Vehicle reconnected')

        # 스테이션 heartbeat 타임아웃 감지
        if state.station_connected and state.station_last_recv > 0:
            if now - state.station_last_recv > station_timeout:
                logger.warning('Station heartbeat timeout — marking disconnected')
                state.update_station_connected(False)

        # 대역폭 계산 (1초 주기)
        proto.calc_bandwidth()

        # 차량에 heartbeat 전송 (서버 책임)
        if now - last_hb >= hb_interval:
            proto.send_heartbeat()
            last_hb = now

        # UI 브로드캐스트
        if now - last_push >= push_interval:
            state._validate()
            state._broadcast_sync()
            last_push = now

        await asyncio.sleep(0.01)
