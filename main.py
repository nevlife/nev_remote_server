#!/usr/bin/env python3
import argparse
import asyncio
import json
import logging
import re
from pathlib import Path

import yaml
import uvicorn
import zenoh

from state import SharedState
from vehicle_bridge import VehicleProtocol, run_send_loop
from station_bridge import StationBridge
from web.server import create_app

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-7s  %(name)s: %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger('main')


def load_config(path: str, overrides: dict) -> dict:
    cfg = {}
    p = Path(path)
    if p.exists():
        cfg = yaml.safe_load(p.read_text()) or {}
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    return cfg


def sync_zenohd_config(locator: str, zenohd_path: str = 'zenohd.json5') -> None:
    """config.yaml 의 zenoh_locator 포트를 zenohd.json5 에 자동 반영."""
    m = re.search(r':(\d+)$', locator)
    if not m:
        logger.warning(f'zenoh_locator 포트 파싱 실패: {locator!r} — zenohd.json5 를 수정하지 않음')
        return
    port = m.group(1)
    Path(zenohd_path).write_text(
        '{\n'
        '  // 이 파일은 main.py 가 config.yaml 의 zenoh_locator 포트로 자동 생성합니다.\n'
        '  // 직접 수정하지 말고 config.yaml 의 zenoh_locator 포트 번호만 바꾸세요.\n'
        '  //\n'
        '  // 실행 방법: zenohd --config zenohd.json5\n'
        '\n'
        '  listen: {\n'
        f'    endpoints: ["tcp/0.0.0.0:{port}"],\n'
        '  },\n'
        '\n'
        '  scouting: {\n'
        '    multicast: {\n'
        '      // WAN 환경에서는 멀티캐스트 비활성화\n'
        '      enabled: false,\n'
        '    },\n'
        '  },\n'
        '}\n'
    )
    logger.info(f'zenohd.json5 → port {port}')


async def run(cfg: dict):
    web_port = cfg.get('web_port', 8080)
    locator  = cfg.get('zenoh_locator', '')

    if locator:
        sync_zenohd_config(locator)

    state = SharedState()
    loop  = asyncio.get_running_loop()

    # Zenoh 세션 생성 (차량 + 스테이션 공용 라우터에 연결)
    zconf = zenoh.Config()
    if locator:
        zconf.insert_json5('connect/endpoints', json.dumps([locator]))
    session = zenoh.open(zconf)
    logger.info(f'Zenoh session opened → {locator or "auto-discovery"}')

    # 차량 브릿지
    vehicle_proto = VehicleProtocol(state, loop)
    vehicle_proto.start(session)

    # 스테이션 브릿지
    station_bridge = StationBridge(state, loop, vehicle_proto)
    station_bridge.start(session)

    app = create_app(state, vehicle_proto)
    uv_cfg = uvicorn.Config(
        app,
        host='0.0.0.0',
        port=web_port,
        log_level='warning',
        loop='none',
    )
    server = uvicorn.Server(uv_cfg)
    logger.info(f'Web  http://0.0.0.0:{web_port}')

    try:
        await asyncio.gather(
            run_send_loop(state, vehicle_proto, cfg),
            server.serve(),
        )
    finally:
        station_bridge.stop()
        vehicle_proto.stop()
        session.close()
        logger.info('Shutdown complete')


def main():
    parser = argparse.ArgumentParser(description='NEV Remote Server')
    parser.add_argument('--config',       default='config.yaml')
    parser.add_argument('--zenoh-locator', default=None)
    parser.add_argument('--web-port',      type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config, {
        'zenoh_locator': args.zenoh_locator,
        'web_port':      args.web_port,
    })

    try:
        asyncio.run(run(cfg))
    except KeyboardInterrupt:
        logger.info('Stopped by user')


if __name__ == '__main__':
    main()
