'use strict';

const $ = id => document.getElementById(id);

const MODE_NAMES   = { '-1': 'IDLE', '0': 'CTRL', '1': 'NAV', '2': 'REMOTE' };
const SRC_NAMES    = { '-1': 'NONE', '0': 'NAV',  '1': 'TELEOP' };
const NS_CODES     = { 0: 'OK', 1: 'HB-DELAY', 2: 'SOCK-ERR' };
const BRIDGE_FLAGS = { 0: 'OK', 1: 'SRV-CMD', 2: 'SOCK-ERR', 3: 'HB-TIMEOUT', 4: 'CTRL-TIMEOUT' };
const MUX_FLAGS    = { 0: 'OK', 1: 'NAV+NO-TELEOP' };

const sgn = v => (v >= 0 ? '+' : '') + v.toFixed(2);
const fmtGB = b => (b / 1073741824).toFixed(1);
const fmtRate = b => {
  if (b < 1024)    return `${b.toFixed(0)}B/s`;
  if (b < 1048576) return `${(b / 1024).toFixed(1)}K/s`;
  return `${(b / 1048576).toFixed(1)}M/s`;
};

function kv(key, val, cls = '') {
  return `<div class="kv">
    <span class="k">${key}</span>
    <span class="v${cls ? ' ' + cls : ''}">${val}</span>
  </div>`;
}

function bar(pct) {
  const w   = Math.min(100, Math.max(0, pct || 0));
  const cls = w > 90 ? 'error' : w > 70 ? 'warn' : '';
  return `<div class="bar-wrap"><div class="bar-fill ${cls}" style="width:${w}%"></div></div>`;
}

function dot(on, color = 'green') {
  return `<span class="dot ${on ? color : ''}"></span>`;
}

function textCls(val, warnAt, errorAt) {
  if (val >= errorAt) return 'red';
  if (val >= warnAt)  return 'yellow';
  return '';
}

class CommandCenter {
  constructor() {
    this.ws    = null;
    this.state = null;
    this._reconnectTimer  = null;
    this._videoWs         = null;
    this._videoRetryTimer = null;
    this._bindUI();
    this._startClock();
    this._connect();
    this._startVideo();
  }

  _connect() {
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    this.ws = new WebSocket(`${proto}//${location.host}/ws`);

    this.ws.onopen = () => {
      $('badge-ws').className = 'badge ok';
      clearTimeout(this._reconnectTimer);
    };
    this.ws.onmessage = e => {
      try { this.state = JSON.parse(e.data); this._render(this.state); }
      catch (err) { console.error(err); }
    };
    this.ws.onclose = () => {
      $('badge-ws').className = 'badge error';
      this._reconnectTimer = setTimeout(() => this._connect(), 2000);
    };
    this.ws.onerror = () => this.ws.close();
  }

  _bindUI() {
    document.querySelectorAll('.mode-btn').forEach(btn =>
      btn.addEventListener('click', () =>
        this._post('/api/cmd_mode', { mode: parseInt(btn.dataset.mode) })));

    $('estop-btn').addEventListener('click', () => {
      const active = !(this.state?.control?.estop ?? false);
      this._post('/api/estop', { active });
    });
  }

  async _post(url, body) {
    try {
      const r = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      const d = await r.json();
      if (!d.ok) console.warn(`${url} failed:`, d);
    } catch (e) { console.error(url, e); }
  }

  _render(s) {
    this._renderHeader(s);
    this._renderHunter(s.hunter);
    this._renderMux(s.mux);
    this._renderNetwork(s.network);
    this._renderTwist(s.twist);
    this._renderEstop(s.estop, s.control);
    this._renderJoy(s.control, s.station_connected);
    this._renderResources(s.resources, s.gpu_list);
    this._renderDisk(s.disk_partitions);
    this._renderNetIfaces(s.net_interfaces, s.resources);
    this._renderAlerts(s.alerts);
  }

  _renderHeader(s) {
    const veh = $('badge-veh');
    if      (s.vehicle_age < 0) { veh.textContent = 'VEH';                              veh.className = 'badge'; }
    else if (s.vehicle_age < 2) { veh.textContent = 'VEH';                              veh.className = 'badge ok'; }
    else                        { veh.textContent = `VEH ${s.vehicle_age.toFixed(0)}s`; veh.className = 'badge error'; }

    const stas = $('badge-stas');
    stas.textContent = 'STAS';
    stas.className   = 'badge ' + (s.station_connected ? 'ok' : 'error');

    const joy = $('badge-joy');
    joy.textContent = 'JOY';
    joy.className   = 'badge ' + (s.control?.joystick_connected ? 'ok' : '');

    const rem = $('badge-rem');
    rem.textContent = 'REM';
    rem.className   = 'badge ' + (s.remote_enabled ? 'ok' : '');

    document.querySelectorAll('.mode-btn').forEach(btn => {
      btn.classList.toggle('active', parseInt(btn.dataset.mode) === s.mux.requested_mode);
      btn.classList.toggle('station-offline', !s.station_connected);
    });

    const eb = $('estop-btn');
    eb.innerHTML = s.control.estop ? '&#9632; RELEASE' : '&#9632; E-STOP';
    eb.classList.toggle('triggered', s.control.estop || s.estop.is_estop);
  }

  _renderHunter(hs) {
    const steerDeg = (hs.steering_angle * 180 / Math.PI).toFixed(1);
    const batCls   = textCls(hs.battery_voltage, 22, 20) ||
                     (hs.battery_voltage >= 22 ? 'green' : '');
    const errCls   = hs.error_code !== 0 ? 'red' : '';

    $('hunter-body').innerHTML =
      kv('vel',   `${sgn(hs.linear_vel)} m/s`) +
      kv('steer', `${steerDeg} °`) +
      kv('state', hs.vehicle_state) +
      kv('ctrl',  hs.control_mode) +
      kv('err',   hs.error_code === 0 ? 'NONE' : `0x${hs.error_code.toString(16).toUpperCase()}`, errCls) +
      kv('bat',   `${hs.battery_voltage.toFixed(2)} V`, batCls);
  }

  _renderMux(mx) {
    const modeCls = mx.requested_mode === 2 ? 'blue' : mx.requested_mode === -1 ? 'muted' : '';

    $('mux-body').innerHTML =
      kv('mode',   MODE_NAMES[mx.requested_mode] ?? mx.requested_mode, modeCls) +
      kv('src',    SRC_NAMES[mx.active_source]   ?? mx.active_source) +
      kv('remote', dot(mx.remote_enabled) + (mx.remote_enabled ? 'YES' : 'NO')) +
      kv('nav',    dot(mx.nav_active)    + (mx.nav_active    ? 'ON'  : 'OFF')) +
      kv('teleop', dot(mx.teleop_active) + (mx.teleop_active ? 'ON'  : 'OFF')) +
      kv('final',  dot(mx.final_active)  + (mx.final_active  ? 'ON'  : 'OFF'));
  }

  _renderNetwork(ns) {
    const stCls  = ns.connected ? 'green' : 'red';
    const rttCls = textCls(ns.ht_rtt, 50, 100);

    const estTotal = (ns.encode_delay ?? 0) + (ns.video_net_delay ?? 0) + (ns.decode_delay ?? 0);

    $('network-body').innerHTML =
      kv('status',    dot(ns.connected, ns.connected ? 'green' : 'red') +
                      (NS_CODES[ns.status_code] ?? ns.status_code), stCls) +
      kv('video tx',  ns.bw_video_tx  > 0 ? `${ns.bw_video_tx.toFixed(2)} Mbps`  : '—') +
      kv('video rx',  ns.bw_video_rx  > 0 ? `${ns.bw_video_rx.toFixed(2)} Mbps`  : '—') +
      kv('tele rx',   ns.bw_telemetry > 0 ? `${ns.bw_telemetry.toFixed(2)} Mbps` : '—') +
      '<div class="kv"><span class="k">─────</span><span class="v"></span></div>' +
      kv('enc delay',   ns.encode_delay    > 0 ? `${ns.encode_delay.toFixed(1)} ms`    : '—') +
      kv('net delay',   ns.video_net_delay > 0 ? `${ns.video_net_delay.toFixed(1)} ms` : '—') +
      kv('dec delay',   ns.decode_delay    > 0 ? `${ns.decode_delay.toFixed(1)} ms`    : '—') +
      kv('total delay', estTotal           > 0 ? `${estTotal.toFixed(1)} ms`            : '—') +
      '<div class="kv"><span class="k">─────</span><span class="v"></span></div>' +
      kv('tele delay', ns.tele_delay_ms > 0 ? `${ns.tele_delay_ms.toFixed(1)} ms` : '—') +
      kv('rtt',        `${ns.ht_rtt.toFixed(1)} ms`, rttCls);
  }

  _renderTwist(tv) {
    const row = (lx, az) => `${sgn(lx)} m/s / ${sgn(az)} rad/s`;

    $('twist-body').innerHTML =
      kv('nav',    row(tv.nav_lx,    tv.nav_az)) +
      kv('teleop', row(tv.teleop_lx, tv.teleop_az)) +
      kv('final',  row(tv.final_lx,  tv.final_az));
  }

  _renderEstop(es, ctrl) {
    const active = es.is_estop || ctrl.estop;
    $('card-estop').classList.toggle('estop-active', active);

    const statusVal = active
      ? `<span class="red">&#9888; E-STOP ACTIVE</span>`
      : `<span class="green">NORMAL</span>`;

    $('estop-body').innerHTML =
      `<div class="kv"><span class="k">status</span><span class="v">${statusVal}</span></div>` +
      kv('bridge', BRIDGE_FLAGS[es.bridge_flag] ?? es.bridge_flag, es.bridge_flag !== 0 ? 'red' : '') +
      kv('mux',    MUX_FLAGS[es.mux_flag]   ?? es.mux_flag,   es.mux_flag   !== 0 ? 'yellow' : '');
  }

  _renderJoy(ctrl, stationConnected) {
    const joyCls  = ctrl.joystick_connected ? 'green' : 'red';
    const stasCls = stationConnected ? 'green' : 'red';

    $('joy-body').innerHTML =
      kv('station',  dot(stationConnected, stasCls) +
                     (stationConnected ? 'CONNECTED' : 'OFFLINE'), stasCls) +
      kv('joystick', dot(ctrl.joystick_connected, joyCls) +
                     (ctrl.joystick_connected ? 'CONNECTED' : 'DISCONNECTED'), joyCls) +
      kv('raw',      `${ctrl.raw_speed.toFixed(3)} / ${ctrl.raw_steer.toFixed(3)}`) +
      kv('cmd',      `${ctrl.linear_x.toFixed(3)} m/s / ${ctrl.steer_angle_deg.toFixed(1)} deg / ${ctrl.angular_z.toFixed(4)} rad/s`);
  }

  _renderResources(r, gpuList) {
    let html = '';

    const cpuCls  = textCls(r.cpu_usage, 70, 90);
    const tempCls = textCls(r.cpu_temp,  60, 80);
    html +=
      kv('CPU', `<span class="${cpuCls}">${r.cpu_usage.toFixed(1)}%</span>` +
                `&ensp;<span class="${tempCls}">${r.cpu_temp.toFixed(1)}°C</span>` +
                `&ensp;load ${r.cpu_load.toFixed(2)}`) +
      bar(r.cpu_usage);

    const ramPct = r.ram_total > 0 ? r.ram_used / r.ram_total * 100 : 0;
    html +=
      kv('RAM', `${r.ram_used} / ${r.ram_total} MB`) +
      bar(ramPct);

    if (gpuList && gpuList.length > 0) {
      gpuList.forEach((g, i) => {
        if (!g || Object.keys(g).length === 0) return;
        const usage = g.gpu_usage    ?? 0;
        const temp  = g.gpu_temp     ?? 0;
        const power = g.gpu_power    ?? 0;
        const memU  = g.gpu_mem_used  ?? 0;
        const memT  = g.gpu_mem_total ?? 0;
        const gCls  = textCls(usage, 70, 90);
        const gtCls = textCls(temp, 60, 80);
        html +=
          kv(`GPU${i}`,
            `<span class="${gCls}">${usage.toFixed(1)}%</span>` +
            `&ensp;<span class="${gtCls}">${temp.toFixed(0)}°C</span>` +
            `&ensp;${power.toFixed(0)}W`) +
          bar(usage) +
          kv('', `mem ${Math.round(memU)} / ${Math.round(memT)} MB`);
      });
    }

    $('res-body').innerHTML = html;
  }

  _renderDisk(partitions) {
    if (!partitions || partitions.length === 0) {
      $('disk-body').innerHTML = '<span class="muted">no data</span>';
      return;
    }
    $('disk-body').innerHTML = partitions
      .filter(p => p && p.mountpoint)
      .map(p => {
        const pct = p.percent ?? 0;
        const cls = textCls(pct, 70, 90);
        return (
          kv(p.mountpoint, `${fmtGB(p.used_bytes ?? 0)} / ${fmtGB(p.total_bytes ?? 1)} GB  <span class="${cls}">${pct.toFixed(0)}%</span>`) +
          bar(pct)
        );
      })
      .join('');
  }

  _renderNetIfaces(ifaces, resources) {
    if (!ifaces || ifaces.length === 0) {
      $('netifaces-body').innerHTML = '<span class="muted">no data</span>';
      return;
    }

    let html = '';
    if (resources.net_total_ifaces > 0) {
      html += kv('total', `${resources.net_active_ifaces} up / ${resources.net_total_ifaces}`);
    }

    ifaces.filter(f => f && f.name).forEach(iface => {
      const upCls = iface.is_up ? 'green' : 'red';
      const spd   = iface.speed_mbps > 0 ? `${iface.speed_mbps}M` : '—';
      html +=
        kv(iface.name,
          `${dot(iface.is_up, upCls)}<span class="${upCls}">${iface.is_up ? 'UP' : 'DOWN'}</span>` +
          `&ensp;${spd}`) +
        kv('', `&#8595;${fmtRate(iface.in_bps ?? 0)}&ensp;&#8593;${fmtRate(iface.out_bps ?? 0)}`);
    });

    $('netifaces-body').innerHTML = html;
  }

  _renderAlerts(alerts) {
    if (!alerts || alerts.length === 0) {
      $('alerts-body').innerHTML = '<span class="muted">—</span>';
      return;
    }
    $('alerts-body').innerHTML = alerts
      .map(a => `<div class="alert-item alert-${a.level}">&#9650; ${a.message}</div>`)
      .join('');
  }

  _startVideo() {
    clearTimeout(this._videoRetryTimer);

    if (this._videoWs) {
      this._videoWs.onclose = null;
      this._videoWs.close();
      this._videoWs = null;
    }

    const imgEl    = $('video-el');
    const statusEl = $('video-status');

    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const ws    = new WebSocket(`${proto}//${location.host}/ws/video`);
    this._videoWs = ws;

    ws.binaryType = 'blob';

    ws.onmessage = (e) => {
      const url = URL.createObjectURL(e.data);
      imgEl.onload = () => URL.revokeObjectURL(url);
      imgEl.src = url;
      imgEl.style.display = 'block';
      $('video-placeholder').style.display = 'none';
      statusEl.textContent = 'LIVE';
    };

    ws.onclose = () => {
      imgEl.style.display = 'none';
      $('video-placeholder').style.display = '';
      statusEl.textContent = 'NO SIGNAL';
      this._videoRetryTimer = setTimeout(() => this._startVideo(), 3000);
    };

    ws.onerror = () => ws.close();
  }

  _startClock() {
    const tick = () => { $('clock').textContent = new Date().toTimeString().slice(0, 8); };
    tick();
    setInterval(tick, 1000);
  }
}

document.addEventListener('DOMContentLoaded', () => new CommandCenter());
