#!/usr/bin/env python3
"""Wi-Fi RSSI helper for Sensing School.

Reads the laptop's current Wi-Fi link metrics from the operating system
and streams them to a browser page over Server-Sent Events. Open
http://localhost:8765 once the helper is running.

Demo: turn on your phone's personal hotspot, connect this laptop to it,
then run this helper, open the page, and walk the phone around the room.

On macOS, RSSI / noise / TX rate / channel / PHY mode update at ~4 Hz via
PyObjC CoreWLAN. MCS index is pulled from `system_profiler` in a slow
background thread (~once every 20 s) because that command takes several
seconds to run.
"""

import sys

if '--help' in sys.argv or '-h' in sys.argv:
    print("""usage: helper.py [--port PORT]

Reads the laptop's current Wi-Fi link metrics from the operating system
and serves them over Server-Sent Events to a browser page at
http://localhost:PORT.

Options:
  --port PORT   Port to listen on (default: 8765)
  --no-browser  Don't auto-open the page in the default browser

Per-platform sources:
  macOS    -- PyObjC CoreWLAN (recommended; pip install pyobjc-framework-CoreWLAN)
              + system_profiler in the background for MCS index
  Linux    -- iw dev <iface> link / station dump
  Windows  -- netsh wlan show interfaces (signal % converted to dBm)
""")
    sys.exit(0)

import os
import re
import json
import time
import queue
import threading
import platform
import subprocess
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PORT = 8765
if '--port' in sys.argv:
    PORT = int(sys.argv[sys.argv.index('--port') + 1])
OPEN_BROWSER = '--no-browser' not in sys.argv

HERE = Path(__file__).resolve().parent
INDEX = HERE / 'index.html'


# ---------- Per-stream rate tables (Mbps) for NSS estimation ----------
# 802.11ax (HE) single-stream rates, long GI (0.8 us). NSS = TxRate / rate.
HE_RATES = {
    20:  [8.6, 17.2, 25.8, 34.4, 51.6, 68.8, 77.4, 86.0, 103.2, 114.7, 129.0, 143.4],
    40:  [17.2, 34.4, 51.6, 68.8, 103.2, 137.6, 154.9, 172.1, 206.5, 229.4, 258.1, 286.8],
    80:  [36.0, 72.1, 108.1, 144.1, 216.2, 288.2, 324.3, 360.3, 432.4, 480.4, 540.4, 600.4],
    160: [72.1, 144.1, 216.2, 288.2, 432.4, 576.5, 648.5, 720.6, 864.7, 960.7, 1080.9, 1201.0],
}
# 802.11ac (VHT) single-stream rates, short GI (0.4 us).
VHT_RATES = {
    20:  [7.2, 14.4, 21.7, 28.9, 43.3, 57.8, 65.0, 72.2, 86.7, None],
    40:  [15.0, 30.0, 45.0, 60.0, 90.0, 120.0, 135.0, 150.0, 180.0, 200.0],
    80:  [32.5, 65.0, 97.5, 130.0, 195.0, 260.0, 292.5, 325.0, 390.0, 433.3],
    160: [65.0, 130.0, 195.0, 260.0, 390.0, 520.0, 585.0, 650.0, 780.0, 866.7],
}
# 802.11n (HT) single-stream rates, short GI (0.4 us).
HT_RATES = {
    20:  [7.2, 14.4, 21.7, 28.9, 43.3, 57.8, 65.0, 72.2],
    40:  [15.0, 30.0, 45.0, 60.0, 90.0, 120.0, 135.0, 150.0],
}


# MCS index -> (modulation, coding rate). Valid for 802.11n MCS 0-7 (and the
# +8/+16/+24 multi-stream variants), 11ac MCS 0-9, 11ax MCS 0-11.
MCS_MOD_CODING = {
    0:  ('BPSK',     '1/2'),
    1:  ('QPSK',     '1/2'),
    2:  ('QPSK',     '3/4'),
    3:  ('16-QAM',   '1/2'),
    4:  ('16-QAM',   '3/4'),
    5:  ('64-QAM',   '2/3'),
    6:  ('64-QAM',   '3/4'),
    7:  ('64-QAM',   '5/6'),
    8:  ('256-QAM',  '3/4'),
    9:  ('256-QAM',  '5/6'),
    10: ('1024-QAM', '3/4'),
    11: ('1024-QAM', '5/6'),
}


def decode_mcs(phy, mcs):
    """Return (modulation, coding) for an MCS index given the PHY mode."""
    if mcs is None:
        return (None, None)
    # 802.11n's MCS encodes both rate and NSS; the modulation/coding repeats every 8.
    idx = mcs % 8 if phy == '802.11n' else mcs
    return MCS_MOD_CODING.get(idx, (None, None))


def estimate_nss(phy, width_mhz, mcs, tx_rate):
    """Return integer NSS or None if it can't be determined cleanly."""
    if phy is None or width_mhz is None or mcs is None or tx_rate is None:
        return None
    table = None
    if phy == '802.11ax':
        table = HE_RATES
    elif phy == '802.11ac':
        table = VHT_RATES
    elif phy == '802.11n':
        # 11n MCS index encodes both rate and NSS already (0-7 = NSS1, 8-15 = NSS2, ...).
        per_stream_mcs = mcs % 8
        derived_nss = (mcs // 8) + 1
        return derived_nss if derived_nss in (1, 2, 3, 4) else None
    if table is None or width_mhz not in table or mcs >= len(table[width_mhz]):
        return None
    base = table[width_mhz][mcs]
    if base is None or base <= 0:
        return None
    raw = tx_rate / base
    nss = round(raw)
    if nss < 1 or nss > 8:
        return None
    if abs(raw - nss) / nss > 0.15:
        return None
    return nss


# ---------- macOS: CoreWLAN fast path + system_profiler slow path ----------

_PHY_MODE = {0: None, 1: '802.11a', 2: '802.11b', 3: '802.11g',
             4: '802.11n', 5: '802.11ac', 6: '802.11ax', 7: '802.11be'}
_WIDTH = {0: None, 1: 20, 2: 40, 3: 80, 4: 160, 5: 320}
_BAND = {0: None, 1: '2.4 GHz', 2: '5 GHz', 3: '6 GHz'}
_SECURITY = {
    0: 'Open', 1: 'WEP', 2: 'WPA Personal', 3: 'WPA Personal Mixed',
    4: 'WPA2 Personal', 5: 'Personal', 6: 'Dynamic WEP',
    7: 'WPA Enterprise', 8: 'WPA Enterprise Mixed', 9: 'WPA2 Enterprise',
    10: 'Enterprise', 11: 'WPA3 Personal', 12: 'WPA3 Enterprise',
    13: 'WPA3 Transition', 14: 'OWE', 15: 'OWE Transition',
}

_corewlan_iface = None
try:
    import CoreWLAN  # type: ignore
    _corewlan_iface = CoreWLAN.CWWiFiClient.sharedWiFiClient().interface()
except Exception:
    _corewlan_iface = None


def read_corewlan():
    if _corewlan_iface is None:
        return None
    i = _corewlan_iface
    with corewlan_lock:
        try:
            rssi = int(i.rssiValue() or 0)
            if rssi == 0:
                return None
            noise = int(i.noiseMeasurement() or 0) or None
            tx = float(i.transmitRate() or 0) or None
            phy = _PHY_MODE.get(int(i.activePHYMode() or 0))
            ssid = i.ssid()
            bssid = i.bssid()
            sec = _SECURITY.get(int(i.security() or 0))
            ch = i.wlanChannel()
            ch_num = int(ch.channelNumber()) if ch else None
            ch_w = _WIDTH.get(int(ch.channelWidth())) if ch else None
            ch_b = _BAND.get(int(ch.channelBand())) if ch else None
            return {
                'rssi': rssi,
                'noise': noise,
                'snr': (rssi - noise) if (rssi is not None and noise is not None) else None,
                'tx_rate_mbps': tx,
                'phy': phy,
                'channel': ch_num,
                'width_mhz': ch_w,
                'band': ch_b,
                'ssid': str(ssid) if ssid else None,
                'bssid': str(bssid) if bssid else None,
                'security': sec,
                'source': 'CoreWLAN',
            }
        except Exception:
            return None


def _run(cmd, timeout=15):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout


def read_system_profiler():
    """Slow (~8 s) but exposes MCS Index. Used to enrich the CoreWLAN data."""
    try:
        out = _run(['system_profiler', 'SPAirPortDataType', '-detailLevel', 'basic'], timeout=20)
    except Exception:
        return None
    block = re.search(r'Current Network Information:\s*\n(.*?)(?:\n\s*\S+:\s*\n|\Z)', out, re.S)
    if not block:
        return None
    text = block.group(1)
    name_m = re.search(r'^\s*([^\n:]+):\s*$', text, re.M)
    rssi_m = re.search(r'Signal\s*/\s*Noise:\s*(-?\d+)\s*dBm\s*/\s*(-?\d+)\s*dBm', text)
    rate_m = re.search(r'Transmit Rate:\s*(\d+)', text)
    mcs_m = re.search(r'MCS Index:\s*(\d+)', text)
    chan_m = re.search(r'Channel:\s*(\d+)\s*\(([^,]+),\s*(\d+)MHz\)', text)
    phy_m = re.search(r'PHY Mode:\s*(\S+)', text)
    sec_m = re.search(r'Security:\s*(.+)', text)
    res = {}
    if name_m: res['ssid_sp'] = name_m.group(1).strip()
    if rssi_m:
        res['rssi_sp'] = int(rssi_m.group(1))
        res['noise_sp'] = int(rssi_m.group(2))
    if rate_m: res['tx_rate_sp'] = float(rate_m.group(1))
    if mcs_m:  res['mcs'] = int(mcs_m.group(1))
    if chan_m:
        res['channel_sp'] = int(chan_m.group(1))
        res['band_sp'] = chan_m.group(2).strip().replace('GHz', ' GHz')
        res['width_sp'] = int(chan_m.group(3))
    if phy_m:  res['phy_sp'] = phy_m.group(1).strip()
    if sec_m:  res['security_sp'] = sec_m.group(1).strip()
    return res or None


# ---------- Scan all nearby APs (macOS only, slow ~6 s per scan) ----------

def scan_corewlan():
    if _corewlan_iface is None:
        return None
    with corewlan_lock:
        try:
            result, err = _corewlan_iface.scanForNetworksWithName_error_(None, None)
        except Exception:
            return None
    if err is not None or result is None:
        return []
    out = []
    for net in result:
        try:
            ch = net.wlanChannel()
            ssid = net.ssid()
            bssid = net.bssid()
            out.append({
                'ssid': str(ssid) if ssid else None,
                'bssid': str(bssid) if bssid else None,
                'rssi': int(net.rssiValue()),
                'noise': int(net.noiseMeasurement() or 0) or None,
                'channel': int(ch.channelNumber()) if ch else None,
                'width_mhz': _WIDTH.get(int(ch.channelWidth())) if ch else None,
                'band': _BAND.get(int(ch.channelBand())) if ch else None,
            })
        except Exception:
            continue
    return out


# ---------- Linux & Windows readers (lighter, single-shot) ----------

def read_linux():
    try:
        ifaces = _run(['iw', 'dev'], timeout=2)
        m = re.search(r'Interface\s+(\S+)', ifaces)
        if not m:
            return None
        iface = m.group(1)
        link = _run(['iw', 'dev', iface, 'link'], timeout=2)
        rssi = re.search(r'signal:\s*(-?\d+)\s*dBm', link)
        ssid = re.search(r'SSID:\s*(.+)', link)
        rate = re.search(r'tx bitrate:\s*([0-9.]+)\s*MBit/s', link)
        bssid = re.search(r'Connected to\s+(\S+)', link)
        freq = re.search(r'freq:\s*(\d+)', link)
        if not rssi:
            return None
        return {
            'rssi': int(rssi.group(1)),
            'noise': None,
            'snr': None,
            'tx_rate_mbps': float(rate.group(1)) if rate else None,
            'channel': None,
            'width_mhz': None,
            'band': ('5 GHz' if freq and int(freq.group(1)) > 4000 else
                     '2.4 GHz' if freq else None),
            'phy': None,
            'ssid': ssid.group(1).strip() if ssid else None,
            'bssid': bssid.group(1).strip() if bssid else None,
            'security': None,
            'source': 'iw',
        }
    except Exception:
        return None


def read_windows():
    try:
        out = _run(['netsh', 'wlan', 'show', 'interfaces'], timeout=2)
        sig = re.search(r'Signal\s*:\s*(\d+)\s*%', out)
        ssid = re.search(r'^\s*SSID\s*:\s*(.+)$', out, re.M)
        bssid = re.search(r'BSSID\s*:\s*([0-9A-Fa-f:]+)', out)
        rate = re.search(r'Receive rate \(Mbps\)\s*:\s*([0-9.]+)', out)
        chan = re.search(r'Channel\s*:\s*(\d+)', out)
        radio = re.search(r'Radio type\s*:\s*(\S+)', out)
        if not sig:
            return None
        pct = int(sig.group(1))
        dbm = -100 + pct / 2.0  # Microsoft's documented linear mapping.
        return {
            'rssi': int(round(dbm)),
            'noise': None,
            'snr': None,
            'tx_rate_mbps': float(rate.group(1)) if rate else None,
            'channel': int(chan.group(1)) if chan else None,
            'width_mhz': None,
            'band': None,
            'phy': radio.group(1) if radio else None,
            'ssid': ssid.group(1).strip() if ssid else None,
            'bssid': bssid.group(1).strip() if bssid else None,
            'security': None,
            'source': 'netsh (estimated dBm)',
        }
    except Exception:
        return None


# ---------- Shared state ----------

state_lock = threading.Lock()
latest = {
    'rssi': None, 'noise': None, 'snr': None,
    'tx_rate_mbps': None,
    'channel': None, 'width_mhz': None, 'band': None,
    'phy': None,
    'ssid': None, 'bssid': None, 'security': None,
    'mcs': None, 'nss': None,
    'modulation': None, 'coding': None,
    'rate_per_stream_mbps': None,
    'source': None, 'ts': None, 'error': None,
}
sp_extras = {}  # last system_profiler result on macOS
subscribers = []
# Scans are now on-demand: an HTTP request to /scan blocks until the scan
# returns. We serialize all CoreWLAN access through a single lock so the
# 4 Hz reads in poll_loop don't run concurrently with the ~6 s scan.
corewlan_lock = threading.Lock()


def _attach_modulation(state):
    mod, coding = decode_mcs(state.get('phy'), state.get('mcs'))
    state['modulation'] = mod
    state['coding'] = coding
    nss = state.get('nss')
    rate = state.get('tx_rate_mbps')
    state['rate_per_stream_mbps'] = round(rate / nss, 1) if (rate and nss) else None
    return state


def merge_macos_state():
    base = read_corewlan()
    if base is None:
        # Fall back to system_profiler-only if CoreWLAN unavailable.
        sp = sp_extras
        if not sp.get('rssi_sp'):
            return None
        snap = {
            'rssi': sp.get('rssi_sp'),
            'noise': sp.get('noise_sp'),
            'snr': (sp['rssi_sp'] - sp['noise_sp']) if sp.get('rssi_sp') is not None and sp.get('noise_sp') is not None else None,
            'tx_rate_mbps': sp.get('tx_rate_sp'),
            'channel': sp.get('channel_sp'),
            'width_mhz': sp.get('width_sp'),
            'band': sp.get('band_sp'),
            'phy': sp.get('phy_sp'),
            'ssid': sp.get('ssid_sp'),
            'bssid': None,
            'security': sp.get('security_sp'),
            'mcs': sp.get('mcs'),
            'nss': estimate_nss(sp.get('phy_sp'), sp.get('width_sp'), sp.get('mcs'), sp.get('tx_rate_sp')),
            'source': 'system_profiler',
        }
        return _attach_modulation(snap)
    # CoreWLAN base + system_profiler enrichment for MCS, SSID.
    sp = sp_extras
    if base.get('ssid') is None and sp.get('ssid_sp'):
        base['ssid'] = sp['ssid_sp']
    base['mcs'] = sp.get('mcs')
    base['nss'] = estimate_nss(base.get('phy'), base.get('width_mhz'), base.get('mcs'), base.get('tx_rate_mbps'))
    return _attach_modulation(base)


def poll_loop():
    sysname = platform.system()
    while True:
        t0 = time.time()
        try:
            if sysname == 'Darwin':
                snap = merge_macos_state()
            elif sysname == 'Linux':
                snap = read_linux()
            elif sysname == 'Windows':
                snap = read_windows()
            else:
                snap = None
        except Exception:
            snap = None

        with state_lock:
            if snap is None:
                latest.update({k: None for k in latest if k != 'error'})
                latest['error'] = 'Could not read Wi-Fi metrics from this OS.'
            else:
                latest.update(snap)
                latest['error'] = None
            latest['ts'] = time.time()
            payload = json.dumps(latest)

        for q in list(subscribers):
            try:
                q.put_nowait(payload)
            except queue.Full:
                pass

        elapsed = time.time() - t0
        if elapsed < 0.25:
            time.sleep(0.25 - elapsed)


def system_profiler_loop():
    """Refresh MCS / SSID / etc. from system_profiler every ~20 s."""
    global sp_extras
    while True:
        try:
            res = read_system_profiler()
            if res:
                sp_extras = res
        except Exception:
            pass
        time.sleep(20)


# ---------- HTTP / SSE ----------

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a, **k):
        pass

    def do_GET(self):
        if self.path in ('/', '/index.html'):
            try:
                body = INDEX.read_bytes()
            except FileNotFoundError:
                self.send_error(500, 'index.html missing next to helper.py')
                return
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path == '/helper.py':
            try:
                body = Path(__file__).read_bytes()
            except Exception:
                self.send_error(500, 'helper.py not readable')
                return
            self.send_response(200)
            self.send_header('Content-Type', 'text/x-python; charset=utf-8')
            self.send_header('Content-Disposition', 'attachment; filename="helper.py"')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path == '/rssi':
            self._serve_sse(subscribers, state_lock, latest)
            return

        if self.path == '/scan':
            t0 = time.time()
            nets = None
            err = None
            try:
                nets = scan_corewlan()
            except Exception as e:
                err = str(e)
            payload = {
                'ts': time.time(),
                'duration_s': round(time.time() - t0, 2),
                'networks': nets if nets is not None else [],
                'error': None if nets is not None else (err or 'Wi-Fi scan failed (Location Services permission?)'),
            }
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        self.send_error(404)

    def _serve_sse(self, sub_list, lock, state_obj):
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('Connection', 'keep-alive')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        q = queue.Queue(maxsize=8)
        sub_list.append(q)
        with lock:
            init = json.dumps(state_obj)
        try:
            self.wfile.write(f'data: {init}\n\n'.encode())
            self.wfile.flush()
            while True:
                msg = q.get()
                self.wfile.write(f'data: {msg}\n\n'.encode())
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            try:
                sub_list.remove(q)
            except ValueError:
                pass


class QuietServer(ThreadingHTTPServer):
    def handle_error(self, request, client_address):
        import sys as _sys
        exc = _sys.exc_info()[1]
        if isinstance(exc, (ConnectionResetError, BrokenPipeError)):
            return  # Browsers routinely drop SSE connections; ignore.
        super().handle_error(request, client_address)


def main():
    threading.Thread(target=poll_loop, daemon=True).start()
    if platform.system() == 'Darwin':
        threading.Thread(target=system_profiler_loop, daemon=True).start()

    srv = QuietServer(('127.0.0.1', PORT), Handler)
    url = f'http://localhost:{PORT}'
    print(f'Wi-Fi RSSI helper running on {platform.system()} -- open {url}')
    if platform.system() == 'Darwin' and _corewlan_iface is None:
        print('  Note: PyObjC CoreWLAN not found, falling back to system_profiler (~8 s per sample).')
        print('        For sub-second updates, run:  pip install pyobjc-framework-CoreWLAN')
    if OPEN_BROWSER:
        # Open after a short delay so the server is accepting connections.
        threading.Timer(0.4, lambda: webbrowser.open(url, new=2)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print()


if __name__ == '__main__':
    main()
