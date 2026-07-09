# ARO live viewer — configuration & operation

How to run the dual-pol / pulsar-fold browser viewer in the ARO operational
topology, where work is split off the archiver.

## Data flow

```
   NODE 1: archiver (squirrel, 10.70.1.1)      NODE 2: receiver (peachfuzz,      BROWSER (anywhere
   ------------------------------------        10.70.1.4)                        that can reach node 2)
   FPGAs ──DPDK──▶ kotekan                      ------------------------          ----------------------
                    ├─ nDiskFileWrite ──▶ disk  livebeam_server.py               http://<node2>:8080/
                    ├─ computeDualpolPower       ├─ listens :PORT (TCP in)  ◀───  HTML + JS + psrcat
                    └─ networkPowerStream ──TCP──▶  from kotekan             ──▶  ws://<node2>:8539
                       (connects OUT to           ├─ HTTP  :8080 (serves viewer)   (frames + fold ctrl)
                        node2:PORT)               └─ WS    :8539 (frames, fold)
```

Node 2 is currently **peachfuzz (10.70.1.4)** — already wired in the recording
config as `stream_peachfuzz → 10.70.1.4:2056`. Any LAN host works (boris
10.70.1.2, peabody 10.70.1.5, cloyd 10.70.1.6, …); just match the IP/port.

Three independent processes, no shared filesystem:

- **kotekan** (node 1) captures from the FPGAs, writes raw to disk, computes
  dual-pol power, and its `networkPowerStream` stage **connects out** over TCP
  to `destination_ip:destination_port`. It is a TCP *client*; it retries the
  connection if the far side isn't up yet, and reconnects if it drops.
- **livebeam_server.py** (node 2) **listens** for that TCP connection, converts
  the power stream to WebSocket frames, and serves the static viewer (HTML/JS
  and `psrcat_b.json`) over HTTP.
- **the browser** talks only to node 2: it loads the page over HTTP and opens a
  WebSocket back to node 2 for the live frames and for fold control. It never
  connects to the archiver.

## Q: is kotekan-on-node-1 / python-on-node-2 / browser-remote supported?

**Yes.** Nothing ties the pieces to one host:

- `networkPowerStream` already supports remote destinations — the production
  `aro_record_46m_prime.yaml` streams to `10.70.1.4:2056` (peachfuzz) alongside
  the local ones, so remote reception is the existing pattern.
- The dual-pol viewer makes **no cross-origin calls**. The airspy viewer poked
  kotekan's REST server for gain/tuning (hence its CORS block); the ARO viewer
  has none of that — fold control goes over the same WebSocket to livebeam, and
  the catalogue/JS are served by livebeam. So the browser only needs to reach
  node 2, and there are no CORS knobs to set.
- The only requirements are: (a) the TCP `PORT` is open from node 1 to node 2,
  (b) livebeam binds an interface node 1 can reach (default `0.0.0.0`), and
  (c) the browser can reach node 2's HTTP + WS ports (directly on the LAN, or
  via an SSH tunnel).

## Node 1 (archiver): kotekan config

In the kotekan config's `power_stream:` block, a `networkPowerStream` points at
node 2. peachfuzz is already there in the recording config — this stanza is what
you receive on node 2:

```yaml
power_stream:
   freq: 600.0
   sample_bw: -400.0          # negative: 2nd-Nyquist band, 800->400 MHz
   destination_protocol: TCP
   stream_peachfuzz:
      kotekan_stage: networkPowerStream
      in_buf: power_buffer
      destination_port: 2056
      destination_ip: 10.70.1.4   # <- node 2; change IP to retarget
```

You can keep multiple `networkPowerStream` stanzas (one per consumer); each
connects independently. For a **monitor-only** run (no disk recording) use
`aro_monitor.yaml` (production config minus the `nDiskFileWrite` stage). For a
**full** run, the recording config already carries remote stream slots — just
set one `destination_ip` to your node 2.

The power stream is `num_elements: 2` (XX/YY), `num_freq: 1024`, uint32 integer
power sums — this is what makes the viewer run in dual-pol mode.

## Node 2 (receiver): run livebeam_server

One-time setup — put the viewer + a venv on node 2 (mirrors `~arofrb/kv/` on
squirrel):

```sh
rsync -a python/scripts/js_viewer/  <node2>:~arofrb/kv/js_viewer/
ssh <node2> 'python3 -m venv ~arofrb/kv/venv-viewer &&
             ~arofrb/kv/venv-viewer/bin/pip install -r ~arofrb/kv/js_viewer/requirements.txt'
```

Run it (listening for kotekan on `PORT`, serving the browser on 8080/8539):

```sh
~arofrb/kv/venv-viewer/bin/python ~arofrb/kv/js_viewer/livebeam_server.py \
    --power-dtype uint32 \        # ARO computeDualpolPower emits uint32 sums
    --sum-freq 4 \               # sum 4 adjacent channels: 1024 -> 256
    --kotekan-port 2056 \        # == the config's destination_port
    --http-port 8080 --ws-port 8539
```

- `--kotekan-host` defaults to `0.0.0.0`, so it accepts the connection coming
  from node 1's IP. Keep it 0.0.0.0 (not 127.0.0.1) for the cross-machine case.
- `--sum-freq` is the frequency-downsample factor; set it in the yaml when the
  viewer is launched via a `SpawnProcess` stage, or on the CLI here.
- Order doesn't matter: start livebeam and kotekan in either order — whichever
  comes up second, kotekan's retry loop connects them.

The `run_viewer.sh` helper does the kill-and-relaunch; edit its `--kotekan-port`
to match the config. `run_monitor.sh` (start the monitor kotekan as a systemd
unit) stays on node 1.

## Browser

- **On the ARO LAN:** open `http://<node2>:8080/` — the page auto-opens
  `ws://<node2>:8539`, no extra config.
- **Remote / off-LAN:** SSH-tunnel node 2's two ports and browse localhost:

  ```sh
  ssh -L 8080:localhost:8080 -L 8539:localhost:8539 <node2>
  # then open http://localhost:8080/
  ```

  If your local ports differ from 8539, pass it on the URL:
  `http://localhost:<httpport>/?ws=<wsport>` (and `&wshost=<host>` to point the
  WebSocket at a different host than the page).

## Ports

| Port | Where       | Direction                         | Flag             |
|------|-------------|-----------------------------------|------------------|
| PORT | node 2 in   | kotekan `networkPowerStream` → livebeam (TCP) | `--kotekan-port` / `destination_port` |
| 8080 | node 2 out  | HTTP: viewer HTML/JS/psrcat       | `--http-port`    |
| 8539 | node 2 out  | WebSocket: frames + fold control  | `--ws-port`      |

## Where things live

- **Repo (`kv/aro`)**, `python/scripts/js_viewer/`: `livebeam_server.py`, `app/`,
  `psrcat_b.json`, `tools/fake_aro_stream.py` (offline synthetic producer), this
  doc. This is the source of truth — deploy to node 2 by rsync.
- **Node 1 (squirrel) `~arofrb/kv/`**: `aro_monitor.yaml` (monitor kotekan
  config), `run_monitor.sh` (start the monitor as the `aro-monitor` systemd
  unit). The kotekan binary/configs live under `~arofrb/kotekan/`.
- **Node 2 `~arofrb/kv/`**: `js_viewer/` (rsynced), `venv-viewer/`,
  `run_viewer.sh`.

Note: during initial testing everything ran on squirrel (livebeam on
`127.0.0.1:2054`). For production, move livebeam + venv to node 2 and repoint one
`networkPowerStream` at it — nothing else changes.

## Operations cheat-sheet

```sh
# node 1: (re)start the monitor kotekan pipeline (systemd, auto-restart)
ssh squirrel.aro 'bash ~arofrb/kv/run_monitor.sh'
ssh squirrel.aro 'sudo systemctl stop aro-monitor'        # stop
ssh squirrel.aro 'sudo journalctl -u aro-monitor -f'      # logs

# node 2: (re)start the viewer;  arg = sum-freq factor (4 -> 256 ch)
ssh <node2> 'bash ~arofrb/kv/run_viewer.sh 4'
ssh <node2> 'pkill -f livebeam_server'                    # stop
ssh <node2> 'tail -f ~arofrb/kv/viewer.log'               # logs
```

In the browser: **Median subtract** removes the standing bandpass; **Pulse
Fold** → pick a target (or type period/DM), toggle **Dedisperse**, watch the
profile build. **Reset layout** (top-left) snaps cards to defaults.
