# js_viewer — browser viewer for kotekan power streams

A live browser viewer for kotekan's `networkPowerStream` output. It currently
ships airspy autocorr and crosscorr UIs, but the front-end is built from a
server-sent `viewer_config`, so it can be pointed at other power-stream
pipelines with mostly server-side changes.

## Pieces

- **`livebeam_server.py`** — the bridge. Accepts kotekan's `networkPowerStream`
  TCP connection, averages frames down to a browser-friendly rate, serves the
  static viewer over HTTP, and pushes frames to browsers over a WebSocket. It
  also emits the `viewer_config` that tells the page which UI to build.
- **`app/`** — ES6 front-end modules: orchestrator (`app.js`), WS client
  (`socket.js`), waterfall / spectrum / crosscorr views, GridStack layout, and
  the control panels under `app/panels/`.
- **`lib/`** — vendored single-file libraries loaded as globals
  (`img_plotting.js` colormaps, `polyfit.js`).
- **`index.html`** — page shell; pulls third-party libs (jQuery/UI, d3 v3,
  Plotly, underscore, GridStack) from CDNs by default.
- **`ccera_rest.py`** — optional CCERA telescope-pointing helper (separate
  process; see `--ccera-pointing`).
- **`notebooks/`** — read back recorded `.dat` files.

## Running

Usually the viewer is launched as a kotekan child via the `SpawnProcess` stage,
already wired into the airspy configs:

```yaml
spawn_pyviewer:
    kotekan_stage: SpawnProcess
    in_buf: post_corr_buf
    exec: '${VIEWER_PYTHON:-python3} ../../python/scripts/js_viewer/livebeam_server.py'
```

Point `VIEWER_PYTHON` at an interpreter that has the viewer deps installed:

```sh
pip install -r python/scripts/js_viewer/requirements.txt
VIEWER_PYTHON=/path/to/venv/bin/python ./kotekan -c config/airspy_crosscorr.yaml
```

Then open <http://HOST:8080/>.

To run it standalone, start it **before** kotekan connects (it blocks on
`accept`):

```sh
cd python/scripts/js_viewer
python livebeam_server.py                              # autocorr
python livebeam_server.py --lag-align-stage lag_align  # crosscorr extras
```

## Ports (all configurable)

| Port  | Direction | Flag                   |
|-------|-----------|------------------------|
| 23401 | TCP in, from `networkPowerStream` | `--kotekan-port` |
| 8539  | WebSocket out, to browsers        | `--ws-port`      |
| 8080  | HTTP, static viewer + `/mode`     | `--http-port`    |
| 12048 | kotekan REST (browser reads gain/freq/adcstat) | `--kotekan-rest-port` |

The browser calls kotekan's REST server cross-origin, so list the viewer
origin(s) under `/rest_server/cors_allow_origins` in the kotekan config (see the
airspy configs) — otherwise the browser blocks the responses.

## Useful options

- `--viewer-integration-ms` — browser-facing integration window; kotekan frames
  arriving within it are averaged into one.
- `--airspy-stages`, `--no-airspy-controls` — which producer stages expose
  gain/LO controls.
- `--lag-align-stage` — enables the crosscorr lag-align panel (the `AirspyAlign`
  stage name).
- `--ccera-pointing`, `--galaxy-view-url` — optional panels.
- `-w/--launch-browser`, `-v/--verbose`.

## Offline assets

`index.html` loads libraries from CDNs by default. To run without internet,
fetch them once with `./fetch_extfiles.sh` and open the viewer with `?offline=1`
(serves from `extlibs/`).

## The viewer_config seam

On WebSocket open the server sends a `viewer_config` JSON — mode, `nvis`,
`vis_labels`, frequency/color ranges, which optional panels to enable, and where
to reach kotekan's REST endpoints. The front-end builds itself from that, so
re-targeting the viewer at a different dataset is mostly a `build_viewer_config()`
change rather than new JS.
