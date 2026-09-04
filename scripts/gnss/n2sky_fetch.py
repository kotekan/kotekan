#!/usr/bin/env python3
"""Pull a REAL sky voltage frame + the seeds that go with it, for the offline path-A/B compare.

The on-sky A/B was blocked because path A exports no per-element correlations -- only the
elem_sum-calibrated header value. This sidesteps that: grab the actual voltage the tracker
despread (peek_hold on the tap buffer) plus the tracked (cp, doppler) per PRN, then run BOTH
paths over the identical bytes offline (n2skyab). Any difference is then purely path B.

Writes <out>.bin  : raw [hop][chan][elem] 4+4b, exactly as GnssChordVoltageTap emits it
       <out>.json : sample_seq, geometry, and per-PRN {prn, cp, doppler, ref_hop}

usage: n2sky_fetch.py [node] [gpu] [outprefix]
"""
import base64
import json
import sys
import urllib.request

node = sys.argv[1] if len(sys.argv) > 1 else 'cx19'
gpu = sys.argv[2] if len(sys.argv) > 2 else '0'
out = sys.argv[3] if len(sys.argv) > 3 else '/tmp/n2sky'

BASE = f"http://{node}:12049"


def get(path, timeout=30):
    return urllib.request.urlopen(BASE + path, timeout=timeout).read()


# --- the voltage frame. peek_hold keeps the newest full frame, but the endpoint still races
# the consumer, so retry.
frame = None
for _ in range(12):
    raw = get(f"/buffer/gnss{gpu}_volt_buf/frame")
    j = json.loads(raw)
    if 'data' in j:
        frame = j
        break
if frame is None:
    sys.exit("could not get a full frame (is peek_hold set on the tap buffer?)")

data = base64.b64decode(frame['data'])
seq = frame['metadata']['sample_seq']
open(out + '.bin', 'wb').write(data)

# --- geometry from the live config (never assume: n_chan differs per node/GPU)
cfg = json.loads(get("/config"))
tap = cfg[f"gnss{gpu}_tap"]
n_chan = len(tap['chan_ids'])
n_elem = tap['n_elements']
fft_len = tap.get('fft_length', 16384)
trk = next(c for c in cfg[f"gnss{gpu}_gpu"]['commands'] if c['name'] == 'cudaGnssChordTrack')

# --- seeds: the combiner's tracked (cp, doppler) plus the hop they belong to. With
# --combine-gpus only gnss0_combine exists and it carries BOTH GPUs' streams.
recs = json.loads(get("/gnss0_combine/get_records"))
sats = []
for s in recs:
    rr = s.get('records') or []
    if not rr:
        continue
    sats.append({'prn': s['prn'], 'cp_chips': s['code_phase_chips'],
                 'doppler_hz': s['doppler_hz'], 'ref_hop': int(max(r[0] for r in rr)),
                 # the live elem_sum'd amplitude at that hop, as a sanity anchor
                 'ref_re': rr[-1][1], 'ref_im': rr[-1][2], 'ref_energy': rr[-1][3]})

meta = {'node': node, 'gpu': int(gpu), 'sample_seq': seq, 'hop0': seq // fft_len,
        'n_hops': len(data) // (n_chan * n_elem), 'n_chan': n_chan, 'n_elem': n_elem,
        'fft_len': fft_len, 'channel_ids': trk['channel_ids'], 'signal': trk['signal'],
        'f_offset_hz': trk['f_offset_hz'], 'sample_rate': trk['sample_rate'],
        'hops_per_record': trk['hops_per_record'], 'conjugate': trk.get('conjugate', False),
        'dll_spacing': trk.get('dll_spacing', 0.5), 'sats': sats}
json.dump(meta, open(out + '.json', 'w'), indent=1)

print(f"wrote {out}.bin ({len(data)} B = {meta['n_hops']} hops x {n_chan} chan x {n_elem} elem)")
print(f"      {out}.json  hop0={meta['hop0']}  signal={meta['signal']}  conj={meta['conjugate']}")
print(f"      {len(sats)} PRNs with records: {[s['prn'] for s in sats]}")
