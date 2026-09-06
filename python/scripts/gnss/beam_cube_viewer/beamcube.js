// Beam-cube viewer. Loads whole days of cube into the browser and does every collapse here,
// so "sum over elements" and "sum over subbands" are one line each instead of a re-export.
//
// ⚠️ THE SUMS ARE OVER ACCUMULATORS, NEVER OVER dB. Each cell carries (n, s1) of LINEAR
// debiased power; a collapse adds s1 and adds n, and 10*log10 happens exactly once at the
// end. Averaging decibels would be wrong in a way that still looks like a beam -- it
// systematically under-weights the bright pixels and flattens the very structure being
// measured. Every accumulation below is therefore linear.

'use strict';

const S = {
  index: null, days: [], cache: new Map(),
  chains: [], sel: new Set(), nsub: 1, nelem: 32,
  elOn: null,            // Set of enabled element indices (the sum AND the scan use it)
  offset: new Map(),     // chain -> dB offset applied when summing/comparing chains
  cmap: 0, grid: true, hover: null,
};
// Elements 4, 5, 12, 13 are dark: their LNAs are broken and are not coming back.
const DARK = [4, 5, 12, 13];
const CMAPS = ['turbo', 'viridis', 'magma'];
const R = 330, CX = 360, CY = 360;   // sky disc, canvas coordinates

// ── healpix ang2pix RING ─────────────────────────────────────────────────────────────────
// A PORT, AND IT IS VERIFIED, NOT TRUSTED. The export ships every pixel's (az, el) centre;
// on load we run this port over those centres and require it to reproduce the pixel index
// for all of them. A wrong pixelisation does not throw -- it draws a perfectly plausible
// beam in the wrong place -- and the Python side of exactly this arithmetic was already
// wrong once (ring2nest, 100% of pixels). So it gets a gate, and the page refuses to draw
// rather than showing a map it cannot vouch for.
function ang2pixRing(nside, theta, phi) {
  const npix = 12 * nside * nside;
  const z = Math.cos(theta), za = Math.abs(z);
  let tt = (phi % (2 * Math.PI)) * (2 / Math.PI);
  if (tt < 0) tt += 4;
  if (za <= 2 / 3) {
    const t1 = nside * (0.5 + tt), t2 = nside * z * 0.75;
    const jp = Math.floor(t1 - t2), jm = Math.floor(t1 + t2);
    const ir = nside + 1 + jp - jm;
    const kshift = 1 - (ir & 1);
    let ip = Math.floor((jp + jm - nside + kshift + 1) / 2);
    ip = ((ip % (4 * nside)) + 4 * nside) % (4 * nside);
    return 2 * nside * (nside - 1) + (ir - 1) * 4 * nside + ip;
  }
  const tp = tt - Math.floor(tt);
  const tmp = nside * Math.sqrt(3 * (1 - za));
  const jp = Math.floor(tp * tmp), jm = Math.floor((1 - tp) * tmp);
  const ir = jp + jm + 1;
  let ip = Math.floor(tt * ir);
  ip = ((ip % (4 * ir)) + 4 * ir) % (4 * ir);
  return z > 0 ? 2 * ir * (ir - 1) + ip : npix - 2 * ir * (ir + 1) + ip;
}
const azelToPix = (nside, az, el) =>
  ang2pixRing(nside, (90 - el) * Math.PI / 180, az * Math.PI / 180);

// ── colour ───────────────────────────────────────────────────────────────────────────────
// Compact anchor-interpolated maps. Perceptual ordering matters more than prettiness here:
// the reader's question is "is this pixel brighter than that one", and a map that is not
// monotone in lightness answers it wrongly.
const ANCHORS = {
  turbo: [[48,18,59],[70,107,227],[36,187,201],[130,231,100],[240,205,55],[246,120,32],[165,24,10]],
  viridis: [[68,1,84],[65,68,135],[42,120,142],[34,168,132],[122,209,81],[253,231,37]],
  magma: [[0,0,4],[59,15,112],[140,41,129],[222,73,104],[254,159,109],[252,253,191]],
};
function ramp(name, x) {
  const a = ANCHORS[name];
  x = Math.max(0, Math.min(0.9999, x));
  const f = x * (a.length - 1), i = Math.floor(f), t = f - i;
  const c0 = a[i], c1 = a[Math.min(a.length - 1, i + 1)];
  return [c0[0] + (c1[0] - c0[0]) * t, c0[1] + (c1[1] - c0[1]) * t, c0[2] + (c1[2] - c0[2]) * t];
}

// ── data ─────────────────────────────────────────────────────────────────────────────────
async function loadDay(day) {
  if (S.cache.has(day)) return S.cache.get(day);
  const man = await (await fetch(`cube_${day}.json`)).json();
  const buf = await (await fetch(`cube_${day}.bin`)).arrayBuffer();
  let off = 0;
  for (const c of man.chains) {
    const P = c.n_pix, N = c.n_sub * c.n_elem * P;
    c.pix = new Int32Array(buf, off, P);           off += 4 * P;
    c.ctr = new Float32Array(buf, off, 2 * P);     off += 8 * P;
    c.n = new Uint32Array(buf, off, N);            off += 4 * N;
    c.s1 = new Float32Array(buf, off, N);          off += 4 * N;
  }
  if (off !== buf.byteLength) {
    throw new Error(`cube_${day}.bin: consumed ${off} of ${buf.byteLength} bytes -- the ` +
      `manifest and the blob disagree, so every array after the first mismatch is ` +
      `misaligned. Re-run gnss_beam_cube.py export.`);
  }
  verifyPixelisation(man);
  S.cache.set(day, man);
  return man;
}

// THE GATE. Reproduce every shipped pixel centre's index with the JS port; any disagreement
// means the port and the exporter do not share a convention and nothing below can be trusted.
function verifyPixelisation(man) {
  for (const c of man.chains) {
    for (let i = 0; i < c.n_pix; i++) {
      const got = azelToPix(man.nside, c.ctr[2 * i], c.ctr[2 * i + 1]);
      if (got !== c.pix[i]) {
        throw new Error(`pixelisation mismatch on ${c.chain}: centre ` +
          `(az ${c.ctr[2 * i].toFixed(3)}, el ${c.ctr[2 * i + 1].toFixed(3)}) -> ${got}, ` +
          `exported as ${c.pix[i]} (nside ${man.nside}). The JS ang2pix port disagrees with ` +
          `healpy; refusing to draw a map that would be plausible and wrong.`);
      }
    }
  }
}

// Collapse the selected days x chains x subbands x elements into per-pixel (n, s1).
// Pure addition, on linear power. Returns a Map pix -> [n, s1].
function collapse() {
  const acc = new Map();
  const subAll = document.getElementById('subsum').checked;
  const elAll = document.getElementById('elsum').checked;
  const subOne = +document.getElementById('sub').value;
  const elOne = +document.getElementById('el').value;
  const [i0, i1] = dayRange();
  const ref = S.cache.get(S.days[i0]);

  for (let d = i0; d <= i1; d++) {
    const man = S.cache.get(S.days[d]);
    if (!man) continue;
    // NEVER SUM TWO CURRENCIES. An elem-archive day is in raw power, a cube day is in
    // pedestal units ((P-F)/F), and a repoint is a different beam altogether. A day whose
    // units or pointing differ from the first selected day is skipped, and syncLabels says so.
    if (!sameCurrency(man, ref)) continue;
    for (const c of man.chains) {
      if (!S.sel.has(c.chain)) continue;
      const P = c.n_pix;
      const s0 = subAll ? 0 : Math.min(subOne, c.n_sub - 1);
      const s1e = subAll ? c.n_sub - 1 : s0;
      const e0 = elAll ? 0 : Math.min(elOne, c.n_elem - 1);
      const e1 = elAll ? c.n_elem - 1 : e0;
      // The per-chain offset brings this chain onto the reference chain's zero (manifest
      // default, reader-tweakable). Applied to the LINEAR accumulator, so the sum over
      // chains is a weighted mean of aligned patterns, not of raw ones.
      const gain = Math.pow(10, -(S.offset.get(c.chain) || 0) / 10);
      for (let s = s0; s <= s1e; s++) {
        for (let e = e0; e <= e1; e++) {
          if (elAll && S.elOn && !S.elOn.has(e)) continue;   // toggled off by the reader
          const base = (s * c.n_elem + e) * P;
          for (let p = 0; p < P; p++) {
            const nn = c.n[base + p];
            if (!nn) continue;                  // absent, not zero power
            const key = c.pix[p];
            const cur = acc.get(key);
            if (cur) { cur[0] += nn; cur[1] += gain * c.s1[base + p]; }
            else acc.set(key, [nn, gain * c.s1[base + p]]);
          }
        }
      }
    }
  }
  return acc;
}

const sameCurrency = (a, b) =>
  (a.units || 'power') === (b.units || 'power') && (a.pointing || null) === (b.pointing || null);

const dayRange = () => {
  const a = +document.getElementById('d0').value, b = +document.getElementById('d1').value;
  return [Math.min(a, b), Math.max(a, b)];
};

// ── render ───────────────────────────────────────────────────────────────────────────────
function draw() {
  const cv = document.getElementById('sky'), g = cv.getContext('2d');
  g.clearRect(0, 0, cv.width, cv.height);
  const man = S.cache.get(S.days[dayRange()[0]]);
  if (!man) return;

  const acc = collapse();
  const minN = +document.getElementById('mn').value;
  // dB once, at the very end, from the summed accumulators.
  const db = new Map();
  let peak = -Infinity;
  for (const [pix, [n, s1]] of acc) {
    if (n < minN || s1 <= 0) continue;
    const v = 10 * Math.log10(s1 / n);
    db.set(pix, v);
    if (v > peak) peak = v;
  }
  const dr = +document.getElementById('dr').value;
  const norm = document.getElementById('peaknorm').checked;
  // Peak-normalise the VALUES, not just the axis labels. Setting the scale to -dr..0 while
  // leaving v in raw dB clips the entire map to one colour -- and because the disc still
  // fills, it reads as "the beam is flat" rather than as a broken scale.
  if (norm && peak > -Infinity)
    for (const [k, v] of db) db.set(k, v - peak);
  const hi = norm ? 0 : peak, lo = hi - dr;

  // Nearest-pixel raster: every screen pixel inside the disc asks which healpix cell it is
  // in. Exact (not a scatter of blobs), and cheap because ang2pix is closed-form.
  const img = g.createImageData(cv.width, cv.height);
  const D = img.data;
  const nside = man.nside;
  for (let y = 0; y < cv.height; y++) {
    for (let x = 0; x < cv.width; x++) {
      const dx = x - CX, dy = y - CY;
      const rr = Math.sqrt(dx * dx + dy * dy);
      if (rr > R) continue;
      // Standard skyplot: zenith centre, horizon rim, North up, azimuth clockwise --
      // the same convention as the live viewer's gps_sky.js, so the two read alike.
      const el = 90 - 90 * (rr / R);
      let az = Math.atan2(dx, -dy) * 180 / Math.PI;
      if (az < 0) az += 360;
      const v = db.get(azelToPix(nside, az, el));
      const o = 4 * (y * cv.width + x);
      if (v === undefined) { D[o] = 26; D[o + 1] = 29; D[o + 2] = 34; D[o + 3] = 255; continue; }
      const c = ramp(CMAPS[S.cmap], (v - lo) / dr);
      D[o] = c[0]; D[o + 1] = c[1]; D[o + 2] = c[2]; D[o + 3] = 255;
    }
  }
  g.putImageData(img, 0, 0);

  if (S.grid) drawGrid(g);
  drawColorbar(lo, hi);

  const [a, b] = dayRange();
  const nDays = b - a + 1;
  document.getElementById('hud').innerHTML =
    `<b>${[...S.sel].join(' + ') || 'no chain selected'}</b><br>` +
    `${nDays} day${nDays > 1 ? 's' : ''} &middot; ${db.size} pixel${db.size === 1 ? '' : 's'} ` +
    `&ge; ${minN} sample${minN === 1 ? '' : 's'}<br>` +
    `peak ${peak === -Infinity ? '—' : peak.toFixed(1) + ' dB'} &middot; ` +
    `range ${dr} dB`;
}

function drawGrid(g) {
  g.save();
  g.strokeStyle = 'rgba(230,233,239,.22)';
  g.fillStyle = 'rgba(139,149,166,.9)';
  g.font = '11px ui-monospace, monospace';
  g.lineWidth = 1;
  for (const el of [0, 30, 60]) {
    const rr = R * (90 - el) / 90;
    g.beginPath(); g.arc(CX, CY, rr, 0, 2 * Math.PI); g.stroke();
    if (el) g.fillText(el + '°', CX + 3, CY - rr + 12);
  }
  for (let az = 0; az < 360; az += 30) {
    const a = az * Math.PI / 180;
    g.beginPath();
    g.moveTo(CX, CY);
    g.lineTo(CX + R * Math.sin(a), CY - R * Math.cos(a));
    g.globalAlpha = 0.11; g.stroke(); g.globalAlpha = 1;
  }
  g.fillStyle = 'rgba(230,233,239,.75)';
  g.font = '12px ui-monospace, monospace';
  for (const [lab, a] of [['N', 0], ['E', 90], ['S', 180], ['W', 270]]) {
    const r = a * Math.PI / 180;
    g.fillText(lab, CX + (R + 13) * Math.sin(r) - 4, CY - (R + 13) * Math.cos(r) + 4);
  }
  // Boresight: az 180, el 81.41 (docs/CHORD_BEAM_MAPS.md). The map should peak here -- it is
  // drawn as a PREDICTION to check the measurement against, never fitted to it.
  const brr = R * (90 - 81.41) / 90, ba = Math.PI;
  const bx = CX + brr * Math.sin(ba), by = CY - brr * Math.cos(ba);
  g.strokeStyle = '#4fd0c7'; g.lineWidth = 1.4;
  g.beginPath(); g.arc(bx, by, 7, 0, 2 * Math.PI); g.stroke();
  g.beginPath(); g.moveTo(bx - 12, by); g.lineTo(bx - 9, by);
  g.moveTo(bx + 9, by); g.lineTo(bx + 12, by); g.stroke();
  g.fillStyle = '#4fd0c7'; g.font = '10px ui-monospace, monospace';
  g.fillText('boresight', bx + 12, by + 15);
  g.restore();
}

function drawColorbar(lo, hi) {
  const cv = document.querySelector('#cbar canvas'), g = cv.getContext('2d');
  const im = g.createImageData(cv.width, cv.height);
  for (let y = 0; y < cv.height; y++) {
    const c = ramp(CMAPS[S.cmap], 1 - y / (cv.height - 1));
    for (let x = 0; x < cv.width; x++) {
      const o = 4 * (y * cv.width + x);
      im.data[o] = c[0]; im.data[o + 1] = c[1]; im.data[o + 2] = c[2]; im.data[o + 3] = 255;
    }
  }
  g.putImageData(im, 0, 0);
  document.getElementById('cbhi').textContent = hi.toFixed(0) + ' dB';
  document.getElementById('cblo').textContent = lo.toFixed(0);
}

// ── chrome ───────────────────────────────────────────────────────────────────────────────
// Band membership, so mixing is flagged rather than silently averaged. Bands sit ~11 dB
// apart in gain, and the cube's dB zero is arbitrary PER CHAIN -- so a sum across bands is
// not a better-sampled beam, it is two beams with different offsets added together.
const BAND = {
  gps_l5: '1176', gal_e5a: '1176', bds_b2a: '1176',
  gal_e5b: '1207', bds_b2b: '1207',
  gps_l2c: '1227', bds_b3i: '1268', gal_e6: '1278',
};

function checkBands() {
  const bands = new Set([...S.sel].map(c => BAND[c] || '?'));
  const el = document.getElementById('bandwarn');
  if (bands.size > 1) {
    el.className = 'note';
    el.innerHTML = `<b>Mixing ${bands.size} bands</b> (${[...bands].join(', ')} MHz). ` +
      `The dB zero is arbitrary <i>per chain</i> and bands differ in gain by ~11 dB, so this ` +
      `sum is two patterns with different zeros added unless the per-chain offsets (dB boxes) ` +
      `align them. The defaults align the main-lobe level; judge the far field with them on ` +
      `and off before trusting a cross-band coadd.`;
  } else el.innerHTML = '';
}

function saveElOn() {
  try { localStorage.setItem('beamcube.elOn', JSON.stringify([...S.elOn])); } catch (e) {}
}

function bindUI() {
  const ids = ['d0', 'd1', 'sub', 'el', 'dr', 'mn', 'subsum', 'elsum', 'peaknorm'];
  for (const id of ids) {
    document.getElementById(id).addEventListener('input', async () => {
      await ensureDaysLoaded();
      syncLabels();
      draw();
    });
  }
  // Scanning single elements SKIPS the toggled-off ones: the slider steps in the direction
  // it was moved until it lands on an enabled element (or stays put if there is none).
  const el = document.getElementById('el');
  let last = +el.value;
  el.addEventListener('input', () => {
    let v = +el.value;
    if (S.elOn && S.elOn.size && !S.elOn.has(v)) {
      const dir = v >= last ? 1 : -1;
      let k = v;
      for (let i = 0; i < S.nelem; i++) { k += dir; if (k < 0 || k >= S.nelem) { k = last; break; } if (S.elOn.has(k)) break; }
      v = k; el.value = v; syncLabels(); draw();
    }
    last = v;
  });
  document.getElementById('cmapbtn').addEventListener('click', e => {
    S.cmap = (S.cmap + 1) % CMAPS.length;
    e.target.textContent = 'colormap: ' + CMAPS[S.cmap];
    draw();
  });
  document.getElementById('gridbtn').addEventListener('click', e => {
    S.grid = !S.grid;
    e.target.classList.toggle('on', S.grid);
    draw();
  });
  const cv = document.getElementById('sky');
  cv.addEventListener('mousemove', ev => {
    const r = cv.getBoundingClientRect();
    const x = (ev.clientX - r.left) * cv.width / r.width - CX;
    const y = (ev.clientY - r.top) * cv.height / r.height - CY;
    const rr = Math.hypot(x, y);
    const out = document.getElementById('readout');
    if (rr > R) { out.textContent = ''; return; }
    const el = 90 - 90 * (rr / R);
    let az = Math.atan2(x, -y) * 180 / Math.PI;
    if (az < 0) az += 360;
    out.textContent = `az ${az.toFixed(1)}°  el ${el.toFixed(1)}°`;
  });
  cv.addEventListener('mouseleave', () => {
    document.getElementById('readout').textContent = '';
  });
}

function syncLabels() {
  const [a, b] = dayRange();
  document.getElementById('d0lab').textContent = S.days[a] || '—';
  document.getElementById('d1lab').textContent = S.days[b] || '—';
  const ref = S.cache.get(S.days[a]);
  let skipped = 0;
  for (let i = a; i <= b; i++) {
    const m = S.cache.get(S.days[i]);
    if (m && ref && !sameCurrency(m, ref)) skipped++;
  }
  document.getElementById('dayinfo').textContent =
    `${b - a + 1} of ${S.days.length} day(s) loaded`
    + (ref ? ` · ${ref.units || 'power'} units · pointing ${ref.pointing || 'unknown'}` : '')
    + (skipped ? ` · ${skipped} day(s) SKIPPED: different units or pointing than ${S.days[a]}` : '');
  const subOn = !document.getElementById('subsum').checked;
  const elOn = !document.getElementById('elsum').checked;
  document.getElementById('sub').disabled = !subOn;
  document.getElementById('el').disabled = !elOn;
  document.getElementById('sublab').textContent =
    subOn ? document.getElementById('sub').value : 'all (' + S.nsub + ')';
  document.getElementById('drlab').textContent = document.getElementById('dr').value + ' dB';
  document.getElementById('mnlab').textContent = document.getElementById('mn').value;
  // Elements 4, 5, 12, 13 are dark: their LNAs are broken and are not coming back. Not a
  // fault to chase -- it is the instrument -- so the viewer says so instead of showing an
  // empty sky and letting the reader diagnose it again.
  const one = +document.getElementById('el').value;
  const nOn = S.elOn ? S.elOn.size : S.nelem;
  document.getElementById('elinfo').textContent =
    elOn
      ? (DARK.includes(one) ? `element ${one} is DARK (broken LNA) — expect an empty map` : '')
      : `${nOn} of ${S.nelem} elements in the sum` + (nOn < S.nelem ? ' (toggle below)' : '');
  document.getElementById('ellab').textContent =
    elOn ? '#' + one : `all (${nOn})`;
  for (const b of document.querySelectorAll('#elgrid button'))
    b.classList.toggle('on', !S.elOn || S.elOn.has(+b.dataset.e));
  const man = S.cache.get(S.days[a]);
  const c = man && man.chains.find(c => S.sel.has(c.chain));
  // Each "subband" is ONE F-engine channel (0.1953125 MHz) of the tracker's sparse comb --
  // 7 channels 16 freq_ids (3.125 MHz) apart across a 20 MHz band, 1 for the narrow L2C.
  // The cube inherits the tracker's channel set; ~100 channels would mean tracking them.
  const fid = c && c.freq_ids[0][0] != null
    ? `freq_id ${c.freq_ids[0][0]}..${c.freq_ids[c.n_sub - 1][1]}` +
      (c.n_sub > 1 ? `, step ${c.freq_ids[1][0] - c.freq_ids[0][0]} (×0.1953 MHz)` : '')
    : 'freq ids missing in this export — rebuild from an archive with freq_id attrs';
  document.getElementById('subinfo').textContent =
    c ? `${c.n_sub} channel${c.n_sub === 1 ? '' : 's'} of the tracker comb: ${fid}` : '';
}

async function ensureDaysLoaded() {
  const [a, b] = dayRange();
  const st = document.getElementById('status');
  for (let i = a; i <= b; i++) {
    if (S.cache.has(S.days[i])) continue;
    st.textContent = `loading ${S.days[i]}…`;
    try { await loadDay(S.days[i]); }
    catch (e) { st.innerHTML = `<div class="note bad">${e.message}</div>`; throw e; }
  }
  st.textContent = '';
}

async function boot() {
  try {
    S.index = await (await fetch('index.json')).json();
  } catch (e) {
    document.getElementById('boot').innerHTML =
      `<div class="note bad">Could not read <b>index.json</b>.<br><br>` +
      `If the page is open as <b>file://</b> that is expected — the browser blocks the ` +
      `cross-origin read and it looks identical to a missing file. Serve the directory ` +
      `instead:<br><br><code>cd /home/kvand/gnss/fixtures/beamcube/web<br>` +
      `python3 -m http.server 8877</code></div>`;
    return;
  }
  S.days = S.index.days.map(d => d.day);
  if (!S.days.length) {
    document.getElementById('boot').textContent =
      'index.json has no days — run: gnss_beam_cube.py export <master.npz>';
    return;
  }
  for (const id of ['d0', 'd1']) {
    const s = document.getElementById(id);
    s.max = S.days.length - 1;
    s.value = S.days.length - 1;
  }
  document.getElementById('d0').value = 0;

  await ensureDaysLoaded();
  const man = S.cache.get(S.days[0]);
  S.chains = man.chains.map(c => c.chain);
  S.nsub = Math.max(...man.chains.map(c => c.n_sub));
  S.nelem = Math.max(...man.chains.map(c => c.n_elem));
  document.getElementById('sub').max = S.nsub - 1;
  document.getElementById('el').max = S.nelem - 1;

  const box = document.getElementById('chains');
  const offOf = ch => { const m = man.chains.find(c => c.chain === ch); return m ? (m.offset_db || 0) : 0; };
  for (const c of S.chains) {
    S.offset.set(c, offOf(c));
    const l = document.createElement('label');
    l.className = 'chainrow';
    l.innerHTML = `<input type="checkbox" value="${c}"> <span class="cname">${c}</span>` +
      `<span class="muted">${BAND[c] || '?'}</span>` +
      `<input type="number" class="off" step="0.1" value="${offOf(c).toFixed(1)}" ` +
      `title="offset (dB) subtracted from ${c} before comparing/summing; default = main-lobe ` +
      `level vs ${man.offset_ref || 'reference'}"><span class="muted">dB</span>`;
    const cb = l.querySelector('input[type=checkbox]');
    cb.addEventListener('change', () => {
      cb.checked ? S.sel.add(c) : S.sel.delete(c);
      checkBands(); syncLabels(); draw();
    });
    const off = l.querySelector('input.off');
    off.addEventListener('input', () => { S.offset.set(c, +off.value || 0); draw(); });
    box.appendChild(l);
  }
  document.getElementById('offreset').addEventListener('click', () => {
    for (const l of box.querySelectorAll('label')) {
      const ch = l.querySelector('input[type=checkbox]').value;
      l.querySelector('input.off').value = offOf(ch).toFixed(1);
      S.offset.set(ch, offOf(ch));
    }
    draw();
  });
  document.getElementById('offzero').addEventListener('click', () => {
    for (const l of box.querySelectorAll('label')) {
      l.querySelector('input.off').value = '0.0';
      S.offset.set(l.querySelector('input[type=checkbox]').value, 0);
    }
    draw();
  });
  document.getElementById('offinfo').textContent = man.offset_ref
    ? `defaults: median main-lobe level ${man.offset_annulus_deg[0]}–${man.offset_annulus_deg[1]}° ` +
      `off boresight, relative to ${man.offset_ref} (day ${S.days[0]})`
    : 'this export carries no offsets (re-run gnss_beam_cube.py export)';

  // Element toggles: one button per antenna, remembered per browser.
  try { const sv = JSON.parse(localStorage.getItem('beamcube.elOn')); if (Array.isArray(sv)) S.elOn = new Set(sv); } catch (e) {}
  if (!S.elOn) S.elOn = new Set([...Array(S.nelem).keys()]);
  const grid = document.getElementById('elgrid');
  for (let e = 0; e < S.nelem; e++) {
    const b = document.createElement('button');
    b.dataset.e = e; b.textContent = e;
    if (DARK.includes(e)) b.title = `element ${e}: DARK (broken LNA)`;
    b.addEventListener('click', () => {
      S.elOn.has(e) ? S.elOn.delete(e) : S.elOn.add(e);
      saveElOn(); syncLabels(); draw();
    });
    grid.appendChild(b);
  }
  const setAll = pred => { S.elOn = new Set([...Array(S.nelem).keys()].filter(pred)); saveElOn(); syncLabels(); draw(); };
  document.getElementById('elall').addEventListener('click', () => setAll(() => true));
  document.getElementById('elnone').addEventListener('click', () => setAll(() => false));
  document.getElementById('ellive').addEventListener('click', () => setAll(e => !DARK.includes(e)));
  document.getElementById('elinv').addEventListener('click', () => { const was = S.elOn; setAll(e => !was.has(e)); });
  // Default to a single chain: a first view that silently mixed bands would teach the wrong
  // reading of the very axis this page exists to separate.
  const first = box.querySelector('input');
  if (first) { first.checked = true; S.sel.add(first.value); }

  document.getElementById('boot').remove();
  bindUI();
  checkBands();
  syncLabels();
  draw();
}

boot();
