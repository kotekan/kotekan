// Stream-health strip: the F-engine front end -- dark instances, per-band rail and power,
// spectral kurtosis and drop counters from the broker's get_rf, which the shared GpsFeed
// already fetches (~1.5 s), so this panel adds no load.
//
// Color language: green = healthy, amber = attention, red = data being lost.

export class AirspyStatsPanel {
    constructor({target, feed}) {
        this.el = document.getElementById(target);
        this._gnss = null;
        feed.on(p => this._update(p));
    }

    _update(payload) {
        if (payload && payload.gnss_rf !== undefined) this._gnss = payload.gnss_rf;
        if (this.el) this.el.innerHTML = this._gnss_block();
    }

    // The F-engine front end: one row per RF band, aggregated across the fleet's instances.
    //
    // Rows key on `band`, which the broker derives from the absolute freq_id the tap serves
    // (GnssChordVoltageTap reads it from the frame's chord metadata; fits.rf_lobes), so it
    // means the same on every instance whatever local channel subset the node holds. The lobe
    // index does not: instances differ in which bands they hold (only some own an L2C
    // channel), and abutting bands (bds_b3i, gal_e6) form one contiguous run. A lobe with no
    // `band` (the tap served no freq_ids, or broker --rf-bands is not declared) still renders,
    // keyed by index, and a banner says those rows are not comparable with the named ones.
    //
    // Partial coverage is not a fault: a narrow band (L2C) lands on only some instances by
    // design, so the coverage cell is grey and its tooltip names the missing instances; the
    // worst-instance column carries fault state.
    //
    // Clip is the headline and it is a max, not a mean: one railing channel on one instance is
    // a real fault, and averaging it across a healthy fleet hides it. The worst instance is
    // named for the same reason.
    //
    // "Not armed" must not look like "quiet": off and unreachable render grey with an explicit
    // word, never as a green zero.
    _gnss_block() {
        const g = this._gnss;
        if (!g || !g.instances) return "";
        const inst = Object.entries(g.instances);
        if (!inst.length) return "";
        const on = inst.filter(([, v]) => v && v.state === "on");
        const unreach = inst.filter(([, v]) => v && v.state === "unreachable").length;
        const short = u => u.replace(/^https?:\/\//, "").replace(/:\d+/, "");
        const cell = (v, color, tip) =>
            `<div style="flex:0 0 5.5em;padding:.15em .5em;" title="${tip}">
               <span style="font-size:1.05em;font-weight:600;color:${color};">${v}</span></div>`;
        const hdr = `<div style="display:flex;align-items:center;font-size:.72em;opacity:.6;
                                 margin-top:.5em;border-top:1px solid #dfe3e8;">
            <div style="flex:0 0 11em;padding:.1em .5em;">F-engine (GNSS)</div>
            <div style="flex:0 0 5.5em;padding:.1em .5em;">rail</div>
            <div style="flex:0 0 5.5em;padding:.1em .5em;">power</div>
            <div style="flex:0 0 5.5em;padding:.1em .5em;">armed</div>
            <div style="flex:0 0 9em;padding:.1em .5em;">worst</div></div>`;
        if (!on.length) {
            const why = unreach === inst.length ? `${unreach} unreachable`
                                                : "monitor not armed";
            return hdr + `<div style="display:flex;align-items:center;
                                      border-top:1px solid #f0f0f0;">
                <div style="flex:0 0 11em;padding:.15em .5em;color:#8a8f98;">—</div>
                <div style="padding:.15em .5em;color:#8a8f98;font-size:.9em;">${why}
                  — these cells are NOT a measurement of zero (the taps' rf_stats is off or
                  unreachable)</div></div>`;
        }
        // Group every instance's lobes by band; fall back to lobe index for any that carry
        // no band, so a fleet with both kinds renders them side by side rather than merging
        // them.
        const groups = new Map();
        let unnamed = 0;
        for (const [u, v] of on) {
            const ls = v.lobes || [];
            if (!ls.length) continue;
            if (!ls.some(l => l && l.band)) unnamed++;
            ls.forEach((l, i) => {
                if (!l) return;
                const key = l.band ? `b:${l.band}` : `i:${i}`;
                if (!groups.has(key))
                    groups.set(key, {band: l.band || null, idx: i,
                                     mhz: l.carrier_mhz != null ? l.carrier_mhz : null,
                                     have: []});
                groups.get(key).have.push([u, l]);
            });
        }
        // Named rows sort by carrier (physical order, identical on every render); unnamed
        // rows keep lobe order and follow them.
        const ordered = [...groups.values()].sort((a, b) => {
            if (a.mhz != null && b.mhz != null) return a.mhz - b.mhz;
            if (a.mhz != null) return -1;
            if (b.mhz != null) return 1;
            return a.idx - b.idx;
        });
        const mixed = unnamed > 0 && unnamed < on.length;
        // With no band names anywhere, lobe index is the only key, so instances disagreeing on
        // lobe count means rows may compare different bands. Named instances legitimately
        // differ in lobe count (only some own an L2C channel), so this applies only when every
        // instance is unnamed.
        const legacyCounts = new Set(on.map(([, v]) => (v.lobes || []).length));
        const legacyMixed = unnamed === on.length && legacyCounts.size > 1;
        let rows = "";
        // Dark instances (broker DARK-INST, get_rf `dark`, per chain): at the floor on every
        // PRN their peers detect while their tap is fed, i.e. replicas on the wrong channels.
        // Such an instance still counts in every fleet sum and shows locks, so this row is
        // where it reads as a fault. No row from a broker that does not run the check.
        const darkChains = Object.entries(g.dark || {});
        if (darkChains.length) {
            const dark = new Map();  // tap url -> the chains that find it dark
            for (const [chain, d] of darkChains)
                for (const u of Object.keys((d && d.instances) || {})) {
                    if (!dark.has(u)) dark.set(u, []);
                    dark.get(u).push(chain);
                }
            const col = dark.size ? "#d64550" : "#3fb26f";
            const list = [...dark].map(([u, cs]) => `${short(u)} (${cs.join(", ")})`);
            const darkTip = "an instance whose deep snr sits at the floor on every PRN its "
                + "peers detect while its tap sees normal per-channel power: a channel map or "
                + "comb wiring fault (compare the tap's freq_ids with the chain's channel_ids)";
            rows += `<div style="display:flex;align-items:center;
                                 border-top:1px solid #f0f0f0;">
                <div style="flex:0 0 11em;font-weight:600;color:#5a6472;padding:.15em .5em;
                            white-space:nowrap;" title="${darkTip}">dark instances</div>`
                + cell(String(dark.size), col,
                       `instances dark on at least one chain; ${darkChains.length} chain(s) `
                       + "checked. Each still counts in every fleet sum.")
                + `<div style="padding:.15em .5em;font-size:.85em;color:${col};">
                     ${list.length ? list.join("; ") : "none"}</div></div>`;
        }
        for (const grp of ordered) {
            const have = grp.have;
            if (!have.length) continue;
            let worstU = null, worstC = -1;
            let pw = [];
            for (const [u, l] of have) {
                // `clip` is one channel's lo + hi; the sum of the separate maxima, which
                // may come from two channels, stands in for a broker that does not serve it.
                const c = l.clip != null ? l.clip : (l.clip_lo || 0) + (l.clip_hi || 0);
                if (c > worstC) { worstC = c; worstU = u; }
                pw.push(l.power || 0);
            }
            pw.sort((a, b) => a - b);
            const med = pw[Math.floor(pw.length / 2)];
            // >1% of nibbles at a rail = the quantiser is discarding signal, not headroom.
            const cCol = worstC > 0.01 ? "#d64550" : worstC > 0.001 ? "#e8a13c" : "#3fb26f";
            const rng = have[0][1];
            const missing = on.map(([u]) => u).filter(u => !have.some(([h]) => h === u));
            const label = grp.band
                ? `${grp.band} · ${grp.mhz != null ? grp.mhz.toFixed(2) : "?"} MHz`
                : `lobe ${grp.idx} · ch ${rng.chan0}-${rng.chan1}`;
            const lblTip = grp.band
                ? `${grp.band} band, carrier ${grp.mhz} MHz. Spans `
                  + `${rng.mhz0}-${rng.mhz1} MHz (freq_id ${rng.freq_id0}-${rng.freq_id1}, `
                  + `local comb ch ${rng.chan0}-${rng.chan1}) on the first reporting instance; `
                  + "nodes hold different channel subsets, so the local range differs per "
                  + "instance while the band does not. Channels are assigned to the nearest "
                  + "declared carrier (broker --rf-bands)."
                : `local comb channels ${rng.chan0}-${rng.chan1}. UNNAMED: this instance's `
                  + "band is unknown -- its tap served no freq_ids (no chord-metadata frame yet), "
                  + "or broker --rf-bands is not declared. Rows keyed by lobe index do not line up "
                  + "across instances once the fleet runs more than two bands.";
            rows += `<div style="display:flex;align-items:center;
                                 border-top:1px solid #f0f0f0;">
                <div style="flex:0 0 11em;font-weight:600;color:#5a6472;padding:.15em .5em;
                            white-space:nowrap;" title="${lblTip}">${label}</div>`
                + cell((worstC * 100).toPrecision(2) + "%", cCol,
                       "WORST instance's fraction of 4+4b nibbles at a rail, this band, over "
                       + "every element in the tap, including full-crate inputs that are not "
                       + "the array's (the live-elements row covers only the array's). "
                       + "Max not mean: one railing channel on one instance is a real fault. "
                       + "Above ~1% the quantiser is losing signal, not just headroom, and "
                       + "every C/N0 below it is understated for a reason that is not the sky.")
                + cell(med != null ? med.toPrecision(3) : "—", "#5a6472",
                       "median |x|^2 across reporting instances, this band. Compare bands: a "
                       + "source that lifts one and not the other is band-selective and "
                       + "external.")
                + cell(`${have.length}/${inst.length}`,
                       have.length === inst.length ? "#3fb26f" : "#8a8f98",
                       "instances reporting this band. Partial is often correct: a narrow "
                       + "band lands on only a few instances by design (gps_l2c's mainlobe is "
                       + "6 freq_ids and 5 land on distinct instances). Not reporting: "
                       + (missing.length ? missing.map(short).join(", ") : "none")
                       + (unreach ? `; ${unreach} unreachable` : ""))
                + `<div style="flex:0 0 9em;padding:.15em .5em;font-size:.85em;
                               color:#8a8f98;white-space:nowrap;overflow:hidden;"
                        title="${worstU || ""}">${worstU ? short(worstU) : "—"}</div>`
                + "</div>";
        }
        if (mixed)
            rows += `<div style="padding:.15em .5em;font-size:.8em;color:#e8a13c;">
                ${unnamed} instance(s) serve no band identity — their rows are keyed by
                lobe index and are not comparable with the named rows above</div>`;
        else if (legacyMixed)
            rows += `<div style="padding:.15em .5em;font-size:.8em;color:#e8a13c;">
                instances disagree on lobe count — rows may compare different bands</div>`;

        // The band rows count every element in the tap, full-crate inputs that are not the
        // array's included. This row is the rail over the array's own elements
        // (array.live_element_ranges), from the broker's elem_clip_live: one figure over all
        // monitored channels, since the tap serves per-element clip only across bands. Worst
        // instance by its mean; the worst single live element rides beside it so one railing
        // input is not averaged away.
        const liveOn = on.filter(([, v]) => v && v.elem_clip_live != null);
        if (liveOn.length) {
            let worstU = null, worstC = -1, maxU = null, maxC = -1, maxE = null;
            for (const [u, v] of liveOn) {
                if (v.elem_clip_live > worstC) { worstC = v.elem_clip_live; worstU = u; }
                if (v.elem_clip_live_max > maxC) {
                    maxC = v.elem_clip_live_max; maxU = u; maxE = v.elem_clip_live_worst;
                }
            }
            const col = c => c > 0.01 ? "#d64550" : c > 0.001 ? "#e8a13c" : "#3fb26f";
            const liveTip = "the array's own elements (array.live_element_ranges), all "
                + "monitored channels together. The band rows above also count full-crate "
                + "inputs that are not the array's, so they can read high while these are "
                + "clean. No per-band split: the tap serves per-element clip only across bands.";
            rows += `<div style="display:flex;align-items:center;
                                 border-top:1px solid #f0f0f0;">
                <div style="flex:0 0 11em;font-weight:600;color:#5a6472;padding:.15em .5em;
                            white-space:nowrap;" title="${liveTip}">live elements · all bands</div>`
                + cell((worstC * 100).toPrecision(2) + "%", col(worstC),
                       "WORST instance's fraction of 4+4b nibbles at a rail, averaged over "
                       + "the array's live elements (dark inputs, at zero power, left out) "
                       + "and every monitored channel.")
                + cell((maxC * 100).toPrecision(2) + "%", col(maxC),
                       `worst single live element: element ${maxE} on `
                       + `${maxU ? short(maxU) : "?"}. One railing input is a fault the `
                       + "mean beside it would hide.")
                + cell(`${liveOn.length}/${inst.length}`,
                       liveOn.length === inst.length ? "#3fb26f" : "#8a8f98",
                       "instances reporting a live-element rail (a broker that cannot read "
                       + "array.live_element_ranges, or a tap serving fewer elements, reports "
                       + "none)")
                + `<div style="flex:0 0 9em;padding:.15em .5em;font-size:.85em;
                               color:#8a8f98;white-space:nowrap;overflow:hidden;"
                        title="${worstU || ""}">${worstU ? short(worstU) : "—"}</div>`
                + "</div>";
        }

        // RFI (spectral kurtosis): the same get_rf instances, folded in by broker --rfi-stats.
        const skOn = on.filter(([, v]) => v && v.sk);
        if (skOn.length) {
            let worstU = null, worstSK = 1, totFlag = 0, band = null;
            for (const [u, v] of skOn) {
                const sk = v.sk; totFlag += sk.sk_flagged || 0; band = sk;
                if (Math.abs((sk.sk_max || 1) - 1) > Math.abs(worstSK - 1)) { worstSK = sk.sk_max; worstU = u; }
            }
            const skCol = totFlag > 0 ? "#d64550" : "#3fb26f";
            rows += `<div style="display:flex;align-items:center;border-top:1px solid #f0f0f0;">
                <div style="flex:0 0 11em;font-weight:600;color:#5a6472;padding:.15em .5em;white-space:nowrap;"
                     title="spectral kurtosis per channel: 1.0 is clean gaussian noise; RFI drives it away in either direction. Flag band ${band ? band.sk_lo + "–" + band.sk_hi : ""} (3σ at ${band ? band.ema_frames : "?"} frames).">RFI · SK &nbsp;max/flag</div>`
                + cell(worstSK != null ? worstSK.toPrecision(3) : "—", skCol,
                       "worst per-channel SK across reporting instances (furthest from 1.0). Inside the band = clean; outside = a channel is jammed.")
                + cell(String(totFlag), totFlag > 0 ? "#d64550" : "#3fb26f",
                       "total channels flagged (SK outside the band) across all instances")
                + cell(`${skOn.length}/${inst.length}`, skOn.length === inst.length ? "#3fb26f" : "#e8a13c",
                       "instances reporting SK")
                + `<div style="flex:0 0 9em;padding:.15em .5em;font-size:.85em;color:#8a8f98;white-space:nowrap;overflow:hidden;" title="${worstU || ""}">${worstU ? short(worstU) : "—"}</div></div>`;
        }

        // Drop counters (cumulative), folded in by broker --drop-stats. Worst instance per class.
        const drOn = on.filter(([, v]) => v && v.drops);
        if (drOn.length) {
            const worst = (key) => { let u = null, val = -1;
                for (const [uu, v] of drOn) { const x = (v.drops || {})[key] || 0; if (x > val) { val = x; u = uu; } }
                return [u, val]; };
            const [suU, su] = worst("srch_send_drops");
            const [, tu] = worst("telem_send_drops");
            const [, mu] = worst("dpdk_missed");
            const dc = (v) => v > 0 ? "#e8a13c" : "#3fb26f";
            rows += `<div style="display:flex;align-items:center;border-top:1px solid #f0f0f0;">
                <div style="flex:0 0 11em;font-weight:600;color:#5a6472;padding:.15em .5em;white-space:nowrap;"
                     title="cumulative dropped-frame counters since instance start — watch the rate across ticks, not the absolute. Worst instance shown per class.">drops &nbsp;srch/telem/nic</div>`
                + cell(su >= 0 ? su.toLocaleString() : "—", dc(su), "worst instance: search-send frames dropped (send buffer full)")
                + cell(tu >= 0 ? tu.toLocaleString() : "—", dc(tu), "worst instance: telemetry-send frames dropped")
                + cell(mu >= 0 ? mu.toLocaleString() : "—", dc(mu), "worst instance: NIC rx-missed packets (dpdk)")
                + `<div style="flex:0 0 9em;padding:.15em .5em;font-size:.85em;color:#8a8f98;white-space:nowrap;overflow:hidden;" title="${suU || ""}">${suU ? short(suU) : "—"}</div></div>`;
        }

        return hdr + rows;
    }
}
