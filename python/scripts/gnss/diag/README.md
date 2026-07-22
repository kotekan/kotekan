# ADR / carrier-phase diagnostic suite (2026-07-21/22, the L5 TEC investigation)

Offline analyzers over the obs jsonl logs and the (re-addable) combiner/assembler
instrumentation dumps. These closed the L5 TEC saga; kept for the next hunt.

- `l5_adr_scatter.py`   single-band ADR scatter (dt.arc_scatter, the validated metric)
- `l5_adr_vs_cn0.py`    per-arc scatter vs C/N0 -> the noise LAW (slope: thermal/fold/flat)
- `l5_common_mode.py`   cross-sat residual correlation: per-sat vs common-mode discriminator
- `l5_dump_analyze.py`  per-record (1 kHz) phase_dump forensics: nulls, wraps, dcmd-vs-dres
- `chan_decompose.py`   per-channel chan_dump: common-mode vs delay-slope split, PCA modes;
                        chan-relative products V_ch*conj(Vsum) are OVERLAY-IMMUNE (the trick)

The instrumentation they consume (combiner `phase_dump_prns`/`phase_dump_path`, assembler
`chan_dump_prn`/`chan_dump_decim`/`chan_dump_path`) was stripped from gnss_node.yaml when the
investigation closed (b6c69655) -- re-add per-chain, BOUNDED to named PRNs, when needed.
