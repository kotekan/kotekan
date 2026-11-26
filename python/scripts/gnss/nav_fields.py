"""Field-by-field reads of the navigation messages nav_decode keeps, the newest of each type per
satellite: what each message carries beside the orbit the decoders already parse (clock
details, ionosphere, group delays, time offsets, Earth orientation, almanacs, health), for a
viewer's message tables (nav_decode --payload-file).

Each row is (label, start, width, kind, scale, unit) on the bits the decoder stores, start
0-based: GPS CNAV the 300-bit message from its preamble; Galileo I/NAV the 128-bit word from
its type; F/NAV the 238-bit page from its type; BeiDou D1 the 300-bit subframe as stored after
BCH (RTKLIB's layout, as bds_d1_predictor); B-CNAV2 the 288-bit frame from its PRN; GLONASS the
string's bits 85..1 (ICD numbering, b[i] = bit i). kind: "u" unsigned, "s" two's complement,
"m" sign and magnitude (GLONASS), "deg" a two's-complement semicircle count shown in degrees.
A row whose start is a tuple reads two pieces (D1 splits fields across words).

Layouts: IS-GPS-705 (CNAV), Galileo OS SIS ICD 2.1 (I/NAV, F/NAV), BDS-SIS-ICD B3I 1.0 and
RTKLIB rcvraw.c (D1), BDS-SIS-ICD B2a 1.0 (B-CNAV2), GLONASS ICD 5.1 (strings 5-15). Checked on
live messages (11-23) against a RINEX 4 broadcast file's ION, STO and EOP records: CNAV types
30/32/33, I/NAV words 5/6/10, F/NAV pages 1/4, B-CNAV2 types 30/34, GLONASS string 5 (RINEX holds
-tau_c and -tau_GPS). B-CNAV2's EOP (type 32) and almanacs (31, 40), and B-CNAV3's BDGIM and
dUT1 (type 30), were placed by finding the reference values in live frames; the GLONASS almanac
by its slots and frequency channels against BRDC. B-CNAV2 type 33's time offset, B-CNAV3's other
fields, PPP-B2b and SBAS other than MT31 and MT32's corrections are titled only.
"""

P = lambda n: 2.0**n  # noqa: E731

# ---- GPS CNAV (IS-GPS-705), bits 0-based ------------------------------------------------------
CNAV_HDR = [("TOW count", 20, 17, "u", 6, "s"), ("alert", 37, 1, "u", 1, "")]
CNAV_CLK = [
    ("t_op", 38, 11, "u", 300, "s of week"),
    ("URA_ED / NED0 index", 49, 5, "s", 1, ""),
    ("URA_NED1 index", 54, 3, "u", 1, ""),
    ("URA_NED2 index", 57, 3, "u", 1, ""),
    ("t_oc", 60, 11, "u", 300, "s of week"),
    ("a_f0", 71, 26, "s", P(-35), "s"),
    ("a_f1", 97, 20, "s", P(-48), "s/s"),
    ("a_f2", 117, 10, "s", P(-60), "s/s²"),
]
CNAV_PACKET = [  # one reduced-almanac packet, 31 bits, offsets from its start
    ("PRN", 0, 6, "u", 1, ""),
    ("δA", 6, 8, "s", P(9), "m"),
    ("Ω0", 14, 7, "deg", P(-6), "°"),
    ("Φ0", 21, 7, "deg", P(-6), "°"),
    ("health L1/L2/L5", 28, 3, "u", 1, "bits"),
]
CNAV = {
    10: (
        "ephemeris 1: orbit",
        [("WN", 38, 13, "u", 1, ""), ("health L1/L2/L5", 51, 3, "u", 1, "bits")],
    ),
    11: ("ephemeris 2: orbit", []),
    30: (
        "clock, group delays, ionosphere",
        CNAV_CLK
        + [
            ("T_GD", 127, 13, "s", P(-35), "s"),
            ("ISC L1C/A", 140, 13, "s", P(-35), "s"),
            ("ISC L2C", 153, 13, "s", P(-35), "s"),
            ("ISC L5I5", 166, 13, "s", P(-35), "s"),
            ("ISC L5Q5", 179, 13, "s", P(-35), "s"),
            ("α0", 192, 8, "s", P(-30), "s"),
            ("α1", 200, 8, "s", P(-27), "s/semicircle"),
            ("α2", 208, 8, "s", P(-24), "s/semicircle²"),
            ("α3", 216, 8, "s", P(-24), "s/semicircle³"),
            ("β0", 224, 8, "s", P(11), "s"),
            ("β1", 232, 8, "s", P(14), "s/semicircle"),
            ("β2", 240, 8, "s", P(16), "s/semicircle²"),
            ("β3", 248, 8, "s", P(16), "s/semicircle³"),
            ("WN_op", 256, 8, "u", 1, ""),
        ],
    ),
    31: (
        "clock and reduced almanac",
        CNAV_CLK + [("WN_a", 127, 13, "u", 1, ""), ("t_oa", 140, 8, "u", P(12), "s")],
    ),
    32: (
        "clock and Earth orientation",
        CNAV_CLK
        + [
            ("t_EOP", 127, 16, "u", 16, "s of week"),
            ("PM_X", 143, 21, "s", P(-20), "arcsec"),
            ("dPM_X/dt", 164, 15, "s", P(-21), "arcsec/day"),
            ("PM_Y", 179, 21, "s", P(-20), "arcsec"),
            ("dPM_Y/dt", 200, 15, "s", P(-21), "arcsec/day"),
            ("ΔUT_GPS", 215, 31, "s", P(-24), "s"),
            ("dΔUT_GPS/dt", 246, 19, "s", P(-25), "s/day"),
        ],
    ),
    33: (
        "clock and GPS-UTC",
        CNAV_CLK
        + [
            ("A0", 127, 16, "s", P(-35), "s"),
            ("A1", 143, 13, "s", P(-51), "s/s"),
            ("A2", 156, 7, "s", P(-68), "s/s²"),
            ("Δt_LS", 163, 8, "s", 1, "s"),
            ("t_ot", 171, 16, "u", 16, "s of week"),
            ("WN_ot", 187, 13, "u", 1, ""),
            ("WN_LSF", 200, 13, "u", 1, ""),
            ("DN", 213, 4, "u", 1, "day"),
            ("Δt_LSF", 217, 8, "s", 1, "s"),
        ],
    ),
    35: (
        "clock and GPS-GNSS time offset",
        CNAV_CLK
        + [
            ("t_GGTO", 127, 16, "u", 16, "s of week"),
            ("WN_GGTO", 143, 13, "u", 1, ""),
            ("GNSS ID", 156, 3, "u", 1, "1 Galileo, 2 GLONASS"),
            ("A0_GGTO", 159, 16, "s", P(-35), "s"),
            ("A1_GGTO", 175, 13, "s", P(-51), "s/s"),
            ("A2_GGTO", 188, 7, "s", P(-68), "s/s²"),
        ],
    ),
    37: (
        "clock and midi almanac",
        CNAV_CLK
        + [
            ("WN_a", 127, 13, "u", 1, ""),
            ("t_oa", 140, 8, "u", P(12), "s"),
            ("almanac PRN", 148, 6, "u", 1, ""),
            ("health L1/L2/L5", 154, 3, "u", 1, "bits"),
            ("e", 157, 11, "u", P(-16), ""),
            ("δi", 168, 11, "deg", P(-14), "°"),
            ("dΩ/dt", 179, 11, "s", P(-33), "semicircle/s"),
            ("√A", 190, 17, "u", P(-4), "m½"),
            ("Ω0", 207, 16, "deg", P(-15), "°"),
            ("ω", 223, 16, "deg", P(-15), "°"),
            ("M0", 239, 16, "deg", P(-15), "°"),
            ("a_f0", 255, 11, "s", P(-20), "s"),
            ("a_f1", 266, 10, "s", P(-37), "s/s"),
        ],
    ),
    12: (
        "reduced almanac",
        [("WN_a", 38, 13, "u", 1, ""), ("t_oa", 51, 8, "u", P(12), "s")],
    ),
    34: ("clock and differential corrections", CNAV_CLK),
    36: ("clock and text", CNAV_CLK),
    15: ("text", []),
}
CNAV_PACKETS = {31: (148, 4), 12: (59, 7)}  # first packet's bit, packets
CNAV_TEXT = {15: (38, 29), 36: (127, 18)}  # first character's bit, characters

# ---- Galileo I/NAV words, 0-based in the 128-bit word ------------------------------------------
GAL_ALM_A = [  # one satellite's almanac fields after its SVID, offsets from the SVID's bit
    ("ΔA½", 6, 13, "s", P(-9), "m½"),
    ("e", 19, 11, "u", P(-16), ""),
    ("ω", 30, 16, "deg", P(-15), "°"),
    ("δi", 46, 11, "deg", P(-14), "°"),
]
INAV = {
    0: (
        "time",
        [
            ("time field", 6, 2, "u", 1, ""),
            ("WN", 96, 12, "u", 1, ""),
            ("TOW", 108, 20, "u", 1, "s"),
        ],
    ),
    1: ("ephemeris 1", [("IODnav", 6, 10, "u", 1, "")]),
    2: ("ephemeris 2", [("IODnav", 6, 10, "u", 1, "")]),
    3: (
        "ephemeris 3 and SISA",
        [("IODnav", 6, 10, "u", 1, ""), ("SISA index", 120, 8, "u", 1, "")],
    ),
    4: ("SVID, clock", [("IODnav", 6, 10, "u", 1, ""), ("SVID", 16, 6, "u", 1, "")]),
    5: (
        "ionosphere, group delays, health, GST",
        [
            ("a_i0", 6, 11, "u", P(-2), "sfu"),
            ("a_i1", 17, 11, "s", P(-8), "sfu/°"),
            ("a_i2", 28, 14, "s", P(-15), "sfu/°²"),
            ("region flags 1-5", 42, 5, "u", 1, "bits"),
            ("BGD E1-E5a", 47, 10, "s", P(-32), "s"),
            ("BGD E1-E5b", 57, 10, "s", P(-32), "s"),
            ("E5b health", 67, 2, "u", 1, ""),
            ("E1-B health", 69, 2, "u", 1, ""),
            ("E5b data validity", 71, 1, "u", 1, ""),
            ("E1-B data validity", 72, 1, "u", 1, ""),
            ("WN", 73, 12, "u", 1, ""),
            ("TOW", 85, 20, "u", 1, "s"),
        ],
    ),
    6: (
        "GST-UTC",
        [
            ("A0", 6, 32, "s", P(-30), "s"),
            ("A1", 38, 24, "s", P(-50), "s/s"),
            ("Δt_LS", 62, 8, "s", 1, "s"),
            ("t_0t", 70, 8, "u", 3600, "s of week"),
            ("WN_0t", 78, 8, "u", 1, ""),
            ("WN_LSF", 86, 8, "u", 1, ""),
            ("DN", 94, 3, "u", 1, "day"),
            ("Δt_LSF", 97, 8, "s", 1, "s"),
            ("TOW", 105, 20, "u", 1, "s"),
        ],
    ),
    7: (
        "almanac (first satellite)",
        [
            ("IODa", 6, 4, "u", 1, ""),
            ("WN_a", 10, 2, "u", 1, ""),
            ("t_0a", 12, 10, "u", 600, "s"),
            ("SVID", 22, 6, "u", 1, ""),
        ]
        + [(l, 22 + o, w, k, sc, u) for l, o, w, k, sc, u in GAL_ALM_A]
        + [
            ("Ω0", 79, 16, "deg", P(-15), "°"),
            ("dΩ/dt", 95, 11, "s", P(-33), "semicircle/s"),
            ("M0", 106, 16, "deg", P(-15), "°"),
        ],
    ),
    8: (
        "almanac (first satellite's clock; second satellite)",
        [
            ("IODa", 6, 4, "u", 1, ""),
            ("a_f0", 10, 16, "s", P(-19), "s"),
            ("a_f1", 26, 13, "s", P(-38), "s/s"),
            ("E5b health", 39, 2, "u", 1, ""),
            ("E1-B health", 41, 2, "u", 1, ""),
            ("SVID", 43, 6, "u", 1, ""),
        ]
        + [(l, 43 + o, w, k, sc, u) for l, o, w, k, sc, u in GAL_ALM_A]
        + [
            ("Ω0", 100, 16, "deg", P(-15), "°"),
            ("dΩ/dt", 116, 11, "s", P(-33), "semicircle/s"),
        ],
    ),
    9: (
        "almanac (second satellite's M0 and clock; third satellite)",
        [
            ("IODa", 6, 4, "u", 1, ""),
            ("WN_a", 10, 2, "u", 1, ""),
            ("t_0a", 12, 10, "u", 600, "s"),
            ("M0", 22, 16, "deg", P(-15), "°"),
            ("a_f0", 38, 16, "s", P(-19), "s"),
            ("a_f1", 54, 13, "s", P(-38), "s/s"),
            ("E5b health", 67, 2, "u", 1, ""),
            ("E1-B health", 69, 2, "u", 1, ""),
            ("SVID", 71, 6, "u", 1, ""),
        ]
        + [(l, 71 + o, w, k, sc, u) for l, o, w, k, sc, u in GAL_ALM_A],
    ),
    10: (
        "almanac (third satellite) and GST-GPS",
        [
            ("IODa", 6, 4, "u", 1, ""),
            ("Ω0", 10, 16, "deg", P(-15), "°"),
            ("dΩ/dt", 26, 11, "s", P(-33), "semicircle/s"),
            ("M0", 37, 16, "deg", P(-15), "°"),
            ("a_f0", 53, 16, "s", P(-19), "s"),
            ("a_f1", 69, 13, "s", P(-38), "s/s"),
            ("E5b health", 82, 2, "u", 1, ""),
            ("E1-B health", 84, 2, "u", 1, ""),
            ("A_0G", 86, 16, "s", P(-35), "s"),
            ("A_1G", 102, 12, "s", P(-51), "s/s"),
            ("t_0G", 114, 8, "u", 3600, "s of week"),
            ("WN_0G", 122, 6, "u", 1, ""),
        ],
    ),
}

# ---- Galileo F/NAV pages, 0-based in the 238-bit page ------------------------------------------
FNAV = {
    1: (
        "clock, ionosphere, group delay, health, GST",
        [
            ("SVID", 6, 6, "u", 1, ""),
            ("IODnav", 12, 10, "u", 1, ""),
            ("SISA index", 94, 8, "u", 1, ""),
            ("a_i0", 102, 11, "u", P(-2), "sfu"),
            ("a_i1", 113, 11, "s", P(-8), "sfu/°"),
            ("a_i2", 124, 14, "s", P(-15), "sfu/°²"),
            ("region flags 1-5", 138, 5, "u", 1, "bits"),
            ("BGD E1-E5a", 143, 10, "s", P(-32), "s"),
            ("E5a health", 153, 2, "u", 1, ""),
            ("WN", 155, 12, "u", 1, ""),
            ("TOW", 167, 20, "u", 1, "s"),
            ("E5a data validity", 187, 1, "u", 1, ""),
        ],
    ),
    2: ("ephemeris 1", [("IODnav", 6, 10, "u", 1, "")]),
    3: ("ephemeris 2", [("IODnav", 6, 10, "u", 1, "")]),
    4: (
        "ephemeris 3, GST-UTC, GST-GPS",
        [
            ("IODnav", 6, 10, "u", 1, ""),
            ("A0", 48, 32, "s", P(-30), "s"),
            ("A1", 80, 24, "s", P(-50), "s/s"),
            ("Δt_LS", 104, 8, "s", 1, "s"),
            ("t_0t", 112, 8, "u", 3600, "s of week"),
            ("WN_0t", 120, 8, "u", 1, ""),
            ("WN_LSF", 128, 8, "u", 1, ""),
            ("DN", 136, 3, "u", 1, "day"),
            ("Δt_LSF", 139, 8, "s", 1, "s"),
            ("t_0G", 147, 8, "u", 3600, "s of week"),
            ("A_0G", 155, 16, "s", P(-35), "s"),
            ("A_1G", 171, 12, "s", P(-51), "s/s"),
            ("WN_0G", 183, 6, "u", 1, ""),
            ("TOW", 189, 20, "u", 1, "s"),
        ],
    ),
    5: (
        "almanac (first two satellites)",
        [
            ("IODa", 6, 4, "u", 1, ""),
            ("WN_a", 10, 2, "u", 1, ""),
            ("t_0a", 12, 10, "u", 600, "s"),
            ("SVID", 22, 6, "u", 1, ""),
        ]
        + [(l, 22 + o, w, k, sc, u) for l, o, w, k, sc, u in GAL_ALM_A]
        + [
            ("Ω0", 79, 16, "deg", P(-15), "°"),
            ("dΩ/dt", 95, 11, "s", P(-33), "semicircle/s"),
            ("M0", 106, 16, "deg", P(-15), "°"),
            ("a_f0", 122, 16, "s", P(-19), "s"),
            ("a_f1", 138, 13, "s", P(-38), "s/s"),
            ("E5a health", 151, 2, "u", 1, ""),
            ("second SVID", 153, 6, "u", 1, ""),
        ],
    ),
    6: (
        "almanac (second and third satellites)",
        [("IODa", 6, 4, "u", 1, "")],
    ),
}

# ---- BeiDou D1, the stored 300-bit subframe (RTKLIB's layout) -----------------------------------
D1_PAGE = ("page number", 43, 7, "u", 1, "")
D1 = {
    1: (
        "clock, group delays, ionosphere",
        [
            ("SatH1", 42, 1, "u", 1, "0 healthy"),
            ("AODC", 43, 5, "u", 1, ""),
            ("URAI", 48, 4, "u", 1, ""),
            ("WN", 60, 13, "u", 1, ""),
            ("T_GD1", 98, 10, "s", 1e-10, "s"),
            ("T_GD2", (108, 4, 120, 6), 10, "s", 1e-10, "s"),
            ("α0", 126, 8, "s", P(-30), "s"),
            ("α1", 134, 8, "s", P(-27), "s/π"),
            ("α2", 150, 8, "s", P(-24), "s/π²"),
            ("α3", 158, 8, "s", P(-24), "s/π³"),
            ("β0", (166, 6, 180, 2), 8, "s", P(11), "s"),
            ("β1", 182, 8, "s", P(14), "s/π"),
            ("β2", 190, 8, "s", P(16), "s/π²"),
            ("β3", (198, 4, 210, 4), 8, "s", P(16), "s/π³"),
        ],
    ),
    2: ("ephemeris 1", []),
    3: ("ephemeris 2", []),
    4: ("almanac pages 1-24", [D1_PAGE]),
    5: ("almanac, health, time offsets", [D1_PAGE]),
}
D1_UTC = [  # subframe 5 page 10
    ("Δt_LS", (50, 2, 60, 6), 8, "s", 1, "s"),
    ("Δt_LSF", 66, 8, "s", 1, "s"),
    ("WN_LSF", 74, 8, "u", 1, ""),
    ("A0_UTC", (90, 22, 120, 10), 32, "s", P(-30), "s"),
    ("A1_UTC", (130, 12, 150, 12), 24, "s", P(-50), "s/s"),
    ("DN", 162, 8, "u", 1, "day"),
]

# ---- BeiDou B-CNAV2 (B2a), 0-based in the 288-bit frame -----------------------------------------
BDGIM = lambda at: [  # noqa: E731
    ("α1", at, 10, "u", 0.125, "TECu"),
    ("α2", at + 10, 8, "s", 0.125, "TECu"),
    ("α3", at + 18, 8, "u", 0.125, "TECu"),
    ("α4", at + 26, 8, "u", 0.125, "TECu"),
    ("α5", at + 34, 8, "u", -0.125, "TECu"),  # a magnitude: α5 is defined <= 0
    ("α6", at + 42, 8, "s", 0.125, "TECu"),
    ("α7", at + 50, 8, "s", 0.125, "TECu"),
    ("α8", at + 58, 8, "s", 0.125, "TECu"),
    ("α9", at + 66, 8, "s", 0.125, "TECu"),
]

BC2_HDR = [
    ("SOW", 12, 18, "u", 3, "s"),
    ("HS", 30, 2, "u", 1, ""),
    ("SISMAI", 35, 4, "u", 1, ""),
]
BC2_CLK = lambda at: [  # noqa: E731
    ("t_oc", at, 11, "u", 300, "s of week"),
    ("a0", at + 11, 25, "s", P(-34), "s"),
    ("a1", at + 36, 22, "s", P(-50), "s/s"),
    ("a2", at + 58, 11, "s", P(-66), "s/s²"),
]
BCNAV2 = {
    10: ("ephemeris 1", [("WN", 30, 13, "u", 1, "")]),
    11: ("ephemeris 2", []),
    30: (
        "clock, group delays, BDGIM ionosphere",
        BC2_HDR
        + BC2_CLK(42)
        + [
            ("IODC", 111, 10, "u", 1, ""),
            ("T_GD B2ap", 121, 12, "s", P(-34), "s"),
            ("ISC B2ad", 133, 12, "s", P(-34), "s"),
        ]
        + BDGIM(145)
        + [("T_GD B1Cp", 219, 12, "s", P(-34), "s")],
    ),
    34: (
        "SISAI, clock, BDT-UTC",
        BC2_HDR
        + [("t_op", 42, 11, "u", 300, "s of week")]
        + BC2_CLK(64)
        + [
            ("IODC", 133, 10, "u", 1, ""),
            ("A0_UTC", 143, 16, "s", P(-35), "s"),
            ("A1_UTC", 159, 13, "s", P(-51), "s/s"),
            ("A2_UTC", 172, 7, "s", P(-68), "s/s²"),
            ("Δt_LS", 179, 8, "s", 1, "s"),
            ("t_ot", 187, 16, "u", 16, "s of week"),
            ("WN_ot", 203, 13, "u", 1, ""),
            ("WN_LSF", 216, 13, "u", 1, ""),
            ("DN", 229, 3, "u", 1, "day"),
            ("Δt_LSF", 232, 8, "s", 1, "s"),
        ],
    ),
    31: (
        "clock, reduced almanac",
        BC2_HDR
        + BC2_CLK(42)
        + [
            ("IODC", 111, 10, "u", 1, ""),
            ("WN_a", 121, 13, "u", 1, ""),
            ("t_oa", 134, 8, "u", P(12), "s"),
        ],
    ),
    32: (
        "clock, Earth orientation",
        BC2_HDR
        + BC2_CLK(42)
        + [
            ("IODC", 111, 10, "u", 1, ""),
            ("t_EOP", 121, 16, "u", 16, "s of week"),
            ("PM_X", 137, 21, "s", P(-20), "arcsec"),
            ("dPM_X/dt", 158, 15, "s", P(-21), "arcsec/day"),
            ("PM_Y", 173, 21, "s", P(-20), "arcsec"),
            ("dPM_Y/dt", 194, 15, "s", P(-21), "arcsec/day"),
            ("ΔUT1", 209, 31, "s", P(-24), "s"),
            ("dΔUT1/dt", 240, 19, "s", P(-25), "s/day"),
        ],
    ),
    33: (
        "clock, BDT-GNSS time offset, reduced almanac",
        BC2_HDR + BC2_CLK(42) + [("IODC", 111, 10, "u", 1, "")],
    ),
    40: (
        "SISAI, midi almanac",
        BC2_HDR
        + [
            ("almanac PRN", 69, 6, "u", 1, ""),
            ("satellite type", 75, 2, "u", 1, "1 GEO, 2 IGSO, 3 MEO"),
            ("WN_a", 77, 13, "u", 1, ""),
            ("t_oa", 90, 8, "u", P(12), "s"),
            ("e", 98, 11, "u", P(-16), ""),
            ("δi", 109, 11, "deg", P(-14), "°"),
            ("√A", 120, 17, "u", P(-4), "m½"),
            ("Ω0", 137, 16, "deg", P(-15), "°"),
            ("dΩ/dt", 153, 11, "s", P(-33), "semicircle/s"),
            ("ω", 164, 16, "deg", P(-15), "°"),
            ("M0", 180, 16, "deg", P(-15), "°"),
            ("a_f0", 196, 11, "s", P(-20), "s"),
            ("a_f1", 207, 10, "s", P(-37), "s/s"),
            ("health", 217, 8, "u", 1, ""),
        ],
    ),
}
BC2_PACKETS = {31: (142, 3)}  # first reduced-almanac packet's bit, packets
BC2_PACKET = [  # 38 bits
    ("PRN", 0, 6, "u", 1, ""),
    ("satellite type", 6, 2, "u", 1, ""),
    ("δA", 8, 8, "s", P(9), "m"),
    ("Ω0", 16, 7, "deg", P(-6), "°"),
    ("Φ0", 23, 7, "deg", P(-6), "°"),
    ("health", 30, 8, "u", 1, ""),
]

BCNAV3 = {
    10: ("ephemeris", []),
    30: (
        "clock, group delay, ionosphere, EOP, time offsets",
        BDGIM(124) + [("ΔUT1", 383, 31, "s", P(-24), "s")],
    ),
    40: ("almanac", []),
    1: ("PPP-B2b satellite mask", []),
    4: ("PPP-B2b clock corrections", []),
}

# DFMC MT32, 0-based in the 250-bit message: the field widths and scales of the SouthPAN SDD
# (SBAS-STN-0001 rev 04, Table 7). The slot starts at bit 11, not 10: placed on live WAAS messages
# (11-26), where IODN then equals the LNAV IODC of the slot's GPS satellite and t_D runs ~40 s
# behind reception. The covariance, DFREI and dRcorr after t_D are not placed.
SBAS_MT32 = [
    ("slot", 11, 8, "u", 1, ""),
    ("IODN", 19, 10, "u", 1, ""),
    ("δx", 29, 11, "s", P(-4), "m"),
    ("δy", 40, 11, "s", P(-4), "m"),
    ("δz", 51, 11, "s", P(-4), "m"),
    ("δB", 62, 12, "s", P(-5), "m"),
    ("δx rate", 74, 8, "s", P(-11), "m/s"),
    ("δy rate", 82, 8, "s", P(-11), "m/s"),
    ("δz rate", 90, 8, "s", P(-11), "m/s"),
    ("δB rate", 98, 9, "s", P(-12), "m/s"),
    ("t_D", 107, 13, "u", 16, "s of day"),
]
SBAS_MT31_IODM = ("IODM", 224, 2, "u", 1, "")

SBAS = {
    0: "test (do not use)",
    31: "satellite mask",
    32: "clock-ephemeris corrections and covariance",
    34: "integrity information",
    35: "integrity information",
    36: "integrity information",
    37: "OBAD parameters and DFRE table",
    39: "SBAS satellite ephemeris and covariance (1 of 2)",
    40: "SBAS satellite ephemeris and covariance (2 of 2)",
    42: "time offsets",
    47: "SBAS satellite almanac",
    62: "internal test",
    63: "null",
}

# ---- GLONASS strings, ICD bit numbers (b[i] = bit i, 85 first on air) ---------------------------
GLO_ALM_A = [  # strings 6, 8, 10, 12, 14: one satellite's first half
    ("C_n, health", 80, 1, "u", 1, ""),
    ("M_n, modification", (79, 2), 2, "u", 1, ""),
    ("n, slot", (77, 5), 5, "u", 1, ""),
    ("τ_n (almanac)", (72, 10), 10, "m", P(-18), "s"),
    ("λ_n, ascending node longitude", (62, 21), 21, "m", P(-20) * 180, "°"),
    ("Δi_n", (41, 18), 18, "m", P(-20) * 180, "°"),
    ("ε_n, eccentricity", (23, 15), 15, "u", P(-20), ""),
]
GLO_ALM_B = [  # strings 7, 9, 11, 13, 15: its second half
    ("ω_n, perigee", (80, 16), 16, "m", P(-15) * 180, "°"),
    ("t_λn, node passage", (64, 21), 21, "u", P(-5), "s"),
    ("ΔT_n, period correction", (43, 22), 22, "m", P(-9), "s/orbit"),
    ("dΔT_n/dt", (21, 7), 7, "m", P(-14), "s/orbit²"),
    ("H_n, frequency channel", (14, 5), 5, "u", 1, "(k = H, or H - 32 above 24)"),
    ("l_n", 9, 1, "u", 1, ""),
]
GLO = {
    1: (
        "immediate data 1: x and t_k",
        [
            ("P1", (78, 2), 2, "u", 1, ""),
            ("t_k hours", (76, 5), 5, "u", 1, "h"),
            ("t_k minutes", (71, 6), 6, "u", 1, "min"),
            ("t_k 30 s", 65, 1, "u", 1, ""),
        ],
    ),
    2: (
        "immediate data 2: y and t_b",
        [
            ("B_n", (80, 3), 3, "u", 1, ""),
            ("P2", 77, 1, "u", 1, ""),
            ("t_b", (76, 7), 7, "u", 15, "min, Moscow day"),
        ],
    ),
    3: (
        "immediate data 3: z and γ_n",
        [
            ("P3", 80, 1, "u", 1, ""),
            ("γ_n", (79, 11), 11, "m", P(-40), "s/s"),
            ("P", (67, 2), 2, "u", 1, ""),
            ("l_n", 65, 1, "u", 1, ""),
        ],
    ),
    4: (
        "immediate data 4: τ_n, age, slot",
        [
            ("τ_n", (80, 22), 22, "m", P(-30), "s"),
            ("Δτ_n", (58, 5), 5, "m", P(-30), "s"),
            ("E_n, age", (53, 5), 5, "u", 1, "days"),
            ("P4", 34, 1, "u", 1, ""),
            ("F_T, accuracy index", (33, 4), 4, "u", 1, ""),
            ("N_T, day of 4-year span", (26, 11), 11, "u", 1, ""),
            ("n, slot", (15, 5), 5, "u", 1, ""),
            ("M, modification", (10, 2), 2, "u", 1, ""),
        ],
    ),
    5: (
        "time scales",
        [
            ("N^A, almanac day", (80, 11), 11, "u", 1, ""),
            ("τ_c, GLONASS-UTC(SU)", (69, 32), 32, "m", P(-31), "s"),
            ("N_4, 4-year span", (36, 5), 5, "u", 1, ""),
            ("τ_GPS, GLONASS-GPS fraction", (31, 22), 22, "m", P(-30), "s"),
            ("l_n", 9, 1, "u", 1, ""),
        ],
    ),
}


def _u(bits, start, width):
    v = 0
    for b in bits[start : start + width]:
        v = (v << 1) | int(b)
    return v


def _read(bits, start, width, kind):
    """One field; start may be (s1, w1, s2, w2) for a field split in two."""
    if isinstance(start, tuple):
        s1, w1, s2, w2 = start
        v = (_u(bits, s1, w1) << w2) | _u(bits, s2, w2)
    else:
        v = _u(bits, start, width)
    if kind in ("s", "deg") and v >> (width - 1):
        v -= 1 << width
    return v


def _glo(b, hi, width):
    """GLONASS field from its highest bit number (b[i] = bit i)."""
    v = 0
    for i in range(hi, hi - width, -1):
        v = (v << 1) | int(b[i])
    return v


# Galileo E6-B C/NAV: the HAS page header (galileo_cnav).
E6B_PAGE = [
    ("HAS status", 14, 2, "u", 1, ""),
    ("message type", 18, 2, "u", 1, ""),
    ("message ID", 20, 5, "u", 1, ""),
    ("message size less one", 25, 5, "u", 1, "pages"),
    ("page ID", 30, 8, "u", 1, ""),
]


def has_rows(msg, sat, kind):
    """[[label, value, unit], ...]: the HAS corrections of one kind ("orbit", "clock" or "bias")
    a parsed type 1 message (galileo_cnav.parse_mt1) carries for one satellite ("E04"),
    unavailable ones left out."""
    d = msg.get("sats", {}).get(sat, {})
    rows = []
    orb = d.get("orbit") if kind == "orbit" else None
    if orb:
        rows.append(["IOD", orb["iod"], ""])
        for k in ("radial", "in_track", "cross_track"):
            if orb[k + "_m"] is not None:
                rows.append([k.replace("_", "-"), orb[k + "_m"], "m"])
    if kind == "clock" and d.get("clock_m") is not None:
        rows.append(["clock", d["clock_m"], "m"])
    for key, label, unit in (
        ("code_bias_m", "code bias", "m"),
        ("phase_bias_cyc", "phase bias", "cyc"),
    ):
        for sig, v in (d.get(key) or {}).items() if kind == "bias" else ():
            if v is not None:
                rows.append(["%s %s" % (label, sig), v, unit])
    return rows


def _rows(bits, layout, glo=False):
    out = []
    for label, start, width, kind, scale, unit in layout:
        try:
            if glo:
                hi = start[0] if isinstance(start, tuple) else start
                v = _glo(bits, hi, width)
                if kind == "m":
                    v = -(v & ((1 << (width - 1)) - 1)) if v >> (width - 1) else v
            else:
                v = _read(bits, start, width, kind)
        except (IndexError, ValueError, TypeError):
            continue
        if kind == "deg":
            v, unit = v * scale * 180.0, "°"
        else:
            v = v * scale
        out.append(
            [label, int(v) if float(v).is_integer() and abs(v) < 1e15 else v, unit]
        )
    return out


def _text(bits, at, n):
    s = "".join(chr(_u(bits, at + 8 * i, 8)) for i in range(n))
    return "".join(c if 32 <= ord(c) < 127 else "·" for c in s)


GLO_FRAME5_14 = [
    ("B1, UT1-UTC(SU)", (80, 11), 11, "m", P(-10), "s"),
    ("B2, its rate", (69, 10), 10, "m", P(-16), "s/day"),
    ("K_P, leap second notice", (59, 2), 2, "u", 1, ""),
]


def parse(signal, kind, num, bits, frame=None):
    """(title, [[label, value, unit], ...]) for one stored message: kind is the message's type
    word ("type", "word", "page", "subframe", "string"), num its number; frame, for a GLONASS
    string, its frame in the superframe (1-5). ("", []) when there is nothing to read.
    """
    if signal in ("GPS_L5_CNAV", "GPS_L2C_CNAV"):
        title, lay = CNAV.get(num, ("", []))
        rows = _rows(bits, CNAV_HDR + lay)
        if num in CNAV_PACKETS:
            at, n = CNAV_PACKETS[num]
            for k in range(n):
                p = _rows(
                    bits,
                    [
                        (l, at + 31 * k + o, w, kd, sc, u)
                        for l, o, w, kd, sc, u in CNAV_PACKET
                    ],
                )
                if p and p[0][1]:
                    rows += [["packet %d %s" % (k + 1, l), v, u] for l, v, u in p]
        if num in CNAV_TEXT:
            rows.append(["text", _text(bits, *CNAV_TEXT[num]), ""])
        return title, rows
    if signal in ("GAL_E5BI_INAV", "GAL_E1B_INAV"):
        title, lay = INAV.get(num, ("", []))
        return title, _rows(bits, lay)
    if signal == "GAL_E5AI_FNAV":
        title, lay = FNAV.get(num, ("", []))
        return title, _rows(bits, lay)
    if signal == "GAL_E6B_CNAV":
        if _read(bits, 14, 24, "u") == 0xAF3BC3:
            return "dummy page (no HAS)", []
        return "HAS page", _rows(bits, E6B_PAGE)
    if signal == "BDS_B3I_D1":
        title, lay = D1.get(num, ("", []))
        rows = _rows(bits, [("SOW", (18, 8, 30, 12), 20, "u", 1, "s")] + lay)
        page = _read(bits, 43, 7, "u") if num in (4, 5) else None
        if num == 5 and page == 10:
            title, rows = "BDT-UTC (page 10)", rows + _rows(bits, D1_UTC)
        elif num == 5 and page in (7, 8):
            title = "satellite health (page %d)" % page
        elif num == 5 and page == 9:
            title = "BDT-GPS, -Galileo, -GLONASS offsets (page 9)"
        elif page:
            title = "almanac (page %d)" % page
        return title, rows
    if signal == "BDS_B2A_BCNAV2":
        title, lay = BCNAV2.get(num, ("", []))
        rows = _rows(bits, lay)
        if num in BC2_PACKETS:
            at, n = BC2_PACKETS[num]
            for k in range(n):
                p = _rows(
                    bits,
                    [
                        (l, at + 38 * k + o, w, kd, sc, u)
                        for l, o, w, kd, sc, u in BC2_PACKET
                    ],
                )
                if p and p[0][1]:
                    rows += [["packet %d %s" % (k + 1, l), v, u] for l, v, u in p]
        return title, rows
    if signal == "BDS_B2B_BCNAV3":
        title, lay = BCNAV3.get(num, ("", []))
        return title, _rows(bits, [("SOW", 6, 20, "u", 1, "s")] + lay)
    if signal == "GLO_L2OF":
        if num in GLO:
            title, lay = GLO[num]
        elif frame == 5 and num == 14:
            title, lay = "UT1 and leap second (frame 5)", GLO_FRAME5_14
        elif frame == 5 and num == 15:
            title, lay = "frame 5 tail", [("l_n", 9, 1, "u", 1, "")]
        elif 6 <= num <= 15:
            title = "almanac (frame %s)" % frame if frame else "almanac"
            lay = GLO_ALM_A if num % 2 == 0 else GLO_ALM_B
        else:
            return "", []
        return title, _rows(bits, lay, glo=True)
    if signal == "SBAS_L5_DFMC":
        if num == 32:
            return SBAS[32], _rows(bits, SBAS_MT32)
        if num == 31 and len(bits) >= 226:
            slots = [str(k + 1) for k in range(214) if bits[10 + k]]
            return SBAS[31], _rows(bits, [SBAS_MT31_IODM]) + [
                ["slots", " ".join(slots), ""]
            ]
        return SBAS.get(num, ""), []
    return "", []
