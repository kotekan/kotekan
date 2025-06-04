#!/usr/bin/env python3
import time
from astropy.time import Time
import astropy.units as units
import astropy.utils.iers
import numpy as np
import matplotlib.pyplot as plt

import updateEOPTable


if __name__ == "__main__":

    # time0_ns = read_time0_ns("http://localhost", 12048)

    # print(time0_ns)

    ts = [
        Time("2015-06-30 00:00:00"),
        Time("2015-06-30 12:00:00"),
        Time("2015-06-30 23:59:58"),
        Time("2015-06-30 23:59:58.5"),
        Time("2015-06-30 23:59:59"),
        Time("2015-06-30 23:59:59.5"),
        Time("2015-06-30 23:59:60"),
        Time("2015-06-30 23:59:60.5"),
        Time("2015-07-01 00:00:00"),
        Time("2015-07-01 00:00:00.5"),
        Time("2020-07-15 06:28:30.0"),
        Time("2020-07-15 06:28:30.123456789"),
        Time("2025-03-01 00:00:00"),
        Time("2025-03-15 00:00:00"),
        Time("2025-04-01 00:00:00"),
        Time("2025-04-15 00:00:00"),
        Time("2025-05-01 00:00:00"),
        Time("2025-05-15 00:00:00"),
        Time("2026-05-15 00:00:00"),
    ]

    for t in ts:
        ns = updateEOPTable.calc_unix_ns_from_t(t)
        t1 = updateEOPTable.calc_astropy_time_from_unix_ns(ns)
        print(t, ns, ":   ", t1)

    t0_ns = updateEOPTable.calc_unix_ns_from_t(Time("2025-03-01 00:00:00", scale="utc"))

    iers = astropy.utils.iers.IERS_Auto.open()

    updateEOPTable.build_EOP_table(ts, t0_ns, iers)

    t0 = Time("2023-01-01 00:00:00", scale="utc")
    t0_ns = updateEOPTable.calc_unix_ns_from_t(t0)

    dt = np.linspace(0.0, 0.03, 10) * units.year

    ts = t0 + dt

    eop_table = updateEOPTable.build_EOP_table(ts, t0_ns, iers)

    t_ns = np.array([eop["time_inst_ns"] for eop in eop_table])
    dut1 = np.array([eop["delta_UT1_inst"] for eop in eop_table])
    x = np.array([eop["x_pm"] for eop in eop_table])
    y = np.array([eop["y_pm"] for eop in eop_table])

    fig, ax = plt.subplots(3, 1, figsize=(12, 9))

    t_x = dt.to_value("day")

    ax[0].plot(t_x, t_ns)
    # ax[1].plot(t_x, dut1)
    ax[1].plot(t_x, dut1 - ts.delta_ut1_utc)
    ax[2].plot(t_x, x)
    ax[2].plot(t_x, y)

    fig.savefig("eop_table.png")

    iers.close()
