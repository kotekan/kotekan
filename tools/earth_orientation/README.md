# Earth Orientation Parameter Management for Kotekan

The Earth Orientation Parameters (EOP) are 3 time dependent real numbers which encode the current orientation and rotational state of the Earth.  Knowledge of these parameters allows one to transform from Earth-fixed coordinates (the terrestrial reference system ITRS) to a celestial frame where the Earth is freely rotating and wobbling (the non-rotating intermediate celestial frame CIRS).  A full transformation to ICRS (where RA and Dec live) also requires knowledge of Earth's precession and nutation, but this is not needed in the Kotekan real-time system.

The three EOP are `delta_UT1_UTC` (the difference between UT1 time and UTC, which with the time encodes the current rotation angle of the Earth), and the measures of polar motion `x_pm` and `y_pm` which locate Earth's rotational axis on its surface.  These three quantities change stochastically in time, and are constantly measured and reported by the Internation Earth Rotation and Reference Systems Service (IERS).

In Kotekan we need the EOP to compute phase factors to allow our visibility matrices and formed beams to track the moving sky. Kotekan manages EOP internally with an EOP table: a list of EOP values and the times at which they apply.  For times between table entries we perform linear interpolation between the adjacent values in the table, the same strategy as `astropy`.

Since EOP are constantly changing in unpredictable ways, we must update the EOP table regularly with new values.  It is important these updates:
- Are applied uniformly to all Kotekan instances (so all instances have identical tables)
- Maintain continuity of the EOP in time (no sudden jumps in values, which would cause jumps in data, when updates come in).
- Are structured so even when nodes receive updates at slightly different times, the current EOP used in _all_ running Kotekan instances are identical.

The Kotekan update endpoint is simple and just replaces the entire table, so we must be smart in constructing an update table which will satisfy these requirements.

To do this:
- The EOP table entries are aligned to a regular grid of times, independent of the specific hour & minute when the updated table are constructed.  By default all EOP entries occur at UTC midnight, the same grid as the IERS data.
- Values currently used by Kotekan are not touched. These are the points marking the beginning and end of the current interval, e.g. the previous and next UTC midnights, which are currently being interpolated between. Only points _after_ the _next_ UTC midnight are updated. The next UTC midnight and all points earlier are copied from the current table, which must be read from a running Kotekan instance. Sufficiently old entries can be dropped.
- The update might take some time (slow networks, etc), so when determining which points to keep and which to update we are conservative and include a short buffer time (1 hour by default) which is added to the current time to determine the active interval.  If updating close to UTC midnight (~7pm Ontario, ~4pm BC), this may cause us to keep an extra days of old EOP.

In this directory are two executable scripts: `generateEOPTable.py` and `broadcastEOPTable.py`.  Both can be invoked with `-h` to get a summary of command-line arguments.

In regular operating conditions these scripts (or equivalent functionality) should be run daily.

### Generating a fresh EOP Table (for starting Kotekan cold)

For a cold start of Kotekan, you need an EOP Table in the initial `config.yaml`. To generate this table you need to get the `frame0` time from `fpga_master`, specify the number of days the table should cover, and optionally provide a filename to write the table as json to.

Run the following, where `FPGA_HOSTNAME` and `FPGA_PORT` are the host (e.g. localhost) and port (e.g. 1234) through which you can access `fpga_master`.

```bash
$ python generateEOPTable.py --frame0-src fpga_master -fh FPGA_HOSTNAME -fp FPGA_PORT -na 7 --enforce-continuity no -o eop_init.json
```

This prints some diagnostic information to the screen, including the full EOP table that can be copy-pasted into a `config.yaml`. The default time spacing (UTC midnights) is used. The table will cover the current day (by default, "current time" is the system time on the machine running the script) as well as 2 days before (controllable with `-nb`/`--num-intervals-before`, default 2) and 7 days after (`-na`/`--num-intervals-after`, default 3). Thus this table will cover 10 days, comprising 11 table entries (start of the first day to end of the last).

By default, the script will use cached IERS data, but re-download if the cached data is more than 10 days old. A download can be forced by passing `--force-iers-download`.

The new table will be constructed entirely using the current IERS data, it will not attempt to talk to Kotekan to enforce the table is continuous (`--enforce-continuity no`, default is `yes`).

The EOP table will also be written to `eop_init.json`. This json is in the format for POSTing to Kotekan.

### Generate a new daily EOP table for a running Kotekan instance

As IERS data is updated, the predictions for the current and near-future EOP improve. Hence we should always use the most up-to-date EOP estimation possible, which means refreshing the table ~daily.

If Kotekan is running we can use it for `frame0` time as well as for its current table, which is needed to enforce continuity between the old table and our new table.

Run the following, where `KOTEKAN_HOSTNAME` and `KOTEKAN_PORT` are the host and port where Kotekan can be accessed.

```bash
$ python generateEOPTable.py --frame0-src kotekan -kh KOTEKAN_HOST -kp KOTEKAN_PORT --force-iers-download --merge-cushion-dt 30min -o eop_daily.json
```

This will generate a new table in `eop_daily.json` which will be identical to Kotekan's current table for at least 30 minutes (default 1hr) into the future, new values will be appended to the table (to fulfill the `num_intervals_after` value, deafult 3), and past values (beyond the ``num_intervals_before` cutoff) will be dropped.

### Broadcast an EOP table to running Kotekan instances

Once you have generated an EOP table, you need to get it into Kotekan.  If Kotekan is running, this is done via a REST POST call. `broadcastEOPTable.py` is a script to update several instances of running Kotekan 

```bash
$ python broadcastEOPTable.py eop_daily.json --eop-post-endpoint earth_rotation_data --broadcast-list HOST1 PORT1 HOST2 PORT2 HOST3 PORT3
```

This checks Kotekan is running on all host/port's, then sends `eop_daily.json` to the `/earth_rotation_data` endpoint (which is specified in the Kotekan config).
