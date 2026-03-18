# Earth Orientation Parameter Management for Kotekan

In this directory are two executable scripts: `generateEOPTable.py` and `broadcastEOPTable.py`.  Both can be invoked with `-h` to get a summary of command-line arguments.

### Generating a fresh EOP Table (for starting Kotekan cold)

For a cold start of Kotekan, you need an EOP Table in the initial `config.yaml`. To generate this table you need to get the `frame0` time from `fpga_master`, specify the number of days the table should cover, and optionally provide a filename to write the table as json to.

Run the following, where `FPGA_HOSTNAME` and `FPGA_PORT` are the host (e.g. localhost) and port (e.g. 1234) through which you can access `fpga_master`.

```bash
$ python generateEOPTable.py --frame0-src fpga_master -fh FPGA_HOSTNAME -fp FPGA_PORT -na 7 --enforce-continuity=no -o eop_init.json
```

This prints some diagnostic information to the screen, including the full EOP table that can be copy-pasted into a `config.yaml`. By default table entries are produced with 1 day spacing at successive UTC midnights, the same cadence as the official IERS entries.  The table will cover the current day (by default, "current time" is the system time on the machine running the script) as well as 2 days before (controllable with `-nb`, default 2) and 7 days after (`-na`, default 3). Thus this table will cover 10 days, comprising 11 table entries (start of the first day to end of the last).

By default, the script will use cached IERS data, but re-download if the cached data is more than 10 days old. A download can be forced by passing `--force-iers-download`.

The new table will be constructed entirely using the current IERS data, it will not attempt to talk to Kotekan to enforce the table is continuous (--enforce-continuity=no).

The EOP table will also be written to `eop_init.json`. This json is in the format for POSTing to Kotekan.

### Generate a new daily EOP table for a running Kotekan instance

As IERS data is updated, the predictions for the current and near-future EOP improve. Hence we should always use the most up-to-date EOP estimation possible, which means refreshing the table ~daily.

If Kotekan is running we can use it for `frame0` time as well as for its current table, which is needed to enforce continuity between the old table and our new table.

Run the following, where `KOTEKAN_HOSTNAME` and `KOTEKAN_PORT` are the host and port where Kotekan can be accessed.

```bash
$ python generateEOPTable.py --frame0-src kotekan -kh KOTEKAN_HOST -kp KOTEKAN_PORT --force-iers-download --merge-cushion-dt=1hr-o eop_daily.json
```

This will generate a new table in `eop_daily.json` which will be identical to `K
