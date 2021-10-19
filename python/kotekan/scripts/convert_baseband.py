"""Runner script to persistently run and convert data for new events."""
from MySQLdb import _mysql
import yaml
import os
import time
import datetime
import sqlite3
import requests
import sys
import baseband_archiver
import multiprocessing

# TODO metrics and slack integration.
ARCHIVER_MOUNT = "/data/chime/baseband/raw"
NUM_THREADS = 20

def convert(file_name, config_file, converted_filenames):
    """Convert the raw data file to hdf5 and then delete the raw file."""
    converted_file = baseband_archiver.convert([file_name], config_file, root=ARCHIVER_MOUNT)[0]
    converted_filenames[file_name] = converted_file
    #TODO: add hook for datatrail here in the future.
    #os.system(f"rm -f {file_name}")

def connect_db():
    """Set up a connection to the L4 database."""
    db_config_file = "chimefrb_db.yaml"
    with open(db_config_file) as f:
        db_config = yaml.safe_load(f)
    return _mysql.connect(**db_config)


def fetch_events(db, event_no):
    """Fetch all events with basebandthat arrived after a specific event."""
    quert_str = (
        "SELECT baseband_raw_data.event_no, event_register.timestamp_utc "
        "FROM baseband_raw_data "
        "JOIN event_register ON baseband_raw_data.event_no=event_register.event_no "
        f"WHERE baseband_raw_data.event_no > {event_no};"
    )

    db.query(quert_str)
    r = db.store_result()
    events = []
    for _ in range(r.num_rows()):
        row = list(r.fetch_row()[0])
        for i in range(len(row)):
            row[i] = row[i].decode("utf-8")
        events.append(tuple(row))
    return events


def is_ready(event):
    """Ask coco about the status of the event and determine if its ready to be converted."""
    ready = False
    url = "http://csBfs:54323/baseband-status"
    response = requests.get(
        f"{url}?event_id={event[0]}",
        headers={"Content-Type": "application/json"},
        data='{"coco_report_type": "FULL"}',
    )
    response.raise_for_status()
    result = response.json()
    if not result or not result["success"]:
        print("Download check %s failed. Continuing to track." % event[0])
        return ready
    files_done = 0
    files_error = 0
    files_inprogress = 0
    num_good_nodes = 0
    num_cannot_contact = 0

    for node, node_result in result["baseband"].items():
        if isinstance(node_result, dict):
            if node_result["status"] == 200:
                for transfer in node_result["reply"]:
                    if transfer["status"] == "done":
                        files_done += 1
                    elif transfer["status"] == "error":
                        files_error += 1
                    elif transfer["status"] == "inprogress":
                        files_inprogress += 1
                num_good_nodes += 1
            else:
                print("Cannot contact %s: %s" % (node, node_result))
                num_cannot_contact += 1
        else:
            print("Cannot parse response for node %s: %s" % (node, node_result))
    if files_inprogress > 0:
        print(f"{files_inprogress} files are still being written out.")
    else:
        if files_error + files_done == num_good_nodes * 4:
            ready = True
            print("Ready (error, done, num good nodes*4) ", files_error, files_done, num_good_nodes*4)
    return ready

def validate_file_existence(files):
    """Confirm that all converted files are located in the appropriate locations."""
    missing = []
    exists = True
    for f in files:
        if not os.path.exists(f):
            missing.append(f)
            exists = False
    return exists, missing

def convert_data(sqlite, conn, e, num_threads):
    """Main conversion function to track the conversion of events."""
    datapath = f"/data/baseband_raw/baseband_raw_{e[0]}"
    convert = is_ready(e)
    if convert is True or datetime.datetime.utcnow() > datetime.datetime.strptime(
        e[1], "%Y-%m-%d %H:%M:%S.%f"
    ) + datetime.timedelta(hours=1):
        if os.path.exists(datapath):
            files = os.listdir(datapath)
            num_files = len(files)
            print(f"Found {num_files} files.")
        else:
            print(f"data path: {datapath} not found")
            files = None 
            # TODO: set database status to `MISSING` and exit.
            print("skipping conversion. Updating state in sqlite DB to MISSING")
            sqlite.execute(
                f"UPDATE conversion SET status = 'MISSING' WHERE event_no = {e[0]}"
            )
            conn.commit()
        if files is not None:
            # make entry in datatrail, local file with local DB with state CONVERTING
            print("starting conversion. Updating state in sqlite DB")
            sqlite.execute(f"INSERT INTO conversion VALUES ({e[0]}, 'CONVERTING')")
            conn.commit()
            converted_files = []
            for i in range(0, len(files), num_threads):
                chunk = files[i:i+num_threads]
                threads = []
                manager = multiprocessing.Manager()
                converted_filenames = manager.dict()
                for f in chunk:
                    th = multiprocessing.Process(target=convert,args=(f, config_file, converted_filenames))
                    th.start()
                    threads.append(th)
                for th in threads:
                    th.join()
                converted_files += converted_filenames.values()
            exists, missing = validate_file_existence(converted_files)
            if exists:
                # UPDATE local DB with state FINISHED
                print("conversion done. Updating state in sqlite DB")
                sqlite.execute(
                    f"UPDATE conversion SET status = 'FINISHED' WHERE event_no = {e[0]}"
                )
                conn.commit()
            else:
                # TODO: send alert/metric
                print("Failed to successfully convert all files. Run the conversion script manually on these files.")
                print(missing)

def connect_conversion_db():
    """Connect to the local tracking database."""
    if not os.path.exists("bb_conversion.db"):
        con = sqlite3.connect("bb_conversion.db")
        sqlite = con.cursor()
        sqlite.execute(
            """CREATE TABLE conversion
                   (event_no int, status text)"""
        )
    else:
        con = sqlite3.connect("bb_conversion.db")
        sqlite = con.cursor()
    return con, sqlite


def fetch_last_converted_event(sqlite):
    """Extract the last fully converted event."""
    event = [0]
    for row in sqlite.execute(
        "SELECT * FROM conversion WHERE status = 'FINISHED' ORDER BY event_no DESC LIMIT 1"
    ):
        event = row
    return event


def main():
    assert os.path.ismount(
        ARCHIVER_MOUNT
    ), f"{ARCHIVER_MOUNT} is not mounted, it is required for this process. Exiting!!!"
    db = connect_db()
    conn, sqlite = connect_conversion_db()
    #sqlite.execute("INSERT INTO conversion VALUES (188946317, 'FINISHED')")
    #conn.commit()
    while True:
        last_event = fetch_last_converted_event(sqlite)
        events = fetch_events(db, last_event[0])
        for e in events:
            convert_data(sqlite, conn, e, NUM_THREADS)
        sys.stdout.flush()
        time.sleep(300)


if __name__ == "__main__":
    main()
