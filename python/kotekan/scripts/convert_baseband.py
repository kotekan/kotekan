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
from glob import glob
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

NUM_THREADS = 5

def convert(file_name, config_file, converted_filenames, backend):
    """Convert the raw data file to hdf5 using parameters specified at the backend level.
    Then delete the raw file.
    """
    backend = conv_backends.get_backend(backend)
    config_file = backend['KOTEKAN_CONFIG']
    root = backend['ARCHIVER_MOUNT']
    converted_file = baseband_archiver.convert(
        [file_name], config_file, root
    )[0]
    converted_filenames[file_name] = converted_file
    # TODO: add hook for datatrail here in the future.
    if os.path.exists(converted_file):
        os.system(f"rm -f {file_name}")


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

def is_ready(event,coco_url):
    """Ask coco about the status of the event and determine if its ready to be converted.
    If coco is down, assumes that the event is ready to be converted.

    Parameters
    ----------
    event : pair
        An event pair as returned by events_from_db().

    coco_url : str
        The coco url to query: it should end in the "baseband-status" endpoint.
    """
    ready = False
    response = requests.get(
        f"{coco_url}?event_id={event[0]}",
        headers={"Content-Type": "application/json"},
        data='{"coco_report_type": "FULL"}',
        timeout=3
    )
    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError:
        return True
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
            print(
                "Ready (error, done, num good nodes*4) ",
                files_error,
                files_done,
                num_good_nodes * 4,
            )
    return ready


def is_unlocked(path):
    """Check for the existence of lock files."""
    locked_files = glob(path + "/.*.lock")
    if len(locked_files) == 0:
        return True
    else:
        print("Still locked: ", len(locked_files))
        return False


def validate_file_existence(files):
    """Confirm that all converted files are located in the appropriate locations."""
    missing = []
    exists = True
    for f in files:
        if not os.path.exists(f):
            missing.append(f)
            exists = False
    return exists, missing


def convert_data(e, num_threads,sqlite, conn, conv_backend):
    """Main conversion function to track the conversion of events.

    Parameters
    ----------
    e : event
        Must have e[0] as event_id, and e[1] as the rough datetime.datetime of the event.

    num_threads : int
        Default 5, but up to 20 works well.

    sqlite :  
        Database connection stuff. If None, will not attempt to update L4-DB.

    conn : Database connection stuff.
        Database connection stuff. If None, will not attempt to update L4-DB.
    """
    try:
        ready = is_ready(e,conv_backend['COCO_URL'])
    except requests.exceptions.HTTPError:
        ready = True # convert if X engine is down
    raw_folder = raw_path_from_event_id(e[0],conv_backend["RAW_PATH"])
    if (
        ready is True
        and not os.path.exists(raw_folder)
        and datetime.datetime.utcnow()
        > datetime.datetime.strptime(e[1], "%Y-%m-%d %H:%M:%S.%f")
        + datetime.timedelta(hours=1)
    ):
        print(f"data path: {raw_folder} not found; skipping conversion")
        # TODO: set database status to `MISSING` and exit.
        if backend['USE_L4_DB']: # only CHIME data uses L4_DB
            print("Updating state in sqlite DB to MISSING")
            sqlite.execute(
                f"UPDATE conversion SET status = 'MISSING' WHERE event_no = {e[0]}"
            )
            conn.commit()
        return
    print("continuing to check if unlocked")
    unlocked = False
    if os.path.exists(raw_folder):
        unlocked = is_unlocked(raw_folder)

    if unlocked is True or datetime.datetime.utcnow() > datetime.datetime.strptime(
        e[1], "%Y-%m-%d %H:%M:%S.%f"
    ) + datetime.timedelta(hours=6):
        dp = os.listdir(raw_folder)
        files = [os.path.join(raw_folder, f) for f in dp]
        if not unlocked:
            for f in dp:
                if os.path.exists(os.path.join(raw_folder, "." + f + ".lock")):
                    fp = os.path.join(raw_folder, "." + f + ".lock")
                    os.system(f"rm -f {fp}")
        num_files = len(files)
        print(f"Found {num_files} files.")

        if num_files > 0:
            # make entry in datatrail, local file with local DB with state CONVERTING
            print("Starting conversion.")
            if sqlite is not None and conn is not None:
                print("Updating state in sqlite DB")
                sqlite.execute(f"INSERT INTO conversion VALUES ({e[0]}, 'CONVERTING')")
                conn.commit()
            converted_files = []
            for i in range(0, len(files), num_threads):
                chunk = files[i : i + num_threads]
                print(chunk)
                threads = []
                manager = multiprocessing.Manager()
                converted_filenames = manager.dict()
                try:
                    config_file = os.path.join(
                        os.path.dirname(os.path.abspath(__file__)),
                        conv_backend["KOTEKAN_CONFIG"], # can this take .j2 files?
                    )
                except FileNotFoundError:
                    print(f'Could not find Kotekan config file at {conv_backend["KOTEKAN_CONFIG"]}, perhaps you need to run python config.py to convert .j2 to .yaml?')
                for f in chunk:
                    th = multiprocessing.Process(
                        target=convert, args=(f, config_file, converted_filenames)
                    )
                    th.start()
                    threads.append(th)
                for th in threads:
                    th.join()
                converted_files += converted_filenames.values()
            exists, missing = validate_file_existence(converted_files)
            if exists:
                # UPDATE local DB with state FINISHED
                print("conversion done.")
                # COMMENTING THIS OUT FOR PCO commissioning os.system(f"rm -rf " + raw_folder)
                if sqlite is not None and conn is not None:
                    print("Updating state in sqlite DB")
                    sqlite.execute(
                        f"UPDATE conversion SET status = 'FINISHED' WHERE event_no = {e[0]}"
                    )
                    conn.commit()
            else:
                # TODO: send alert/metric
                print(
                    "Failed to successfully convert all files. Run the conversion script manually on these files."
                )
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


def get_size(start_path="."):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size


def check_inventory(raw_filepath):
    total_volume = 0
    for event in os.listdir(raw_filepath):
        total_volume += round(get_size(os.path.join(raw_filepath, event)) / 1024 ** 3, 2)
    return total_volume

def events_from_db():
    """Returns a list of events since the last one converted, and the last event converted."""
    db = connect_db()
    conn, sqlite = connect_conversion_db()
    last_event = fetch_last_converted_event(sqlite)
    events = fetch_events(db, last_event[0])
    return events, last_event

def yyyymmdd_from_event_id(event_id,archiver_mount):
    if str(event_id)[0:4] in ["2019", "2020", "2021", "2022", "2023", "2024"]:
        return event_id[0:4], event_id[4:6], event_id[6:8]
    candidates = []
    p = os.path.join(
        archiver_mount, "20[0-9][0-9]/[0-9]*/[0-9]*/*"
    )  # yyyy/mm/dd/KNOWN-PULSAR_[event_id] gets caught by this pattern
    all_dirs = glob(p)
    candidates += [
        os.path.relpath(d, base_dir) for d in all_dirs if str(event_id) in d
    ]
    if len(candidates) == 0:
        raise OSError("Event {event_id} not found in {archiver_mount}/YYYY/MM/DD") 
    yyyy = candidates[0][0:4]  # e.g. 2021/06/04, no leading or trailing slashes
    mm = candidates[0][5:7]
    dd = candidates[0][8:10]
    return yyyy, mm, dd

def raw_path_from_event_id(event_id,raw_filepath):
    return os.path.join(raw_filepath,f'baseband_raw_{event_id}')

def h5_path_from_event_id(event_id,archiver_mount):
    path = glob(os.path.join(archiver_mount, f'*/*/*/*_{event_id}'))
    if len(path) == 0:
        raise OSError("Event {event_id} not found in {archiver_mount}/YYYY/MM/DD") 
    else:
        return path[0]

def events_from_filepath(raw_filepath,archiver_mount):
    """If we are not using the L4-db to get the event to be converted, we need to return an adequate list of events. Do this by comparing the raw and converted directory trees.

    raw_filepath : str
        Path to a folder which has sub-folders of the form baseband_raw_[event_id], as seen relative to the conversion node e.g. "/data/baseband_raw" on cfdn9.

    archiver_mount : str
        Path to a folder which has sub-folders of the form YYYY/MM/DD as seen relative to the conversion node e.g. "/data/chime/baseband/raw" on cfdn9.

    To remain compatible with events_from_db, events_from_filepath must support the following interface:
    - e[0] must be an integer.
    - e[1] must be a datetime object corresponding roughly to the file creation time.
    """
    all_raw_folders = glob(os.path.join(raw_filepath, '*'))
    raw_event_ids = [int(folder.split('_')[-1]) for folder in all_raw_folders]
    
    all_h5_folders = glob(os.path.join(archiver_mount, '*/*/*/*'))
    h5_event_ids = [int(folder.split('_')[-1]) for folder in all_h5_folders]
    h5_event_ids.sort()
    # Which conversions need to be done?
    # all conversions which have not begun...
    unstarted = set(raw_event_ids) - set(h5_event_ids)
    # ...and all the unfinished conversions which are started but do not have the same number of files present in the .data and .h5 folders. 
    started = list(set(raw_event_ids).intersection(h5_event_ids))
    started.sort()
    unfinished = []
    for event_id in started:
        num_raw = len(glob(os.path.join(raw_path_from_event_id(event_id, raw_filepath),'*.data')))
        num_h5 = len(glob(h5_path_from_event_id(event_id, archiver_mount)))
        if num_raw < num_h5:
            unfinished.append(event_id)
        
    finished = set(raw_event_ids) - set(unstarted) - set(unfinished)
    finished = list(finished)
    finished.sort()
    last_event_id = finished[-1]
    todo = unfinished + list(unstarted)
    todo.sort()
    
    events = []
    for event_id in todo:
        dt = datetime.datetime.fromtimestamp(os.path.getmtime(raw_path_from_event_id(event_id, raw_filepath = raw_filepath)))
        events.append((event_id, dt.strftime("%Y-%m-%d %H:%M:%S.%f")))
    # get last_event
    
    last_dt = datetime.datetime.fromtimestamp(os.path.getmtime(h5_path_from_event_id(last_event_id, archiver_mount = archiver_mount))).strftime("%Y-%m-%d %H:%M:%S.%f")
    last_event = (last_event_id, last_dt)

    return events, last_event

def main():
    registry = CollectorRegistry()

    bce = Gauge(
        "baseband_conversion_event",
        "Event for which baseband data is being converted.",
        registry=registry,
    )
    bcle = Gauge(
        "baseband_conversion_last_event",
        "Last event for which baseband data was properly converted.",
        registry=registry,
    )
    last_active = Gauge(
        "baseband_conversion_last_active_timestamp",
        "Timestamp when the baseband conversion loop was last run..",
        registry=registry,
    )
    inventory = Gauge(
        "baseband_conversion_raw_data_inventory",
        "Inventory of the raw data on /data/baseband_raw.",
        registry=registry,
    )
    conv_backend = pco_backend # TODO: test with CHIME backend as well.
    assert os.path.exists(
        conv_backend["ARCHIVER_MOUNT"]
    ), f"{conv_backend['ARCHIVER_MOUNT']} is not mounted, it is required for this process. Exiting!!!"

    while True:
        # set last_active, inventory, and bcle.
        last_active.set(time.time())
        inv = check_inventory(conv_backend['RAW_PATH'])
        inventory.set(inv)
        if conv_backend["USE_L4_DB"]:
            # use db to keep track of conversion to do list
            events, last_event = events_from_db() 
        else:
            # directly check filepaths for conversion to do list -- at outriggers
            events, last_event = events_from_filepath(raw_filepath = conv_backend["RAW_PATH"], archiver_mount = conv_backend["ARCHIVER_MOUNT"])
        print(events)
        bcle.set(last_event[0])

        # push metrics
        push_to_gateway(
            conv_backend["PROMETHEUS_GW"], job="baseband_conversion", registry=registry
        )
        if len(events) == 0:
            bce.set(0)
            push_to_gateway(
                conv_backend["PROMETHEUS_GW"], job="baseband_conversion", registry=registry
            )
        else:
            for e in events:
                print(f"converting event {e}")
                bce.set(e[0])
                push_to_gateway(
                    conv_backend["PROMETHEUS_GW"], job="baseband_conversion", registry=registry
                )
                if conv_backend["USE_L4_DB"]:
                    conn, sqlite = connect_conversion_db()
                else:
                    conn = None
                    sqlite = None
                convert_data(e, NUM_THREADS,sqlite, conn,conv_backend) # if sqlite or conn are None, will look for events using filepaths.
        sys.stdout.flush()
        time.sleep(300)


if __name__ == "__main__":
    main()
