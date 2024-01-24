"""Full-array baseband conversion backends for CHIME Outriggers"""

"""
    ARCHIVER_MOUNT : The directory hosting a YYYY/MM/DD/astro_EVENTID/baseband_EVENTID_FREQID.h5 filetree.
    NUM_THREADS : The number of threads to use in multiprocessing.
    KOTEKAN_CONFIG : A path to the kotekan config for the baseband receiver. This tells us the size of the kotekan buffer frames to use.
    USE_L4_DB : Whether to use L4 DB to check whether the events are done converting. If False, we will look for the last modified timestamp to determine whether .data files are ready to convert to .h5
    RAW_PATH : Path holding filetree of the format baseband_raw_EVENTID/baseband_EVENTID_FREQID.data
    PROMETHEUS_GW : The local hostname and port at which Prometheus is receiving metrics.
    COCO_URL : The HTTP endpoint which reports baseband status (i.e. where is coco running?)
"""


def get_backend(name):
    chime_backend = {
        "ARCHIVER_MOUNT": "/data/chime/baseband/raw",
        "NUM_THREADS": 20,
        "KOTEKAN_CONFIG": "/home/calvin/kotekan/config/chime_science_run_recv_baseband.yaml",
        "USE_L4_DB": False,
        "RAW_PATH": "/data/baseband_raw/",
        "PROMETHEUS_GW": "frb-vsop.chime:9091",
        "COCO_URL": "http://csBfs:54323/baseband-status",
    }

    kko_backend = {
        "ARCHIVER_MOUNT": "/data/kko/baseband/raw",
        "NUM_THREADS": 5,
        "KOTEKAN_CONFIG": "/home/calvin/baseband_commissioning/kotekan/config/chime_kko_baseband_recv.j2", # POINT TO MOUNTED FILE
        "USE_L4_DB": False,
        "RAW_PATH": "/tank/baseband_raw",
        "PROMETHEUS_GW": "aux:9091",
        "COCO_URL": "http://aux:54323/baseband-status",
    }  # a conv_backend for prometheus integration without L4DB integration (is this needed for outriggers?) or datatrails integration.

    if name == "chime":
        return chime_backend
    if name == "kko":
        return kko_backend
