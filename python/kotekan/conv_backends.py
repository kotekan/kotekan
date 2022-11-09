"""Full-array baseband conversion backends for CHIME Outriggers"""
def get_backend(name):
    chime_backend = {
                'ARCHIVER_MOUNT' : '/data/chime/baseband/raw',
                'NUM_THREADS' : 5,
                'BB_STATUS_URL' : "http://csBfs:54323/baseband-status",
                'KOTEKAN_CONFIG' : "../../../config/baseband_commissioning/kotekan/config/chime_science_run_recv_baseband.yaml",
                'USE_L4_DB' : True,
                'RAW_PATH' : '/data/baseband_raw/',
                'PROMETHEUS_GW' : "frb-vsop.chime:9091",
                'COCO_URL' : 'http://csBfs:54323/baseband-status'
                }

    pco_backend = {
                'ARCHIVER_MOUNT' : '/data/princeton/baseband/raw',
                'NUM_THREADS' : 5,
                'BB_STATUS_URL' : "http://aux:54323/baseband-status",
                'KOTEKAN_CONFIG' : "/home/calvin/baseband_commissioning/kotekan/config/chime_pco_gpu.j2",
                'USE_L4_DB' : False,
                'RAW_PATH' : '/data/baseband_raw/',
                'PROMETHEUS_GW' : "aux:9091",
                'COCO_URL' : 'http://aux:54323/baseband-status'
                } # a conv_backend for prometheus integration without L4DB integration (is this needed for outriggers?) or datatrails integration.


    if name == 'chime':
        return chime_backend
    if name == 'pco':
        return pco_backend
