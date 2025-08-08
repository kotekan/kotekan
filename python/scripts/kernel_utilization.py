# === Start Python 2/3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614

# === End Python 2/3 compatibility

import requests
from tabulate import tabulate
import os
import argparse

parser = argparse.ArgumentParser(
    description="Query Kotekan GPU timings and display the results"
)
parser.add_argument(
    "-i",
    help="ID(s) of GPUs",
    dest="gpu_ids",
    type=int,
    default=[0, 1, 2, 3],
    nargs="+",
)
parser.add_argument(
    "-n", help="Host name", dest="host_name", type=str, default="localhost"
)

args = parser.parse_args()

headers = {"Content-type": "application/json"}
r = []
json_data = []
gpu_ids = args.gpu_ids
host_name = args.host_name

api_base = "http://" + host_name + ":12048"

endpoints = requests.get(api_base + "/endpoints")
endpoints = endpoints.json()["GET"]
endpoints = [e for e in endpoints if e.startswith("/gpu_profile/")]

pipelines = []
for i, gpu_id in enumerate(gpu_ids):
    for e in endpoints:
        if not e.endswith("/gpu_%i" % gpu_id):
            continue
        pipeline_id = e.replace("/gpu_profile/", "").replace("/gpu_%i" % gpu_id, "")
        try:
            R = requests.get(api_base + e)
        except requests.exceptions.RequestException as err:
            print("Error getting data: ", err)
            exit(-1)
        pipelines.append((gpu_id, pipeline_id, R.json()))

kernels = {}
copy_ins = {}
copy_outs = {}

for gpu_id, pipeline_id, json in pipelines:
    if not gpu_id in kernels:
        kernels[gpu_id] = []
        copy_ins[gpu_id] = []
        copy_outs[gpu_id] = []
    # Kernel tables
    try:
        for kernel in json["kernel"]:
            ktime = kernel["time"]
            if ktime is None:
                ktime = 0.0
            kernels[gpu_id].append(
                [
                    os.path.basename(kernel["name"]),
                    "%.6f" % ktime,
                    "%.4f" % ((kernel["utilization"] or 0.0) * 100) + "%",
                ]
            )
    except Exception:
        print("No kernel profiling data returned for GPU " + str(gpu_ids[gpu_id]))
        print("You may need to set `-i` to not include some of the GPUs")
        exit(-1)
    kernels[gpu_id].append(
        [
            "Total for " + pipeline_id,
            "%.6f" % (json["kernel_total_time"] or 0.0),
            "%.4f" % ((json["kernel_utilization"] or 0.0) * 100) + "%",
        ]
    )

    for copy_in in json["copy_in"]:
        copy_ins[gpu_id].append(
            [
                copy_in["name"],
                "%.6f" % copy_in["time"],
                "%.4f" % (copy_in["utilization"] * 100) + "%",
            ]
        )
    copy_ins[gpu_id].append(
        [
            "Total for " + pipeline_id,
            "%.6f" % json["copy_in_total_time"],
            "%.4f" % (json["copy_in_utilization"] * 100) + "%",
        ]
    )

    for copy_out in json["copy_out"]:
        copy_outs[gpu_id].append(
            [
                copy_out["name"],
                "%.6f" % (copy_out["time"] or 0.0),
                "%.4f" % ((copy_out["utilization"] or 0.0) * 100) + "%",
            ]
        )
    copy_outs[gpu_id].append(
        [
            "Total for " + pipeline_id,
            "%.6f" % (json["copy_out_total_time"] or 0.0),
            "%.4f" % ((json["copy_out_utilization"] or 0.0) * 100) + "%",
        ]
    )

for gpu_id in range(0, len(gpu_ids)):
    if not gpu_id in kernels:
        continue
    print("| -------- GPU[" + str(gpu_ids[gpu_id]) + "] Kernel timing --------")
    print(
        tabulate(
            kernels[gpu_id],
            headers=["Kernel name", "time", "utilization"],
            tablefmt="orgtbl",
        )
    )
    print("| -------- Host->GPU DMA timing --------")
    print(
        tabulate(
            copy_ins[gpu_id],
            headers=["Copy in name", "time", "utilization"],
            tablefmt="orgtbl",
        )
    )
    print("| -------- GPU->Host DMA timing --------")
    print(
        tabulate(
            copy_outs[gpu_id],
            headers=["Copy out (GPU->host)", "time", "utilization"],
            tablefmt="orgtbl",
        )
    )
    print("")
