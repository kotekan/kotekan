import requests
from tabulate import tabulate
import json
import os

headers = {'Content-type': 'application/json'}

r = []
json_data = []
n_gpus = 4

for gpu_id in range(0,n_gpus):
    r.append(requests.get('http://localhost:12048/gpu_profile/' + str(gpu_id)))
    json_data.append(r[gpu_id].json())

kernels = []
copy_ins = []
copy_outs = []

for gpu_id in range(0,n_gpus):
    kernels.append([])
    # Kernel tables
    for kernel in json_data[gpu_id]["kernel"]:
        kernels[gpu_id].append([os.path.basename(kernel["name"]), '%.6f' % kernel["time"], '%.4f' % (kernel["utilization"]*100) + "%"])
    kernels[gpu_id].append(["Total:", '%.6f' % json_data[gpu_id]["kernel_total_time"], '%.4f' % (json_data[gpu_id]["kernel_utilization"]*100) + "%"])

    copy_ins.append([])
    for copy_in in json_data[gpu_id]["copy_in"]:
        copy_ins[gpu_id].append([copy_in["name"], '%.6f' % copy_in["time"], '%.4f' % (copy_in["utilization"]*100) + "%"])
    copy_ins[gpu_id].append(["Total:", '%.6f' % json_data[gpu_id]["copy_in_total_time"], '%.4f' % (json_data[gpu_id]["copy_in_utilization"]*100) + "%"])

    copy_outs.append([])
    for copy_out in json_data[gpu_id]["copy_out"]:
        copy_outs[gpu_id].append([copy_out["name"], '%.6f' % copy_out["time"], '%.4f' % (copy_out["utilization"]*100) + "%"])
    copy_outs[gpu_id].append(["Total:", '%.6f' % json_data[gpu_id]["copy_out_total_time"], '%.4f' % (json_data[gpu_id]["copy_out_utilization"]*100) + "%"])

for gpu_id in range(0,n_gpus):
    print "| -------- GPU[" + str(gpu_id) + "] Kernel timing --------"
    print tabulate(kernels[gpu_id], headers=["Kernel name", "time", "utilization"], tablefmt='orgtbl')
    print "| -------- Host->GPU DMA timing --------"
    print tabulate(copy_ins[gpu_id], headers=["Copy in name", "time", "utilization"], tablefmt='orgtbl')
    print "| -------- GPU->Host DMA timing --------"
    print tabulate(copy_outs[gpu_id], headers=["Copy out (GPU->host)", "time", "utilization"], tablefmt='orgtbl')
    print ""
