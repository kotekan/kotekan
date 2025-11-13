#!/usr/bin/env python
"""
Script for getting network data from kotekan DPDK ports and displaying it.

For example:

    # ./network_stat.py green 4
    -------------------------------------------------------------
    Port: 0; RX Rate: 7.3113 Gbps; pps: 195280.0; loss: 0.024733%
    Port: 1; RX Rate: 7.3104 Gbps; pps: 195257.0; loss: 0.022017%
    Port: 2; RX Rate: 7.3133 Gbps; pps: 195334.4; loss: 0.000000%
    Port: 3; RX Rate: 7.3125 Gbps; pps: 195313.1; loss: 0.000000%

"""

import json
import requests
import time
import argparse


class Port(object):
    def __init__(self, hostname, port_number):
        """Return a Customer object whose name is *name* and starting
        balance is *balance*."""
        self.hostname = hostname
        self.port_number = port_number
        self.rx_bytes = 0
        self.rx_packets = 0
        self.rx_lost_packets = 0
        self.last_update = -1

    def update(self):
        resp = requests.get(
            "http://"
            + self.hostname
            + ":12048/dpdk/handlers/"
            + str(self.port_number)
            + "/port_data"
        )
        json_data = json.loads(resp.text)
        cur_rx_bytes = json_data["rx_bytes_total"]
        cur_rx_packets = json_data["rx_packets_total"]
        cur_rx_lost_packets = json_data["lost_packets"]

        cur_time = time.time()

        time_diff = cur_time - self.last_update
        rx_bytes_diff = cur_rx_bytes - self.rx_bytes
        rx_packets_diff = cur_rx_packets - self.rx_packets
        rx_lost_packets_diff = cur_rx_lost_packets - self.rx_lost_packets

        ret = {}
        ret["Gbps"] = 8 * rx_bytes_diff / time_diff / (1000 * 1000 * 1000)
        ret["loss_percent"] = (
            100.0
            * float(rx_lost_packets_diff)
            / float(rx_packets_diff + rx_lost_packets_diff)
        )
        ret["pps"] = rx_packets_diff / time_diff

        self.rx_bytes = cur_rx_bytes
        self.rx_packets = cur_rx_packets
        self.rx_lost_packets = cur_rx_lost_packets
        self.last_update = cur_time

        return ret


parser = argparse.ArgumentParser()
parser.add_argument(
    "host", help="The host name of the system you want to get stats from."
)
parser.add_argument("num_ports", help="The number of ports on the system")
args = parser.parse_args()

hostname = str(args.host)
num_ports = int(args.num_ports)

ports = []
first_time = True

for i in range(num_ports):
    ports.append(Port(hostname, i))

while True:

    for i in range(num_ports):
        stats = ports[i].update()
        if first_time == False:
            print(
                "Port: "
                + str(i)
                + "; RX Rate: "
                + "{:6.4f}".format(stats["Gbps"])
                + " Gbps; pps: "
                + "{:8.1f}".format(stats["pps"])
                + "; loss: "
                + "{:3.6f}".format(stats["loss_percent"])
                + "%"
            )

    first_time = False
    print("-------------------------------------------------------------")
    time.sleep(2)
