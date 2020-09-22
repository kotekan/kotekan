# === Start Python 2/3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614

# === End Python 2/3 compatibility

from future import standard_library

standard_library.install_aliases()
import time
import threading
import socket
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

UDP_IP = "127.0.0.1"
UDP_PORT = 2054
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

length = 1024
freqs = 1024
times = 128
integration = 128
waterfall = np.zeros((freqs // 4, times), dtype=np.float32)

"""
idx=0
while True:
	idx=(idx+1)%times
	print(idx)
	for i in np.arange(integration):
		data, addr = sock.recvfrom(length)
		d=np.fromstring(data,dtype=np.int8)
		waterfall[:,idx]+=d
"""


def updatefig(*args):
    global waterfall
    p.set_data(waterfall)
    return (p,)


def data_listener():
    idx = 0
    debuf = False
    while True:
        idx = (idx + 1) % times
        for _ in np.arange(integration):
            data, addr = sock.recvfrom(length)
            d = np.fromstring(data, dtype=np.int8)
            waterfall[:, idx] += d.reshape(-1, 4).mean(axis=1)


thread = threading.Thread(target=data_listener)
thread.daemon = True
thread.start()

f, ax = plt.subplots()
plt.ioff()
p = ax.imshow(waterfall, aspect="auto", animated=True)
ani = animation.FuncAnimation(f, updatefig, frames=100, interval=100)
f.show()
