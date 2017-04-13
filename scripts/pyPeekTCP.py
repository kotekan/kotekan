import time
import threading
import socket
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

TCP_IP="127.0.0.1"
TCP_PORT = 2054
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind((TCP_IP, TCP_PORT))
sock.listen(1)

length =1024
freqs=1024
times=128
integration=128
waterfall = np.zeros((freqs/4,times),dtype=np.float32)

def updatefig(*args):
    global waterfall
    p.set_data(waterfall)
    return p,

def data_listener():
	idx=0
	while True:
		idx=(idx+1)%times
		d=np.zeros(freqs)
		for i in np.arange(integration):
			data = connection.recv(length)
			d = d + np.fromstring(data,dtype=np.int8)
		waterfall[:,idx]=10*np.log10(d.reshape(-1,4).mean(axis=1))

connection, client_address = sock.accept()

thread = threading.Thread(target=data_listener)
thread.daemon = True
thread.start()

time.sleep(1)

f, ax = plt.subplots()
plt.ioff()
p=ax.imshow(waterfall,aspect='auto',animated=True,interpolation='nearest',cmap='gray',vmin=38, vmax=42)
c = f.colorbar(p)
ax.set_xlabel('Time')
ax.set_ylabel('Freq')
c.set_label('Power (dB, arbitrary)')
ani = animation.FuncAnimation(f, updatefig, frames=100, interval=500)
f.show()

