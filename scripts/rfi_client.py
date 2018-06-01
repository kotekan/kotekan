"""
/*********************************************************************************
* RFI Documentation Header Block
* File: rfi_client.py
* Purpose: A client to receive and display real-time rfi data from rfi_receiver.py
* Python Version: 3.6 
* Dependencies: Matplotlib, yaml, numpy, argparse
* Help: Run "python3 rfi_client.py" -H (or --Help) for how to use.
*********************************************************************************/
"""
import threading
import socket
import numpy as np
import matplotlib.animation as animation
import matplotlib.dates as md
from matplotlib.widgets import Button
import datetime
import matplotlib.pyplot as plt
import random
import time
import matplotlib.dates as mdates
import os
import argparse
import yaml

def parse_dict(cmd, _dict):
    for key, value in _dict.items():
        if(type(value) == dict):
            parse_dict(cmd,value)
        else:
            if key in cmd.config.keys():
                if(type(cmd.config[key]) == type(value)):
                    print("Setting Config Paramter %s to %s" %(key,str(value)))
                    cmd.config[key] = value

class CommandLine:

    def __init__(self):

        #Defaults
        self.TCP_IP = '127.0.0.1'
        self.TCP_PORT = 2901
        self.config = {'samples_per_data_set':32768, 'timestep':2.56e-6, 'waterfallX': 1024,
                       'waterfallY': 1024, 'waterfall_request_delay': 10, 'colorscale': 1.0,
                       'sk_step': 256, 'num_elements': 2048}
        self.mode = 'pathfinder'
        self.supportedModes = ['vdif','pathfinder', 'chime']
        parser = argparse.ArgumentParser(description = "RFI Receiver Script")
        parser.add_argument("-H", "--Help", help = "Example: Help argument", required = False, default = "")
        parser.add_argument("-r", "--receive", help = "Example: 127.0.0.1:2900", required = False, default = "")
        parser.add_argument("-c", "--config", help = "Example: ../kotekan/kotekan_opencl_rfi.yaml", required = False, default = "")
        parser.add_argument("-m", "--mode", help = "Example: vdif, pathfinder", required = False, default = "")
        argument = parser.parse_args()
        status = False

        if argument.Help:
            print("You have used '-H' or '--Help' with argument: {0}".format(argument.Help))
            status = True
        if argument.receive:
            print("You have used '-r' or '--receive' with argument: {0}".format(argument.receive))
            self.TCP_IP = argument.receive[:argument.receive.index(':')]
            self.TCP_PORT = int(argument.receive[argument.receive.index(':')+1:]) 
            print("Setting TCP IP: %s PORT: %d"%(self.TCP_IP ,self.TCP_PORT ))
            status = True
        if argument.config:
            print("You have used '-c' or '--config' with argument: {0}".format(argument.config))
            parse_dict(self,yaml.load(open(argument.config)))
            print(self.config)
            status = True
        if argument.mode:
            print("You have used '-m' or '--mode' with argument: {0}".format(argument.mode))
            if(argument.mode in self.supportedModes):
                self.mode = argument.mode
                print("Setting mode to %s mode."%(argument.mode))
            else:
                print("This mode in currently not supported, reverting to default")
                print("Supported Modes Include:")
                for mode in self.supportedModes:
                    print("- ",mode)
            status = True
        if not status:
            print("Maybe you want to use -H or -s or -p or -p as arguments ?") 

def init():
    im.set_data(waterfall)

def animate(i):
    im.set_data(waterfall)
    x_lims = mdates.date2num([t_min,t_min + datetime.timedelta(seconds=waterfall.shape[1]*app.config['samples_per_data_set']*app.config['timestep'])])
    im.set_extent([x_lims[0],x_lims[1],400,800])
    return im

def recvall(sock, n):
    # Helper function to recv n bytes or return None if EOF is hit
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data

def savewaterfall():

    global  waterfall, t_min

    if not os.path.exists("PathfinderLiveData"):
        os.makedirs("PathfinderLiveData")

    np.save("PathfinderLiveData/" + t_min.strftime("%d%m%YT%H%M%SZ") ,waterfall)
    return

def data_listener():

    global sock_tcp, waterfall, addr, t_min, app

    WATERFALLMESSAGE = "W"
    TIMEMESSAGE = "T"

    timesize = len(t_min.strftime('%d-%m-%YT%H:%M:%S:%f'))
    waterfallsize = 8*waterfall.size #Bytes
    delay = app.config['waterfall_request_delay']

    while True:

        sock_tcp.send(WATERFALLMESSAGE.encode())

        data = recvall(sock_tcp, waterfallsize)
        #data = sock_tcp.recv(waterfallsize)
        if(data == None):
            print("Connection to %s:%s Broken... Exiting"%(addr[0],str(addr[1])))
            break

        waterfall = np.fromstring(data).reshape(waterfall.shape)
        #if(app.mode == 'pathfinder'):
        #    savewaterfall()
        print(waterfall)

        sock_tcp.send(TIMEMESSAGE.encode())

        data = recvall(sock_tcp, timesize).decode()

        if(data == None):
            print("Connection to %s:%s Broken... Exiting"%(addr[0],str(addr[1])))
            break

        t_min  = datetime.datetime.strptime(data, '%d-%m-%YT%H:%M:%S:%f')
        print(t_min)

        time.sleep(delay)

class Callback(object):

    def SaveData(self, event): 
        if not os.path.exists("RFIData"):
            os.makedirs("RFIData")
        newDir = "RFIData/" + datetime.datetime.utcnow().strftime('%Y%m%dT%H:%M:%S') #Create New Folder
        os.makedirs(newDir)
        np.save(newDir + '/data.npy', waterfall)
        plt.savefig(newDir + '/image.png')
        f = open(newDir + '/info.txt',"w") #Create Info Text File
        f.write("X Lims: %f - %f\n"%(x_lims[0], x_lims[1]))
        f.close()
        print("Saved Data! Path: ./" + newDir)

def to_confidence(app, ticks):
    M = app.config["sk_step"]*app.config["num_elements"]
    sigma = np.sqrt(4/M)
    return np.round(np.abs(1-ticks)/(sigma),1)

if( __name__ == '__main__'):

    app = CommandLine()

    plt.ion()

    #Initialize Plot
    nx, ny = app.config['waterfallY'], app.config['waterfallX']
    t_min = datetime.datetime.utcnow()
    waterfall = -1*np.ones([nx,ny])

    fig = plt.figure()

    x_lims = mdates.date2num([t_min,t_min + datetime.timedelta(seconds=waterfall.shape[1]*app.config['samples_per_data_set']*app.config['timestep'])])
    im = plt.imshow(waterfall, aspect = 'auto',cmap='viridis',extent=[x_lims[0],x_lims[1],400,800], vmin=1-app.config['colorscale'],vmax=1+app.config['colorscale'])
    ticks = np.linspace(1-app.config['colorscale'], 1+app.config['colorscale'], num = 10)
    cbar = plt.colorbar(label = "Detection Confidence", ticks = ticks)
    cbar.ax.set_yticklabels(to_confidence(app, ticks))
    plt.title("RFI Viewer (Mode: "+app.mode+")")
    plt.xlabel("Time")
    plt.ylabel("Frequency[MHz]")
    ax = plt.gca()
    ax.xaxis_date()
    date_format = mdates.DateFormatter('%H:%M:%S')
    ax.xaxis.set_major_formatter(date_format)
    fig.autofmt_xdate()
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=waterfall.size, interval=50)

    time.sleep(1)

    #Intialize TCP
    TCP_IP= app.TCP_IP
    TCP_PORT = app.TCP_PORT
    addr = (TCP_IP, TCP_PORT)
    sock_tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print("Trying to connect to %s:%s" %(addr[0],addr[1]))
    Connected = False

    while not Connected:
        try:
            sock_tcp.connect(addr)
            Connected = True
        except:
            print("Could not connect to %s:%s Trying again in 5 seconds" %(addr[0],addr[1]))
            time.sleep(5)

    thread = threading.Thread(target=data_listener)
    thread.daemon = True
    thread.start()

    callback = Callback()
    buttonLocation = plt.axes([0.81, 0.05, 0.1, 0.075])
    saveButton = Button(buttonLocation, 'Save')
    saveButton.on_clicked(callback.SaveData)


    input()
