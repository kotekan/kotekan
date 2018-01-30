/*********************************************************************************
* RFI Documentation Header Block
* File: rfi_receiver.py
* Purpose: A server to receive real-time rfi data from kotekan and send to rfi_client.py
* Python Version: 3.6 
* Dependencies: yaml, numpy, argparse
* Help: Run "python3 rfi_receiver.py" -H (or --Help) for how to use.
*********************************************************************************/

import threading
import socket
import numpy as np
import datetime
import os
import random
import time
import argparse
import yaml
import json
import sys

class CommandLine:

    def __init__(self):

        #Defaults
        self.UDP_IP= "0.0.0.0"
        self.UDP_PORT = 2900
        self.TCP_IP = '10.10.10.2'
        self.TCP_PORT = 41214
        self.mode = 'pathfinder'
        self.config = {'frames_per_packet': 64, 'num_freq': 1024, 'num_local_freq': 8, 'samples_per_data_set':32768, 'num_elements': 2,
                        'timestep':2.56e-6, 'bytes_per_freq': 16, 'waterfallX': 1024, 'waterfallY': 1024, 'vdif_rfi_header_size': 21, 'sk_step': 256}
        self.supportedModes = ['vdif','pathfinder']
        parser = argparse.ArgumentParser(description = "RFI Receiver Script")
        parser.add_argument("-H", "--Help", help = "Example: Help argument", required = False, default = "")
        parser.add_argument("-r", "--receive", help = "Example: 127.0.0.1:2900", required = False, default = "")
        parser.add_argument("-s", "--send", help = "Example: 10.10.10.2:41214", required = False, default = "")
        parser.add_argument("-c", "--config", help = "Example: ../kotekan/kotekan_opencl_rfi.yaml", required = False, default = "")
        parser.add_argument("-m", "--mode", help = "Example: vdif, pathfinder", required = False, default = "")
        argument = parser.parse_args()
        status = False

        if argument.Help:
            print("You have used '-H' or '--Help' with argument: {0}".format(argument.Help))
            status = True
        if argument.send:
            print("You have used '-s' or '--send' with argument: {0}".format(argument.send))
            self.TCP_IP = argument.send[:argument.send.index(':')]
            self.TCP_PORT = int(argument.send[argument.send.index(':')+1:])
            print("Setting TCP IP: %s PORT: %d"%(self.TCP_IP ,self.TCP_PORT ))
            status = True
        if argument.receive:
            print("You have used '-r' or '--receive' with argument: {0}".format(argument.receive))
            self.UDP_IP = argument.receive[:argument.receive.index(':')]
            self.UDP_PORT = int(argument.receive[argument.receive.index(':')+1:]) 
            print("Setting UDP IP: %s PORT: %d"%(self.UDP_IP ,self.UDP_PORT ))
            status = True
        if argument.config:
            print("You have used '-c' or '--config' with argument: {0}".format(argument.config))
            for key, value in yaml.load(open(argument.config)).items():
                if(type(value) == dict):
                    if('kotekan_process' in value.keys() and value['kotekan_process'] == 'rfiBroadcast'):
                        for k in value.keys():
                            if k in self.config.keys():
                                if(type(self.config[k]) == type(value[k])):
                                    print("Setting Config Paramter %s to %s" %(k,str(value[k])))
                                    self.config[k] = value[k]
                else:
                    if key in self.config.keys():
                        if(type(self.config[key]) == type(value)):
                            print("Setting Config Paramter %s to %s" %(key,str(value)))
                            self.config[key] = value
            print(self.config)
            status = True
        if argument.mode:
            print("You have used '-m' or '--mode' with argument: {0}".format(argument.mode))
            if(argument.mode in self.supportedModes):
                self.mode = argument.mode
                print("Setting mode to %s mode."%(argument.mode))
            else:
                print("This mode in currently not supported, reverting to default")
            status = True
        if not status:
            print("Maybe you want to use -H or -s or -p or -p as arguments ?") 

def VDIFheadercheck(header,app):
    
    if(header['combined_flag'] != 1):
        print("VDIF Header Error: Only Combined RFI values are currently supported ")
        return False
    if(header['sk_step'] != app.config['sk_step']):
        print("VDIF Header Error: SK Step does not match config")
        return False
    if(header['num_elements'] != app.config['num_elements']):
        print("VDIF Header Error: Number of Elements does not match config")
        return False
    if(header['num_times_per_frame'] != app.config['samples_per_data_set']):
        print("VDIF Header Error: Samples per Dataset does not match config")
        return False
    if(header['num_freq'] != app.config['num_freq']):
        print("VDIF Header Error: Number of Frequencies does not match config")
        return False
    return True
def data_listener():

    global sock_udp, waterfall, t_min, app

    #Config Variables
    frames_per_packet = app.config['frames_per_packet']
    local_freq = app.config['num_local_freq']
    timesteps_per_frame = app.config['samples_per_data_set']
    timestep = app.config['timestep']
    bytesPerFreq = app.config['bytes_per_freq']
    global_freq = app.config['num_freq']
    sk_step = app.config['sk_step']
    vdifRFIHeaderSize = app.config['vdif_rfi_header_size']
    mode = app.mode
    #Intialization Variables
    firstPacket = True
    vdifPacketSize = global_freq*4 + vdifRFIHeaderSize

    while True:

        if (mode == 'pathfinder'):

            #Receive packet from port
            packet, addr = sock_udp.recvfrom(frames_per_packet*local_freq*bytesPerFreq)
            
            if(packet != ''):

                print('Receiving UDP Packet...')
                data = np.fromstring(packet,dtype=np.dtype([('bin', 'i4',1), ('seq', 'i8',1), ('mask', 'f4',1)]))

                if(firstPacket):

                    t_min = datetime.datetime.utcnow()
                    min_seq = np.min(data['seq'])
                    max_seq = min_seq + (waterfall.shape[1]-1)*timesteps_per_frame
                    firstPacket = False

                else:

                    new_max = np.max(data['seq'])

                    if(new_max > max_seq):

                        roll_amount = int(-1*max((new_max-max_seq)/timesteps_per_frame,waterfall.shape[1]/8))
                        
                        #DO THE ROLL
                        waterfall = np.roll(waterfall,roll_amount,axis=1)
                        waterfall[:,roll_amount:] = -1

                        #Adjust Times
                        min_seq += -1*roll_amount*timesteps_per_frame
                        max_seq += -1*roll_amount*timesteps_per_frame
                        t_min += datetime.timedelta(seconds=-1*roll_amount*timestep*timesteps_per_frame)

                waterfall[(data['bin']).astype(int),((data['seq']-min_seq)/timesteps_per_frame ).astype(int)] = data['mask']
                #print(data['bin'])

        elif (mode == 'vdif'):

            packet, addr = sock_udp.recvfrom(vdifPacketSize)

            if(packet != ''):

                print('Receiving UDP Packet...')

                header = np.fromstring(packet[:vdifRFIHeaderSize],dtype=np.dtype([('combined_flag', np.uint8 ,1), ('sk_step', np.int32,1), ('num_elements', np.int32,1),
                                                             ('num_times_per_frame', np.int32,1), ('num_freq', np.int32,1), ('seq', np.uint32 ,1)]))
                data = np.fromstring(packet[vdifRFIHeaderSize:],dtype=np.float32)

                if(firstPacket):

                    #TODO add header check here
                    if(VDIFheadercheck(header,app) == False):
                        break
                    t_min = datetime.datetime.utcnow()
                    min_seq = header['seq']
                    max_seq = min_seq + (waterfall.shape[1] - 1)
                    firstPacket = False

                else:

                    if(header['seq'] > max_seq):

                        roll_amount = int(-1*waterfall.shape[1]/8)
                        
                        #DO THE ROLL
                        waterfall = np.roll(waterfall,roll_amount,axis=1)
                        waterfall[:,roll_amount:] = -1
                        min_seq -= roll_amount
                        max_seq -= roll_amount
                        
                        #Adjust Times
                        t_min += datetime.timedelta(seconds=-1*roll_amount*timestep*timesteps_per_frame)

                waterfall[:,int(header['seq']-min_seq)] = data
                    
                
def TCP_stream():

    global sock_tcp, waterfall, t_min

    sock_tcp.listen(1)

    while True:

        conn, addr = sock_tcp.accept()
        print('Established Connection to %s:%s' %(addr[0],addr[1]))

        while True:

            MESSAGE = conn.recv(1).decode() #Client Message

            if not MESSAGE: break

            elif MESSAGE == "W":
                print("Sending Watefall Data %d ..."%(len(waterfall.tostring())))
                conn.send(waterfall.tostring())  #Send Watefall
            elif MESSAGE == "T":
                print("Sending Time Data ...")
                print(len(t_min.strftime('%d-%m-%YT%H:%M:%S:%f')))
                conn.send(t_min.strftime('%d-%m-%YT%H:%M:%S:%f').encode())  #Send Watefall
            print(MESSAGE)
        print("Closing Connection to %s:%s ..."%(addr[0],str(addr[1])))
        conn.close()

if( __name__ == '__main__'):

    app = CommandLine()

    #Intialize UDP
    UDP_IP= app.UDP_IP
    UDP_PORT = app.UDP_PORT
    sock_udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_udp.bind((UDP_IP, UDP_PORT))

    #Intialize TCP
    TCP_IP= app.TCP_IP
    TCP_PORT = app.TCP_PORT
    sock_tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock_tcp.bind((TCP_IP, TCP_PORT))

    #Intialize Time
    t_min = datetime.datetime.utcnow()

    #Initialize Plot
    nx, ny = app.config['waterfallY'], app.config['waterfallX']
    waterfall = -1*np.ones([nx,ny],dtype=float)

    time.sleep(1)
   
    thread = threading.Thread(target=data_listener)
    thread.daemon = True
    thread.start()

    thread2 = threading.Thread(target=TCP_stream)
    thread2.daemon = True
    thread2.start()

    input()


