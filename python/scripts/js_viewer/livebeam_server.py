#!/usr/bin/env python

from autobahn.twisted.websocket import WebSocketServerProtocol, \
                                       WebSocketServerFactory
import numpy as np
import json

import time
import threading
import socket
import datetime
import struct
import signal

from twisted.python import log
from twisted.internet import reactor


MSG_TYPE = {'header':0,
            'freqlist':1,
            'timestep':2}

class KotekanPowerStream():
   def __init__(self,host='localhost',port=23401):
      self.host = host
      self.port = port
      self.initialize_tcp()

   def initialize_tcp(self):
      self.header_fmt = "=iiiidiiiId"
      TCP_IP = self.host
      TCP_PORT = self.port
      self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
      self.sock.bind((TCP_IP, TCP_PORT))
      self.sock.listen(1)
      self.connect_tcp()
      self.looph=reactor.callLater(0.001, self.receive_frame)

   def receive(self,length):
      chunks = []
      bytes_recd = 0
      while bytes_recd < length:
         self.sock.settimeout(0.1)
         chunk = self.connection.recv(min(length - bytes_recd, 2048))
         if chunk == b"":
            self.close_tcp()
            raise RuntimeError("socket connection broken")
         chunks.append(chunk)
         bytes_recd = bytes_recd + len(chunk)
      return b"".join(chunks)

   def receive_frame(self):
      try:
         d = self.receive(self.frame_length+self.frame_header)
      except:
         return
      self.frame_idx, self.elem_idx, self.samples_summed = struct.unpack("III", d[:self.frame_header])
      self.data = np.frombuffer(d[self.frame_header:], dtype=np.float32)
      WebSockConn.sendPowerFrame()
      self.looph=reactor.callLater(0.001, self.receive_frame)

   def connect_tcp(self):
      print("Connecting to Kotekan")
      self.connection, self.client_address = self.sock.accept()
      self.connected=True
      self.packed_header = self.connection.recv(48)

      print("Connected to Kotekan")
      tcp_header = struct.unpack(self.header_fmt, self.packed_header)
      self.frame_length = tcp_header[0]  # packet_length
      self.frame_header = tcp_header[1]  # header_length
      self.frame_samples= tcp_header[2]  # samples_per_packet
      self.frame_dtype  = tcp_header[3]  # sample_type
      self.frame_raw_cad= tcp_header[4]  # raw_cadence
      self.frame_nfreq  = tcp_header[5]  # num_freqs
      self.frame_nvis   = tcp_header[6]  # num_elems = num_vis
      self.frame_int_len= tcp_header[7]  # samples_summed
      self.frame_idx0   = tcp_header[8]  # handshake_idx
      self.frame_utc0   = tcp_header[9]  # handshake_utc

      info_header = self.connection.recv(self.frame_nfreq * 4 * 2 + self.frame_nvis * 1)
      self.frame_freqs = np.frombuffer(info_header[:self.frame_nfreq * 4 * 2], dtype=np.float32).reshape(-1, 2)
      self.frame_elems = np.frombuffer(info_header[self.frame_nfreq * 4 * 2 :], dtype=np.int8)

   def close_tcp(self):
      self.connected=False
      self.connection.close()
      if (self.looph.active()):
         try:
            self.looph.cancel()
         except:
            print("Kotekan loop already closed?...")
      WebSockConn.close()
      print("Closed connection to Kotekan")


class MyWSServerFactory(WebSocketServerFactory):
   def __init__(self, url):
      WebSocketServerFactory.__init__(self, url)
      self.clients = []

   def register(self, client):
      print("Register websock")
      if client not in self.clients:
         print("registered client {}".format(client.peer))
         self.clients.append(client)

   def unregister(self, client):
      if client in self.clients:
         print("unregistered client {}".format(client.peer))
         self.clients.remove(client)

   def sendPowerFrame(self):
      for c in self.clients:
         c.sendPowerFrame()
   
   def close(self):
      for c in self.clients:
         c.sendClose()



class MyServerProtocol(WebSocketServerProtocol):
   def onConnect(self, request):
      print("Client connecting: {0}".format(request.peer))

   def onOpen(self):
      print("WebSocket connection open.")
      self.factory.register(self)
      self.nfreq=KotekanConn.frame_nfreq
      self.nvis =KotekanConn.frame_nvis
      self.sendfreq=128 #number of freqs to transmit
      self.ntime=1      #number of timesteps to transmit
      self.target_name='Undefined'
      self.databuf = np.zeros((self.ntime, self.nvis, self.nfreq), dtype=np.float32)
      self.output_file = ""
      self.recording = False

      ### ADD NVIS TO PROTOCOL ###
      header_info=json.dumps({'nfreq':self.sendfreq})
      self.sendMessage(header_info.encode('utf8'), isBinary = False)

      ### SEND FREQ LIST ###
      send_data = np.int8(MSG_TYPE['freqlist']).tobytes() + \
                  np.mean((KotekanConn.frame_freqs/1e6).reshape(self.sendfreq,-1),axis=1).tobytes()
      self.sendMessage(send_data,isBinary=True)

   def sendPowerFrame(self):
      for i in np.arange(KotekanConn.frame_nvis):
         if not KotekanConn.connected:
            self.transport.loseConnection() 
            return
         if (KotekanConn.elem_idx == i):
            self.databuf[:,i,:] = KotekanConn.data;
         else:
            print("Missed data from vis {}".format(i))

      sample_time = KotekanConn.frame_utc0 + KotekanConn.frame_raw_cad * KotekanConn.frame_int_len * (KotekanConn.frame_idx - KotekanConn.frame_idx0);
      send_data = np.int8(MSG_TYPE['timestep']).tobytes() + \
                  np.float64(sample_time).tobytes() + \
                  np.mean(self.databuf.reshape(self.ntime, self.nvis, self.sendfreq, -1),axis=3).tobytes()
      self.sendMessage(send_data,isBinary=True)
      if self.recording:
         self.output_file.write(np.float64(sample_time).tobytes() + self.databuf.tobytes())

   def onMessage(self, payload, isBinary):
      if isBinary:
         print("Binary message received: {0} bytes".format(len(payload)))
      else:
         print("Text message received: {0}".format(payload.decode('utf8')))
         request = json.loads(payload.decode('utf8'))
         if ('type' in request):
            if request["type"] == "record":
               if (request["state"]):
                  if (self.recording):
                     self.output_file.close()
                     self.recording=False
                  else:
                     fn = request["file"]
                     self.output_file = open(fn, "wb")
                     self.output_file.write(KotekanConn.packed_header)
                     self.output_file.write(KotekanConn.frame_freqs.tobytes())
                     self.output_file.write(KotekanConn.frame_elems.tobytes())
                     self.recording=True
               else:
                  self.output_file.close()
                  self.recording=False

         else:
            print("Got a bad request from a client")

   def onClose(self, wasClean, code, reason):
      print("WebSocket connection closed: {0}".format(reason))
      self.factory.unregister(self)
      if (self.recording):
         self.output_file.close()

def shutdown(sig, frame):
   KotekanConn.close_tcp()
   reactor.stop()

KotekanConn = KotekanPowerStream()
WebSockConn = MyWSServerFactory("ws://localhost:8539")
signal.signal(signal.SIGINT, shutdown)

from twisted.web import static, server
class Site(server.Site):
    def getResourceFor(self, request):
      request.setHeader('Access-Control-Allow-Origin', '*')
      request.setHeader('Access-Control-Allow-Methods', 'GET')
      request.setHeader('Access-Control-Allow-Headers', 'x-prototype-version,x-requested-with')
      request.setHeader('Access-Control-Max-Age', '2520')
      return server.Site.getResourceFor(self, request)

if __name__ == '__main__':
   import os
   import sys
   dir = os.path.dirname(os.path.abspath(sys.argv[0]))
   os.chdir(dir)

   log.startLogging(sys.stdout)

   factory = WebSockConn
   factory.protocol = MyServerProtocol

   reactor.listenTCP(8539, factory)

   root = static.File("./")
   site = Site(root)

   reactor.listenTCP(8080, site)

   import webbrowser
   webbrowser.open('http://localhost:8080')

   reactor.run()



