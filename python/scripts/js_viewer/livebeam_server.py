###############################################################################
##
##  Copyright (C) 2011-2014 Tavendo GmbH
##
##  Licensed under the Apache License, Version 2.0 (the "License");
##  you may not use this file except in compliance with the License.
##  You may obtain a copy of the License at
##
##      http://www.apache.org/licenses/LICENSE-2.0
##
##  Unless required by applicable law or agreed to in writing, software
##  distributed under the License is distributed on an "AS IS" BASIS,
##  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
##  See the License for the specific language governing permissions and
##  limitations under the License.
##
###############################################################################

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

from enum import Enum

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

   def receive(self,length):
      chunks = []
      bytes_recd = 0
      while bytes_recd < length:
         chunk = self.connection.recv(min(length - bytes_recd, 2048))
         if chunk == b"":
            raise RuntimeError("socket connection broken")
         chunks.append(chunk)
         bytes_recd = bytes_recd + len(chunk)
      return b"".join(chunks)

   def receive_frame(self):
      d = self.receive(self.frame_length+self.frame_header)
      self.frame_idx, self.elem_idx, self.samples_summed = struct.unpack("III", d[:self.frame_header])
      self.data = np.frombuffer(d[self.frame_header:], dtype=np.float32)

   def connect_tcp(self):
      self.connection, self.client_address = self.sock.accept()
      self.packed_header = self.connection.recv(48)

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

      sec_per_pkt_frame = self.frame_raw_cad * self.frame_int_len

      info_header = self.connection.recv(self.frame_nfreq * 4 * 2 + self.frame_nvis * 1)
      self.frame_freqs = np.frombuffer(info_header[:self.frame_nfreq * 4 * 2], dtype=np.float32).reshape(-1, 2)
      self.frame_elems = np.frombuffer(info_header[self.frame_nfreq * 4 * 2 :], dtype=np.int8)

   def close_tcp(self):
      self.connection.close()


class MyServerProtocol(WebSocketServerProtocol):

   def onConnect(self, request):
      print("Client connecting: {0}".format(request.peer))
      KotekanConn.connect_tcp()

   def onOpen(self):
      print("WebSocket connection open.")
      KotekanConn.connect_tcp()
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


      self.looph=reactor.callLater(0.001, self.dataLoop)


   def dataLoop(self):
      for i in np.arange(KotekanConn.frame_nvis):
         KotekanConn.receive_frame()
         if (KotekanConn.elem_idx == i):
            self.databuf[:,i,:] = KotekanConn.data;
         else:
            print("Missed data from vis {}".format(i))

#      print(KotekanConn.frame_idx, KotekanConn.frame_raw_cad, KotekanConn.frame_utc0)

      sample_time = KotekanConn.frame_utc0 + KotekanConn.frame_raw_cad * KotekanConn.frame_int_len * (KotekanConn.frame_idx - KotekanConn.frame_idx0);
      send_data = np.int8(MSG_TYPE['timestep']).tobytes() + \
                  np.float64(sample_time).tobytes() + \
                  np.mean(self.databuf.reshape(self.ntime, self.nvis, self.sendfreq, -1),axis=3).tobytes()
      self.sendMessage(send_data,isBinary=True)
      if self.recording:
         self.output_file.write(np.float64(sample_time).tobytes() + self.databuf.tobytes())

      self.looph=reactor.callLater(0.001, self.dataLoop)

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
            printf("Got a bad request from a client")

      ## echo back message verbatim
#      self.sendMessage(payload, isBinary)

   def onClose(self, wasClean, code, reason):
      self.looph.cancel();
      print("WebSocket connection closed: {0}".format(reason))
      KotekanConn.close_tcp()
      if (self.recording):
         self.output_file.close()


KotekanConn = KotekanPowerStream()

from twisted.web import static, server
class Site(server.Site):
    def getResourceFor(self, request):
      request.setHeader('Access-Control-Allow-Origin', '*')
      request.setHeader('Access-Control-Allow-Methods', 'GET')
      request.setHeader('Access-Control-Allow-Headers', 'x-prototype-version,x-requested-with')
      request.setHeader('Access-Control-Max-Age', '2520')
      return server.Site.getResourceFor(self, request)

if __name__ == '__main__':

   import sys

   from twisted.python import log
   from twisted.internet import reactor

   log.startLogging(sys.stdout)

   factory = WebSocketServerFactory("ws://localhost:8539")
   factory.protocol = MyServerProtocol

   reactor.listenTCP(8539, factory)

   root = static.File("./")
   site = Site(root)

   reactor.listenTCP(8080, site)

   import webbrowser
   webbrowser.open('http://localhost:8080')

   reactor.run()



