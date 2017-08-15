import threading
import socket
import sys
import numpy as np
import matplotlib.animation as animation
import matplotlib.dates as md
import datetime
import struct
import json

import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt

import random
import time
import cmath
import math

from astropy.coordinates import SkyCoord, EarthLocation, Angle, AltAz
from astropy import units as u
from astropy.time import Time

class Settings(QDialog):
    def __init__(self, parent):
        super(Settings, self).__init__(parent)
	self.main = parent
	self.settingsfigure = plt.figure(2)
	self.settingscanvas = FigureCanvas(self.settingsfigure)	

	#self.resize(self.main.frameGeometry().width()/2,self.main.frameGeometry().height()/2)
	self.layout = QVBoxLayout()

	self.timer = QTimer()
	self.timer.timeout.connect(self.UpdateGraph)
	self.timer.start(3000)
	hbox = QHBoxLayout()
	self.mode = QCheckBox("Median Subtract")
      	self.mode.setChecked(self.main.medsub)
      	self.mode.stateChanged.connect(parent.ChangeMode)
	hbox.addWidget(self.mode)
	self.dispersed_button = QCheckBox("Dispersion Correction")
	self.dispersed_button.setChecked(self.main.DeDisperse)
	self.dispersed_button.stateChanged.connect(self.ChangeDispersion)
	hbox.addWidget(self.dispersed_button)

	self.layout.addLayout(hbox)

	hbox = QHBoxLayout()
	self.pul_label = QLabel("Pulsar: ")
	hbox.addWidget(self.pul_label)

	self.pls_text = QLineEdit(self.main.target)
	self.pls_text.textChanged[str].connect(self.UpdateSettings)
	hbox.addWidget(self.pls_text)
	self.layout.addLayout(hbox)
	
	hbox = QHBoxLayout()
	self.dm_text = QLabel("Dispersion Measure: "+ str(self.main.psrdata['dmeasure']))
	hbox.addWidget(self.dm_text)
	
	self.fold_text = QLabel("Folding Period: ")# + str(self.main.fold_period))
	hbox.addWidget(self.fold_text)
	self.fold_edit = QLineEdit(str(self.main.fold_period))
	self.fold_edit.textChanged[str].connect(self.UpdateFold)
	hbox.addWidget(self.fold_edit)
	self.layout.addLayout(hbox)

	hbox = QHBoxLayout()
	self.GraphSlider = QSlider(Qt.Vertical)
	self.GraphSlider.setMinimum(1)
        self.GraphSlider.setMaximum(10)
        self.GraphSlider.setValue(3)
	self.GraphSlider.setFocusPolicy(Qt.StrongFocus)
	self.GraphSlider.setTickPosition(QSlider.TicksBothSides)
        self.GraphSlider.setTickInterval(1)
	self.GraphSlider.valueChanged.connect(self.UpdateGraph)
	hbox.addWidget(self.GraphSlider)

	hbox.addWidget(self.settingscanvas)

	self.layout.addLayout(hbox)

	self.setLayout(self.layout)
	self.UpdateGraph()

    def ChangeDispersion(self):
	self.main.DeDisperse = not self.main.DeDisperse

    def UpdateSettings(self, text):
	try: 
		x = self.main.psrcat[text]
		self.main.target = text
		self.main.psrdata = self.main.psrcat[self.main.target]
		self.main.fold_period = 1./self.main.psrdata['frequency']
		self.dm_text.setText("Dispersion Measure: "+ str(self.main.psrdata['dmeasure']))
		self.fold_edit.setText(str(self.main.fold_period))
		self.main.temp_target = ""
		self.main.UpdateLabel()
	except:
		self.main.temp_target = text
		self.main.UpdateLabel()
		return

    def UpdateFold(self, text):
	try:
		self.main.fold_period = float(text)
		self.main.UpdateLabel()
	except:
		return

    def UpdateGraph(self):

	self.settingsfigure.clear()
	x = np.arange(0,1,1.0/self.main.dedispersed.shape[0]) #phase
	y_lower = -1*(self.GraphSlider.value()/100.0)
	y_upper = self.GraphSlider.value()/100.0
	plt.ylim([y_lower,y_upper])
	plt.title('Pulse Profile')
	plt.xlabel('Phase')
	plt.ylabel('Power (arb)')
	y = np.zeros_like(self.main.MedSubbed[:,0])
	for i in range(self.main.MedSubbed.shape[0]):
		y[y.size-1-i] = np.mean(self.main.MedSubbed[i,:][(self.main.MedSubbed[i,:]>(5*y_lower))*(self.main.MedSubbed[i,:]<(5*y_upper))])
	#y /= self.main.MedSubbed.shape[1]
	p10 = np.poly1d(np.polyfit(x, y, 15))
	plt.plot(x,y,'.',x,p10(x))
	self.settingscanvas.draw()

class Window(QDialog):
    def receive(self,connection,length):
	    chunks = []
	    bytes_recd = 0
	    while bytes_recd < length:
		chunk = connection.recv(min(length - bytes_recd, 2048))
		if chunk == b'':
		    raise RuntimeError("socket connection broken")
		chunks.append(chunk)
		bytes_recd = bytes_recd + len(chunk)
	    return b''.join(chunks)

    def __init__(self, parent=None):

        super(Window, self).__init__(parent)
	self.header_fmt = '=iiiidiiiId'
	self.stokes_lookup = ['YX','XY','YY','XX','LR','RL','LL','RR','I','Q','U','V']
	self.curtime = 0
	self.curpoint = 0
	self.TCP_IP="0.0.0.0"
	self.TCP_PORT = 2054
	self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
	self.sock.bind((self.TCP_IP, self.TCP_PORT))
	self.sock.listen(1)
	self.psrcat = json.load(open('psrcat/psrcat_b.json'))['pulsars']
	self.target = 'B1933+16'
	self.temp_target = ""
	self.psrdata = self.psrcat[self.target]
	
	self.connection, self.client_address = self.sock.accept()

	self.packed_header = self.receive(self.connection,48)

	self.tcp_header  = struct.unpack(self.header_fmt,self.packed_header)

	self.pkt_length  = self.tcp_header[0] # packet_length
	self.pkt_header  = self.tcp_header[1] # header_length
	self.pkt_samples = self.tcp_header[2] # samples_per_packet
	self.pkt_dtype   = self.tcp_header[3] # sample_type
	self.pkt_raw_cad = self.tcp_header[4] # raw_cadence
	self.pkt_freqs   = self.tcp_header[5] # num_freqs
	self.pkt_elems   = self.tcp_header[6] # num_freqs
	self.pkt_int_len = self.tcp_header[7] # samples_summed
	self.pkt_idx0	= self.tcp_header[8] # handshake_idx
	self.pkt_utc0	= self.tcp_header[9] # handshake_utc


	self.sec_per_pkt_frame = self.pkt_raw_cad * self.pkt_int_len

	self.info_header = self.receive(self.connection, self.pkt_freqs * 4 * 2 + self.pkt_elems * 1)
	self.freqlist = np.fromstring(self.info_header[:self.pkt_freqs * 4*2],dtype=np.float32).reshape(-1,2)
	self.freqlist = self.freqlist/1e6
	self.elemlist = np.fromstring(self.info_header[self.pkt_freqs*4*2:],dtype=np.int8)

	self.plot_freqs = self.pkt_freqs/8

	self.plot_times=256
	self.plot_phase=128
	self.total_integration=1024*8

	self.DeDisperse = True
        

	if (self.pkt_int_len > self.total_integration):
		print "Pre-integrated to longer than desired time!"
		print "{} vs {}".format(self.pkt_int_len, self.total_integration)
		print "Resetting integration length to {}".format(self.pkt_int_len)
		self.total_integration=self.pkt_int_len
	self.local_integration=self.total_integration / self.pkt_int_len

	self.waterfall = np.zeros((self.plot_times,self.plot_freqs,self.pkt_elems),dtype=np.float32) + np.nan;
	self.countfold = np.zeros((self.plot_phase,self.plot_freqs,self.pkt_elems),dtype=np.float32);
	self.fold_period = 1./self.psrdata['frequency']
	self.waterfold = np.zeros((self.plot_phase,self.plot_freqs,self.pkt_elems),dtype=np.float32);
	self.times = np.zeros(self.plot_times)

	time.sleep(1)
	self.f, self.ax = plt.subplots(2,self.pkt_elems,gridspec_kw = {'height_ratios':[2, 1]})
	self.f.subplots_adjust(right=0.8)
	if (self.pkt_elems == 1):
		self.ax=[self.ax]
	plt.ioff()
	self.p=[]
	
	self.tmin=md.date2num(datetime.datetime.fromtimestamp(self.pkt_utc0 - self.plot_times*self.local_integration*self.sec_per_pkt_frame))
	self.tmax=md.date2num(datetime.datetime.fromtimestamp(self.pkt_utc0))
	self.times = self.pkt_utc0 - np.arange(self.plot_times)*self.local_integration*self.sec_per_pkt_frame
	self.date_format = md.DateFormatter('%H:%M:%S')
	self.medsub=True
	self.colorscale=[-.5,.5]

	for i in np.arange(self.pkt_elems):
		self.p.append(self.ax[0,i].imshow(self.waterfall[:,:,i],aspect='auto',animated=True,origin='upper',interpolation='nearest', \
				cmap='viridis',vmin=self.colorscale[0], vmax=self.colorscale[1], extent=[self.freqlist[0,0],self.freqlist[-1,-1],self.tmin,self.tmax]))
		self.ax[0,i].set_yticklabels([])
		self.ax[0,i].yaxis_date()

	self.ax[0,0].set_title(self.stokes_lookup[self.elemlist[0]+8])
	self.ax[0,1].set_title(self.stokes_lookup[self.elemlist[1]+8])

	self.ax[0,0].set_ylabel('Local Time')
	self.ax[0,0].yaxis_date()
	self.ax[0,0].yaxis.set_major_formatter(self.date_format)

	for i in np.arange(self.pkt_elems):
		self.p.append(self.ax[1,i].imshow(self.waterfold[:,:,i],aspect='auto',animated=True,origin='upper',interpolation='nearest', \
				cmap='viridis',vmin=self.colorscale[0], vmax=self.colorscale[1], extent=[self.freqlist[0,0],self.freqlist[-1,-1],0,1]))
		self.ax[1,i].set_xlabel('Freq (MHz)')

	self.ax[1,0].set_ylabel('Pulse Phase')

	self.cbar_ax = self.f.add_axes([0.85, 0.15, 0.05, 0.7])
	self.c = self.f.colorbar(self.p[0], cax=self.cbar_ax)
	self.c.set_label('Power (dB, arbitrary)')
	
        self.figure = self.f
	
        self.canvas = FigureCanvas(self.figure)
	self.ani = animation.FuncAnimation(self.f, self.updatefig, frames=100, interval=100)
       
	self.set_button = QPushButton('Show Settings')
	self.set_button.clicked.connect(self.ShowSettings)

	newfont = QFont("Times", 15, QFont.Bold) 
	self.title_text = QLabel(self)
	self.title_text.setAlignment(Qt.AlignCenter)	
        self.title_text.setFont(newfont)
	self.title_text.setFixedHeight(25)
	
	hbox1 = QHBoxLayout()
	self.pos_text_1 = QLabel(self)	
        self.pos_text_1.setFont(newfont)
	self.pos_text_1.setFixedHeight(25)
	self.pos_text_1.setAlignment(Qt.AlignLeft)
	hbox1.addWidget(self.pos_text_1)
	self.pos_text_2 = QLabel(self)	
        self.pos_text_2.setFont(newfont)
	self.pos_text_2.setFixedHeight(25)
	self.pos_text_2.setAlignment(Qt.AlignRight)
	hbox1.addWidget(self.pos_text_2)
	hbox2 = QHBoxLayout()
	self.info_text_1 = QLabel(self)	
        self.info_text_1.setFont(newfont)
	self.info_text_1.setFixedHeight(25)
	self.info_text_1.setAlignment(Qt.AlignLeft)
	hbox2.addWidget(self.info_text_1)
	self.info_text_2 = QLabel(self)	
        self.info_text_2.setFont(newfont)
	self.info_text_2.setFixedHeight(25)
	self.info_text_2.setAlignment(Qt.AlignRight)
	hbox2.addWidget(self.info_text_2)
        # set the layout
        vbox = QVBoxLayout()
	vbox.addWidget(self.title_text)
	vbox.addLayout(hbox1)
	vbox.addLayout(hbox2)
        vbox.addWidget(self.canvas)	
	vbox.addWidget(self.set_button)

	self.UpdateLabel()
        self.setLayout(vbox)
	self.showMaximized()
    def ShowSettings(self):
	print
	self.new_child = Settings(self)
	self.new_child.show()
    def ChangeMode(self):
	self.medsub = not self.medsub
	if self.medsub:
		self.colorscale=[-0.5,0.5]
	else:
		self.colorscale=[-10,10]

    def UpdateLabel(self):

	if(self.curtime != 0):
		Pointing = self.curtime.isot +'        ' + self.curpoint.altaz.to_string('dms')
		c = SkyCoord(self.curpoint.altaz.to_string('hmsdms').split(' ')[0], self.curpoint.altaz.to_string('hmsdms').split(' ')[1], frame='icrs')
		c = c.transform_to('fk5').to_string('hmsdms')
	else:
		Pointing = 'Unavailiable'
		c = 'Unavailiable'
	if(self.temp_target == ""):
		self.title_text.setText(self.target)
	else:
		self.title_text.setText(self.temp_target + ' (PSRCAT: '+self.target + ')')
	self.pos_text_1.setText("Pointing: " + Pointing)
	self.pos_text_2.setText("FK5: " + c)
	self.info_text_1.setText("Dispersion Measure: " + str(self.psrdata['dmeasure']))
	self.info_text_2.setText("Fold Period: " + str(self.fold_period))

    def updatefig(self,*args):
	self.tmin=md.date2num(datetime.datetime.fromtimestamp(np.amin(self.times)))
	self.tmax=md.date2num(datetime.datetime.fromtimestamp(np.amax(self.times)))
	self.dedispersed = np.copy(self.waterfold)
	self.dedispersed_count = np.copy(self.countfold)
	self.MedSubbed = np.zeros_like(self.dedispersed[:,:,0])
	for i in np.arange(self.pkt_elems):
		if(self.DeDisperse):
			for j in range(self.dedispersed.shape[1]):
				#time = (float(j)/self.dedispersed.shape[1])*+((202.0/600)**3)*self.psrdata['dmeasure']*(0.4)
				time = (4.148808/1000)*((1/0.4)**2 - (1/(0.4+(128-j)*(0.4/self.dedispersed.shape[1])))**2)*self.psrdata['dmeasure'] #Works
				location = 1*int((time/self.fold_period)*self.dedispersed.shape[0])
				#location = -1*int((time/(self.plot_times*self.local_integration*self.sec_per_pkt_frame))*self.dedispersed.shape[0]) #Works for waterfall
				#print j,time,location
				self.dedispersed[:,j,i] = np.roll(self.dedispersed[:,j,i],location)
				self.dedispersed_count[:,j,i] = np.roll(self.dedispersed_count[:,j,i],location)
		"""
		int(j*0.01490597512*self.psrdata['dmeasure'])
		for j in range(self.waterfall.shape[1]):
			if(float(np.count_nonzero(np.isnan(self.waterfall[:,j,i])))/self.waterfall[:,j,i].size > 0.1):
				self.countfold[:,j,i] = 0"""
		if self.medsub:
			self.p[i].set_data(self.waterfall[:,:,i]-np.nanmedian(self.waterfall[:,:,i],axis=0)[np.newaxis,:])
			tmpdata = 10*np.log10(self.dedispersed[:,:,i]/self.dedispersed_count[:,:,i])
			self.p[self.pkt_elems+i].set_data(tmpdata-np.median(tmpdata,axis=0)[np.newaxis,:])
		else:
			self.p[i].set_data(self.waterfall[:,:,i])
			tmpdata = 10*np.log10(self.dedispersed[:,:,i]/self.dedispersed_count[:,:,i])
			self.p[self.pkt_elems+i].set_data(tmpdata)
		self.MedSubbed += (self.dedispersed[:,:,i]/self.dedispersed_count[:,:,i])-np.median((self.dedispersed[:,:,i]/self.dedispersed_count[:,:,i]),axis=0)[np.newaxis,:]
		self.p[i].set_extent([self.freqlist[0,0],self.freqlist[-1,-1], self.tmin,self.tmax])
		self.p[i].set_clim(vmin=self.colorscale[0], vmax=self.colorscale[1])
		self.p[self.pkt_elems+i].set_clim(vmin=self.colorscale[0]/10, vmax=self.colorscale[1]/10)
	return self.p


def data_listener():
	global main
	last_idx = main.pkt_idx0
	data_pkt_frame_idx = 0;
	data_pkt_samples_summed = 1;
	idx=0
	while True:
		try:
			d=np.zeros([main.pkt_freqs,main.pkt_elems])
			n=np.zeros([main.pkt_freqs,main.pkt_elems])
			t=np.zeros(main.plot_times)
			main.waterfold*= 0.999
			main.countfold*= 0.999
			for i in np.arange(main.local_integration*main.pkt_elems):
				data = main.receive(main.connection,main.pkt_length+main.pkt_header)
				if (len(data) != main.pkt_length+main.pkt_header):
					print("Lost Connection!")
					main.connection.close()
					return;
				data_pkt_frame_idx, data_pkt_elem_idx, data_pkt_samples_summed = struct.unpack('III',data[:main.pkt_header])
				data_array = np.fromstring(data[main.pkt_header:],dtype=np.uint32)
				d[:,data_pkt_elem_idx] += data_array * 1.0
				n[:,data_pkt_elem_idx] += data_pkt_samples_summed * 1.0
				#n[:,data_pkt_elem_idx][data_array != 0] += data_pkt_samples_summed * 1.0
				fold_idx = np.array(((main.sec_per_pkt_frame * data_pkt_frame_idx + 0.5*main.fold_period) % main.fold_period) /main.fold_period * main.plot_phase,dtype=np.int32)
				main.waterfold[fold_idx,:,data_pkt_elem_idx] += data_array.reshape(-1,main.pkt_freqs / main.plot_freqs).mean(axis=1)
				main.countfold[fold_idx,:,data_pkt_elem_idx] += data_pkt_samples_summed
				#main.countfold[:,15:70,:] = 0
			roll_idx = (data_pkt_frame_idx - last_idx)/main.local_integration
			main.times = np.roll(main.times,roll_idx)
			main.times[0] = main.sec_per_pkt_frame * (data_pkt_frame_idx - main.pkt_idx0) + main.pkt_utc0
			main.waterfall = np.roll(main.waterfall,roll_idx,axis=0)
			main.waterfall[0,:,:]=10*np.log10((d/n).reshape(-1,main.pkt_freqs / main.plot_freqs,main.pkt_elems).mean(axis=1))
			#main.waterfall[0,:,:] = (d/1000).reshape(-1,main.pkt_freqs / main.plot_freqs,main.pkt_elems).mean(axis=1)
			last_idx = data_pkt_frame_idx
			
		except:
			main.connection, main.client_address = main.sock.accept()
			main.packed_header = main.receive(main.connection,48)
			main.info_header = main.receive(main.connection, main.pkt_freqs * 4 * 2 + main.pkt_elems * 1)
			
			main.pkt_length  = main.tcp_header[0] # packet_length
			main.pkt_header  = main.tcp_header[1] # header_length
			main.pkt_samples = main.tcp_header[2] # samples_per_packet
			main.pkt_dtype   = main.tcp_header[3] # sample_type
			main.pkt_raw_cad = main.tcp_header[4] # raw_cadence
			main.pkt_freqs   = main.tcp_header[5] # num_freqs
			main.pkt_elems   = main.tcp_header[6] # num_freqs
			main.pkt_int_len = main.tcp_header[7] # samples_summed
			main.pkt_idx0	= main.tcp_header[8] # handshake_idx
			main.pkt_utc0	= main.tcp_header[9] # handshake_utc

			main.sec_per_pkt_frame = main.pkt_raw_cad * main.pkt_int_len


			#main.waterfall = np.zeros((self.plot_times,self.plot_freqs,self.pkt_elems),dtype=np.float32) + np.nan;
			print("Reconnected!")

def get_pointing():
    global main
    while True:
        ARO = EarthLocation(lat=45.95550333*u.deg, lon=-78.073040402778*u.deg, height=260.4*u.m)
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect(("192.168.3.105", 6350))
            s.send("A")
            b = s.recv(100).rstrip()
            main.curtime = Time("{}-{}-{}:{}:{}".format(b.split(' ')[0][0:4],b.split(' ')[0][4:6],b.split(' ')[0][6:11],b.split(' ')[0][11:13],b.split(' ')[0][13:-1]))
            main.curpoint = SkyCoord(alt = Angle(b.split(' ')[2],unit=u.deg), az = Angle(b.split(' ')[1], unit=u.deg), obstime = main.curtime, frame = 'altaz', location = ARO )
            s.close()
	    main.UpdateLabel()
	    """
	    for k,v in self.psrcat.iteritems():
		if v['RA'][0:6] == '01h47m':
			print k
	    """
            time.sleep(1)
        except:
	    print("disconnected from mooncake, retrying in 30 seconds...")
            time.sleep(30)
            

if __name__ == '__main__':

    np.seterr(divide='ignore', invalid='ignore')

    app = QApplication(sys.argv)
    main = Window()
    main.show()
    
    thread = threading.Thread(target=data_listener)
    thread.daemon = True
    thread.start()

    thread2 = threading.Thread(target=get_pointing)
    thread2.daemon= True
    thread2.start()


    sys.exit(app.exec_())
