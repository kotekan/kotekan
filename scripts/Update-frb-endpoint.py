import sys
import os

"""
A temporary piece of code for FRB endpoint communication. Eventually should be done via ch_master
Mode is either gain or beam , for now
Usage: python Update-frb-endpoint.py gain [new_gain_path]  for gain update; or
       python Update-frb-endpoint.py beam [beam_offset_id] for shifting central beam_id for column mode
"""

#set a gain directory just in case, it should read in whatever is set in sys.argv[2] though
gaindir = '/root/FRB-GainFiles/eigen_cygA_03_22'
#gaindir = '/etc/kotekan/gains/tauA_03_09'

mode = sys.argv[1]
if (mode == 'beam'):
    beamID = sys.argv[2]
elif (mode == 'gain'):
    gaindir = sys.argv[2]

#Targeted list as of March 22 from /etc/carillon/node_keepalive_config.json (25 racks)
rack_list = ['cn0g', 'cn1g','cn2g', 'cn3g','cn4g','cn5g','cn6g','cn8g','cn9g','cnAg','cnBg','cnCg','cnDg','cs0g','cs1g','cs2g','cs3g','cs4g','cs5g','cs6g','cs8g','cs9g','csAg','csBg','csCg']

for rack in rack_list:
    for node in range(10):
        for gpu in range(4):
            uname = "gpu/gpu_"+str(gpu)
            if (mode == 'gain'):
                sys.stdout.write("curl "+str(rack)+str(node)+":12048/"+str(uname)+"/frb/update_gains/"+str(gpu)+" -X POST -H \'Content-Type: application/json\' -d \'{\"gain_dir\":\""+str(gaindir)+"\"}\'\n")
                os.system("curl "+str(rack)+str(node)+":12048/"+str(uname)+"/frb/update_gains/"+str(gpu)+" -X POST -H \'Content-Type: application/json\' -d \'{\"gain_dir\":\""+str(gaindir)+"\"}\'\n")
            elif (mode == 'beam'):
                sys.stdout.write("curl "+str(rack)+str(node)+":12048/"+str(uname)+"/frb/update_beam_offset -X POST -H \'Content-Type: application/json\' -d \'{\"beam_offset\":"+str(beamID)+"}\'\n")
                os.system("curl "+str(rack)+str(node)+":12048/"+str(uname)+"/frb/update_beam_offset -X POST -H \'Content-Type: application/json\' -d \'{\"beam_offset\":"+str(beamID)+"}\'\n")
            else :
                print "Mode", mode, "Not recognised"
                
