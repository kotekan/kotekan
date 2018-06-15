import datetime
import os.path
import time
import numpy as np
import dateutil.parser
import sys
import requests
import json

#Targeted list from /etc/carillon/node_keepalive_config.json
rack_list = ['cn0g', 'cn1g','cn2g', 'cn3g','cn4g','cn5g','cn6g','cn8g','cn9g','cnAg','cnBg','cnCg','cnDg','cs0g','cs1g','cs2g','cs3g','cs4g','cs5g','cs6g','cs8g','cs9g','csAg','csBg','csCg']

TIMEOUT = 5.
def send_post(url, json_data=""):
    """ Send a put request and JSON content to the specified URL. """
    header = {'Content-type': 'application/json'}
    try:
        r = requests.post(url, timeout=TIMEOUT, json=json_data, headers=header)
    except:
        print("Unsuccessful at "+str(url))

timenow = datetime.datetime.utcfromtimestamp(time.time())
yy = timenow.year
mm= timenow.month
dd = timenow.day
A = int(yy/100.)
B = 2-A+int(A/4.)
C = int(365.25*yy)
D = int(30.6001*(mm+1))
sch_JD = B+C+D+dd+1720994.5
sch_file = "Schedule_JD"+str(sch_JD)+".dat"
print "Starting with", sch_file
Plan = np.genfromtxt(sch_file,dtype=['S14','f','f','f','f','i','i'])
lastline = 0
while (1):
    time_now = time.time()
    print "UTC time now", time.time(), timenow

    if (lastline >= len(Plan)-1):
        #Reached the end of a scheduled day
        sch_JD = float(sch_JD) + 1
        sch_file = "Schedule_JD"+str(sch_JD)+".dat"
        Plan = np.genfromtxt(sch_file,dtype=['S14','f','f','f','f','i','i'])
        print "Loading from new schedule file", sch_file
        lastline = 0
    for i in range(lastline,len(Plan),1):
        offset = time_now - Plan[i][1]
        if (offset>0): #Rise time before time_now
            offset2 = time_now - Plan[i][2]
            if (offset2<0): #Time_now before set time
                print "    curl at i=", i, Plan[i], datetime.datetime.utcfromtimestamp(Plan[i][1]), datetime.datetime.utcfromtimestamp(Plan[i][2])
                for rack in rack_list:
                    for node in range(10):
                        for gpu in range(4):
                            send_post("http://"+str(rack)+str(node)+":12048/gpu/gpu_"+str(gpu)+"/update_pulsar/"+str(gpu), json_data={"beam":int(Plan[i][5]),"ra":float(Plan[i][3]),"dec":float(Plan[i][4]),"scaling":int(Plan[i][6])})
                lastline = i+1
        else:
            break
    time.sleep(1)

#curl localhost:12048/gpu/gpu_<gpu_id>/update_pulsar/<gpu_id> -X POST -H 'Content-Type: application/json' -d '{"beam":<value>,"ra":<value>, "dec":<value>, "scaling":<value>}'
                


