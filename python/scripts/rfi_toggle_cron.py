import datetime
import requests
import json
import time

#Wait until the correct UTC time (deals with daylight savings time)
while(datetime.datetime.utcnow().time() < datetime.time(19,10)):
    time.sleep(5)

downtime = 60 #minutes

#Endpoint parameters
url = 'http://csBfs:54323/toggle-rfi-zeroing'
headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}

#Payload
payload = {'rfi_zeroing': False}
payload['rfi_zeroing'] = False

#Turn Off
on = True
try:
    r = requests.post(url, data=json.dumps(payload), headers=headers)
    if(not r.ok):
        print("RFI CRONJOB: Something went wrong in the request.")
    else:
        on = False
        print("RFI CRONJOB: Request sent")
except:
    print("RFI CRONJOB: Failure to Contact Kotekan Master, is it running?")

#If we successfully turned RFI Zeroing off
if(not on):

    #Wait until sun has passed
    time.sleep(downtime*60)

    #Payload
    payload = dict()
    payload['rfi_zeroing'] = True

    #Turn rfi zeroing back on
    try:
        r = requests.post(url, data=json.dumps(payload), headers=headers)
        if(not r.ok):
            print("RFI CRONJOB: Something went wrong in the request.")
        else:
            on = True
            print("RFI CRONJOB: Request sent")
    except:
        print("RFI CRONJOB: Failure to Contact Kotekan Master, is it running?")

#Exit
if(on):
    print("RFI CRONJOB: RFI zeroing has successfully been turned back on")
else:
    print("RFI CRONJOB: There was a failure, RFI zeroing has not been turned back on")
