
import argparse
from collections import namedtuple
import datetime
import math
import os
from pytz import timezone
from requests.structures import CaseInsensitiveDict
import time
import requests


kotekan_params = namedtuple('KotekanParams',['kotekan_master','h5_dir','raw_ext'])

parser = argparse.ArgumentParser()
parser.add_argument("-start_unix_seconds", 
        help='start_unix_seconds')
parser.add_argument("-L", "--length", help="Dump length in seconds (default: 0.1)",
        type=float)
parser.add_argument("-start_unix_nano", "--start_unix_nano", help="start_unix_nano",
        type=float)
parser.add_argument("-event_id", "--event_id", help="event_id",
        type=float)

args = parser.parse_args()
event_id = args.event_id
start_unix_seconds = int(args.start_unix_seconds)
start_unix_nano = int(args.start_unix_nano)

transit_time=float(start_unix_seconds)
now = datetime.datetime.utcnow() + datetime.timedelta(seconds=0) 

local_tz = timezone('Canada/Pacific')
tz = timezone('UTC') #all timestamps should now be in UTC
kotekan_params = namedtuple('KotekanParams',['kotekan_master','h5_dir','raw_ext'])

# Note: h5_dir has to be specified relative to whatever machine runs this script.
dest = kotekan_params(kotekan_master="http://csBfs:54323/baseband",h5_dir= "",raw_ext="")

transit_dt= datetime.datetime.utcfromtimestamp(transit_time)

print(f"{transit_dt - now} time left until trigger")
print(datetime.timedelta(minutes=6))

event_time = math.modf(transit_time)
event_id = transit_dt.strftime("%Y%m%d%H%M%S")
event_dir = f'{transit_dt.strftime("%Y/%m/%d")}/astro_{event_id}/{dest.raw_ext}'
full_path = os.path.join(dest.h5_dir,event_dir)

payload={'event_id': int(event_id), 'file_path': str(event_id), 'start_unix_seconds': start_unix_seconds, 'start_unix_nano': start_unix_nano, 'duration_nano': int(args.length * 1e9), 
         'dm': 0.0, 'dm_error': 0.0}

print (payload)

wait=True

while wait:
    now = datetime.datetime.utcnow() + datetime.timedelta(seconds=0) 
    if transit_dt - now < datetime.timedelta(minutes=2):  # send trigger 2 minutes before the event
        wait = False
print("========================================")
print(f"The following will dump to: {full_path}")
print(f'Sending {event_id} dump request to {dest.kotekan_master}')
try:
    response = requests.post(url=dest.kotekan_master, json=payload)
    print (response)
except Exception as e: 
    print(e)