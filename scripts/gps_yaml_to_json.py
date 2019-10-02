import yaml, json, sys
import requests

with open(sys.argv[1], 'r') as stream:
    try:
        config_json = yaml.load(stream)
    except yaml.YAMLError as exc:
        sys.stderr.write(exc)

try:
    gps_request = requests.get('http://10.10.10.2:54321/get-frame0-time')
    #gps_request = requests.get('http://carillon.chime:54231/get-frame0-time')
    config_json['gps_time'] = gps_request.json()
except requests.exceptions.RequestException as rex:
    config_json['gps_time'] = {}
    config_json['gps_time']['error'] = rex

sys.stdout.write(json.dumps(config_json))
