#!/usr/bin/env python
import yaml, json, sys

with open(sys.argv[1], 'r') as stream:
    try:
        sys.stdout.write(json.dumps(yaml.load(stream)))
    except yaml.YAMLError as exc:
        sys.stderr.write(exc)


