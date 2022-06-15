#!/usr/bin/env python
"""
This script converts YAML or Jinja files into JSON for use by kotekan

All files without a ".j2" extension are treated as YMAL, files with a ".j2"
extension are processed by the Jinja template render.

Extra variables can be passed in with the -e flag, in json/dict format:
 -e '{"my_val": 20, "my_val2": 30, "my_array": [0,2,1]}'

The JSON is returned in STDOUT, and error messages are returned in STDERR
"""

import argparse
from kotekan.config import load_config_file


# Setup arg parser
parser = argparse.ArgumentParser(description="Convert YAML or Jinja files into JSON")
parser.add_argument("name", help="Config file name", type=str)
parser.add_argument(
    "-d", "--dump", help="Dump the yaml, useful with .j2 files", action="store_true"
)
parser.add_argument(
    "-e", "--variables", help="Add extra jinja variables, JSON format", type=str
)
args = parser.parse_args()

options = args.variables

print(load_config_file(args.name, dump=args.dump, jinja_options=options))
