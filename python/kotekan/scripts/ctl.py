# Python 2/3 compatibility
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import (ascii, bytes, chr, dict, filter, hex, input,
                      int, map, next, oct, open, pow, range, round,
                      str, super, zip)

import requests
from urlparse import urljoin, urlsplit
import click
import yaml

TIMEOUT = 5.


def send_get(url):
    """ Send a get request to the specified URL. """
    try:
        r = requests.get(url, timeout=TIMEOUT)
    except requests.exceptions.ReadTimeout:
        print("Server response timed out.")
        return
    if r.status_code != 200:
        print("Request unsuccessful.")
        print(r.headers)
    return r


def send_post(url, json_data=""):
    """ Send a put request and JSON content to the specified URL. """
    header = {'Content-type': 'application/json'}
    try:
        r = requests.post(url, timeout=TIMEOUT, json=json_data, headers=header)
    except requests.exceptions.ReadTimeout:
        print("Server response timed out.")
        return
    if r.status_code != 200:
        print("Request unsuccessful.")
        print(r.headers)
    return r


@click.group()
def cli():
    """ Send commands to the kotekan REST server. """
    pass


@cli.command()
@click.argument('config')
@click.option("--url", "-u", default="http://localhost:12048/",
              help="The URL where the kotekan server can be reached.")
def start(config, url):
    """ Start kotekan with yaml CONFIG. """
    with open(config, 'r') as stream:
        cfg_yaml = yaml.load(stream)
    send_post(urljoin(url, "start"), json_data=cfg_yaml)


@cli.command()
@click.option("--url", "-u", default="http://localhost:12048/",
              help="The URL where the kotekan server can be reached.")
def stop(url):
    """ Stop kotekan. """
    send_get(urljoin(url, "stop"))


@cli.command()
@click.option("--url", "-u", default="http://localhost:12048/",
              help="The URL where the kotekan server can be reached.")
def status(url):
    """ Query kotekan status. """
    r = send_get(urljoin(url, "status"))
    if r.status_code == 200:
        host_addr = urlsplit(url).netloc
        state = "running" if r.json()['running'] else "idle"
        print("Kotekan instance on {} is {}.".format(host_addr, state))
