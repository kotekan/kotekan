# === Start Python 2/3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614

# === End Python 2/3 compatibility


import requests
from requests.compat import urljoin, urlsplit
import click
import yaml

click.disable_unicode_literals_warning = True

TIMEOUT = 10.0


def send_get(url, timeout=TIMEOUT):
    """Send a get request to the specified URL."""
    try:
        r = requests.get(url, timeout=timeout)
    except requests.exceptions.ReadTimeout:
        print("Server response timed out.")
        return
    except requests.exceptions.ConnectionError as e:
        print("Could not reach server.")
        print(str(e))
        return
    if r.status_code != 200:
        print("Request unsuccessful.")
        print(r.headers)
    return r


def send_post(url, json_data="", timeout=TIMEOUT):
    """Send a put request and JSON content to the specified URL."""
    header = {"Content-type": "application/json"}
    try:
        r = requests.post(url, timeout=timeout, json=json_data, headers=header)
    except requests.exceptions.ReadTimeout:
        print("Server response timed out.")
        return
    except requests.exceptions.ConnectionError as e:
        print("Could not reach server.")
        print(str(e))
        return
    if r.status_code != 200:
        print("Request unsuccessful.")
        print(r.headers)
    return r


@click.group()
def cli():
    """Send commands to the kotekan REST server."""
    pass


@cli.command()
@click.argument("config")
@click.option(
    "--url",
    "-u",
    default="http://localhost:12048/",
    help="The URL where the kotekan server can be reached.",
)
def start(config, url):
    """Start kotekan with yaml CONFIG."""
    with open(config, "r") as stream:
        cfg_yaml = yaml.load(stream)
    send_post(urljoin(url, "start"), json_data=cfg_yaml)


@cli.command()
@click.option(
    "--url",
    "-u",
    default="http://localhost:12048/",
    help="The URL where the kotekan server can be reached.",
)
def stop(url):
    """Stop kotekan."""
    send_get(urljoin(url, "stop"), timeout=30.0)


@cli.command()
@click.option(
    "--url",
    "-u",
    default="http://localhost:12048/",
    help="The URL where the kotekan server can be reached.",
)
def status(url):
    """Query kotekan status."""
    r = send_get(urljoin(url, "status"))
    if r is None:
        return
    if r.status_code == 200:
        host_addr = urlsplit(url).netloc
        state = "running" if r.json()["running"] else "idle"
        print("Kotekan instance on {} is {}.".format(host_addr, state))
