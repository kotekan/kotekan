import requests
from urlparse import urljoin, urlsplit
import click
import yaml
import json
import os

TIMEOUT = 1.


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


def send_put(url, json_data=""):
    """ Send a put request and JSON content to the specified URL. """
    try:
        r = requests.put(url, timeout=TIMEOUT, json=json_data)
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
    send_put(urljoin(url, "start"), json_data=cfg_yaml)


@cli.command()
#@click.option("--url", "-u", default="http://localhost:12048/",
#              help="The URL where the kotekan server can be reached.")
def stop():
    """ Stop kotekan.
        Requires sudo.
        Will only work for instances running on localhost since stop endpoint is
        not reliable at this point.
    """
    # TODO: looks like stop endpoint blocks on join()
    #       just send SIGINT for now
    #send_put(urljoin(url, "stop"))
    os.system("killall -s SIGINT kotekan")


@cli.command()
@click.option("--url", "-u", default="http://localhost:12048/",
              help="The URL where the kotekan server can be reached.")
def status(url):
    """ Query kotekan status. """
    # Right now status uses PUT
    r = send_put(urljoin(url, "status"))
    if r.status_code == 200:
        host_addr = urlsplit(url).netloc
        state = "running" if r.json()['running'] else "idle"
        print("Kotekan instance on {} is {}.".format(host_addr, state))


if __name__ == "__main__":
    cli()
