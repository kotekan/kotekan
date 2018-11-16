#!/usr/bin/env python
"""
REST Server and clients for the Dataset Broker.

"""

import sys
import thread
import datetime

from rest import AsyncRESTClient, AsyncRESTServer, endpoint
from rest import coroutine, coroutine_return
from rest import run_client  # generic REST servers and clients
import toro  # conditional variables for tornado coroutines

import log  # logging helper functions


class DSBrokerAsyncRESTServer(AsyncRESTServer):
    """
    REST interface for Dataset Broker.
    """

    DEFAULT_PORT = 12050

    def __init__(self, address='', port=DEFAULT_PORT, logging_params={}):
        """
        List of dict with entries 'type', 'name', and 'address'
        """
        self.states = dict()
        self.datasets = dict()
        self.signal_states_updated = toro.Condition()
        self.signal_datasets_updated = toro.Condition()
        self.lock_datasets = thread.allocate_lock()
        self.lock_states = thread.allocate_lock()
        super(DSBrokerAsyncRESTServer, self).__init__(address=address,
                                                      port=port,
                                                      heartbeat_string='Gs')

    ##################
    # Server commands
    ##################

    @coroutine
    @endpoint('status')
    def status(self, handler):
        """ Get status of dataset-broker.

        Shows all datasets and states registered by the broker.

        curl
        -X GET
        http://localhost:12050/status
        """
        self.log.debug('%.32r: Received status request' % self)
        reply = dict()
        with self.lock_datasets:
            self.log.debug('%.32r: states: %r' % (self, self.states))
            reply["states"] = self.states
        with self.lock_states:
            self.log.debug('%.32r: datasets: %r' % (self, self.datasets))
            reply["datasets"] = self.datasets
        coroutine_return(reply)

    @coroutine
    @endpoint('register-state')
    def registerState(self, handler, hash):
        """ Register a dataset state with the broker.

        This should only ever be called by kotekan's datasetManager.
        """
        self.log.debug('%.32r: Received register state request, hash: %r'
                       % (self, hash))
        reply = dict(result="success")
        with self.lock_states:
            if self.states.get(hash) is None:
                # we don't know this state, ask for it
                reply['request'] = "get_state"
                reply['hash'] = hash
        coroutine_return(reply)

    @coroutine
    @endpoint('send-state')
    def sendState(self, handler, hash, state):
        """ Send a dataset state to the broker.

        This should only ever be called by kotekan's datasetManager.
        """
        self.log.debug('%.32r: Received state %r : %r' % (self, hash, state))
        reply = dict()

        # do we have this state already?
        with self.lock_states:
            found = self.states.get(hash)
            if found is not None:
                # if we know it already, does it differ?
                if found != state:
                    reply['result'] = "error: a different state is know to " \
                                      "the broker with this hash: %r" % found
                    self.log.warn('%.32r: Failure receiving state: a '
                                  'different state with the same hash is: %r'
                                  % (self, found))
                else:
                    reply['result'] = "success"
            else:
                self.states[hash] = state
                reply['result'] = "success"
                self.signal_states_updated.notify_all()
        coroutine_return(reply)

    @coroutine
    @endpoint('register-dataset')
    def registerDataset(self, handler, hash, dataset):
        """ Register a dataset with the broker.

        This should only ever be called by kotekan's datasetManager.
        """
        self.log.debug('%.32r: Registering new dataset with hash %r : %r' %
                       (self, hash, dataset))
        reply = dict(result="success")

        # dataset already known?
        with self.lock_datasets:
            found = self.datasets.get(hash)
            if found is not None:
                # if we know it already, does it differ?
                if found != dataset:
                    reply['result'] = "error: a different dataset is know to" \
                                      " the broker with this hash: %r" % found
                    self.log.warn('%.32r: Failure receiving dataset: a'
                                  ' different dataset with the same hash is: %r'
                                  % (self, found))
                else:
                    reply['result'] = "success"
            else:
                self.datasets[hash] = dataset
                reply['result'] = "success"
                self.signal_datasets_updated.notify_all()

            coroutine_return(reply)

    @coroutine
    @endpoint('request-ancestor')
    def requestAncestor(self, handler, ds_id, type):
        """ Request the ancestor of a given type closest to the dataset with the
        given ID.

        This is called by kotekan's datasetManager.

        curl
        -d '{"ds_id":123,"type":"10freqState"}'
        -X POST
        -H "Content-Type: application/json"
        http://localhost:12050/request-ancestor
        """
        self.log.debug(
            '%.32r: Received request for ancestor of type %r of dataset %r'
            % (self, type, ds_id))
        reply = dict()

        # Do we know this dset ID?
        found = yield self.wait_for_dset(ds_id)
        if not found:
            reply['result'] = "error: dataset ID %r unknown to broker." % ds_id
            self.log.info('%.32r: Dataset %r unknown to broker' % (self, ds_id))
            coroutine_return(reply)

        try:
            # default parameter doesn't work in subroutines?
            ancestor = yield self.ancestor(ds_id, type, js=dict(datasets=dict(),
                                                                states=dict()))
            reply.update(ancestor)
        except Exception as error:
            reply["result"] = error.message
            self.log.error('%.32r: %r' % (self, error))
            coroutine_return(reply)

        reply['result'] = "success"
        coroutine_return(reply)

    @coroutine
    def wait_for_dset(self, id):
        found = True
        self.lock_datasets.acquire()

        if self.datasets.get(id) is None:
            # wait for half of kotekans timeout before we admit we don't have it
            self.lock_datasets.release()
            notified = True
            try:
                while notified:
                    notified = yield self.signal_datasets_updated.wait(
                        deadline=datetime.timedelta(seconds=15))
                    # did someone send it to us by now?
                    with self.lock_datasets:
                        if self.datasets.get(id) is not None:
                            break
            except toro.Timeout as e:
                self.log.debug('%.32r: %r' % (self, e.message))
                pass
            self.lock_datasets.acquire()
            if self.datasets.get(id) is None:
                self.log.warn('%.32r: Timeout when waiting for dataset %r'
                              % (self, id))
                found = False
        self.lock_datasets.release()

        coroutine_return(found)

    @coroutine
    def wait_for_state(self, id):
        found = True
        self.lock_states.acquire()
        if self.states.get(id) is None:
            # wait for half of kotekans timeout before we admit we don't have it
            self.lock_states.release()
            notified = True
            try:
                while notified:
                    notified = yield self.signal_states_updated.wait(
                        deadline=datetime.timedelta(seconds=15))
                    # did someone send it to us by now?
                    with self.lock_states:
                        if self.states.get(id) is not None:
                            break
            except toro.Timeout as e:
                self.log.debug('%.32r: %r' % (self, e.message))
                pass
            self.lock_states.acquire()
            if self.states.get(id) is None:
                self.log.warn('%.32r: Timeout when waiting for state %r'
                              % (self, id))
                found = False
        self.lock_states.release()

        coroutine_return(found)

    @coroutine
    def ancestor(self, ds_id, type, js):
        with self.lock_datasets:
            js["datasets"][ds_id] = self.datasets[ds_id]
            state_id = self.datasets[ds_id]['state']

        found = yield self.wait_for_state(state_id)
        if not found:
            raise Exception("Error: dataset-broker is in a bad state."
                            " Found reference to not existing state ID.")

        # look for the state of requested type
        with self.lock_states:
            if self.states[state_id]["type"] == type:
                js["states"][state_id] = self.states[state_id]
                coroutine_return(js)

            # loop through the inner states
            state = self.states[state_id].get("inner", None)
            while state != None:
                if state["type"] == type:
                    js["states"][state_id] = self.states[state_id]
                    coroutine_return(js)
                state = state.get("inner", None)

        # look for the requested type in parent dataset states
        with self.lock_datasets:
            if self.datasets[ds_id]['is_root']:
                coroutine_return(js)
            next_ds = self.datasets[ds_id]['base_dset']

        found = yield self.wait_for_dset(next_ds)
        if not found:
            raise Exception("Error: Broker is in bad state."
                            " Found reference to not existing dataset ID.")
        js = yield self.ancestor(next_ds, type, js)
        coroutine_return(js)


#########################################
# Dataset Broker REST client
#########################################

class DSBrokerAsyncRESTClient(AsyncRESTClient):
    """
    Implements an asynchronous client that exposes the functions of the
    specified remote dataset broker.

    The client is implemented using a Tornado AsyncHTTPClient. It exposes the
    dataset broker methods
    (i.e REST endpoints) as local methods. The local methods are Tornado
    coroutines so requests to
    multiple clients can be made in parallel. This is especially beneficial
    since the data requests
    from the server are slow IO operations which benefit the mist from
    co-execution.

    The client will operate only if the IOloop in which is was created is
    running.

    Parameters:

        name (str): Name of the client, to be used in logging etc.

        hostname (str): The hostname of the dataset broker. If `host` is None,
        an (experimental,
             Python-based) dataset broker REST server will be created locally.

        port (int): The port number to which the dataset broker REST server is
        listening. Default is port 80.
    """
    DEFAULT_PORT = DSBrokerAsyncRESTServer.DEFAULT_PORT

    def __init__(self, hostname='localhost', port=DEFAULT_PORT):
        super(DSBrokerAsyncRESTClient, self).__init__(
            hostname=hostname, port=port,
            server_class=DSBrokerAsyncRESTServer,
            heartbeat_string='Gc')

    @coroutine
    def registerState(self, hash):
        result = yield self.post('register-state', hash)
        coroutine_return(result)

    @coroutine
    def sendState(self, hash, state):
        result = yield self.post('send-state', hash, state)
        coroutine_return(result)

    @coroutine
    def registerDataset(self, state_id, base_ds_id):
        result = yield self.post('register-dataset', state_id, base_ds_id)
        coroutine_return(result)

    @coroutine
    def status(self):
        result = yield self.get('status')
        coroutine_return(result)

    @coroutine
    def requestAncestor(self, ds_id, type):
        result = yield self.post('request-ancestor', ds_id, type)
        coroutine_return(result)


def main():
    """ Command-line interface to launch and operate the dataset broker.
    """
    # Setup logging
    log.setup_basic_logging('DEBUG')
    client, server = run_client(sys.argv[1:],
                                DSBrokerAsyncRESTServer,
                                DSBrokerAsyncRESTClient,
                                object_name='DSETBROKER',
                                server_config_path='dsetbroker.servers')
    return client, server


if __name__ == '__main__':
    client, server = main()
