# Kotekan docker environment instructions
The files in this directory set up an array of docker containers and distributes
config files to mimick kotekan nodes on a network for testing. It was built to be
run on the receiver nodes and small changes may be necessary to use it on systems
other than Ubuntu 16.04.

**To run:**

1. Build kotekan on the host machine. This is the executable that will be run on the
Docker containers so the image they are built on needs to be similar to the host system.
(It would be possible to use a different image or compile kotekan on the container by
modifying the Dockerfile.)
2. Modify the docker-compose-example.yaml file to suit your needs. The example file has
service entries for a gpu node and a receiver node that can be used as templates for
additional services (which spawn containers). The kotekan config that will be provided
to the node is specified by the KOTEKAN_CONFIG environment variable
3. Start the Docker containers with the command
`docker-compose -f docker-compose-example.yaml up`, substituting in your own yaml file.
