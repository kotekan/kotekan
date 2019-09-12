# ARO-live viewer

Developpers:

Jacob Taylor: jacob.taylor@mail.utoronto.ca

Keith Vanderlinde: vanderlinde@dunlap.utoronto.ca

##Dependencies

The live-view was written to accomodate a python 2 or 3 environment, 
however python 3 should be used whenever possible.

The live-view was written for a linux environment, preferably the latest version
of ubuntu. However, Centos or Mac work also. 

numpy: sudo apt-get install python3-numpy

pyqt5: sudo apt-get install python3-pyqt5

yaml: sudo apt-get install python3-yaml

astropy: sudo apt-get install python3-astropy

matplotlib: sudo apt-get install python3-matplotlib

psrcat: should be included with kotekan in kotekan/scripts/viewer/psrcat

gnome-terminal: The current live-view opens a gnome-terminal to create the port forward.
This will need to be updated to include more systems.

##Set-up

The live-viewer requires key-less entry to the system running kotekan.

Instructions for keyless entry set-up:

On the machine you wish to run the live-viewer enter the following:

```
ssh-keygen -t rsa

ssh-copy-id -i ~/.ssh/id_rsa.pub user@hostname
```

Where user and hostname refer to the machine running kotekan.

For easier ssh access add the following to ~/.ssh/config

Host **user_specified_name**
    HostName **hostname**
    User **user**

With the bold parameters indicating the information of the machine running kotekan. 

##Launching Kotekan

```
ssh **user**@**hostname**
cd **path_to_kotekan**/kotekan/build/kotekan
sudo ./kotekan -c **path_to_config**
```

Make sure the config you are using has a networkPowerStream process, Take note of the
port it is streaming to. 

##Launching Live-view

In a terminal on your local machine, navigate to the kotekan/scripts/viewer directory
and run the following.

```
python3 jacob_qt5_viewer.py
```
When prompted, enter the following information of the machine currently running kotekan:

user: example squirrel
IP address: example 192.168.52.35
port: example 2051 (this value can be found in the current kotekan config under the networkPowerStream process)

This should open a second terminal which should create a tunnel to the kotekan machine

##Features
