import numpy as np
import datetime

import ch_util
from ch_util import ephemeris, tools

import chimedb.core


print("feed_positions:")

# Get the latest correlator inputs from the Layout DB.
chime_inputs = ch_util.tools.get_correlator_inputs(datetime.datetime.now(), correlator="chime")

# The feed positions in the kotekan config file correspond to chan_ids [0-255], [512-767], [1024-1279], [1536-1791].
# For a map, see: https://bao.chimenet.ca/doc/documents/538/2/get/chime_map_cyl_based.pdf?n=1942
cyl1_chan_id = np.arange(0, 256, 1)
cyl2_chan_id = np.arange(512, 768, 1)
cyl3_chan_id = np.arange(1024, 1280, 1)
cyl4_chan_id = np.arange(1536, 1792, 1)
cyl_chan_id = np.concatenate((cyl1_chan_id, cyl2_chan_id, cyl3_chan_id, cyl4_chan_id))

for i in cyl_chan_id:
    
    antenna_type = chime_inputs[i].__class__.__name__
    id = chime_inputs[i].id
    crate = chime_inputs[i].crate
    slot = chime_inputs[i].slot
    sma = chime_inputs[i].sma
    corr_order = chime_inputs[i].corr_order
    delay = chime_inputs[i].delay
    input_sn = chime_inputs[i].input_sn
    corr = chime_inputs[i].corr
    
    pos_x = 0.0
    pos_y = 0.0

    if ((antenna_type != "NoiseSource") and (antenna_type != "Blank") and (antenna_type != "HolographyAntenna")):
        pos_x = chime_inputs[i].pos[0]
        pos_y = chime_inputs[i].pos[1]
    
    antenna_str = f"# {antenna_type}(id={id}, crate={crate}, slot={slot}, sma={sma}, corr_order={corr_order}, delay={delay}, input_sn='{input_sn}', corr='{corr}')"

    if (i == 0):
        prefix = f"  [[{pos_x:.20f}, {pos_y:.20f}], "
        
        if ((pos_x == 0.0) and (pos_y == 0.0)):
            prefix = f"  [[0.0, 0.0], "

        pad = max(0, 58 - len(prefix))
        print(prefix + " " * pad + antenna_str)

    elif (i == 1791):
        prefix = f"   [{pos_x:.20f}, {pos_y:.20f}]] "

        if ((pos_x == 0.0) and (pos_y == 0.0)):
            prefix = f"   [0.0, 0.0]] "

        pad = max(0, 58 - len(prefix))
        print(prefix + " " * pad + antenna_str)

    else:
        prefix = f"   [{pos_x:.20f}, {pos_y:.20f}], "

        if ((pos_x == 0.0) and (pos_y == 0.0)):
            prefix = f"   [0.0, 0.0], "

        pad = max(0, 58 - len(prefix))
        print(prefix + " " * pad + antenna_str)
