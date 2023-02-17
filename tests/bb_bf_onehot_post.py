"""
See documentation in bb_bf_onehot_pre.py
"""
print("VOLTAGE    CPU(re,im)       CPU(re,im)sh     CPU       GPU")
for i in range(18):
    if not i in onehot:
        break
    bbi = bb.get(i)
    cpu = (0, 0)
    cpush = (0, 0)
    if bbi is not None:
        cpu = (bbi[1][0], bbi[1][1])
        cpush = (bbi[2][0], bbi[2][1])
    print(
        "0x%02x      (% 4i, % 4i)      (% 4i, % 4i)     0x%02x      0x%02x"
        % (
            onehot[i][1],
            cpu[0],
            cpu[1],
            cpush[0],
            cpush[1],
            sparse["simulated_formed_beams_buffer"][i][1],
            sparse["host_formed_beams_buffer"][i][1],
        )
    )
