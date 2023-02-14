print("VOLTAGE    CPU(re,im)       CPU(re,im)sh     CPU       GPU")
for i in range(18):
    if not i in onehot:
        break
    print(
        "0x%02x      (% 4i, % 4i)      (% 4i, % 4i)     0x%02x      0x%02x"
        % (
            onehot[i][1],
            bb[i][1][0],
            bb[i][1][1],
            bb[i][2][0],
            bb[i][2][1],
            sparse["simulated_formed_beams_buffer"][i][1],
            sparse["host_formed_beams_buffer"][i][1],
        )
    )
