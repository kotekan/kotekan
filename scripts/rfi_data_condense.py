import numpy as np
import os
try:
    import matplotlib.pyplot as plt
except:
    print("No Plotting Availiable")
import sys

"""
rfi_file_list = []
for f in os.listdir("."):
    if f.endswith(".rfi"):
        rfi_file_list.append(f)
data = np.empty([0])
for f in rfi_file_list:
	data = np.append(data, np.loadtxt(f,delimiter =','))
np.save("rfi_recorder_data_condensed",data)
"""
def LocateFiles():
    #Locate all the files
    rfi_file_list = []
    for f in os.listdir("."): 
        if f.endswith(".rfi"):
            rfi_file_list.append(f)
    for i in np.arange(16):
        flag = True
        for j in range(len(rfi_file_list)):
            if("_%d_"%(i) in rfi_file_list[j]):
                flag = False
                break
        if flag:
            print("Missing Node %d"%(i+1))
    return rfi_file_list

def ExtractData(file_list):
    #Extract All the data
    data_list, min_seq_list, max_seq_list = [], [], []
    for f in file_list:
        data = np.fromfile(f,dtype=np.dtype([('bin', 'i4',1), ('seq', 'i8',1), ('mask', 'f4',1)]))
        data_list.append((data,f))
        min_seq_list.append(np.min(data['seq']))
        max_seq_list.append(np.max(data['seq']))
        #print(f,np.min(data['seq']),np.max(data['seq']))
    min_seq, max_seq = np.median(np.array(min_seq_list)), np.median(np.array(max_seq_list))
    return data_list, min_seq, max_seq

def CondenseData():
    files = LocateFiles()
    data_list, min_seq, max_seq = ExtractData(files)

    Time_Dim = int((max_seq-min_seq)/32768)
    Total_data = np.zeros([1024,Time_Dim])

    for data,f in data_list:
        data = data[(data['seq'] < max_seq) & (data['seq'] > min_seq)]
        if(data.size < 1):
            print("Bad File: " + f)
            continue
        Total_data[data['bin'],((data['seq']-min_seq)/32768).astype(int)] = data['mask']
    
    np.save("RFI_Time_Mask_Waterfall",Total_data)
    Spectrum = np.mean(Total_data,axis=1)
    np.save("RFI_Spectrum",Spectrum)

def LoadTimeData():
    return np.load("RFI_Time_Mask_Waterfall.npy")

def LoadSpectrumData():
    return np.load("RFI_Spectrum.npy")

def PlotWaterfall(data,filename=''):
    plt.imshow(data,cmap = 'viridis',extent=[0,data.shape[1]*.08388608,400,800],interpolation='none', aspect = 'auto')
    plt.colorbar()
    plt.xlabel("Time[s]")
    plt.ylabel("Frequency[MHz]")
    if(filename != ''):
        plt.savefig("plots/"+filename+".pdf")
    plt.show()

def PlotSpectrum(data,filename=''):
    link_avg = [0]*8
    node_avg = [0]*16
    data_to_plot = np.zeros_like(data)
    for node in range(5):
        for link in range(2):
            for freq in range(8):
                CurrentBin = node + 16*link + 128*freq
                data_to_plot[CurrentBin] = data[CurrentBin]
                link_avg[link] += data[CurrentBin]
                node_avg[node] += data[CurrentBin]
    for i in range(len(node_avg)):
        print("Chi %02d: %f"%(i+1,node_avg[i]*100/64))
    for i in range(len(link_avg)):
        print("Link %d: %f"%(i+1,link_avg[i]*100/128))


    plt.plot(np.linspace(800,400,num=1024),data_to_plot,'.')
    plt.xlabel("Frequency[MHz]")
    plt.ylabel("RFI Mask Percentage")
    plt.show()

if __name__ == '__main__':
    
    try:
        mode=sys.argv[1]
    except:
        print("Please Select a Mode")
        sys.exit(0)
    try:
        filename=sys.argv[2]
    except:
        filename = ''
    if(mode == "condense"):
        CondenseData()
    elif(mode == "waterfall"):
        PlotWaterfall(LoadTimeData(),filename=filename)
    elif(mode == "spectrum"):
        PlotSpectrum(LoadSpectrumData(),filename=filename)
    else:
        print("Invalid Mode.... Exitting")

