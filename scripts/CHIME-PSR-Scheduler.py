"""
==================================
    Pulsar Scheduler (version 1) - Cherry Ng (15 June 2018)

    Used for generating observing schedule file for the 10-beam pulsar beamformer.
    Requires an input source file with PSRnames,RA,Dec,type,and some default priorities.
    Currently set high for 42 NANOGrav pulsars and known glitches.
    The scheduler also re-weights the priority based on how frequent each source has been scheduled previously
    Roughly 10 days is needed to cycle through all N-hemi pulsars, given the current allocation scheme.
    It is optional to generate associated summary plot to visualize the daily obs. schedule

    TO DOs:
    - figure out appropriate scaling factor
    - interaction with manual input (human override, new sources ...)
    - refine priority setting 
    - check lastobs by psr_id not name

==================================
"""
import numpy as np
import fnmatch
import subprocess
import sys, os
import math
import time
from operator import itemgetter
from optparse import OptionParser
import operator
import matplotlib as mpl
import pylab as plt
import datetime
mpl.rcParams.update({'font.size': 13,'font.family': 'serif'})

usage = "usage: %prog [options] arg"
parser = OptionParser()
parser.add_option("-f", dest="INF", help="Input file with list of pulasr RA DEC", default="Pulsar_Sources.lis")
parser.add_option("-t", dest="Date", help="Input date YYYY.MM.DD (default today)", default=time.strftime("%x"))
parser.add_option("-p", dest="PLOT", help="make summary plot (default \"false\")", default="false")

(options, args) = parser.parse_args(args=sys.argv[1:])
print options

d2r = math.pi/180.
r2d = 180./math.pi

#CHIME site specific
Direction = "W"
LAT = 49.4911
Long = 119.5886
Diff_Long = Long / 15. #In hour
Horizon_limit = 20 #Only consider those above horizon of 20 deg

#Parse command line options
Date = options.Date
INF = options.INF
PLOT = options.PLOT

if (PLOT == 'true'):
    fig = plt.figure(figsize=[14,8])
    plt.subplots_adjust(wspace = 1)
    TSTART = 0 
    TEND = 24

#Field of view is in radian
FOV400 = 2.5 / 180. * (math.pi) #it is 2.5 deg in documentations, although it is strictly speaking not c/Freq/20
FOV800 = 1.3 / 180. * (math.pi)

#Sidereal rate is the rotation of the Earth, in radian per hr.
Rsid = (2*(math.pi) ) / (23.93447)

#Convert obs date to JD
import astropy.time
import dateutil.parser
dt = []
dt.append(dateutil.parser.parse(Date))
time = astropy.time.Time(dt)
JD = time[0].jd  

#----Remove any previously genearted schedule file of this day
os.system("rm Sch*"+str(JD)+"*.dat")

#Read in file, these lists have the same length as the input source list file
PSRnames = np.genfromtxt(INF, dtype=str, unpack=True)[1]
Pindex = np.genfromtxt(INF, dtype=str, unpack=True)[0]
RA = np.genfromtxt(INF).transpose()[2]
DEC = np.genfromtxt(INF).transpose()[3]
Priorities = np.genfromtxt(INF).transpose()[5]
Type = np.genfromtxt(INF, dtype=str, unpack=True)[4]

#These lists will have longer length, depending on the observing time
Sche_UTrise = []
Sche_UTset = []
Sche_ALT = []
Sche_PSR = []
Sche_Priori = []
Sche_Type = []
Sche_ID = []
Sche_RA = []
Sche_DEC = []

#Total days in directory
nday = int(subprocess.check_output("ls Sch*dat | wc -l ",shell=True).strip())
#Histogram of lastobs
LastObs = []
for j in range(len(PSRnames)): #Loop through each pulsars from the input file
    if ( DEC[j] > (Horizon_limit-90+LAT)) : 
        #----Figure out some last obs statistics---------------
        #-----TODO: better to use pulsar index because of some degenerate names
        #---- and maybe eventually replaced by a count of the observations, instead of the schedule, require feedback from pulsarbackend
        penalty_gal = 0
        if (nday > 0):
            penalty = int(subprocess.check_output("grep "+str(PSRnames[j].split("_cp")[0])+" Sch*dat | wc -l ",shell=True).strip())
            bonus = (nday-penalty)*200.
        else:
            penalty = 0
            bonus = 0
        LastObs.append(penalty)

        #Depending on the priority, chose the tracking duration (FOV)
        if (Priorities[j] > 500): 
            FOV = FOV400
        else:
            FOV = FOV800
        #Drift time is always the same for a particular pulsar
        DriftAngle = math.acos((np.cos(FOV) - (np.sin(DEC[j]*d2r))**2 ) / (np.cos(DEC[j]*d2r))**2 )
        TIME = DriftAngle / Rsid

        #Find the LST rannge (based on RA only)
        LST = RA[j]/15.
        LST_rise = LST - TIME/2.
        LST_set = LST + TIME/2.

        #Convert LST to GST
        GST_rise = (LST_rise + Diff_Long) % 24
        GST_set = (LST_set + Diff_Long) % 24

        S = JD - 2451545.0    #at noon TT on 1 January AD 2000, JD 2,451,545 started.
        T = S / 36525.
        T0=6.697374558 + (2400.051336*T) + (0.000025862*T*T)
        while (T0>24):
            T0 = T0-24

        #Convert GST to UT        
        r = ((GST_rise - T0) % 24 ) * 0.9972695663
        s = ((GST_set - T0) % 24 ) * 0.9972695663 

        if (r > s):  
            #To take care of some pulsars rising just near 24h (end of day), make the set time "a day later"
            T_tmr = (S+1) / 36525.
            T0_tmr = 6.697374558 + (2400.051336*T_tmr) + (0.000025862*T_tmr*T_tmr)
            while (T0_tmr>24):
                T0_tmr = T0_tmr-24
            s_tmr = ((GST_set - T0_tmr) % 24 ) * 0.9972695663
            if (s<s_tmr): #set time tomorrow is 23hr, end of file
                r2 = ((24+GST_rise-T0) % 24 ) * 0.9972695663
                s2 = r2 + TIME*0.9972695663 
                Sche_ALT.append(90 - LAT + DEC[j])
                Sche_PSR.append(str(PSRnames[j]))
                Sche_Priori.append( Priorities[j] + bonus )
                Sche_Type.append(Type[j])
                Sche_ID.append(Pindex[j])
                Sche_RA.append(RA[j])
                Sche_DEC.append(DEC[j])
                Sche_UTrise.append(r2)
                Sche_UTset.append(s2)
            # "Corner case", force the transit to be the day before
            r = (GST_rise - T0) *0.9972695663
            s = (GST_set - T0) *  0.9972695663

        #If circumpol, list it again after/before 12 hrs
        if ( DEC[j] >= (90-LAT)) :
            penalty_gal_cir = 200 
            if (RA[j]>225) and (RA[j]<285):  #in galactic plane, increase penalty in the main transit
                penalty_gal = 500
                penalty_gal_cir = -100  #encoruage it to be observed out of the galactic plane
            RAapp = (RA[j] + 180) % 360
            if (RAapp>225)    and (RAapp<285):
                penalty_gal_cir  = 500   #in galactic plane, increase penalty in the circumpol re-transit
            GST_rise_app = (RAapp/15. - TIME/2. + Diff_Long) % 24
            GST_set_app = (RAapp/15. + TIME/2. + Diff_Long) % 24
            rc = ((GST_rise_app - T0) % 24 ) * 0.9972695663 
            sc = ((GST_set_app - T0) % 24 ) * 0.9972695663
            if (rc > sc) :
                # "Rare case", take care of the day after
                T_tmr = (S+1) / 36525.
                T0_tmr = 6.697374558 + (2400.051336*T_tmr) + (0.000025862*T_tmr*T_tmr)
                while (T0_tmr>24):
                    T0_tmr = T0_tmr-24    
                s_tmr = ((GST_set_app - T0_tmr) % 24 ) * 0.9972695663
                if (sc < s_tmr): #set time tomorrow is 23hr, end of file
                    rc2 = ((24+GST_rise_app-T0) % 24 ) * 0.9972695663
                    sc2 = rc2 + TIME*0.9972695663
                    Sche_UTrise.append(rc2)
                    Sche_UTset.append(sc2)
                    Sche_ALT.append(270 - LAT - DEC[j])
                    Sche_PSR.append(str(PSRnames[j])+"_cp")
                    Sche_Type.append(Type[j])
                    Sche_ID.append(Pindex[j])
                    Sche_RA.append(RAapp)
                    Sche_DEC.append(180-DEC[j])
                    Sche_Priori.append( Priorities[j] + bonus - penalty_gal_cir)
                # "Corner case", force the transit to be the day before
                rc = (GST_rise_app - T0) *0.9972695663
                sc = (GST_set_app - T0) *  0.9972695663
            Sche_UTrise.append(rc)
            Sche_UTset.append(sc)
            Sche_ALT.append(270 - LAT - DEC[j])
            Sche_PSR.append(str(PSRnames[j])+"_cp")
            Sche_Type.append(Type[j])
            Sche_ID.append(Pindex[j])
            Sche_RA.append(RAapp)
            Sche_DEC.append(180-DEC[j])
            Sche_Priori.append( Priorities[j] + bonus - penalty_gal_cir)

        Sche_ALT.append(90 - LAT + DEC[j])
        Sche_PSR.append(str(PSRnames[j]))
        Sche_Priori.append( Priorities[j] + bonus - penalty_gal)
        Sche_Type.append(Type[j])
        Sche_ID.append(Pindex[j])
        Sche_RA.append(RA[j])
        Sche_DEC.append(DEC[j])
        Sche_UTrise.append(r)
        Sche_UTset.append(s)

#--------------------------------------------------------------------------------------------------
if (PLOT == 'true'):
    ax_main = plt.subplot2grid((5,5),(1,0), rowspan=3,colspan=4)
    #Plot every visible stars in red once
    for j in range(len(Sche_UTrise)):
        #---Bright sources for sanity check -----
        if (Sche_PSR[j].split("_cp")[0] == "B0329+54") or (Sche_PSR[j].split("_cp")[0] == "B2016+28") :
            ax_main.annotate(str(Sche_PSR[j]),xy=(Sche_UTrise[j],Sche_ALT[j]),fontsize=9,horizontalalignment='right',verticalalignment='bottom',zorder=20)
            ax_main.plot( (Sche_UTrise[j],Sche_UTset[j]),(Sche_ALT[j],Sche_ALT[j]),lw=4.6,c='k',zorder=8)
        if (Sche_Type[j] == 'glitch'): #Glitch
            ax_main.plot( (Sche_UTrise[j],Sche_UTset[j]),(Sche_ALT[j],Sche_ALT[j]),lw=4.6,c='c',zorder=7)
        elif (Sche_Type[j] == 'nanograv'): #NANOGrav
            ax_main.plot( (Sche_UTrise[j],Sche_UTset[j]),(Sche_ALT[j],Sche_ALT[j]),lw=4.6,c='m',zorder=7)
    #These are just a cheeky way to get pretty legend
    ax_main.plot( (0,0),(0,0),lw=2,c='r',zorder=0, label="Unobserved")
    ax_main.plot( (0,0),(0,0),lw=2,c='m',zorder=0, label='NANOGrav')
    ax_main.plot( (0,0),(0,0),lw=2,c='c',zorder=0, label='Glitch')

    ax_lastobs = plt.subplot2grid((5,5),(4,0), rowspan=1,colspan=4)
    bins = np.arange(0,nday*2+2,1)
    offset = 0.5
    ax_lastobs.hist(LastObs, color='gainsboro',bins=bins)
    ax_lastobs.set_xticks(bins+ offset)
    ax_lastobs.set_xticklabels(bins)
    ax_lastobs.set_xlim([0,nday*2+1])
    ax_lastobs.set_xlabel('Number of observations within the last '+str(nday)+' days')
    ax_lastobs.set_ylabel('Number of pulsars')
            

#------------------------------------------------------
          
#Sort times in ascending order
Sorting = np.argsort(Sche_UTrise)
Sche_ALT = np.array(Sche_ALT)[Sorting]
Sche_PSR = np.array(Sche_PSR)[Sorting]  
Sche_RA = np.array(Sche_RA)[Sorting] 
Sche_DEC = np.array(Sche_DEC)[Sorting] 
Sche_Type = np.array(Sche_Type)[Sorting] 
Sche_ID = np.array(Sche_ID)[Sorting]
Sche_UTrise = np.array(Sche_UTrise)[Sorting] 
Sche_UTset = np.array(Sche_UTset)[Sorting] 
Sche_Priori = np.array(Sche_Priori)[Sorting]

 
if (PLOT == 'true'):
    #Do a star count for the whole sky, at every time step where a new source comes (Red line on the top panel)
    CountAtEachStep = np.zeros(len(Sche_UTrise),dtype=int)
    for i in range(len(Sche_UTrise)):
        c = 0 
        for j, (r, s) in enumerate(zip(Sche_UTrise,Sche_UTset)):
            if (r <= Sche_UTrise[i]) and (s >= Sche_UTrise[i])  :
                c = c+1
        CountAtEachStep[i] = c
    ax_skyct = plt.subplot2grid((5,5),(0,0), rowspan=1,colspan=4)
    ax_skyct.plot(Sche_UTrise, CountAtEachStep, c='r', alpha=0.3,zorder=0, label='All pulsars on sky')
    ax_skyct.fill_between(Sche_UTrise, 0, CountAtEachStep, color='r',alpha=0.3,zorder=0)
#-----------------------

print "Full lengths (UTr, UTs, PSR, Priori, Sche_RA, Sche_DEC)      ", len(Sche_UTrise), len(Sche_UTset), len(Sche_PSR), len(Sche_Priori), len(Sche_RA), len(Sche_DEC)

n_del = 0
#Find the times when >10 pulsars are on sky and dump some sources, according to the priority
for i in range(len(Sche_UTrise)):
    ii = i - n_del  #Take care of the case if deleted earlier sources
    if (ii<len(Sche_UTrise)):  #Because the length gets shortened as we delete entries
        Cur_Ind = []
        for j, (r, s) in enumerate(zip(Sche_UTrise,Sche_UTset)):
            if (r <= Sche_UTrise[ii]) and (s >= Sche_UTrise[ii])  :
                Cur_Ind.append(j)
        b = len(Cur_Ind) 
        if (b>10):
            Cur_Priori = np.zeros(b)
            Cur_rise = np.zeros(b)
            for k in range(b):
                Cur_Priori[k] = Sche_Priori[Cur_Ind[k]]
                Cur_rise[k] = Sche_UTrise[Cur_Ind[k]]
            
            order = np.lexsort((Cur_rise, -(Cur_Priori) )) #Sort by reverse priority and also the later sources
            while (b > 10):  #These ones get dumped
                INDEX = Cur_Ind[order[-1]]
                if (INDEX <= ii):
                    n_del = n_del + 1
                    if (PLOT == 'true'):
                        ax_main.plot( (Sche_UTrise[INDEX],Sche_UTset[INDEX]),(Sche_ALT[INDEX],Sche_ALT[INDEX]),lw=2,c='r',zorder=9)

                Sche_UTrise = np.delete(Sche_UTrise,INDEX)
                Sche_UTset =  np.delete(Sche_UTset,INDEX)
                Sche_PSR = np.delete(Sche_PSR,INDEX)
                Sche_Priori = np.delete(Sche_Priori,INDEX)
                Sche_RA = np.delete(Sche_RA,INDEX)
                Sche_DEC = np.delete(Sche_DEC,INDEX)
                Sche_ALT = np.delete(Sche_ALT,INDEX)
                Sche_Type = np.delete(Sche_Type,INDEX)
                Sche_ID = np.delete(Sche_ID,INDEX)
                order = np.delete(order, -1)
                b = b - 1

                
if (PLOT == 'true') : #=---------------------Re-count, for plotting purpose
    CountAtEachStepFinal = np.zeros(len(Sche_UTrise),dtype=int)
    for i in range(len(Sche_UTrise)):
        c = 0 
        for j, (r, s) in enumerate(zip(Sche_UTrise,Sche_UTset)):
            if (r <= Sche_UTrise[i]) and (s >= Sche_UTrise[i])  :
                c = c+1
        CountAtEachStepFinal[i] = c
    ax_skyct.plot(Sche_UTrise, CountAtEachStepFinal, c='Gray',zorder=1, label='Final allocations')
    ax_skyct.fill_between(Sche_UTrise, 0, CountAtEachStepFinal, color='gainsboro',zorder=1)
#---------------------------

print "After dumping: Full lengths (UTr, UTs, PSR, Priori, Sche_RA, Sche_DEC)      ", len(Sche_UTrise), len(Sche_UTset), len(Sche_PSR), len(Sche_Priori), len(Sche_RA), len(Sche_DEC)

#----- Allocate to each of the 10 beams
beam_now = 0 
beam_last_set = np.zeros(10)
beam_ids = np.zeros(len(Sche_UTrise),dtype=int)
for i in range(10):
    beam_ids[i] = beam_now
    beam_last_set[beam_now] = Sche_UTset[i]
    beam_now = (beam_now + 1) %10
    if (PLOT == 'true'):
        ax_main.plot(  (Sche_UTrise[i],Sche_UTset[i]),(Sche_ALT[i],Sche_ALT[i]) ,lw=2,c='gainsboro',zorder=10)
for i in range(10,len(Sche_UTrise),1):
    while (beam_last_set[beam_now] > Sche_UTrise[i]):
        beam_now = (beam_now + 1) %10
    beam_ids[i] = beam_now
    beam_last_set[beam_now] = Sche_UTset[i]
    beam_now = (beam_now + 1) %10
    if (PLOT == 'true'):
        ax_main.plot(  (Sche_UTrise[i],Sche_UTset[i]),(Sche_ALT[i],Sche_ALT[i]) ,lw=2,c='gainsboro',zorder=10)

print "Going to write schedule to: Schedule_JD"+str(JD)+".dat"
f = open("Schedule_JD"+str(JD)+".dat", "w")
f.write('%-15s %-17s %-17s %-13s %-11s %-2s %-5s\n' % ("#psrname", "Unix_utc_start", "Unix_utc_end", "ra", "dec", "beam", "scaling"))
for j, (psr, t, ts, ra, dec, b) in enumerate(zip(Sche_PSR,Sche_UTrise, Sche_UTset, Sche_RA,Sche_DEC,beam_ids)):
    tt = (dt[0] - datetime.datetime(1970,1,1)).total_seconds()+t*3600.
    tts = (dt[0] - datetime.datetime(1970,1,1)).total_seconds()+ts*3600.
    f.write('%-15s %-17s %-17s %-13s %-14s %-5s 48\n' % (psr, tt, tts, ra, dec, b))
f.close()

if (PLOT == 'true'):
    ax_main.axhline(y=20, c='k') #label='Elevation=20')
    ax_main.axhline(y=180-LAT, c='k') #label='Elevation=20')
    ax_main.axhline(y=180-LAT*2, c='k',ls=':') #label='Elevation=20')

    ax_main.set_xlim([TSTART,TEND])
    ax_main.set_xticks(np.arange(TSTART, TEND+1, 5))
    minor_ticks = np.arange(TSTART, TEND, 1)
    ax_main.set_xticks(minor_ticks, minor=True)
    ax_main.set_xticklabels([]) 

    ax_main.set_ylim([20,180])
    ax_main.set_yticks(np.arange(20, 180, 20))
    axY2 = ax_main.twinx()
    axY2.set_ylim([20-90+LAT,180-90+LAT])
    axY2.set_yticks(np.arange(-20, 140, 20))
    legend = ax_main.legend(loc='upper right', prop={'size':9})
    legend.set_zorder(200)
    frame = legend.get_frame()
    frame.set_facecolor('w')
    for legobj in legend.legendHandles:
        legobj.set_linewidth(4.0)

    ax_skyct.set_xlim([TSTART,TEND])
    ax_skyct.set_ylim([0,np.max(CountAtEachStep)])
    ax_skyct.set_xticks(np.arange(TSTART, TEND+1, 5))
    minor_ticks = np.arange(TSTART, TEND, 1)
    ax_skyct.set_xticks(minor_ticks, minor=True)
    ax_skyct.axhline(y=10, c='gainsboro',ls=":")
    leg = ax_skyct.legend(loc="upper left", prop={'size':10} )
    for legobj in leg.legendHandles:
        legobj.set_linewidth(4.0)

    #Plot the number of pulsars per beam (right panel)-------------------------------
    ax_beamload = plt.subplot2grid((5,5),(0,4), rowspan=5,colspan=1)
    beamload = np.zeros(10,dtype=int)
    for i in range(len(Sche_UTrise)):
        beamload[beam_ids[i]] = beamload[beam_ids[i]] + 1
    npsr = np.arange(10)
    ax_beamload.plot(beamload, npsr ,'k',zorder=10)
    for i,j in zip(beamload, npsr):
        ax_beamload.annotate(str(i),xy=(i,j),fontsize=13,horizontalalignment='right',verticalalignment='bottom',zorder=20)


    for i in range(len(beamload)-1 ):
        fx = [beamload[i], beamload[i+1], 0, 0]
        fy = [npsr[i], npsr[i+1], npsr[i+1], npsr[i]]
        plt.fill(fx,fy, color='gainsboro',zorder=0)

    vmax = np.max(beamload)
    vhalf = int(vmax/2.)
    ax_beamload.set_xlim([0,vmax+5])
    ax_beamload.set_xticks([0,vhalf,vmax])

    plt.figtext(0.2,0.59,"Circumpolar",fontdict={'fontsize':12},horizontalalignment='center')
    plt.figtext(0.2,0.57,"Region",fontdict={'fontsize':12},horizontalalignment='center')
    plt.figtext(0.43,0.74,"UT time (hr)",fontdict={'fontsize':12},horizontalalignment='center')
    plt.figtext(0.86,0.05,"N$_{\mathrm{psr}}$/beam",fontdict={'fontsize':12},horizontalalignment='center')
    plt.figtext(0.09,0.5,"Elevation ($^{\circ}$)",fontdict={'fontsize':12},rotation=90,horizontalalignment='center',verticalalignment='center')
    plt.figtext(0.755,0.5,"Declination ($^{\circ}$)",fontdict={'fontsize':12},rotation=-90,horizontalalignment='center',verticalalignment='center')
    plt.figtext(0.792,0.5,"Pulsar Beam ID",fontdict={'fontsize':12},rotation=90,horizontalalignment='center',verticalalignment='center')
    plt.figtext(0.09,0.82,"N$_{\mathrm{psr}}$/instance",fontdict={'fontsize':12},rotation=90,horizontalalignment='center',verticalalignment='center')
    plt.figtext(0.5,0.92,"CHIME Pulsar observation scheduled on "+str(Date), fontdict={'fontsize':16},horizontalalignment='center',verticalalignment='center')

    plt.savefig("SummaryPlan_JD"+str(JD)+".pdf",transparent=False, bbox_inches='tight')
