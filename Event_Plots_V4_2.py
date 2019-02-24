#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 13:08:18 2018

@author: matthew
"""
import spacepy.pycdf as cdf
import numpy as np
import matplotlib.pyplot as plt
import time
import rbspfetchdata.updateCDFs as up
import datetime as dt
import cdfstuff.ripper_V2 as rip
from scipy import signal
import mysql.connector as conn
import scipy
import dbstuff.databaseFunctions as dbFun
from matplotlib.gridspec import GridSpec
from scipy.signal import blackmanharris
from scipy import signal
from scipy.ndimage import gaussian_filter1d
from scipy.fftpack import fft
import matplotlib.dates as mdates
import matplotlib.colors as colors
import matplotlib.ticker as ticker
#==============================================================================
#==============================================================================
def format_func(value, tick_number):
  return '{0:.4g}'.format(1/value)

def sort_Mag(magEntries):
  cutoff = 500
  mag_Epoch = []
  mag_Mag = []
  mag_Coord = []
  for entry in magEntries:      
      mag_Epoch.append(entry[0])
      if entry[1] > cutoff:
          mag_Mag.append(cutoff)
      else:
          mag_Mag.append(entry[1])
      mag_Coord.append(entry[12]/6371.2)
  mag_Epoch = np.array(mag_Epoch)
  mag_Mag = np.array(mag_Mag)
  mag_Coord = np.array(mag_Coord)
  return mag_Epoch, mag_Mag, mag_Coord
   
def get_FFT_Power_Info(mag_Mag, mag_Epoch): 
  datesNum = mdates.date2num(mag_Epoch)
  span = len(datesNum)
  power = []   
  # Number of sample points(must be even!!!)
  N = 4096
  # sample spacing
  T = 1
  
  frequency = np.linspace(0.0, 1.0/(2.0*T), N/2)
  for i in range(0,len(mag_Mag)):
      if i > int(N/2) and len(mag_Mag) - i > int(N/2):
          magSlice = mag_Mag[i-int(N/2):i+int(N/2)]
          timeSlice = datesNum[i-int(N/2):i+int(N/2)]
      if i < int(N/2):
          firstSlice = mag_Mag[0:int(N/2)-i]
          magSlice = np.append(firstSlice, mag_Mag[0:i+int(N/2)])
          secondSlice = datesNum[0:int(N/2)-i]
          timeSlice = np.append(secondSlice, datesNum[0:i+int(N/2)])
      if len(mag_Mag) - i < int(N/2):
          firstSlice = mag_Mag[i-int(N/2):]
          magSlice = np.append(firstSlice, mag_Mag[len(mag_Mag) + 
                              (len(mag_Mag)-(int(N/2)+i)):])
          secondSlice = datesNum[i-int(N/2):]
          timeSlice = np.append(secondSlice, datesNum[len(mag_Mag) + 
                              (len(mag_Mag)-(int(N/2)+i)):])
      
#      fitMag = get_Fit_Values(timeSlice, magSlice)
      y = np.array(magSlice) - np.mean(magSlice)
      wbh = blackmanharris(N)
      ywbhf = fft(y*wbh)
      power.append(np.abs(ywbhf))
      
  power = np.array(power)
  return power, frequency, datesNum, span, N

def get_Power_Spectra_Subplot(fig, ax, axT, magDict, betaDict):
    ax.text(.98, .8, 'Magnetic Field Power Spectra', horizontalalignment='right', transform=ax.transAxes, 
      bbox=dict(facecolor='white', alpha=0.7))
    ax.set_facecolor('black')
    lowest = 3
    cutoff = 1200
    power, frequency, datesNum, span, N = get_FFT_Power_Info(magDict['Magnitude'], magDict['Epoch'])
    newN = len(frequency) - cutoff
    newFreq = frequency[lowest:newN]
    y,x = np.meshgrid(newFreq, datesNum)
    newPower = power[:,lowest:newN]
    im = ax.pcolormesh(x, y, newPower, cmap='RdBu_r', norm=colors.LogNorm(vmin=0.01, vmax=newPower.max()))
    ax.set_ylim(np.max(newFreq), np.min(newFreq))
    ax.set_xlim(datesNum[0], datesNum[span-1])
    Hours = mdates.HourLocator()   
    Minutes = mdates.MinuteLocator()
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(format_func))
    ax.xaxis.set_major_locator(Hours)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.set_xticks(datesNum[::int(np.ceil(span/15))])
    fig.colorbar(im, ax=ax, fraction = .05, pad = .07)   
    ax.set_ylabel('Period [sec]', labelpad=5)
    
    axT.set_ylabel('Beta Value', rotation=270, labelpad=11)
    axT.plot(betaDict['Epoch'], betaDict['Total'], color='black')
    axT.axhline(y=1, color='black')
    
def get_Particle_Heatmap_SubPlot_EnergyBinned(fig, ax, axT, enBin, TOFxEHDict):
  ax.text(.98, .8, 'Particle Count By Energy Channel', horizontalalignment='right', transform=ax.transAxes, 
       bbox=dict(facecolor='white', alpha=0.7))
  ax.set_facecolor('black')
  ax.set_ylabel('Energy [keV]')
  TOFxE_Epoch = TOFxEHDict['Epoch']
  enBin[np.where(enBin == 0)] = 0.01
  energy = np.linspace(10, 1000, 14)
  x,y = np.meshgrid(TOFxEHDict['Epoch'], energy)
  im = ax.pcolormesh(x, y, enBin.transpose(), cmap='jet', norm=colors.LogNorm(vmin=0.1, vmax=enBin.max()+3e12))
  span = np.ceil(int(TOFxE_Epoch[len(TOFxE_Epoch)-1].timestamp() - TOFxE_Epoch[0].timestamp())/11)
  Hours = mdates.HourLocator()   
  Minutes = mdates.MinuteLocator()
  ax.xaxis.set_major_locator(Minutes)
  ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
  ax.set_xticks(TOFxE_Epoch[::int(span/15)])
  fig.colorbar(im, ax=ax, fraction = .05, pad = .07)
  ax.set_yscale('linear')
  
  axT.plot(TOFxEHDict['Epoch'], TOFxEHDict['FPDU_Density'], color='maroon')  
  axT.set_ylabel('Particle Density', rotation = 270, labelpad = 11)
  
def fmt(x, pos):
    a, b = '{:.0e}'.format(x).split('e')
    b = int(b)
    return r'${}*10^{{{}}}$'.format(a, b)

def get_Particle_Heatmap_SubPlot_AngleBinned(fig, ax, axT, angleBin, SC):
  TOFxEHDict = rip.get_CDF_Dict('TOFxEH_'+SC, strtDate, stpDate)
  kappaDict = dbFun.get_Kappa_For_Span('TOFxEH_'+SC, strtDate, stpDate)
  ax.text(.98, .8, 'Particle Count By Pitch Angle', horizontalalignment='right',
          transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.7))
  ax.set_facecolor('black')
  ax.set_ylabel('Angle [degrees]')  
  angle = TOFxEHDict['PA_Midpoint'][0,:]
  TOFxE_Epoch = TOFxEHDict['Epoch']
  x,y = np.meshgrid(TOFxE_Epoch, angle)
  im = ax.pcolormesh(x, y, angleBin.transpose(), cmap='jet', vmin=0, 
                     vmax=angleBin.max())
  span = np.ceil(int(TOFxE_Epoch[len(TOFxE_Epoch)-1].timestamp() - 
                                 TOFxE_Epoch[0].timestamp())/11)
  Hours = mdates.HourLocator()   
  Minutes = mdates.MinuteLocator()
  ax.xaxis.set_major_locator(Hours)
  ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
  ax.set_xticks(TOFxE_Epoch[::int(span/15)])
  fig.colorbar(im, ax=ax, fraction = .05, pad = .07, 
               format=ticker.FuncFormatter(fmt))
  ax.set_yscale('linear')
  ax.set_yticks(angle[::4])
  ax.set_yticklabels(angle[::4])
  
  axT.plot(kappaDict['Epoch'], kappaDict['Kappa'], color='oldlace')  
  axT.set_ylabel('Kappa', rotation = 270, labelpad = 11)
  axT.set_ylim(0,2)
    
def get_Mag_Pressure(magnitude):
  mu = (4 * np.pi) * 1e-7
  magVals = magnitude*1e-9
  magPress = (magVals**2)/(2*mu)
  return magPress

def get_MagPress_BelowThreshold(cutoff, magnitudes):
    mag_MagPress = []
    for entry in magnitudes:      
        if entry <= cutoff:
            mag_MagPress.append(get_Mag_Pressure(entry))
    mag_MagPress = np.array(mag_MagPress)
    return mag_MagPress

#==============================================================================
def get_Totaled_Entry(entry):
    newEnt = convert_PerChannel_From_Bytes(entry)
    paraTot = 0
    perpTot = 0
    totalTot = 0
    for i in range(len(newEnt[2])):
        paraTot = paraTot + newEnt[2][i]
        perpTot = perpTot + newEnt[3][i]
        totalTot = totalTot + newEnt[4][i]
    totEnt = [newEnt[0], newEnt[1], float(paraTot), float(perpTot), 
              float(totalTot), newEnt[5], newEnt[6], newEnt[7]]
    return totEnt

def convert_PerChannel_From_Bytes(entry):
    paraArr = np.fromstring(entry[2])
    perpArr = np.fromstring(entry[3])
    totalArr = np.fromstring(entry[4])
    coordArr = np.fromstring(entry[7], dtype='float32')
    newEntry = [entry[0], entry[1], paraArr, perpArr, totalArr, 
                entry[5],entry[6], coordArr]
    return newEntry  
  
def get_Pressure_Total_From_PerChannel(TABLE_NAME, START_DATE, STOP_DATE):
  DB_NAME = 'TOFxE_Pressure_PerChannel'
  cnx = conn.connect(user='root', passwd ='root')
  curs = cnx.cursor() 
  curs.execute("USE `{}`".format(DB_NAME))
  query = ('SELECT * FROM `' + TABLE_NAME + '` WHERE Epoch BETWEEN %s AND %s' 
       + 'ORDER BY Epoch ASC')
  curs.execute(query, (START_DATE, STOP_DATE))
  entries = curs.fetchall()  
  newEntries = []
  for entry in entries:
    newEntries.append(get_Totaled_Entry(entry))
  cnx.close()
  return newEntries

def get_Beta_Total_From_PerChannel(TABLE_NAME, START_DATE, STOP_DATE):
  DB_NAME = 'TOFxE_Beta_PerChannel'
  cnx = conn.connect(user='root', passwd ='root')
  curs = cnx.cursor() 
  curs.execute("USE `{}`".format(DB_NAME))
  query = ('SELECT * FROM `' + TABLE_NAME + '` WHERE Epoch BETWEEN %s AND %s' 
       + 'ORDER BY Epoch ASC')
  curs.execute(query, (START_DATE, STOP_DATE))
  entries = curs.fetchall()  
  newEntries = []
  for entry in entries:
    newEntries.append(get_Totaled_Entry(entry))
  cnx.close()
  return newEntries
#==============================================================================
#==============================================================================
dates = [dt.datetime(2013,5,1,12,30,16), dt.datetime(2013,5,1,12,50,16),.5]
#2013 05/01, 1230:16, 16.5 minute duration
strtDate = dates[0] - dt.timedelta(minutes=180)
stpDate = dates[1] + dt.timedelta(minutes=180) 
#==============================================================================
magDict = rip.get_CDF_Dict('Mag_1Sec_A', strtDate, stpDate)
TOFxEHDict = rip.get_CDF_Dict('TOFxEH_A', strtDate, stpDate)
magPress = get_MagPress_BelowThreshold(300, magDict['Magnitude'])
betaDict = dbFun.get_Beta_Total_For_Span('TOFxEH_A', strtDate, stpDate)
pressDict = dbFun.get_Pressure_Total_For_Span('TOFxEH_A', strtDate, stpDate)
kappaDict = dbFun.get_Kappa_For_Span('TOFxEH_A', strtDate, stpDate)
#==============================================================================
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, sharey=False, 
     figsize=(17,12), dpi=166) 
ax1T = ax1.twinx()
ax2T = ax2.twinx()
ax3T = ax3.twinx()
fig.subplots_adjust(hspace=0.02)
ax3.set_xlabel('Time Stamp')
#------------------------------------------------------------------------------
#==============================================================================
holder = TOFxEHDict['FPDU']
holder[np.where(holder<0)] = 0
binnedByEnergy = np.sum(holder, 2)
binnedByAngle = np.sum(holder, 1)

get_Particle_Heatmap_SubPlot_EnergyBinned(fig, ax1, ax1T, binnedByEnergy, 
                                          TOFxEHDict)
get_Particle_Heatmap_SubPlot_AngleBinned(fig, ax2, ax2T, binnedByAngle, 
                                         'A')
get_Power_Spectra_Subplot(fig, ax3, ax3T, magDict, betaDict)

pos1 = list(ax1.get_position().bounds)
pos1T = list(ax1T.get_position().bounds)
pos2T = list(ax2T.get_position().bounds)
pos3T = list(ax3T.get_position().bounds)
pos1T[2] = pos1[2]
pos2T[2] = pos1[2]
pos3T[2] = pos1[2]
ax1T.set_position(pos1T)
ax2T.set_position(pos2T)
ax3T.set_position(pos3T)

ax1.axvline(x=dates[0], color='black')
ax1.axvline(x=dates[1], color='black')
ax2.axvline(x=dates[0], color='black')
ax2.axvline(x=dates[1], color='black')
ax3.axvline(x=dates[0], color='black')
ax3.axvline(x=dates[1], color='black')
plt.plot()

Dir = '/home/matthew/CSTR/PlotFolder/FixedPlots/'
fig.savefig(Dir + 'TEST.jpeg')

#==============================================================================
