# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 14:54:51 2016

@author: cenpa
"""

import numpy as np
#from obspy.core import UTCDateTime
import obspy.clients.fdsn as fdsn
import datetime as dt
import matplotlib.pyplot as plt
import scipy.signal as sig
import pykalman as pyk

client = fdsn.Client()
t1 = dt.datetime.now()
t2 = t1-dt.timedelta(seconds = 400)

#Used http://www.latlong.net/ to find lat, long of cenpa circle room

report = client.get_stations(t2,t1,network='UW',level='response',
                             latitude=47.6603,longitude = -122.3031,
                             maxradius = .05)


# Attempt at getting all waveforms from all of the stations in some radius:

channels = report.get_contents()['channels']
bulk = [channel.split('.') for channel in channels]
for item in bulk:
    item.append(str(t1))
    item.append(str(t2))
    
# This didn't work
#waveForms = client.get_waveforms_bulk(bulk)

# Option 2:

waveForms = []
for k,item in enumerate(bulk):
    if k < 3:
        waveForms.append(np.copy(client.get_waveforms(item[0],item[1],item[2],item[3],item[5],item[4])[0]))


for data in waveForms:
    #wf = client.get_waveforms('UW','NOWS','','ENE',t2,t1)
    #data = np.copy(wf)[0]
    times = np.arange(len(data))*.01
    freq, psd =sig.welch(data, 100.,nperseg=2**11,noverlap=2**10, detrend='linear')
    fig,ax=plt.subplots(1,2)
    ax[0].plot(times,data-np.polyval(np.polyfit(times,data,1),times))
    ax[1].loglog(freq, psd)
    
# Example
kf = pyk.KalmanFilter(initial_state_mean=np.average(waveForms[0]),n_dim_obs=1)
em = kf.em(waveForms[0])
sm = em.smooth(waveForms[0])
fig,ax=plt.subplots(1,1);ax.plot(np.arange(len(waveForms[0]))*.01,waveForms[0],label='seismometer');ax.plot(np.arange(len(waveForms[0]))*.01,sm[0],c='r',alpha=.5,label='smoothed')
ax.set_title('.'.join(item[:4]))