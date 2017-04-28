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

client = fdsn.Client()
t1 = dt.datetime.now()
t2 = t1-dt.timedelta(seconds = 400)

#Used http://www.latlong.net/ to find lat, long of cenpa circle room

report = client.get_stations(t2,t1,network='UW',level='response',
                             latitude=47.6603,longitude = -122.3031,
                             maxradius = .15)
                             
def haversine_np(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)

    All args must be of equal length.    

    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2

    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km


# Attempt at getting all waveforms from all of the stations in some radius:

#channels = report.get_contents()['channels']
channels = [[u'UW', u'ALKI', u'', u'EHZ'],[u'UW', u'ALKI', u'', u'ENE'],
[u'UW', u'ALKI', u'', u'ENN'],[u'UW', u'ALKI', u'', u'ENZ'],[u'UW', u'BRKS', u'', u'ENE'],
[u'UW', u'BRKS', u'', u'ENN'],[u'UW', u'BRKS', u'', u'ENZ'],[u'UW', u'BSFP', u'', u'ENE'],
[u'UW', u'BSFP', u'', u'ENN'],[u'UW', u'BSFP', u'', u'ENZ'],[u'UW', u'FINN', u'', u'ENE'],
[u'UW', u'FINN', u'', u'ENN'],[u'UW', u'FINN', u'', u'ENZ'],[u'UW', u'GTWN', u'', u'ENE'],
[u'UW', u'GTWN', u'', u'ENN'],[u'UW', u'GTWN', u'', u'ENZ'],[u'UW', u'HOLY', u'', u'ENE'],
[u'UW', u'HOLY', u'', u'ENN'],[u'UW', u'HOLY', u'', u'ENZ'],[u'UW', u'KCAM', u'', u'ENE'],
[u'UW', u'KCAM', u'', u'ENN'],[u'UW', u'KCAM', u'', u'ENZ'],[u'UW', u'KDK', u'', u'ENE'],
[u'UW', u'KDK', u'', u'ENN'],[u'UW', u'KDK', u'', u'ENZ'],[u'UW', u'KIMB', u'', u'ENE'],
[u'UW', u'KIMB', u'', u'ENN'],[u'UW', u'KIMB', u'', u'ENZ'],[u'UW', u'LAWT', u'', u'ENE'],
[u'UW', u'LAWT', u'', u'ENN'],[u'UW', u'LAWT', u'', u'ENZ'],[u'UW', u'MARY', u'', u'ENE'],
[u'UW', u'MARY', u'', u'ENN'],[u'UW', u'MARY', u'', u'ENZ'],[u'UW', u'NIHS', u'', u'ENE'],
[u'UW', u'NIHS', u'', u'ENN'],[u'UW', u'NIHS', u'', u'ENZ'],[u'UW', u'NOWS', u'', u'ENE'],
[u'UW', u'NOWS', u'', u'ENN'],[u'UW', u'NOWS', u'', u'ENZ'],[u'UW', u'SCC', u'', u'ENE'],
[u'UW', u'SCC', u'', u'ENN'],[u'UW', u'SCC', u'', u'ENZ'],[u'UW', u'SEA', u'', u'ENE'],
[u'UW', u'SEA', u'', u'ENN'],[u'UW', u'SEA', u'', u'ENZ'],[u'UW', u'SLA', u'00', u'HN1'],
[u'UW', u'SLA', u'00', u'HN2'],[u'UW', u'SLA', u'00', u'HNZ'],[u'UW', u'SLA', u'01', u'HNE'],
[u'UW', u'SLA', u'01', u'HNN'],[u'UW', u'SLA', u'01', u'HNZ'],[u'UW', u'SLA', u'02', u'HNE'],
[u'UW', u'SLA', u'02', u'HNN'],[u'UW', u'SLA', u'02', u'HNZ'],[u'UW', u'SLA', u'03', u'HNE'],
[u'UW', u'SLA', u'03', u'HNN'],[u'UW', u'SLA', u'03', u'HNZ'],[u'UW', u'SLA', u'60', u'HDO'],
[u'UW', u'SLA', u'61', u'HDD'],[u'UW', u'SLA', u'62', u'HDD'],[u'UW', u'SLA', u'63', u'HDD'],
[u'UW', u'SLA', u'64', u'HDD'],[u'UW', u'SLA', u'65', u'HDD'],[u'UW', u'SLA', u'66', u'HDD'],
[u'UW', u'SP2', u'', u'BHE'],[u'UW', u'SP2', u'', u'BHN'],[u'UW', u'SP2', u'', u'BHZ'],
[u'UW', u'SP2', u'', u'ENE'],[u'UW', u'SP2', u'', u'ENN'],[u'UW', u'SP2', u'', u'ENZ'],
[u'UW', u'TKCO', u'', u'ENE'],[u'UW', u'TKCO', u'', u'ENN'],[u'UW', u'TKCO', u'', u'ENZ'],
[u'UW', u'WISC', u'', u'ENE'],[u'UW', u'WISC', u'', u'ENN'],[u'UW', u'WISC', u'', u'ENZ']]
#bulk = [channel.split('.') for channel in channels]
bulk = channels
for item in bulk:
    item.append(str(t1))
    item.append(str(t2))
    
# This didn't work
#waveForms = client.get_waveforms_bulk(bulk)

# Option 2:

waveForms = []
titles = []
for k,item in enumerate(bulk):
    try:
        if k < 10:
            waveForms.append(np.copy(client.get_waveforms(item[0],item[1],item[2],item[3],item[5],item[4])[0]))
            titles.append(''.join(item[:-2]))
    except:
        pass

for k,data in enumerate(waveForms):
    #wf = client.get_waveforms('UW','NOWS','','ENE',t2,t1)
    #data = np.copy(wf)[0]
    times = np.arange(len(data))*.01
    freq, psd =sig.welch(data, 100.,nperseg=2**11,noverlap=2**10, detrend='linear')
    fig,ax=plt.subplots(1,2)
    ax[0].plot(times,data-np.polyval(np.polyfit(times,data,1),times))
    ax[1].loglog(freq, psd)
    ax[0].set_title(titles[k])