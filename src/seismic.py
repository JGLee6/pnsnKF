import numpy as np
import obspy.core.utcdatetime as UTCDateTime
import obspy.clients.fdsn as fdsn  # seismic network
import datetime as dt
import matplotlib.pyplot as plt
import scipy.signal as sig
import pykalman as pyk

# plt.interactive(False)

class SeismicReader(object):

    def __init__(self, t1, t2):
        """
        :param t1: beginning timestamp for timeseries slice
        :param t2: end timestamp
        """
        self.client = fdsn.Client()  # open a client to talk to database
        # This gives a list of seismometer stations around some latlong in Seattle
        # Used http://www.latlong.net/ to find lat, long of cenpa circle room
        # self.report = self.client.get_stations(t2, t1, network='UW', level='response', latitude=47.6603, longitude=-122.3031,
        #                                 maxradius=.05)

        self.inventory = self.client.get_stations(t2, t1, network='UW', level='response', latitude=47.695,
                                               longitude=-127.955, maxradius=2.3)

        print(self.inventory)

        # self.inventory.plot()

        # Attempt at getting all waveforms from all of the stations in some radius:
        channels = self.inventory.get_contents()['channels']
        print(channels)
        # bulk = [channel.split('.') for channel in channels]
        # for item in bulk:
        #     item.append(str(t2))
        #     item.append(str(t1))
        #
        # print(bulk)
        #
        # self.waveForms = []
        # for k, item in enumerate(bulk):
        #     if k > 2:
        #         break
        #     print(item)
        #     self.waveForms.append(np.copy(self.client.get_waveforms(*item)[0]))
        #     print(self.waveForms[-1])

        # for data in waveForms:
        #     #wf = client.get_waveforms('UW','NOWS','','ENE',t2,t1)
        #     #data = np.copy(wf)[0]
        #     times = np.arange(len(data))*.01
        #     freq, psd =sig.welch(data, 100.,nperseg=2**11,noverlap=2**10, detrend='linear')
        #     fig,ax=plt.subplots(1,2)
        #     ax[0].plot(times,data-np.polyval(np.polyfit(times,data,1),times))
        #     ax[1].loglog(freq, psd)
    
    # # Example
    # def example(self):
    #     kf = pyk.KalmanFilter(initial_state_mean=np.average(waveForms[0]), n_dim_obs=1)
    #     em = kf.em(waveForms[0])
    #     sm = em.smooth(waveForms[0])
    #     fig, ax = plt.subplots(1,1);ax.plot(np.arange(len(waveForms[0]))*.01,waveForms[0],label='seismometer');ax.plot(np.arange(len(waveForms[0]))*.01,sm[0],c='r',alpha=.5,label='smoothed')
    #     ax.set_title('.'.join(item[:4]))

if __name__ == "__main__":
    # t1 = dt.datetime.now()
    t1 = dt.datetime(2017, 05, 19, 12, 7)
    t2 = t1 - dt.timedelta(seconds=300)
    seis = SeismicReader(t1, t2)
