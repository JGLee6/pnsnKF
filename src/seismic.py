import numpy as np
import obspy.core.utcdatetime as UTCDateTime
import obspy.clients.fdsn as fdsn  # seismic network
import datetime as dt
import matplotlib.pyplot as plt
import scipy.signal as sig
import pykalman as pyk
import re
import numpy.polynomial.polynomial as poly


# plt.interactive(False)

class SeismicReader(object):
    """
    Seismic Reader takes in time points and creates an object with seismic response data in the form

    ARMA coefficients phi_j, theta_k

    """
    def __init__(self, t1, t2):
        """
        :param t1: beginning timestamp for timeseries slice
        :param t2: end timestamp
        """
        self.numSeries = 3
        self.client = fdsn.Client()  # open a client to talk to database
        # This gives a list of seismometer stations around some latlong in Seattle
        # Used http://www.latlong.net/ to find lat, long of cenpa circle room
        self.inventory = self.client.get_stations(t2, t1, network='UW',
                                                  level='response',
                                                  latitude=47.695,
                                                  longitude=-127.955,
                                                  maxradius=2.3)

        # Attempt at getting all waveforms from all of the stations in some radius:
        channels = self.inventory.get_contents()['channels']

        bulk = [channel.split('.') for channel in channels]
        for item in bulk:
            item.append(t2)
            item.append(t1)

        self.waveForms = []
        self.guess = []
        self.ARMA = []

        for k, item in enumerate(bulk):
            if k > self.numSeries - 1:
                break
            self.waveForms.append(np.copy(self.client.get_waveforms(*item)[0]))

            self.guess.append(np.copy(self.client.get_waveforms(*item).remove_response(self.inventory)[0]))

            inv = self.client.get_stations(t2, t1, network=item[0],
                                           station=item[1], location=item[2],
                                           channel=item[3], level='response')

            self.ARMA.append(self.getPoleZeroGain(inv))

    def getPoleZeroGain(self, inventory):
        """
        Takes output of zero,pole query as string. Splits into appropriate 
        roots and then converts to coefficients of a polynomial.
        Returns
        -------
        AR : ndarray
            List of AR coefficients normalized so first AR coefficient is 1.
        MA : ndarray
            List of MA coefficients normalized so first AR coefficient is 1.
        """
        _, zeros, poles, K = re.split('ZEROS|POLES|CONSTANT', inventory[0][0][0].response.get_sacpz())

        zeros = zeros.split()

        # first value is number of zeros, get roots as complex floats
        nZ = int(zeros[0])
        zVals = np.zeros(nZ, dtype='complex')

        for k in range(nZ):
            zVals[k] = float(zeros[2 * k + 1]) + 1j * float(zeros[2 * k + 2])

        poles = poles.split()

        # first value is number of poles, get roots as complex floats
        nP = int(poles[0])
        pVals = np.zeros(nP, dtype='complex')
        for k in range(nP):
            pVals[k] = float(poles[2 * k + 1]) + 1j * float(poles[2 * k + 2])

        # only single value for gain coefficient
        K = float(K.split()[0])

        # Multiply out roots of polynomials to get coefficients of ARMA model
        ARp = poly.polyfromroots(pVals)[::-1]
        MAq = poly.polyfromroots(zVals)[::-1]
        p = len(ARp)
        q = len(MAq)

        # Scale so that coefficient on largest order pole coefficient is 1
        # and scale inputs coefficients (zeros) by gain (conversion factor)
        MAq *= K

        # Because the transition matrices and covariance matrices depend on 
        # the number of AR and MA coefficients, we'll extend the outputs as 
        # zeros beyond the last value
        if p > q:
            AR = ARp
            MA = np.zeros(p, dtype='complex')
            MA[:q] = MAq
        elif q > p:
            MA = MAq
            AR = np.zeros(q, dtype='complex')
            AR[:p] = ARp
        else:
            AR = ARp
            MA = MAq

        return AR, MA

    def matrices(self, indx):
        """
        Calls getPoleZeroGain to find ARMA coefficients and then defines the transition
        matrix of the hidden state, the selection matrix of the hidden state, 
        and the transition matrix of the observed state.
        """
        AR, MA = self.ARMA[indx]
        r = len(AR)
        H = np.zeros(r)
        H[0] = 1.0

        G = np.zeros([r, r], dtype='complex')
        G[:, 0] = AR
        np.fill_diagonal(G[:-1, 1:], 1)

        R = np.copy(MA)

        return H, G, R

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
    print(seis.waveForms)
    print(seis.guess)
    print(seis.ARMA)
