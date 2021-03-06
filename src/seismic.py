import numpy as np
import obspy.core.utcdatetime as UTCDateTime
import obspy.clients.fdsn as fdsn  # seismic network
import datetime as dt
import matplotlib.pyplot as plt
import scipy.signal as sig
import scipy.linalg as la
import pykalman as pyk
import re
import numpy.polynomial.polynomial as poly


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
        self.channels = self.inventory.get_contents()['channels']

        bulk = [channel.split('.') for channel in self.channels]
        for item in bulk:
            item.append(t2)
            item.append(t1)

        self.zs = []  # array of z's in K* m/s^2 (counts)
        self.fs = []  # their estimate of outputs
        self.ARMA = []  # array of AR coeff, MA coeff, and gain K
        self.r = []
        self.p = []
        self.q = []
        self.N = []
        self.Sk = []
        self.Hk = []
        self.Gk = []
        self.sigW = []
        self.sigF = []
        self.sigR = []
        self.qInvk = []
        self.rInvk = []
        

        for k, item in enumerate(bulk):
            if k > self.numSeries - 1:
                break
            self.zs.append(np.copy(self.client.get_waveforms(*item)[0]))
            self.N.append(len(self.zs[k]))
            

            self.fs.append(np.copy(self.client.get_waveforms(*item).remove_response(self.inventory)[0]))

            inv = self.client.get_stations(t2, t1, network=item[0],
                                           station=item[1], location=item[2],
                                           channel=item[3], level='response')

            ar,ma,K,arp0,p,q,r = self.getPoleZeroGain(inv)
            self.ARMA.append([ar,ma,K,arp0])
            self.p.append(p)
            self.q.append(q)
            self.r.append(r)
            Hk,Gk,Sk = self.kARMA_matrices(k)
            self.Hk.append(Hk)
            self.Gk.append(Gk)
            self.Sk.append(Sk)
            self.sigW.append(1e8)  # 1e8??
            self.sigF.append(1e-3)
            self.sigR.append(1)
            qinv,rinv = self.covar_matrices(k,self.sigF[k],self.sigW[k],self.sigR[k])
            self.qInvk.append(qinv)
            self.rInvk.append(rinv)
            
            

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
        zVals = np.zeros(nZ, dtype=np.complex_)

        for k in range(nZ):
            zVals[k] = float(zeros[2 * k + 1]) + 1j * float(zeros[2 * k + 2])

        poles = poles.split()

        # first value is number of poles, get roots as complex floats
        nP = int(poles[0])
        pVals = np.zeros(nP, dtype=np.complex_)
        for k in range(nP):
            pVals[k] = float(poles[2 * k + 1]) + 1j * float(poles[2 * k + 2])

        # only single value for gain coefficient
        K = float(K.split()[0])

        # Multiply out roots of polynomials to get coefficients of ARMA model
        ARp = poly.polyfromroots(pVals)
        MAq = poly.polyfromroots(zVals)
        p = len(ARp)
        q = len(MAq)

        # Divide by leading coefficient of AR process
        # Scale input's coefficients (zeros) by gain (conversion factor)
        ARp0 = ARp[0]
        ARp /= ARp[0]  # why is this a negative, removed
        MAq = -MAq*K/ARp[0]
        # We also treat AR coefficients (aside from leading) to be the negative
        ARp[1:] *= -1
        
        r = np.max([p, q+1])

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

        return AR, MA, K, ARp0, p, q, r

    def kARMA_matrices(self, indx):
        """
        Calls getPoleZeroGain to find ARMA coefficients and then defines the transition
        matrix of the hidden state, the selection matrix of the hidden state, 
        and the transition matrix of the observed state.
        """
        AR, MA, K, AR0 = self.ARMA[indx]
    
        r = self.r[indx]    
        Hk = np.zeros(r)
        Hk[0] = 1.0
        Hk = np.reshape(Hk, (1,r))

        Gk = np.zeros([r, r], dtype='complex')
        Gk[:, 0] = AR
        np.fill_diagonal(Gk[:-1, 1:], 1)

        Sk = np.zeros(r, dtype=np.complex_)
        Sk[0] = 1.0
        Sk[1:] = MA[:r - 1]

        return Hk, Gk, Sk
    
    def covar_matrices(self, indx, sigF, sigW, sigR):
        """
        Creates covariance matrices for ARMA kalman filter model
        """
        Sk = self.Sk[indx]
        Qk = np.outer(Sk,Sk.conj())*sigF + sigW*np.eye(len(Sk))
        Qinv = np.linalg.inv(Qk)
        Rinv = np.reshape(1./sigR, (1,1))
        
        return Qinv, Rinv
        
        

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

    def gen_inputs(self, channel):
        # Start defining matrices for the time series of size N
        zAve = np.average(self.zs[channel])
        z = np.reshape(self.zs[channel] - zAve, (self.N[channel], 1))
        # z = np.reshape(self.zs[channel], (self.N[channel], 1))

        G = np.array([self.Gk[channel] for _ in xrange(self.N[channel])])
        H = np.array([self.Hk[channel] for _ in xrange(self.N[channel])])
        g = np.array([np.zeros(self.r[channel]) for _ in xrange(self.N[channel])])
        h = np.array([[0] for _ in xrange(self.N[channel])])
        qinv = np.array([self.qInvk[channel] for _ in xrange(self.N[channel])])
        rinv = np.array([self.rInvk[channel] for _ in xrange(self.N[channel])])

        return z, g, h, G, H, qinv, rinv, zAve


if __name__ == "__main__":
    # t1 = dt.datetime.now()
    t1 = dt.datetime(2017, 05, 19, 23, 47)
    t2 = t1 - dt.timedelta(seconds=1200)
    seis = SeismicReader(t1, t2)
    #for key, value in 
