import numpy as np
import scipy.linalg as la
import pykalman as pyk
import seismic as seism
import ckbs_l2
import test

if __name__ == "__main__":
    t1 = seism.dt.datetime(2017, 05, 19, 12, 22)
    t2 = t1 - seism.dt.timedelta(seconds=1000)
    seis = seism.SeismicReader(t1, t2)
    channel = 0
    x, f, zAve, info = test.smooth_seis(seis, channel, l1=False)

