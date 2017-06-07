import numpy as np
import scipy.linalg as la
import pykalman as pyk
import seismic as seism
import ckbs_l2
import test
import matplotlib.pyplot as plt


if __name__ == "__main__":
    t1 = seism.dt.datetime(2017, 05, 19, 12, 22)
    t2 = t1 - seism.dt.timedelta(seconds=1000)
    seis = seism.SeismicReader(t1, t2)
    channel = 0
    z, g, h, G, H, qinv, rinv, zAve = seis.gen_inputs(channel)
    ar, ma, k, ar0 = seis.ARMA[channel]

    diag, subDiag, rhs = ckbs_l2.l2_affine(z, g, h, G, H, qinv, rinv)

    # x2, f2, zAve2, info2 = test.smooth_seis(seis, channel, l1=False)

    x2 = ckbs_l2.tridiag_solve_b(diag, subDiag, rhs)
    z2 = np.array([np.dot(H[i], x2[i]) for i in xrange(len(H))])  # + zAve

    seis.zs[channel]

    f2 = test.back_solve(seis, channel, x2)

    ratios = np.mean(np.divide(seis.fs[channel], np.real(f2)))

    # for i in range(10):
    #     resid = ckbs_l2.check_optimality(diag, subDiag, rhs, x2)
    #     print(np.linalg.norm(resid))
    #     xnew = ckbs_l2.tridiag_solve_b(diag, subDiag, resid)
    #     x2 -= xnew


    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].set_title(seis.channels[channel])
    ax[0].plot(np.arange(len(f2)), seis.zs[channel], label='observed')
    ax[0].plot(np.arange(len(f2)), np.real(z2), label='Kalman-l2', alpha=.5)

    ax[1].plot(np.arange(len(f2)), seis.fs[channel], label='z-transf')
    ax[1].plot(np.arange(len(f2)), -5e4*np.real(f2), label='Kalman-l2', alpha=.5)


    test.summary_plot(seis, channel)
