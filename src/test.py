import numpy as np
import pykalman as pyk
import seismic as seism
import ckbs_l2 as ksl2
import ckbs_l1 as ksl1
import matplotlib.pyplot as plt


def back_solve(seismicReader, channel, x):
    """
    given seis, channel, and the solution x, reconstruct the ground motion f
    """
    r = seismicReader.r[channel]
    q = seismicReader.q[channel]
    p = seismicReader.p[channel]
    phis, thetas, K, phi0 = seismicReader.ARMA[channel]
    N, n = np.shape(x)
    f = np.zeros([N, 1], dtype=np.complex_)
    if (r == q + 1) and (r != p):
        for k in xrange(N):
            f[k] = x[k, -1] / thetas[-1]
    elif (r == q + 1) and (r == p):
        f[0] = x[0, -1] / thetas[-1]
        for k in xrange(1, N):
            f[k] = (x[k, -1] - phis[-1] * x[k - 1, 0]) / thetas[-1]
    else:
        d = p - q + 1
        f[0] = x[0, -d] / thetas[-d]
        for k in xrange(1, d):
            f[k] = (x[k, -d] - np.dot(phis[-k:], x[:k, 0])) / thetas[-d]
        for k in xrange(d, N):
            f[k] = (x[k, -d] - np.dot(phis[-d:], x[k - d:k, 0])) / thetas[-d]

    return f


def smooth_seis(seis, channel, l1=True):
    # Start defining matrices for the time series of size N
    z, g, h, G, H, qinv, rinv, zAve = seis.gen_inputs(channel)

    if l1:
        x, r, s, pPos, pNeg, info = ksl1.l1_affine(z, g, h, G, H, qinv, rinv)
    else:
        diag, subDiag, rhs = ksl2.l2_affine(z, g, h, G, H, qinv, rinv)
        x = ksl2.tridiag_solve_b(diag, subDiag, rhs)
        print("Norm of residual is {}".format(np.linalg.norm(ksl2.check_optimality(diag, subDiag, rhs, x))))
        info = None
    f = back_solve(seis, channel, x)
    f = np.reshape(f, (len(f)))

    return x, f, zAve, info


def EM_l2(seis, channel, maxIter=5):
    for k in xrange(maxIter):
        x, f, zAve, info = smooth_seis(seis, channel, False)
        seis.sigF[channel] = np.var(f)
        seis.sigR[channel] = np.var(x[:, 0])
        seis.rInvk[channel] = np.reshape(1 / seis.sigR[channel], (1, 1))
        seis.qInvk[channel] = np.linalg.inv(np.cov(x.T))

    return x, f


def EM_l1(seis, channel, maxIter=5):
    for k in xrange(maxIter):
        x, f, zAve, info = smooth_seis(seis, channel)
        seis.sigF[channel] = np.var(f)
        seis.sigR[channel] = np.average(np.abs(x[:, 0] - np.average(x[:, 0])))
        seis.rInvk[channel] = np.reshape(1 / seis.sigR[channel], (1, 1))
        seis.qInvk[channel] = np.linalg.inv(np.cov(x.T))

    return x, f


def summary_plot(seis, channel):
    z, g, h, G, H, qinv, rinv, zAve = seis.gen_inputs(0)
    # x1, f1, zAve1, info1 = smooth_seis(seis, channel)
    x2, f2, zAve2, info2 = smooth_seis(seis, channel, l1=False)

    ar, ma, k, ar0 = seis.ARMA[channel]
    # z1 = np.array([np.dot(H[i], y1[i]) for i in xrange(len(H))])
    z2 = np.array([np.dot(H[i], x2[i]) for i in xrange(len(H))])

    # heur = [17.,1]

    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].set_title(seis.channels[channel])
    ax[0].plot(np.arange(len(f2)), seis.zs[channel], label='observed')
    # ax[0].plot(np.arange(len(f1)), np.real((z1 * k / ar0) + zAve), label='Kalman-l1', alpha=.5)
    # ax[0].plot(np.arange(len(f2)), np.real(z2), label='Kalman-l2', alpha=.5)
    ax[0].plot(np.arange(len(f2)), np.real((z2) + zAve), label='Kalman-l2', alpha=.5)
    ax[1].plot(np.arange(len(f2)), seis.fs[channel], label='z-transf')
    # ax[1].plot(np.arange(len(f1)), -np.real(f1 * ar0), label='Kalman-l1', alpha=.5)
    ax[1].plot(np.arange(len(f2)), np.real(f2), label='Kalman-l2', alpha=.5)
    ax[1].set_ylim([-5e-5, 5e-5])
    ax[0].set_ylabel('seismometer output [counts]')
    ax[1].set_xlabel('time [s]')
    ax[1].set_ylabel(r'accel. [$m/s^2$]')
    ax[0].legend()
    ax[1].legend()

    fig.savefig("asdf.png")

    # return y1, y2, f1, f2, info1, info2
    return x2, f2, info2

if __name__ == "__main__":
    t1 = seism.dt.datetime(2017, 05, 19, 12, 22)
    t2 = t1 - seism.dt.timedelta(seconds=1200)
    seis = seism.SeismicReader(t1, t2)

    # Start defining matrices for each time
    # x1, x2, f1, f2, info1, info2 = summary_plot(seis, 0)
    x2, f2, info2 = summary_plot(seis, 0)