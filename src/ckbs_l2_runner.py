import numpy as np
import scipy.linalg as la
import pykalman as pyk
import seismic as seism
import ckbs_l2

t1 = seism.dt.datetime(2017, 05, 19, 12, 22)
t2 = t1 - seism.dt.timedelta(seconds=1000)
seis = seism.SeismicReader(t1, t2)

channel = 0

z = np.reshape(seis.zs[channel], (seis.N[channel], 1))
G = np.array([seis.Gk[channel] for k in xrange(seis.N[channel])])
H = np.array([seis.Hk[channel] for k in xrange(seis.N[channel])])
g = np.array([np.zeros(seis.r[channel]) for k in xrange(seis.N[channel])])
h = np.array([[channel] for k in xrange(seis.N[channel])])
qinv = np.array([seis.qInvk[channel] for k in xrange(seis.N[channel])])
rinv = np.array([seis.rInvk[channel] for k in xrange(seis.N[channel])])

diag, subDiag, rhs = ckbs_l2.l2_affine(z, g, h, G, H, qinv, rinv)
x = ckbs_l2.tridiag_solve_b(diag, subDiag, rhs)

lhs = np.dot(diag[0, :, :], x[0, :]) + np.dot(subDiag[0, :, :].conj().T, x[1, :])
print(lhs - rhs[0, :])

lhs2 = np.dot(subDiag[-1, :, :], x[-2, :]) + np.dot(diag[-1, :, :], x[-1, :])
rhs[-1, :]

lhs2 - rhs[-1, :]

# tmp = la.block_diag(*[diag[i, :, :] for i in range(seis.N[channel])])
#
#
# r = seis.r[channel]
# # populate subdiagonal
# for i in range(1, seis.N[channel]):
#     row_start = i * r
#     col_start = (i - 1) * r
#     tmp[row_start: row_start + r, col_start: col_start + r] = subDiag[(i - 1), :, :] # subdiag
#     tmp[col_start: col_start + r, row_start: row_start + r] = subDiag[(i - 1), :, :].conj().T
#
# xx = x.reshape(40000*r)
# bb = rhs.reshape(40000*r)
#
# resid = np.dot(tmp, xx) - bb