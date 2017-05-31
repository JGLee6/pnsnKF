import numpy as np
import scipy.linalg as la
import pykalman as pyk
from src import seismic as seism

"""
This analytically solves the linear system corresponding to the least squares problem with normal-normal errors.
"""

t1 = seism.dt.datetime(2017, 05, 19, 12, 7)
t2 = t1 - seism.dt.timedelta(seconds=10)
seis = seism.SeismicReader(t1, t2)

Hk = seis.Hk[0]
Gk = seis.Gk[0]
qInvk = seis.qInvk[0]
rInvk = seis.rInvk[0]

N = seis.N[0]
r = seis.r[0]


def create_bigH():
    """
    dimensions are N by N * r
    :return:
    """
    H = np.zeros((N, N*r))
    for i in range(N):
        start = r * i
        stop = start + r
        H[i, start:stop][:, None] = Hk  # to broadcast the assignment properly.
    return H


def create_bigG():
    G = np.eye(r*N, dtype=np.complex_)
    for i in range(1, N):
        rowStart = i*r
        columnStart = (i-1)*r
        G[rowStart:(rowStart + r), columnStart:(columnStart + r)] = -Gk
    return G


def create_bigqInv():
    return la.block_diag(*[qInvk for _ in range(N)])


def create_bigrInv():
    return rInvk * np.eye(N)

H = create_bigH()
G = create_bigG()
Qinv = create_bigqInv()
Rinv = create_bigrInv()

w = np.zeros(r*N)
z = seis.zs[0]

LHS = np.dot(np.dot(H.T, Rinv), H) + np.dot(np.dot(G.T, Qinv), G)
RHS = np.dot(np.dot(H.T, Rinv), z)

x = np.linalg.solve(LHS, RHS)
