import numpy as np
import scipy.linalg as la
import pykalman as pyk
import seismic as seism
import ckbs_l2

if __name__ == "__main__":
    t1 = seism.dt.datetime(2017, 05, 19, 12, 22)
    t2 = t1 - seism.dt.timedelta(seconds=1000)
    seis = seism.SeismicReader(t1, t2)
    channel = 0
    z, g, h, G, H, qinv, rinv, zAve = seis.gen_inputs(channel)

    diag, subDiag, rhs = ckbs_l2.l2_affine(z, g, h, G, H, qinv, rinv)
    x = ckbs_l2.tridiag_solve_b(diag, subDiag, rhs)
    resid = ckbs_l2.check_optimality(diag, subDiag, rhs, x)

