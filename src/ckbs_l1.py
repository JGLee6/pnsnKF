#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun May  7 16:22:55 2017

@author: john
"""
import numpy as np
import pykalman as pyk
import seismic as seism
import ckbs_l2 as ksl2
import matplotlib.pyplot as plt

"""
Conventions:
N : length of time series
n : range size. Expect r = max(p,q+1)

"""


def l1_affine(z, g, h, G, H, qinv, rinv, maxIter=10, epsilon=1e-2):
    """
    Inputs
    ------
    z : ndarray
        N x m array of observed state variable (time series). N observations of 
        m dimensional data.
    g : ndarray
        N x n array of affine transition map translation between hidden states.
        N instances of additive component to affine transition from n 
        dimensional hidden state data.
    h : ndarray
        N x m array of observation affine map translation from hidden state to 
        observed state. Dependence of observed state on hidden state.
    G : ndarray
        N x n x n array of transition matrices of hidden state. N instances of 
        transition matrix from n dimensional hidden state data.
    H : ndarray
        N x m x n observation affine map from hidden state to observed state. 
        N instances of transition matrix from n dimensional hidden state data.
    qinv : ndarray
        n x n Inverse covariance matrix of L2 observed state loss.
    rinv : ndarray
        m x m Inverse covariance matrix of L1 hidden state loss.
    maxIter : int, optional
        Maximum number of iterations in smoother (iterate on Gauss-Newton to 
        find direction of gradient and then perform line search)
    epsilon : float, optional
        Convergence parameter
    """
    # Hard-coded variables
    # initial duality gap
    mu = 100

    # fraction of true slope (aka c)
    gamma = .01

    # max num of backtracking tries
    maxKount = 18

    # control parameter for backtracking (aka gamma)
    beta = .5

    # corrector period, how many iterations before updating duality gap parameter
    cT = 3

    # things returned by the function
    info = []

    # Size of problem
    n = np.shape(g)[-1]  # should be r
    N, m = np.shape(z)  # num time steps, dimension of observed data, expect m = 1 for us

    # Create vector of zero state values
    xZero = np.zeros([N, n])

    # update D and A to not include measurements
    # compute tridiagonal elements of smoothing program
    diagHid, subDiagHid = ksl2.l2_tridiag_hidden(G, qinv)

    # Compute gradient of l2-loss wrt hidden state
    c = ksl2.l2_grad_hidden(xZero, g, G, qinv)

    # Runs block-tridiagonal solver to find smoothed value
    # Gets Gauss-Newton step in linearization of observed state variable
    y = ksl2.tridiag_solve_b(diagHid, subDiagHid, -c)

    # block-tridiagonal multiplication with D as diagonals, A as off-diags and
    # y as vector
    cRec = blktridiag_mul(diagHid, subDiagHid, y)

    # Double checks that y is the solution to block-tridiag matrix(D,A)y = c
    if np.linalg.norm(-c - cRec, ord=np.inf) > 1e-5:
        print 'tridiagonal solver failed'
        # return c, cRec

    # Initiallize constraint variables and linearization of h(y)~h(x) + H*(x-y)
    b = np.zeros([N, m], dtype=np.complex_)
    r = np.zeros([N, m], dtype=np.complex_)
    s = np.zeros([N, m], dtype=np.complex_)
    crinv = np.zeros([N, m, m], dtype=np.complex_)  # Cholesky decomposition of rinv
    B = np.zeros([N, m, n], dtype=np.complex_)
    pPos = np.zeros([N, m], dtype=np.complex_)  # max(0, b + By)
    pNeg = np.zeros([N, m], dtype=np.complex_)  # max(0,-b + By)

    for k in xrange(N):
        crinv[k] = np.linalg.cholesky(rinv[k])  # prev.: crinv[k] = np.sqrt(rinv[k])
        B[k] = np.dot(-crinv[k], H[k])
        b[k] = np.dot(crinv[k], (z[k] - h[k]))
        # Version 1
        temp = b[k] + np.dot(B[k], y[k])
        # No idea what this step is about (500???)
        pPos[k] = 500 + max(0, temp)  # initialization
        pNeg[k] = 500 + max(0, -temp)
        r[k] = np.sqrt(2) * np.ones([1, m])
        s[k] = np.sqrt(2) * np.ones([1, m])

    # These should have been constructed to be zero
    # if (np.min(s) < 0 or np.min(r) <= 0 or np.min(pPos) <= 0 or np.min(pNeg) <= 0):
    #    print 'L1_affine: initial s, r, pPlus, pMinus not all positive'
    #    return
    # determine the value of y that solves the problem
    converge = False
    itr = 0
    while (not converge) and (itr < maxIter):
        itr += 1
        # Computes updates for KKT conditions
        F = kuhn_tucker_l1(mu, s, y, r, b, c, B, diagHid, subDiagHid, pPos, pNeg)
        # Computes newton descent step
        dpPos, dpNeg, dr, ds, dy = newton_step_l1(mu, s, y, r, b, c, B, 
                                                  diagHid, subDiagHid, pPos, pNeg)
        # Check that expressions involving updates of slack variables sum to 0
        check_newton_l1(pNeg, dpNeg, s, ds, mu, 1)
        check_newton_l1(pPos, dpPos, r, dr, mu, 2)
        # determine maximum allowable step factor lambda
        ratio = np.array([dr, ds, dpPos, dpNeg]) / np.array([r, s, pPos, pNeg])
        ratioMax = np.max(-ratio)
        # if ratioMax is not negative, pick from ratio not including ratioMax
        if ratioMax <= 0:
            lmbd = 1
        else:
            rNeg = -1. / ratio[ratio < 0]
            maxNeg = np.min(rNeg)
            lmbd = .99 * np.min([maxNeg, 1])

        # Line Search
        ok = False
        kount = 0
        lmbd = lmbd / beta
        while (not ok) and (kount < maxKount):
            kount += 1
            lmbd *= beta

            # update parameters with step size of lmbd
            sNew = s + lmbd * ds
            yNew = y + lmbd * dy
            rNew = r + lmbd * dr
            pPosNew = pPos + lmbd * dpPos
            pNegNew = pNeg + lmbd * dpNeg

            # Check for feasibility
            if (np.min(sNew) <= 0) or (np.min(rNew) <= 0) or (np.min(pPosNew) <= 0) or (np.min(pNegNew) <= 0):
                print 'L1_affine: program error, negative entries'
                return
            # Compute the KKT conditions at the new values of y,p+,p-,s,r
            FNew = kuhn_tucker_l1(mu, sNew, yNew, rNew, b, c, B, diagHid, 
                                  subDiagHid, pPosNew, pNegNew)
            FMax = np.max(np.abs(F), 0)
            FNewMax = np.max(np.abs(FNew), 0)
            # Check that all of the variables have decreased
            # XXX: I believe we want all, not any
            ok = all(FNewMax <= (1 - gamma * lmbd) * FMax)
        if not ok:
            df = np.max(F - FNew, 0)
            if all(df <= epsilon):
                print 'L1_affine: line search failed'
                # return

        s = sNew
        y = yNew
        r = rNew
        pPos = pPosNew
        pNeg = pNegNew
        # Compute the loss after the update and compare to inital state loss
        VP = l2l1_obj(y, z, g, h, G, H, qinv, rinv)
        Kxnu = l2l1_obj(xZero, z, g, h, G, H, qinv, rinv)
        G1 = np.sum(r * pPos) + np.sum(s * pNeg)
        converge = (G1 < np.min([Kxnu - VP, epsilon]))
        # Every third step is a corrector (That is update the mu parameter)
        if (itr % cT == 0):
            temp = np.sum(r * pPos) + np.sum(s * pNeg)
            compMuFrac = temp / (2. * m * N)
            mu = .1 * compMuFrac
        info.append([FMax, G1, Kxnu - VP, mu, kount])
    kktLast = kuhn_tucker_l1(0, s, y, r, b, c, B, diagHid, subDiagHid, pPos, pNeg)
    info.append(kktLast)

    return y, r, s, pPos, pNeg, info


def check_newton_l1(p, dp, s, ds, mu, num):
    """
    Checks update step from Newton step for constraint on expression involving 
    slack variables to sum to zero. The variable 'num' just allows one to track 
    which expression fails.
    """
    expr = np.linalg.norm(p * s + s * dp + p * ds - mu, ord=np.inf)
    if (np.abs(expr) > 1e-4):
        print 'L1_affine: Newton Solver not working, expr', num, ' not 0'
    return


def blktridiag_mul(A, B, v):
    r"""
    Multiplies block tridiagonal matrix with diagonal blocks given by A matrices,
    and off-diagonal blocks given by B and $B^{H}$  above and below respectively.
    
    .. math::
        C \cdot v = w
        
    where 
    
    .. math::
        C = \left(
        \begin{matrix} 
        A_{0} & B^{T}_{0} & ... & ... \\ 
        B_{0} & A_{1} & B^{T}_{1} & ...  \\
        ... & ... & ... & ... \\
        ... & ... & B_{N-2} & A_{N-1} \\ 
        \end{matrix}
        \right)
    
    .. math::
        v = \left(
        \begin{matrix}
        v_{0} \\
        v_{1} \\
        ... \\
        v_{N-1} \\
        \end{matrix}
        \right)
        
        
    Inputs
    ------
    A : ndarray
        N blocks of nxm matrices in one ndarray of shape Nxnxm
    B : ndarray
        N-1 blocks of nxm matrices in one
    """
    N, n, m = np.shape(A)
    w = np.zeros([N, m], dtype=np.complex_)
    w[0] = np.dot(A[0], v[0]) + np.dot(B[0].T, v[1])
    w[-1] = np.dot(A[-1], v[-1]) + np.dot(B[-1], v[-2])
    for k in xrange(1, N - 1):
        w[k] = np.dot(A[k], v[k]) + np.dot(B[k].conj().T, v[k + 1]) + np.dot(B[k - 1], v[k - 1])

    return w


def blkdiag_mul(A, v):
    """
    Block diagonal matrix multiplication with A_i as diagonal blocks.
    """
    N = np.shape(v)[0]
    w = np.array([np.dot(A[k], v[k]) for k in xrange(N)])
    return w


def blkdiag_mul_h(A, v):
    """
    Block diagonal matrix multiplication with A_i^H as diagonal blocks.
    """
    N = np.shape(v)[0]
    w = np.array([np.dot(A[k].conj().T, v[k]) for k in xrange(N)])
    return w


def kuhn_tucker_l1(mu, s, y, r, b, d, Bdia, Hdia, Hlow, pPos, pNeg):
    """
    Update step for KKT solver and l1 cost on observed variable
    """
    Hy = blktridiag_mul(Hdia, Hlow, y)
    Bt_SmR = blkdiag_mul_h(Bdia, s - r)
    By = blkdiag_mul(Bdia, y)

    
    F = np.hstack([pPos - pNeg - b - By,
                  pNeg * s - mu,
                  r + s - 2 * np.sqrt(2),
                  pPos * r - mu,
                  Hy + d + 0.5 * Bt_SmR])

    return F


def l2l1_obj(x, z, g, h, G, H, Qinv, Rinv):
    """
    Compute the loss objective
    
    .. math::
        f(x_{k}) = \frac{1}{2}\sum |x-g(x)|_{Q^{-1}}^2 + \sqrt(2)\sum|z-h(x)|_{R^{-1}}
    
    """
    N, n = np.shape(x)
    # chQ = np.linalg.cholesky(Qinv[0])
    chR = np.linalg.cholesky(Rinv[0])
    zRes = z[0] - h[0] - np.dot(H[0], x[0])
    xRes = x[0] - g[0]
    loss = np.sqrt(2) * np.linalg.norm(np.dot(chR, zRes), 1)
    loss += .5 * np.dot(xRes.T, np.dot(Qinv[0], xRes))  # np.linalg.norm(np.dot(chQ,xRes),2)**2.
    for k in xrange(1, N):
        # chQ = np.linalg.cholesky(Qinv[k])
        chR = np.linalg.cholesky(Rinv[k])
        zRes = z[k] - h[k] - np.dot(H[k], x[k])
        xRes = x[k] - g[k]
        xRes = x[k] - g[k] - np.dot(G[k], x[k - 1])
        loss += np.sqrt(2) * np.linalg.norm(np.dot(chR, zRes), 1)
        loss += .5 * np.dot(xRes.T, np.dot(Qinv[k], xRes))  # np.linalg.norm(np.dot(chQ,xRes),2)**2.

    return loss


def newton_step_l1(mu, s, y, r, b, d, BDia, HDia, HLow, pPos, pNeg):
    r"""
    Inputs
    ------
    mu : float
        Duality Gap variable
    s : float
        Non-negativity constraint on pNeg
    y : ndarray
        Newton method estimate of l2 step
    r : float
        Non-negativity constraint on pPos
    b : ndarray
        Additive component on linearization of L1 loss
    d : ndarray
        Gradient of loss wrt y
    BDia : ndarray
        Derivative of linearization of L1 loss
    HDia : ndarray
        Diagonal element of block-tridiagonal hessian matrix of loss wrt y
    HLow : ndarray
        Off-diagonal elements of block-tridiagonal hessian matrix of loss wrt y
    pPos : float
        First slack variable for L1 portion of objective
    pNeg : float
        Second slack variable for L1 portion of objective
    """
    N, n = np.shape(y)
    m = np.shape(b)[-1]
    if (np.min(s) < 0 or np.min(r) <= 0 or np.min(pPos) <= 0 or np.min(pNeg) <= 0):
        print 'L1_affine: initial s, r, pPlus, pMinus not all positive'
        return
    rs = r * s
    rpN = r * pNeg
    spP = s * pPos
    tinv = rs / (rpN + spP)
    tinvir = s / (rpN + spP)
    tinvis = r / (rpN + spP)

    # Create a new array for modified diagonal blocs
    modCDiag = np.array([HDia[k] + np.dot(BDia[k].conj().T, np.dot(np.diag(tinv[k]), BDia[k])) for k in xrange(N)])

    By = blkdiag_mul(BDia, y)
    Cy = blktridiag_mul(HDia, HLow, y)
    constraint = b + By - pPos
    eBar = blkdiag_mul_h(BDia, np.sqrt(2)-s) - Cy - d.conj()
    tInvFBar = mu*(tinvir-tinvis) - tinv*constraint - tinvir*pPos*(2*np.sqrt(2)-s)
    # Compute D(pPos) and D(pNeg)    

    BTtemp = blkdiag_mul_h(BDia, tInvFBar)

    R5 = eBar + BTtemp

    dy = ksl2.tridiag_solve_b(modCDiag, HLow, R5)

    #Bypdy = blkdiag_mul(BDia, y + dy)
    Bdy = blkdiag_mul(BDia, dy)

    #ds = tinv * (b + Bypdy - pPos) + tinvis * mu + tinvir * (2. * np.sqrt(2) * pPos - mu - pPos * s)
    ds = tinv*Bdy - tInvFBar
    dr = 2. * np.sqrt(2) - r - s - ds
    dpNeg = (mu - pNeg * ds) / s - pNeg
    #dpPos = dpNeg + Bypdy + b + pNeg - pPos
    dpPos = dpNeg + Bdy + constraint  + pNeg

    dpPos = np.reshape(dpPos, (N, m))
    dpNeg = np.reshape(dpNeg, (N, m))
    dy = np.reshape(dy, (N, n))
    dr = np.reshape(dr, (N, m))
    ds = np.reshape(ds, (N, m))

    return dpPos, dpNeg, dr, ds, dy
    
    
def back_solve(seismicReader, indx, x):
    """    
    """
    r = seismicReader.r[indx]
    q = seismicReader.q[indx]
    p = seismicReader.p[indx]
    phis, thetas, K, phi0 = seismicReader.ARMA[indx]
    N, n = np.shape(x)
    f = np.zeros([N,1], dtype=np.complex_)
    if (r == q+1) and (r!=p):
        for k in xrange(N):
            f[k] = x[k,-1]/thetas[-1]
    elif (r == q+1) and (r == p):
        f[0] = x[0,-1]/thetas[-1]
        for k in xrange(1,N):
            f[k] = (x[k,-1]-phis[-1]*x[k-1,0])/thetas[-1]
    else:
        d = p-q+1
        f[0] = x[0,-d]/thetas[-d]
        for k in xrange(1,d):
            f[k] = (x[k,-d] - np.dot(phis[-k:],x[:k,0]))/thetas[-d]
        for k in xrange(d,N):
            f[k] = (x[k,-d] - np.dot(phis[-d:],x[k-d:k,0]))/thetas[-d]
            
    return f


def smooth_seis(seis, channel):
    # Start defining matrices for the time series of size N
    z = np.reshape(seis.zs[channel]-np.average(seis.zs[channel]), (seis.N[channel], 1))

    G = np.array([np.real(seis.Gk[channel]) for k in xrange(seis.N[channel])])
    H = np.array([seis.Hk[channel] for k in xrange(seis.N[channel])])
    g = np.array([np.zeros(seis.r[channel]) for k in xrange(seis.N[channel])])
    h = np.array([[channel] for k in xrange(seis.N[channel])])
    qinv = np.array([seis.qInvk[channel] for k in xrange(seis.N[channel])])
    rinv = np.array([seis.rInvk[channel] for k in xrange(seis.N[channel])])

    x,r,s,pPos,pNeg,info = l1_affine(z, g, h, G, H, qinv, rinv)
    f = back_solve(seis, 0, x)
    f = np.reshape(f, (len(f)))
    
    return x, f
    
def summary_plot(seis, channel):
    x, f = smooth_seis(seis, channel)
    
    ar,ma,k,ar0 = seis.ARMA[channel]
    heur = [1000.,1.]
    
    fig,ax=plt.subplots(2,1,sharex=True)
    ax[0].set_title(seis.channels[channel])
    ax[0].plot(np.arange(len(f)),seis.zs[channel],label='observed')
    ax[0].plot(np.arange(len(f)),np.real(x[:,0]*k/ar0*heur[0]),label='Kalman')
    ax[1].plot(np.arange(len(f)),seis.fs[channel],label='z-transf')
    ax[1].plot(np.arange(len(f)),-np.real(f)*heur[1],label='Kalman',alpha=.5)
    #ax[1].plot(np.arange(len(f)),np.abs(f)*heur[1],label='abs: Kalman',alpha=.5)
    ax[1].set_ylim([-5e-5,5e-5])
    ax[0].set_ylabel('seismometer output [counts]')
    ax[1].set_xlabel('time [s]')
    ax[1].set_ylabel(r'accel. [$m/s^2$]')
    ax[0].legend()
    ax[1].legend()
    
    return x,f
            
    
if __name__== "__main__":
    t1 = seism.dt.datetime(2017, 05, 19, 12, 22)
    t2 = t1 - seism.dt.timedelta(seconds=1200)
    seis = seism.SeismicReader(t1, t2)
    
    # Start defining matrices for each time
    x,f = summary_plot(seis, 0)