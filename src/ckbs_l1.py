#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun May  7 16:22:55 2017

@author: john
"""
import numpy as np
import scipy.linalg as la
import pykalman as pyk
import seismic as seism

"""
Conventions:
N : length of time series
n : range size. Expect r = max(p,q+1)

"""


def ckbs_l1_affine(z, g, h, G, H, qinv, rinv, maxIter=10, epsilon=1e-2):
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
    D, A = l2_tridiag_hidden(G, qinv)

    # Compute gradient of l2-loss wrt hidden state
    c = l2_grad_hidden(xZero, g, G, qinv)

    # Runs block-tridiagonal solver to find smoothed value
    # Gets Gauss-Newton step in linearization of observed state variable
    y = tridiag_solve_b(D, A, -c)

    # block-tridiagonal multiplication with D as diagonals, A as off-diags and
    # y as vector
    cRec = blktridiag_mul(D, A, y)

    # Double checks that y is the solution to block-tridiag matrix(D,A)y = c
    if np.linalg.norm(-c - cRec, ord=np.inf) > 1e-5:
        print 'tridiagonal solver failed'
        # return c, cRec

    # Initiallize constraint variables and linearization of h(y)~h(x) + H*(x-y)
    b = np.zeros([N, m])
    r = np.zeros([N, m])
    s = np.zeros([N, m])
    crinv = np.zeros([N, m, m])  # Cholesky decomposition of rinv
    B = np.zeros([N, m, n])
    pPos = np.zeros([N, m])  # max(0, b + By)
    pNeg = np.zeros([N, m])  # max(0,-b + By)

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
        F = kuhn_tucker_l1(mu, s, y, r, b, c, B, D, A, pPos, pNeg)
        # Computes newton descent step
        dpPos, dpNeg, dr, ds, dy = newton_step_l1(mu, s, y, r, b, c, B, D, A, pPos, pNeg)
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
            FNew = kuhn_tucker_l1(mu, sNew, yNew, rNew, b, c, B, D, A, pPosNew, pNegNew)
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
    and off-diagonal blocks given by B and $B^{T}$  above and below respectively.
    
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
    w = np.zeros([n * N, m])
    w[0] = np.dot(A[0], v[0]) + np.dot(B[0].T, v[1])
    w[-1] = np.dot(A[-1], v[-1]) + np.dot(B[-1], v[-2])
    for k in xrange(1, N - 1):
        w[k] = np.dot(A[k], v[k]) + np.dot(B[k].T, v[k + 1]) + np.dot(B[k - 1], v[k - 1])

    return w


def blkdiag_mul(A, v):
    """
    Block diagonal matrix multiplication with A_i as diagonal blocks.
    """
    m = np.shape(v)[-1]
    w = np.array([np.dot(A[k], v[k]) for k in xrange(m)])
    return w


def blkdiag_mul_t(A, v):
    """
    Block diagonal matrix multiplication with A_i^T as diagonal blocks.
    """
    m = np.shape(v)[-1]
    w = np.array([np.dot(A[k].T, v[k]) for k in xrange(m)])
    return w


def l2_tridiag_observed(H, Rinv):
    """
    Creates contribution to block-tridiagonal 'Hessian' matrix for Kalman 
    Smoother when observed loss is l2. Really only contributes a diagonal 
    block.
    
    Inputs
    ------
    H : ndarray
        N x m x n observation affine map from hidden state to observed state. 
        N instances of transition matrix from n dimensional hidden state data.
    rinv : ndarray
        m x m Inverse covariance matrix of L1 hidden state loss.
    """
    N, n, m = np.shape(H)
    # diagonal block matrices
    D = np.zeros([N, n, n])
    for k in xrange(N):
        D[k] = np.dot(H[k], np.dot(Rinv[k], H[k]).T)

    return D


def l2_tridiag_hidden(G, Qinv):
    """
    Creates block-tridiagonal 'Hessian' matrix for Kalman Smoother.
    """
    N, n = np.shape(G)[:-1]

    # off-diagonal block matrices (below diagonal)
    subD = np.zeros([N - 1, n, n])

    # diagonal block matrices
    D = np.zeros([N, n, n])
    D[-1] = Qinv[-1]
    for k in xrange(N - 1):
        D[k] = Qinv[k - 1] + np.dot(G[k].T, np.dot(Qinv[k], G[k]))

    subD = np.array([-np.dot(Qinv[k], G[k]) for k in xrange(N - 1)])

    return D, subD


def l2_grad_observed(z, h, H, Rinv):
    r"""
    Computes gradient of loss function (just hidden variable term)
    
    .. math::
        f(x_k) = \frac{1}{2}\sum [z_k-H_k z_{k-1}-h_k]^T R_k^{-1}[z_k-H_k z_{k-1}-h_k]
        
    .. math::
        \frac{\partial f}{\partial x_j} =  
             -H_k^TR_k^{-1}[z_k - H_k x_k-h_k]
    """
    N, n, m = np.shape(H)
    grad = np.zeros([N, n])
    for k in xrange(N):
        grad[k] = np.dot(H[k], np.dot(Rinv[k],z[k]-h[k]))

    return grad


def l2_grad_hidden(x, g, G, Qinv):
    r"""
    Computes gradient of loss function (just hidden variable term)
    
    .. math::
        f(x_k) = \frac{1}{2}\sum [x_k-G_k x_{k-1}-g_k]^TQ_k^{-1}[x_k-G_k x_{k-1}-g_k]
        
    .. math::
        \frac{\partial f}{\partial x_k} = Q_k^{-1}[x_k-G_k x_{k-1}-g_k] 
             -G_{k+1}^TQ_k^{-1}[x_{k+1}-G_{k+1}x_k-g_{k+1}]
    """
    N, n = np.shape(G)[:-1]
    grad = np.zeros([N, n])
    x01 = x[1] - g[1] - np.dot(G[1], x[0])
    xm1 = x[-1] - g[-1] - np.dot(G[-1], x[-2])
    grad[0] = -np.dot(G[1].T, np.dot(Qinv[1], x01))
    grad[-1] = np.dot(Qinv[-1], xm1)
    for k in xrange(1, N - 1):
        xk1 = x[k + 1] - g[k + 1] - np.dot(G[k + 1], x[k])
        xk = x[k] - g[k] - np.dot(G[k], x[k - 1])
        grad[k] = np.dot(Qinv[k], xk) - np.dot(G[k + 1].T, np.dot(Qinv[k + 1], xk1))

    return grad


def kuhn_tucker_l1(mu, s, y, r, b, d, Bdia, Hdia, Hlow, pPos, pNeg):
    """
    Update step for KKT solver and l1 cost on observed variable
    """
    Hy = blktridiag_mul(Hdia, Hlow, y)
    Bt_SmR = blkdiag_mul_t(Bdia, s - r)
    By = blkdiag_mul(Bdia, y)

    F = np.array([pPos - pNeg - b - By,
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
    modCDiag = np.array([HDia[k] + np.dot(BDia[k].T, np.dot(np.diag(tinv[k]), BDia[k])) for k in xrange(N)])

    # Compute D(pPos) and D(pNeg)


    Cy = blktridiag_mul(HDia, HLow, y)
    By = blkdiag_mul(BDia, y)

    temp = s - np.sqrt(2) + tinv * (b + By + pPos) + mu * tinvis + tinvir * (2. * np.sqrt(2) * pPos - mu - pPos * s)
    BTtemp = blkdiag_mul_t(BDia, temp)

    R5 = Cy + d + BTtemp

    dy = -tridiag_solve_b(modCDiag, HLow, R5)

    Bypdy = blkdiag_mul(BDia, y + dy)

    ds = tinv * (b + Bypdy - pPos) + tinvis * mu + tinvir * (2. * np.sqrt(2) * pPos - mu - pPos * s)
    dr = 2. * np.sqrt(2) - r - s - ds
    dpNeg = (mu - pNeg * ds) / s - pNeg
    dpPos = dpNeg + Bypdy + b + pNeg - pPos

    dpPos = np.reshape(dpPos, (N, m))
    dpNeg = np.reshape(dpNeg, (N, m))
    dy = np.reshape(dy, (N, n))
    dr = np.reshape(dr, (N, m))
    ds = np.reshape(ds, (N, m))

    return dpPos, dpNeg, dr, ds, dy


def tridiag_solve_b(C, A, r):
    """
    Solves block-tridiagonal system of equations, where C are the diagonal 
    matrix elements, A and A^{T} are the above and below diagonal elements 
    respectively, and r is the vector of observations.
    
    References
    ----------
    Algorithm 2.1 from https://arxiv.org/pdf/1303.1993.pdf
    """
    N, n = np.shape(C)[:-1]
    D = np.zeros([N, n, n])
    s = np.zeros([N, n])
    D[0] = C[0]
    s[0] = r[0]
    for k in xrange(1, N):
        D[k] = C[k] - np.dot(A[k - 1].T, cho_solver(D[k - 1], A[k - 1]))
        s[k] = r[k] - np.dot(A[k - 1], cho_solver(D[k - 1], s[k - 1]))

    e = np.zeros([N, n])
    e[-1] = cho_solver(D[-1], s[-1])
    for k in xrange(N - 1, 0, -1):
        e[k] = cho_solver(D[k], (s[k] - np.dot(A[k - 1], e[k])))

    return e


def cho_solver(A, B):
    """
    Solve the linear equations A x = b, given the Cholesky factorization of A.
    """
    return la.cho_solve((np.linalg.cholesky(A), True), B)



def test_l2l2():
    """
    Tests l2l2 affine solver with a random walk by comparing to pyKalman 
    smoother.
    
    """
    # Script to test with random walk as x, 2 x identity as observation dependence
    N = 10000  # number of time steps
    x = np.cumsum(np.random.randn(N))  # create hidden state
    x = np.reshape(x, [N,1])  # reshape so matrix like
    G = np.array([[[0]] for k in xrange(N)])  # Create transition matrices
    H = np.array([[[2]] for k in xrange(N)])  # Create observation matrices
    # Create observations
    z = np.array([np.dot(H[k],x[k])+np.random.randn(1) for k in xrange(N)])
    # Don't want translation for affine transformation, so just set to zero
    g = np.array([[0] for k in xrange(N)])
    h = np.copy(g)
    # Create covariance matrices for hidden and observed processes
    qinv = np.array([[[1]] for k in xrange(N)])
    rinv = np.copy(qinv)
    
    #y,r,s,pPos,pNeg,info = ckbs_l1_affine(z,g,h,G,H,qinv,rinv,maxIter=18,epsilon=1e-5)
    # Run tri-diagonal smoother (This is almost a factor 2 faster!)
    y = l2_affine(z, g, h, G, H, qinv, rinv)
    print 'Smoothing with tri-diagonal solver'
    
    # Compare to pykalman on test case
    kf = pyk.KalmanFilter(transition_matrices=G[0],observation_matrices=H[0],
                          transition_covariance=qinv[0],
                          observation_covariance=rinv[0])
    y2 = kf.smooth(z)
    print 'Smoothing with pyKalman solver'
    
    # It seems to not nail the first and last points to the same precision...
    print 'Comparing results...'
    assert all (np.abs(y-y2[0])[1:-1] < 1e-5)
    
    
def back_solve(seismicReader, indx, x):
    """    
    """
    r = seismicReader.r[indx]
    q = seismicReader.q[indx]
    p = seismicReader.p[indx]
    phis, thetas, K = seismicReader.ARMA[indx]
    N, n = np.shape(x)
    f = np.zeros([N,1])
    if (r == q+1) and (r!=p):
        for k in xrange(N):
            f[k] = x[k,-1]/thetas[-1]
    elif (r == q+1) and (r == p):
        f[0] = x[0,-1]/thetas[-1]
        for k in xrange(1,N):
            f[k] = (x[k,-1]-phis[-1]*x[k-1,0])/thetas[-1]
    else:
        d = p-q
        # For each time step, must solve lower triangular linear system of eq
        for k in xrange(d,N):
            mat = np.tri(d,d,0,dtype='complex')
            for l in xrange(d):
                mat = np.fill_diagonal(mat[l:,:-l],phis[r-l])
            f[k] = cho_solver(mat, x[k,-d:])/thetas[-1]
            
    return f
            
    
if __name__== "__main__":
    t1 = seism.dt.datetime(2017, 05, 19, 12, 7)
    t2 = t1 - seism.dt.timedelta(seconds=300)
    seis = seism.SeismicReader(t1, t2)
    
    # Start defining matrices for each time
    z = np.reshape(seis.zs[0], (seis.N[0], 1))
    G = np.array([np.real(seis.Gk[0]) for k in xrange(seis.N[0])])
    H = np.array([seis.Hk[0] for k in xrange(seis.N[0])])
    g = np.array([np.zeros(seis.r[0]) for k in xrange(seis.N[0])])
    h = np.array([[0] for k in xrange(seis.N[0])])
    qinv = np.array([np.real(seis.qInvk[0]) for k in xrange(seis.N[0])])  # why np.real
    rinv = np.array([np.real(seis.rInvk[0]) for k in xrange(seis.N[0])])  # why np.real?
    
    y = ckbs_l1_affine(z, g, h, G, H, qinv, rinv)
