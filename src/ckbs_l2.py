import numpy as np
import scipy.linalg as la
import pykalman as pyk
#from src import seismic as seism
import seismic as seism
import matplotlib.pyplot as plt

def l2_affine(z, g, h, G, H, qinv, rinv):
    """
    Uses tri_diagonal solver to solve l2-l2 system of Kalman Smoothing.

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
        m x m Inverse covariance matrix of L2 hidden state loss.
    """
    # Size of problem
    n = np.shape(g)[-1]  # should be r
    N, m = np.shape(z)  # length of time series, dimension of observed data, expect m = 1 for us

    # Create vector of zero state values
    xZero = np.zeros([N, n])

    # compute tridiagonal elements of smoothing program
    # Hidden state contribution (G^T Q^{-1} G)
    diagHid, subDiagHid = l2_tridiag_hidden(G, qinv)

    # Observed state contribution (H^T R^{-1} H)
    diagObs = l2_tridiag_observed(H, rinv)

    # Combine into single set of block tridiagonal matrix
    diag = diagHid + diagObs
    subDiag = subDiagHid

    # Compute gradient of loss wrt hidden state
    gHid = l2_grad_hidden(xZero, g, G, qinv)
    gObs = l2_grad_observed(z, h, H, rinv)
    g = gHid + gObs

    return tridiag_solve_b(diag, subDiag, g)


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
    N, m, n= np.shape(H)
    # diagonal block matrices
    D = np.zeros([N, n, n])
    for k in xrange(N):
        D[k] = np.dot(H[k].conj().T, np.dot(Rinv[k], H[k]))

    return D


def l2_tridiag_hidden(G, Qinv):
    """
    Creates block-tridiagonal 'Hessian' matrix for Kalman Smoother.
    """
    N, n = np.shape(G)[:-1]

    # off-diagonal block matrices (below diagonal)
    subD = np.zeros([N - 1, n, n])

    # diagonal block matrices
    D = np.zeros([N, n, n], dtype=np.complex_)
    # This commented portion is maybe physically motivated for a process
    # that continues with the same covariance and transition matrices, but is 
    # not the optimal solution for the problem
    D[-1] = Qinv[-1] # + np.dot(G[-1].T,np.dot(Qinv[-1],G[-1]))
    for k in xrange(N - 1):
        D[k] = Qinv[k - 1] + np.dot(G[k].conj().T, np.dot(Qinv[k], G[k]))

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
    N, m, n = np.shape(H)
    grad = np.zeros([N, n])
    for k in xrange(N):
        grad[k] = np.dot(H[k].conj().T, np.dot(Rinv[k], z[k] - h[k]))

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
    grad = np.zeros([N, n], dtype=np.complex_)
    x01 = x[1] - g[1] - np.dot(G[1], x[0])
    xm1 = x[-1] - g[-1] - np.dot(G[-1], x[-2])
    grad[0] = -np.dot(G[1].conj().T, np.dot(Qinv[1], x01))
    grad[-1] = np.dot(Qinv[-1], xm1)
    for k in xrange(1, N - 1):
        xk1 = x[k + 1] - g[k + 1] - np.dot(G[k + 1], x[k])
        xk = x[k] - g[k] - np.dot(G[k], x[k - 1])
        grad[k] = np.dot(Qinv[k], xk) - np.dot(G[k + 1].conj().T, np.dot(Qinv[k + 1], xk1))

    return grad


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
    D = np.zeros([N, n, n], dtype=np.complex_)
    s = np.zeros([N, n], dtype=np.complex_)
    D[0] = C[0]
    s[0] = r[0]
    for k in xrange(1, N):
        D[k] = C[k] - np.dot(A[k - 1].T, cho_solver(D[k - 1], A[k - 1]))
        s[k] = r[k] - np.dot(A[k - 1], cho_solver(D[k - 1], s[k - 1]))

    e = np.zeros([N, n], dtype=np.complex_)
    e[-1] = la.solve(D[-1], s[-1])
    for k in xrange(N - 1, 0, -1):
        e[k] = la.solve(D[k], (s[k] - np.dot(A[k - 1], e[k])))

    return e


def cho_solver(A, B):
    """
    Solve the linear equations A x = b, given the Cholesky factorization of A.
    """
    return la.cho_solve((np.linalg.cholesky(A), True), B)


def back_solve(seismicReader, indx, x):
    """    
    """
    r = seismicReader.r[indx]
    q = seismicReader.q[indx]
    p = seismicReader.p[indx]
    phis, thetas, K = seismicReader.ARMA[indx]
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


def test_l2l2():
    """
    Tests l2l2 affine solver with a random walk by comparing to pyKalman
    smoother.

    """
    # Script to test with random walk as x, 2 x identity as observation dependence
    N = 10000  # number of time steps
    x = np.cumsum(np.random.randn(N))  # create hidden state
    x = np.reshape(x, [N, 1])  # reshape so matrix like
    G = np.array([[[0]] for k in xrange(N)])  # Create transition matrices
    H = np.array([[[2]] for k in xrange(N)])  # Create observation matrices
    # Create observations
    z = np.array([np.dot(H[k].T, x[k]) + np.random.randn(1) for k in xrange(N)])
    # Don't want translation for affine transformation, so just set to zero
    g = np.array([[0] for k in xrange(N)])
    h = np.copy(g)
    # Create covariance matrices for hidden and observed processes
    qinv = np.array([[[1]] for k in xrange(N)])
    rinv = np.copy(qinv)

    # y,r,s,pPos,pNeg,info = ckbs_l1_affine(z,g,h,G,H,qinv,rinv,maxIter=18,epsilon=1e-5)
    # Run tri-diagonal smoother (This is almost a factor 2 faster!)
    y = l2_affine(z, g, h, G, H, qinv, rinv)
    print 'Smoothing with tri-diagonal solver'

    # Compare to pykalman on test case
    kf = pyk.KalmanFilter(transition_matrices=G[0], observation_matrices=H[0],
                          transition_covariance=qinv[0],
                          observation_covariance=rinv[0])
    y2 = kf.smooth(z)
    print 'Smoothing with pyKalman solver'

    # It seems to not nail the first and last points to the same precision...
    print 'Comparing results...'
    assert all(np.abs(y - y2[0])[1:] < 1e-5)
    
    
def summary_plot(seis, channel):
    # Start defining matrices for the time series of size N
    z = np.reshape(seis.zs[channel], (seis.N[channel], 1))

    G = np.array([np.real(seis.Gk[channel]) for k in xrange(seis.N[channel])])
    H = np.array([seis.Hk[channel] for k in xrange(seis.N[channel])])
    g = np.array([np.zeros(seis.r[channel]) for k in xrange(seis.N[channel])])
    h = np.array([[channel] for k in xrange(seis.N[channel])])
    qinv = np.array([seis.qInvk[channel] for k in xrange(seis.N[channel])])
    rinv = np.array([seis.rInvk[channel] for k in xrange(seis.N[channel])])

    y = l2_affine(z, g, h, G, H, qinv, rinv)
    f = back_solve(seis, channel, y)
    
    fig,ax=plt.subplots(2,1,sharex=True)
    ax[0].set_title(seis.channels[channel])
    ax[0].plot(np.arange(len(f)),seis.zs[channel],label='observed')
    ax[1].plot(np.arange(len(f)),seis.fs[channel],label='z-transf')
    ax[1].plot(np.arange(len(f)),-np.real(f)/1e5,label='Kalman')
    ax[1].set_ylim([-5e-5,5e-5])
    ax[0].set_ylabel('seismometer output [counts]')
    ax[1].set_xlabel('time [s]')
    ax[1].set_ylabel(r'accel. [$m/s^2$]')
    ax[0].legend()
    ax[1].legend()
    
    return f

if __name__ == "__main__":
    t1 = seism.dt.datetime(2017, 05, 19, 12, 22)
    t2 = t1 - seism.dt.timedelta(seconds=1200)
    seis = seism.SeismicReader(t1, t2)

    # get the plots from seis.
    f0 = summary_plot(seis, 0)
    f1 = summary_plot(seis, 1)
    f2 = summary_plot(seis, 2)

    #print(D)

    #print(sD)
    # y = l2_affine(z, g, h, G, H, qinv, rinv)
