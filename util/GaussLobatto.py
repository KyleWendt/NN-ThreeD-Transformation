'''
Created on Oct 18, 2013

@author: wendt
'''

def _tridiag_ev(b):
    pass

def fx(n):
    from numpy import arange, asarray, diag, sqrt, zeros
    from scipy.linalg import eigh

    if n < 2:
        raise ValueError("n must be > 1")
    if n == 2:
        return asarray((-1., 1.)), asarray((1., 1.))

    xi = zeros(n)
    wi = zeros(n)
    Pn = zeros(n)
    i = arange(1, n)
    b = i / sqrt((2.*i - 1) * (2.*i + 1))
    M = diag(b, -1) + diag(b, 1)
    xi, wi = eigh(M)
    wi = 2 * wi[0, :] ** 2
    return xi, wi


def _gauss_lobatto_mesh(n):
    from numpy import arange, asarray, diag, sqrt, zeros
    from scipy.linalg import eigvalsh

    if n < 2:
        raise ValueError("n must be > 1")
    if n == 2:
        return asarray((-1., 1.)), asarray((1., 1.))

    xi = zeros(n)
    wi = zeros(n)
    Pn = zeros(n)
    i = arange(1, n - 2)
    b = sqrt((i * (2. + i)) / (3. + 4.*i * (2. + i)))  # coeff for Jacobi Poly with a=b=1

    M = diag(b, -1) + diag(b, 1)
    xi[1:n - 1] = eigvalsh(M)
    xi[0] = -1.; xi[-1] = 1.

    Pim2 = 1.  # P_{i-2}
    Pim1 = xi  # P_{i-1}
    for i in range(2, n):  # want P_{n-1}
        wi = (1. / i) * ((2 * i - 1) * xi * Pim1 - (i - 1) * Pim2)
        Pim2 = Pim1
        Pim1 = wi
    wi = 2. / (n * (n - 1) * wi ** 2)
    wi[0] = wi[-1] = 2. / (n * (n - 1))
    return xi, wi


def _gauss_legendre_mesh(n):
    from numpy import arange, asarray, diag, sqrt, zeros
    from scipy.linalg import eigh

    if n < 2:
        raise ValueError("n must be > 1")
    if n == 2:
        return asarray((-1., 1.)), asarray((1., 1.))

    xi = zeros(n)
    wi = zeros(n)
    Pn = zeros(n)
    i = arange(1, n)
    b = i / sqrt((2.*i - 1) * (2.*i + 1))

    M = diag(b, -1) + diag(b, 1)
    xi, wi = eigh(M)
    wi = 2 * wi[0, :] ** 2
    return xi, wi


def GLL_Mesh(bounds, npts):
    from numpy import zeros
    if isinstance(npts, str):
        na = npts.upper().split(" ")
        if len(bounds) != len(npts) + 1:
            ValueError("bounds should be 1 element longer that npts")
        T = []; N = []
        for x in na:
            if x[-1] == 'L':
                T.append('L')
                n = int(x[:-1])
                if n < 3:
                    raise ValueError("Lobatto interval must have >2 points")
                N.append(n)
            else:
                T.append('G')
                N.append(int(x))
    else:
        N = list(npts)
        T = ["G"] * len(N)
    ntot = sum(N)
    xi = zeros(ntot); wi = zeros(ntot)
    o = 0
    for i, (n, t) in  enumerate(zip(N, T)):
        A = bounds[i]
        B = bounds[i + 1]
        if t == 'L':
            if i > 0:
                if T[i - 1] == 'L':
                    X, W = _gauss_lobatto_mesh(n + 1)
                    wi[o - 1] += W[0]
                    xi[o:o + n] = X[1:] * (B - A) / 2. + (A + B) / 2
                    wi[o:o + n] = W[1:] * (B - A) / 2.
                    o += n
                    continue
            X, W = _gauss_lobatto_mesh(n)
        else:
            X, W = _gauss_legendre_mesh(n)        

        xi[o:o + n] = X * (B - A) / 2. + (A + B) / 2.
        wi[o:o + n] = W * (B - A) / 2.
        o += n
    return xi, wi