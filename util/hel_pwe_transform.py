import pywigxjpf
pywigxjpf.wig_table_init(2*100, 9)
pywigxjpf.wig_temp_init(2*100)

def clebsch_gordan(two_j1,  two_j2,  two_j3,  two_m1,  two_m2,  two_m3):
    from math import sqrt
    threej = pywigxjpf.wig3jj((two_j1,  two_j2,  two_j3,
                             two_m1,  two_m2,  -two_m3))
    phase = -1 if (two_m3+two_j1-two_j2) % 4 == 2 else 1
    return phase * sqrt(float(two_j3+1)) * threej

def Wigner_d(j, m1, m2, x):
    # using definition from A.R. Edmonds, eq 4.1.23
    from scipy.special import factorial, eval_jacobi
    from numpy import sqrt
    mp, m = m1, m2
    p = 1
    if mp + m < 0:
        mp, m = -mp, -m
        p *= (-1) ** (mp - m)
    if mp < m:
        mp, m = m, mp
        p *= (-1) ** (mp - m)
    return (p * 
        sqrt(
            factorial(j + mp) * factorial(j - mp) / 
            (factorial(j + m) * factorial(j - m))
            ) * 
        sqrt(.5 + .5 * x) ** (mp + m) * 
        sqrt(.5 - .5 * x) ** (mp - m) * 
        eval_jacobi(j - mp, mp - m, mp + m, x)
    )
    
    
    
def Wigner_d_mat(j, x):
    from numpy import  array, isscalar, zeros
    X = array(x)
    W = zeros([2 * j + 1, 2 * j + 1] + list(X.shape), X.dtype)
    for mp in range(-j, j + 1):
        for m in range(-j, j + 1):
            W[mp, m] = Wigner_d(j, mp, m, X)
    return W

def LSJ_to_Hel(V, x, S, T):
    from numpy import  zeros, pi, sqrt

    nk = V['kmesh'].size
    nx = x.size
    if S == 1:
        Vhel = zeros((3, 3, nk, nk, nx), float)
    else: 
        Vhel = zeros((1, 1, nk, nk, nx), float)
    JMax=  V['J Max']
    eta = -1 if S == T else 1
    for J in range(JMax + 1):
        if J == 0:
            D = zeros((3, 3, nx))
            D[0,0] = Wigner_d_mat(J, x)[0,0]
        else:
            D = Wigner_d_mat(J, x)
     
        for L in range(abs(J - S), J + S + 1):
            for Lp in range(abs(J - S), J + S + 1):
                if (L + S + T) & 1 == 0 or  (Lp + S + T) & 1 == 0: continue
                v = V['pwe'][(Lp, L, J, S, T)]
                c = (1 + eta * (-1) ** Lp) * (1 + eta * (-1) ** L)
                for M in range(-S, S + 1):
                    c1 = clebsch_gordan(2*L, 2*S, 2*J, 0, 2*M, 2*M) * sqrt((2.*L + 1.) / (4. * pi))
                    for Mp in range(-S, S + 1):
                        c2 = clebsch_gordan(2*Lp, 2*S, 2*J, 0, 2*Mp, 2*Mp) * sqrt((2.*Lp + 1.) / (4. * pi))
                        Vhel[Mp, M] += v[:, :, None] * D[M, Mp, None, None,:] * c * c1 * c2
    V['hel S=%d T=%d' % (S,T)] = Vhel
    V['xmesh'] = x
    return Vhel

def Hel_to_LSJ(V, xmesh, xweights, Lp, L, J, S, T):
    from numpy import  zeros, pi, sqrt
    
    if isinstance(V, dict):
        nk = V['kmesh'].size
        Vhel = V['hel S=%d T=%d' % (S,T)]
    else:
        nk = V.shape[-2]
        Vhel = V
        
    Vkk = zeros((nk, nk), float)
    
    if J == 0:
        D = zeros((3, 3, xmesh.size))
        D[0,0] = Wigner_d_mat(J, xmesh)[0,0]
    else:
        D = Wigner_d_mat(J, xmesh)
    D *= xweights[None, None, :]
    
    eta = -1 if S == T else 1
    
    c = 4. * pi * sqrt((2.*L + 1.) * (2.*Lp + 1.)) / (2.*J + 1) 
    c /= 2 * (1 + eta * (-1) ** Lp) * (1 + eta * (-1) ** L)
    
    for M in range(-S, S + 1):
        c1 = clebsch_gordan(2*L, 2*S, 2*J, 0, 2*M, 2*M)
        for Mp in range(-S, S + 1):
            c2 = clebsch_gordan(2*Lp, 2*S, 2*J, 0, 2*Mp, 2*Mp)
            if min(M, Mp) == -1 and max(M, Mp) != -1:
                phase = (eta * (-1)**(S+M) if Mp ==-1 else 1) * (eta * (-1)**(S+Mp) if M == -1 else 1)
                Vkk += Vhel[abs(Mp), abs(M), :, :, ::-1].dot(D[M, Mp]) * c1 * c2 * c * phase             
            else:
                Vkk += Vhel[abs(Mp), abs(M)].dot(D[M, Mp]) * c1 * c2 * c                                
    return Vkk