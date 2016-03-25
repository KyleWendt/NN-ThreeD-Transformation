
def main():
    from util import io, hel_pwe_transform as hel
    from util.GaussLobatto import GLL_Mesh
    # vllsjt = io.load_lsj_from_hdf5(None, None, 0, 'PN')
    vllsjt = io.load_lsj_from_tarball('data/n3lo_lsj.tar.gz')
    x, x_weights = GLL_Mesh((-1, 1), "20L")  # 20 gauss-lobatto points (includes +- 1)
    for S in (0, 1):
        for T in (0, 1):
            hel.LSJ_to_Hel(vllsjt, x, S, T)
            # vllsjt["hel S=0 T=0"] is now the 3d summed version for S = T = 0

    for lb, lk, j, s, t in vllsjt['pwe']:
        vkk = hel.Hel_to_LSJ(vllsjt, x, x_weights, lb, lk, j, s, t)
        vkk = hel.Hel_to_LSJ(vllsjt['hel S=%d T=%d' % (s, t)], x, x_weights, lb, lk, j, s, t) # this also works!, here we are just passing numpy array instead of the dictionary
        err = abs(vkk - vllsjt['pwe'][(lb, lk, j, s, t)]).min()
        if err > 1e-10:
            print('large error in lljst=%s' % str((lb, lk, j, s, t)))

if __name__ == '__main__':
    main()