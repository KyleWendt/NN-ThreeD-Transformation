


def main():
    from util.io import load_lsj_from_hdf5
    vllsjt = load_lsj_from_hdf5(None, None, 0, 'PN')
    # save_lsj_to_tarball('data/n3lo_lsj.tar.gz', vllsjt)
    # exit()
    from util.io import load_lsj_from_tarball

    vllsjt2 = load_lsj_from_tarball('data/n3lo_lsj.tar.gz')
    # vllsjt2 = load_lsj_from_directory('data/n3lo_lsj')

    for x in vllsjt['pwe']:
        err = abs(vllsjt['pwe'][x] - vllsjt2['pwe'][x]).max()
        if err > 1e-12:
            print(x, err)

    for x in vllsjt2['pwe']:
        err = abs(vllsjt['pwe'][x] - vllsjt2['pwe'][x]).max()
        if err > 1e-12:
            print(x, err)

    print(abs(vllsjt['kmesh'] - vllsjt2['kmesh']).max())
    print(abs(vllsjt['kweights'] - vllsjt2['kweights']).max())

if __name__ == '__main__':

    main()
    # from numpy import pi
    # print('{:+21.13e} {:+21.13e}'.format(+pi, pi))
    # print('{:+21.13e} {:+21.13e}'.format(0,0))
    # print('{:<+21.13e} {:+21.13e}'.format(-pi,pi))
    # print('{:<+21.13e} {:+21.13e}'.format(-pi * 1e100, pi))
