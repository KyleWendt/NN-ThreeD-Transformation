

def load_lsj_from_hdf5(hdf_file_path=None, potential_name=None, which=0, ch='PN'):
    import h5py
    if hdf_file_path is None:
        import os.path
        hdf_file_path = os.path.expanduser('~/Data/ME/Evolved_Potentials.h5')

    if potential_name is None:
        potential_name = "N3LO EM 500"

    with h5py.File(hdf_file_path, 'r') as Fin:
        pot = Fin[potential_name]
        kmesh = pot['kmesh'][...]
        kweights = pot['kweights'][...]
        lambdas = pot['lambdas'][...]
        node_bounds = pot['node bounds'][...]
        node_count = pot['node count'][...]
        vllsjt = dict(kmesh=kmesh, kweights=kweights, lam=lambdas[which], pwe={})
        jmax = 0
        for wave in pot['PWE']:
            wave_grp=pot['PWE'][wave]
            l, s, j, t = [wave_grp.attrs[l] for l in 'LSJT']
            vkk = wave_grp[ch]['Trel']['V %d' % which][...]
            if l == -1:
                for a, lk in enumerate((j-1, j+1)):
                    for b, lb in enumerate((j-1, j+1)):
                        vllsjt['pwe'][(lb, lk, j, s, t)] = vkk[b, a]
            else:
                vllsjt['pwe'][(l, l, j, s, t)] = vkk
            jmax = max(jmax, j)
            vllsjt['J Max'] = jmax
        return vllsjt


def save_lsj_to_tarball(file_path, v_dict):
    import tarfile
    import io

    me_fmt = '{:<+21.13e}'
    vpot_fmt = ' {{:4d}} {{:4d}} {:s}\n'.format(me_fmt)
    mesh_fmt = ' {{:4d}} {:s} {:s}\n'.format(me_fmt, me_fmt)

    with tarfile.open(file_path, 'w:gz') as fout:
        readme_string = \
"""Notes on format:
Only the top triangle is stored.
The file names are "l_bra l_ket j_tot s_tot t_tot".
For l_bra == l_ket, only elements indexed as i <= j are included.
For l_bra < l_ket, all i,j are included.
No files wilth l_ket > l_bra are included in the archive.

"""
        readme_file = io.BytesIO(readme_string.encode('utf8'))
        info = tarfile.TarInfo(name='README')
        info.size = len(readme_string)
        fout.addfile(tarinfo=info, fileobj=readme_file)
        del readme_string, info, readme_file

        mesh_string = ''
        kmesh = v_dict['kmesh']
        kweights = v_dict['kweights']
        nk = kmesh.size
        for i, (m, w) in enumerate(zip(kmesh, kweights)):
            mesh_string += mesh_fmt.format(i+1, m, w)
        mesh_file = io.BytesIO(mesh_string.encode('utf8'))
        info = tarfile.TarInfo(name='mesh')
        info.size = len(mesh_string)
        fout.addfile(tarinfo=info, fileobj=mesh_file)
        del mesh_file, info, mesh_string

        for (lb, lk, j, s, t), vkk in v_dict['pwe'].items():
            pot_string = ''
            if lb > lk:
                continue
            elif lb < lk:
                for a in range(nk):
                    for b in range(nk):
                        pot_string += vpot_fmt.format(a + 1, b + 1, vkk[a, b])
            else:
                for a in range(nk):
                    for b in range(a + 1):
                        pot_string += vpot_fmt.format(a + 1, b + 1, vkk[a, b])
            info = tarfile.TarInfo(name='lljst/V {:d} {:d} {:d} {:d} {:d}'.format(lb, lk, j, s, t))
            pot_file = io.BytesIO(pot_string.encode('utf8'))
            info.size = len(pot_string)
            fout.addfile(tarinfo=info, fileobj=pot_file)
            del pot_file, info, pot_string


def load_lsj_from_tarball(file_path):
    import tarfile
    import numpy

    with tarfile.open(file_path, 'r:gz') as fin:
        mesh_file = fin.extractfile('mesh')
        data = numpy.genfromtxt(mesh_file, dtype=[('a', int), ('kmesh', float), ('kweights', float)])
        kmesh, kweights = data['kmesh'], data['kweights']
        nk = len(kmesh)
        jmax = 0
        pwe = dict()
        for member in fin.getmembers():
            if member.name.startswith('lljst'):
                wave = lb, lk, j, s, t = tuple(map(int, member.name[8:].split()))
                jmax = max(j, jmax)
                pot_file = fin.extractfile(member)
                data = numpy.genfromtxt(pot_file, dtype=[('a', int), ('b', int), ('me', float)])
                vkk = numpy.zeros((nk, nk))
                if lb == lk:
                    vkk[data['a'] - 1, data['b'] - 1] = vkk[data['b'] - 1, data['a'] - 1] = data['me']
                    pwe[(lb, lk, j, s, t)] = vkk
                else:
                    vkk[data['a'] - 1, data['b'] - 1] = data['me']
                    pwe[(lb, lk, j, s, t)] = vkk
                    pwe[(lk, lb, j, s, t)] = vkk.T
    v_dict = {
        'kmesh': kmesh,
        'kweights': kweights,
        'J Max': jmax,
        'pwe':pwe
    }
    return v_dict


def load_lsj_from_directory(directory_path):
    import numpy
    import os.path
    import glob

    jmax = 0
    pwe = {}
    kmesh, kweights = numpy.loadtxt(os.path.join(directory_path, 'mesh'), unpack=True)[1:]
    nk = len(kmesh)
    for wave_path in glob.glob(os.path.join(directory_path, 'lljst/V *')):
        try:
            wave = lb, lk, j, s, t = tuple(map(int, os.path.basename(wave_path)[2:].split()))
        except Exception as e:
            continue
        jmax = max(j, jmax)
        data = numpy.loadtxt(wave_path, dtype=[('a', int), ('b', int), ('me', float)])
        vkk = numpy.zeros((nk, nk))
        if lb == lk:
            vkk[data['a'] - 1, data['b'] - 1] = vkk[data['b'] - 1, data['a'] - 1] = data['me']
            pwe[(lb, lk, j, s, t)] = vkk
        else:
            vkk[data['a'] - 1, data['b'] - 1] = data['me']
            pwe[(lb, lk, j, s, t)] = vkk
            pwe[(lk, lb, j, s, t)] = vkk.T
    v_dict = {
        'kmesh': kmesh,
        'kweights': kweights,
        'J Max': jmax,
        'pwe': pwe
    }
    return v_dict




