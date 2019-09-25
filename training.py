
import joblib
import rmsd
import json
import csv

import numpy as np
import pandas as pd
import qml
from qml.kernels import get_atomic_local_kernel
from qml.kernels import gaussian_kernel, laplacian_kernel
from qml.math import svd_solve
from qml.representations import generate_fchl_acsf

from chemhelp import cheminfo

from scipy import stats
import matplotlib.pyplot as plt

from qml.kernels import kpca

cachedir = '.pycache'
memory = joblib.Memory(cachedir, verbose=0)

np.random.seed(43)

def rmse(X, Y):
    """
    Root-Mean-Square Error

    Lower Error = RMSE \left( 1- \sqrt{ 1- \frac{1.96\sqrt{2}}{\sqrt{N-1}} }  \right )
    Upper Error = RMSE \left(    \sqrt{ 1+ \frac{1.96\sqrt{2}}{\sqrt{N-1}} } - 1 \right )

    This only works for N >= 8.6832, otherwise the lower error will be
    imaginary.

    Parameters:
    X -- One dimensional Numpy array of floats
    Y -- One dimensional Numpy array of floats

    Returns:
    rmse -- Root-mean-square error between X and Y
    le -- Lower error on the RMSE value
    ue -- Upper error on the RMSE value
    """

    assert X.shape == Y.shape

    N = X.shape[0]

    if N < 9:
        print("Not enough points. {} datapoints given. At least 9 is required".format(N))
        return

    diff = X - Y
    diff = diff**2
    rmse = np.sqrt(diff.mean())

    le = rmse * (1.0 - np.sqrt(1-1.96*np.sqrt(2.0)/np.sqrt(N-1)))
    ue = rmse * (np.sqrt(1 + 1.96*np.sqrt(2.0)/np.sqrt(N-1))-1)

    return rmse, le, ue


@memory.cache
def prepare_training_data():

    # distance_cut = 10.0
    # parameters = {
    #     "pad": 25, # max atoms
    #     "rcut": distance_cut,
    #     "acut": distance_cut,
    #     "elements": [1, 6, 7, 8],
    # }

    distance_cut = 20.0
    parameters = {
        "pad": 25,
	'nRs2': 22,
	'nRs3': 17,
	'eta2': 0.41,
	'eta3': 0.97,
	'three_body_weight': 45.83,
	'three_body_decay': 2.39,
	'two_body_decay': 2.39,
        "rcut": distance_cut,
        "acut": distance_cut,
    }

    dirprefix = "../data/qmepa890/"
    filename = dirprefix + "data.csv"

    # 1. File ID (e.g. 0415 means the information pertains to the files `0415.xyz` and `0415_+.xyz`)
    # 2. Index of the proton (in the `XXXX_+.xyz` file listed in the same row)
    # 3. Gas-phase energy of neutral molecule plus thermal corrections from vibrational analysis
    # 4. Gas-phase energy of protonated molecule plus thermal corrections from vibrational analysis
    # 5. Gas-phase energy of neutral molecule
    # 6. Gas-phase energy of protonated molecule
    # 7. Energy of neutral molecule using SMD implicit solvent model
    # 8. Energy of protonated molecule using SMD implicit solvent model
    # 9. PM6 heat-of-formation of neutral molecule using COSMO implicit solvent model
    # 10. PM6 heat-of-formation of protonated molecule using COSMO implicit solvent model

    df = pd.read_csv(filename, sep=",", header=None)

    molecule_names = df.iloc[:,0]
    proton_idxs = df.iloc[:,1]
    energies = df.iloc[:,2:]

    representations = []
    coord_list = []
    atoms_list = []

    for name in molecule_names:

        name = str(name).zfill(4)
        atoms, coord = rmsd.get_coordinates_xyz(dirprefix + "structures/" + name + "_+.xyz")
        atoms = [cheminfo.convert_atom(atom) for atom in atoms]

        representation = generate_fchl_acsf(atoms, coord, **parameters)

        representations.append(representation)

        coord_list.append(coord)
        atoms_list.append(atoms)

    representations = np.array(representations)
    proton_idxs = np.array(proton_idxs)

    print(representations.shape)

    return coord_list, atoms_list, proton_idxs, representations, energies


def krr(kernel, properties, rcond=1e-11):

    alpha = svd_solve(kernel, properties, rcond=rcond)

    return alpha


def get_kernel(X1, X2, charges1, charges2, sigma=1):

    if len(X1.shape) > 2:

        K = get_atomic_local_kernel(X1, X2,  charges1, charges2, sigma)

    else:

        K = laplacian_kernel(X2, X1, sigma)

    return K


def check_learning(representations, atomss, properties, select_atoms=None):

    properties = np.array(properties)

    if select_atoms is not None:
        representations = [representations[i,j] for i,j in enumerate(select_atoms)]
        representations = np.array(representations)

    n_points = len(properties)
    indexes = np.arange(n_points, dtype=int)
    np.random.shuffle(indexes)

    n_valid = 50
    v_idxs = indexes[-n_valid:]
    v_repr = representations[v_idxs]
    v_props = properties[v_idxs]
    v_atoms = [atomss[i] for i in v_idxs]

    n_training = [2**x for x in range(1, 8)]

    for n in n_training:

        t_idxs = indexes[:n]
        t_repr = representations[t_idxs]
        t_props = properties[t_idxs]
        t_atoms = [atomss[i] for i in t_idxs]

        # Train
        t_K = get_kernel(t_repr, t_repr, t_atoms, t_atoms)

        # PCA
        if select_atoms is not None:
            pca = kpca(t_K, n=2)

            # plot
            fig, axs = plt.subplots(2, 1, figsize=(5,10))

            sc = axs[0].scatter(*pca, c=t_props)
            fig.colorbar(sc, ax=axs[0])

            im = axs[1].imshow(t_K)
            fig.colorbar(im, ax=axs[1])

            fig.savefig("_tmp_pca_{:}.png".format(n))

        else:
            fig, axs = plt.subplots(1, 1)
            im = axs.imshow(t_K)
            fig.colorbar(im, ax=axs)
            fig.savefig("_tmp_pca_{:}.png".format(n))


        # Train model
        t_alpha = krr(t_K, t_props)

        # Test and predict
        p_K = get_kernel(t_repr, v_repr, t_atoms, v_atoms)
        p_props = np.dot(p_K, t_alpha)

        # rmse error
        p_rmse, le, ue = rmse(p_props, v_props)

        print("{:5d}".format(n), "{:10.2f} ± {:4.2f}".format(p_rmse, ue))






    return


def overview(properties):

    # 0. Gas-phase energy of neutral molecule plus thermal corrections from vibrational analysis
    # 1. Gas-phase energy of protonated molecule plus thermal corrections from vibrational analysis
    # 2. Gas-phase energy of neutral molecule
    # 3. Gas-phase energy of protonated molecule
    # 4. Energy of neutral molecule using SMD implicit solvent model
    # 5. Energy of protonated molecule using SMD implicit solvent model
    # 6. PM6 heat-of-formation of neutral molecule using COSMO implicit solvent model
    # 7. PM6 heat-of-formation of protonated molecule using COSMO implicit solvent model

    neutral = properties.iloc[:,4]
    protonated = properties.iloc[:,5]

    neutral = np.array(neutral)
    protonated = np.array(protonated)
    protonation = protonated - neutral
    protonation -= protonation.mean()

    kde = stats.gaussian_kde(protonation)
    xx = np.linspace(min(protonation), max(protonation), 2000)
    fig_en = plt.figure()
    ax_en = fig_en.add_subplot(111)
    ax_en.plot(xx, kde(xx))
    ax_en.set_xlabel("Energy [kcal/mol]")
    # ax_en.set_ylabel("Density")
    fig_en.savefig("_tmp_energy_overview.png")

    print(protonation)

    return


def main():

    coord_list, atoms_list, proton_idxs, representations, properties = prepare_training_data()

    neutral = properties.iloc[:,4]
    protonated = properties.iloc[:,5]

    neutral = np.array(neutral)
    protonated = np.array(protonated)
    protonation = protonated - neutral
    # protonation -= protonation.mean()

    # energies = properties.iloc[:,0]
    check_learning(representations, atoms_list, protonation, select_atoms=proton_idxs)


    return


if __name__ == '__main__':
    main()
