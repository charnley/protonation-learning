
import joblib
import rmsd
import json
import csv

import numpy as np
import pandas as pd
import qml
from qml.kernels import get_atomic_local_kernel, get_local_kernel
from qml.kernels import gaussian_kernel, laplacian_kernel
from qml.math import svd_solve, cho_solve
from qml.representations import generate_fchl_acsf

from chemhelp import cheminfo

from scipy import stats
import matplotlib.pyplot as plt

from qml.kernels import kpca

cachedir = '.pycache'
memory = joblib.Memory(cachedir, verbose=0)

np.random.seed(44)

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
def prepare_training_data_protonafinity():

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
        "elements": [1, 6, 7, 8, 9, 12]
    }

    dirprefix = "../data/dataset-proton-affinity/data/"
    filename = dirprefix + "pm3_properties.csv"
    df = pd.read_csv(filename, sep=",")

    # column names
    col_neuidx = "MoleculeIdx"
    col_proidx = "ProtonatedIdx"
    col_refsmi = "ReferenceSmiles"
    col_prosmi = "ProtonatedSmiles"
    col_neueng = "NeutralEnergy"
    col_proeng = "ProtonatedEnergy"


    # Collect energies
    energies_neutr = df[col_neueng]
    energies_proto = df[col_proeng]

    energies = [energies_neutr, energies_proto]
    energies = np.array(energies)

    # Protonated representation
    p_representations = []
    p_coord_list = []
    p_atoms_list = []

    # Neutral representation
    n_representations = []
    n_coord_list = []
    n_atoms_list = []

    for idx, row in df.iterrows():

        # name = str(name).zfill(4)

        print(row)

        nidx = row[col_neuidx]
        pidx = row[col_proidx]

        nname = f"xyz{nidx}_n.xyz"
        pname = f"xyz{nidx}_{pidx}.xyz"

        # Neutral state
        atoms, coord = rmsd.get_coordinates_xyz(dirprefix + "pm3.cosmo.mop/" + nname)
        atoms = [cheminfo.convert_atom(atom) for atom in atoms]

        n_representation = generate_fchl_acsf(atoms, coord, **parameters)
        n_representations.append(n_representation)
        n_coord_list.append(coord)
        n_atoms_list.append(atoms)

        # Protonated state
        atoms, coord = rmsd.get_coordinates_xyz(dirprefix + "pm3.cosmo.mop/" + pname)
        atoms = [cheminfo.convert_atom(atom) for atom in atoms]

        # Find protonated atom
        smiles = row[col_prosmi]
        molobj, status = cheminfo.smiles_to_molobj(smiles)

        assert molobj is not None, "Molobj failed for {smiles}"

        smi_atoms = molobj.GetAtoms()
        atom_charges = [atom.GetFormalCharge() for atom in smi_atoms]
        atom_charges = np.array(atom_charges)
        idx, = np.where(atom_charges > 0)

        assert len(idx) == 1, f"Should only be one charged atom in {pname}"

        idx = idx[0]

        # Set nitrogen to heavy atom
        atoms[idx] = 12

        p_representation = generate_fchl_acsf(atoms, coord, **parameters)
        p_representations.append(n_representation)
        p_coord_list.append(coord)
        p_atoms_list.append(atoms)

    # proton_idxs = np.array(proton_idxs)

    n_representations = np.array(n_representations)
    p_representations = np.array(p_representations)

    return n_representations, p_representations, n_coord_list, p_coord_list, n_atoms_list, p_atoms_list, energies


@memory.cache
def prepare_training_data_qmepa890():

    # distance_cut = 10.0
    # parameters = {
    #     "pad": 25, # max atoms
    #     "rcut": distance_cut,
    #     "acut": distance_cut,
    #     "elements": [1, 6, 7, 8],
    # }

    # Table 5. Free atom energies from DFT/PBE0/def2TZVP.
    # H   C   N   O   S
    # Multiplicity    2   3   4   3   3
    # Energy / Eh     −0.501036   −37.8054    −54.5438    −75.0186    −397.974

    atom_energies = {}
    # TODO 



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
        "elements": [1, 6, 7, 8, 12]
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

    p_representations = []
    p_coord_list = []
    p_atoms_list = []

    n_representations = []
    n_coord_list = []
    n_atoms_list = []

    for h_idx, name in zip(proton_idxs, molecule_names):

        name = str(name).zfill(4)

        atoms, coord = rmsd.get_coordinates_xyz(dirprefix + "structures/" + name + ".xyz")
        atoms = [cheminfo.convert_atom(atom) for atom in atoms]
        n_representation = generate_fchl_acsf(atoms, coord, **parameters)
        n_representations.append(n_representation)
        n_coord_list.append(coord)
        n_atoms_list.append(atoms)

        atoms, coord = rmsd.get_coordinates_xyz(dirprefix + "structures/" + name + "_+.xyz")
        atoms = [cheminfo.convert_atom(atom) for atom in atoms]
        atoms[h_idx-1] = 12
        p_representation = generate_fchl_acsf(atoms, coord, **parameters)
        p_representations.append(n_representation)
        p_coord_list.append(coord)
        p_atoms_list.append(atoms)

        print(f"representing {name}")

    proton_idxs = np.array(proton_idxs)

    n_representations = np.array(n_representations)
    p_representations = np.array(p_representations)

    return n_representations, p_representations, n_coord_list, p_coord_list, n_atoms_list, p_atoms_list, proton_idxs, energies



def krr(kernel, properties, rcond=1e-9, solver="cho"):
    # rcond = 1e-4

    if solver == "cho":
        alpha = cho_solve(kernel, properties, l2reg=rcond)
    else:
        alpha = svd_solve(kernel, properties, rcond=rcond)

    return alpha


def create_local_kernel(X1, X2, charges1, charges2, sigma=2.0, mode="local"):

    K = qml.kernels.get_local_kernel(X1, X2,  charges1, charges2, sigma)

    return K

def get_kernel(X1, X2, charges1, charges2, sigma=1, mode="local"):
    """

    mode local or atomic
    """

    if len(X1.shape) > 2:

        K = get_atomic_local_kernel(X1, X2,  charges1, charges2, sigma)

    else:

        K = laplacian_kernel(X2, X1, sigma)

    return K


def check_learning_atom(representations, atomss, properties, select_atoms=None):

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


def check_learning_atomization(
    representations,
    atoms_list,
    properties):
    """

    check learning of atomization energy

    """

    properties = np.array(properties)

    n_points = len(properties)
    indexes = np.arange(n_points, dtype=int)
    np.random.shuffle(indexes)

    n_valid = 50
    v_idxs = indexes[-n_valid:]
    v_repr = representations[v_idxs]
    v_atoms = [atoms_list[i] for i in v_idxs]
    v_props = properties[v_idxs]

    # n_training = [2**x for x in range(1, 5)]
    n_training = [2**x for x in range(1, 10)]
    # n_training = [2**6]

    for n in n_training:

        t_idxs = indexes[:n]
        t_props = properties[t_idxs]

        t_repr = representations[t_idxs]
        t_atoms = [atoms_list[i] for i in t_idxs]

        # Train
        t_K = create_local_kernel(t_repr, t_repr, t_atoms, t_atoms)

        if False:

            pca = kpca(K_bind, n=2)

            # plot
            fig, axs = plt.subplots(2, 1, figsize=(5,10))

            sc = axs[0].scatter(*pca, c=t_props)
            fig.colorbar(sc, ax=axs[0])

            im = axs[1].imshow(K_bind)
            fig.colorbar(im, ax=axs[1])

            fig.savefig("_tmp_pca_{:}.png".format(n))


        # Train model
        t_alpha = krr(t_K, t_props)

        # Test and predict
        v_K = create_local_kernel(t_repr, v_repr, t_atoms, v_atoms)

        p_props = np.dot(v_K, t_alpha)

        # rmse error
        p_rmse, le, ue = rmse(p_props, v_props)

        print("{:5d}".format(n), "{:10.2f} ± {:4.2f}".format(p_rmse, ue))


    return


def check_learning_mol(
    n_representations,
    p_representations,
    n_atoms_list,
    p_atoms_list,
    properties):
    """

    n_rep - neutral representations
    p_rep - protonated representations

    kernel = kernel_p - kernel_n

    """

    properties = np.array(properties)

    n_points = len(properties)
    indexes = np.arange(n_points, dtype=int)
    np.random.shuffle(indexes)

    n_valid = 50
    v_idxs = indexes[-n_valid:]
    vn_repr = n_representations[v_idxs]
    vp_repr = p_representations[v_idxs]
    vn_atoms = [n_atoms_list[i] for i in v_idxs]
    vp_atoms = [p_atoms_list[i] for i in v_idxs]
    v_props = properties[v_idxs]

    n_training = [2**x for x in range(1, 10)]
    # n_training = [2**6]

    for n in n_training:

        t_idxs = indexes[:n]
        t_props = properties[t_idxs]

        tn_repr = n_representations[t_idxs]
        tn_atoms = [n_atoms_list[i] for i in t_idxs]

        tp_repr = p_representations[t_idxs]
        tp_atoms = [p_atoms_list[i] for i in t_idxs]

        # Train
        tn_K = create_local_kernel(tn_repr, tn_repr, tn_atoms, tn_atoms)
        tp_K = create_local_kernel(tp_repr, tp_repr, tp_atoms, tp_atoms)
        # tnp_K = create_local_kernel(tn_repr, tp_repr, tn_atoms, tp_atoms)
        # tpn_K = create_local_kernel(tp_repr, tn_repr, tp_atoms, tn_atoms)

        # print(tn_K)
        # print(tp_K)
        # print(tnp_K)
        # print(tpn_K)

        # K_bind = tn_K + tp_K - tnp_K - tpn_K
        K_bind = tp_K - tn_K

        if False:

            pca = kpca(K_bind, n=2)

            # plot
            fig, axs = plt.subplots(2, 1, figsize=(5,10))

            sc = axs[0].scatter(*pca, c=t_props)
            fig.colorbar(sc, ax=axs[0])

            im = axs[1].imshow(K_bind)
            fig.colorbar(im, ax=axs[1])

            fig.savefig("_tmp_pca_{:}.png".format(n))


        # Train model
        t_alpha = krr(K_bind, t_props, solver="svd")

        # Test and predict
        vn_K = create_local_kernel(tn_repr, vn_repr, tn_atoms, vn_atoms)
        vp_K = create_local_kernel(tp_repr, vp_repr, tp_atoms, vp_atoms)

        v_K = vp_K - vn_K

        p_props = np.dot(v_K, t_alpha)

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

    try:
        neutral = properties.iloc[:,0]
        protonated = properties.iloc[:,1]
    except:
        neutral = properties[:,0]
        protonated = properties[:,1]

    neutral = np.array(neutral)
    protonated = np.array(protonated)
    protonation = protonated - neutral
    # protonation -= protonation.mean()

    kde = stats.gaussian_kde(protonation)
    xx = np.linspace(min(protonation), max(protonation), 2000)
    fig_en = plt.figure()
    ax_en = fig_en.add_subplot(111)
    ax_en.plot(xx, kde(xx))
    ax_en.set_xlabel("Energy [kcal/mol]")
    # ax_en.set_ylabel("Density")
    fig_en.savefig("_tmp_energy_overview.png")

    print("n_prop", len(protonation))

    # print(protonation)

    return


def main():

    # n_representations, p_representations, \
    # n_coord_list, p_coord_list, \
    # n_atoms_list, p_atoms_list, \
    # proton_idxs, properties = prepare_training_data_protonafinity()

    n_representations, p_representations, \
    n_coord_list, p_coord_list, \
    n_atoms_list, p_atoms_list, \
    properties = prepare_training_data_protonafinity()

    overview(properties)

    atomization = properties.iloc[:,6]
    neutral = properties.iloc[:,4]
    protonated = properties.iloc[:,5]

    neutral = np.array(neutral)
    protonated = np.array(protonated)
    protonation = protonated - neutral
    # protonation -= protonation.mean()


    # results = check_learning_atomization(
    #     n_representations,
    #     n_atoms_list,
    #     atomization)
    #
    # quit()

    check_learning_mol(
        n_representations,
        p_representations,
        n_atoms_list,
        p_atoms_list,
        protonation)


    # energies = properties.iloc[:,0]
    # check_learning(representations, atoms_list, protonation, select_atoms=proton_idxs)


    return


if __name__ == '__main__':
    main()
