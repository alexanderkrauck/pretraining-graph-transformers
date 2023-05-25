"""ZINC dataset with ability to conert to RDKit molecules."""

__author__ = "Alexander Krauck"
__email__ = "alexander.krauck@gmail.com"
__date__ = "2023-05-25"

from typing import Callable, Optional

from tqdm import tqdm
from torch_geometric.datasets import ZINC
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

zinc_mapping = {
    0: [6, 0, -1],  # 'C': 0 -> Carbon atomic number is 6
    1: [8, 0, -1],  # 'O': 1 -> Oxygen atomic number is 8
    2: [7, 0, -1],  # 'N': 2 -> Nitrogen atomic number is 7
    3: [9, 0, -1],  # 'F': 3 -> Fluorine atomic number is 9
    4: [
        6,
        0,
        -1,
    ],  # 'C H1': 4 -> empirically this only occurs when the carbon is bonded to 3 other atoms (non hydrogen). so i can just set it to carbon
    5: [16, 0, -1],  # 'S': 5 -> Sulfur atomic number is 16
    6: [17, 0, -1],  # 'Cl': 6 -> Chlorine atomic number is 17
    7: [
        16,
        -1,
        -1,
    ],  # 'O -': 7 -> Oxygen atomic number is 8. seems fine if i set formal charge to -1
    8: [7, 1, 1],  # 'N H1 +': 8 ->
    9: [35, 0, -1],  # 'Br': 9 -> Bromine atomic number is 35
    10: [7, 1, 3],  # 'N H3 +': 10 -> Nitrogen atomic number is 7
    11: [7, 1, 2],  # 'N H2 +': 11 -> Nitrogen atomic number is 7
    12: [7, 1, -1],  # 'N +': 12 -> Nitrogen atomic number is 7
    13: [
        7,
        -1,
        -1,
    ],  # 'N -': 13 -> Nitrogen atomic number is 7: Can appear aromatic for rdkit even tho only 2 bonds.
    14: [16, -1, -1],  # 'S -': 14 -> Sulfur atomic number is 16
    15: [53, 0, -1],  # 'I': 15 -> Iodine atomic number is 53
    16: [15, 0, -1],  # 'P': 16 -> Phosphorus atomic number is 15
    17: [8, 1, 1],  # 'O H1 +': 17 -> Oxygen atomic number is 8
    18: [7, -1, 1],  # 'N H1 -': 18 -> Nitrogen atomic number is 7
    19: [8, 1, -1],  # 'O +': 19 -> Oxygen atomic number is 8
    20: [16, 1, -1],  # 'S +': 20 -> Sulfur atomic number is 16
    21: [15, 0, 1],  # 'P H1': 21 -> Phosphorus atomic number is 15
    22: [15, 0, 2],  # 'P H2': 22 -> Phosphorus atomic number is 15
    23: [6, -1, 2],  # 'C H2 -': 23 -> Carbon atomic number is 6
    24: [15, 1, -1],  # 'P +': 24 -> Phosphorus atomic number is 15
    25: [16, 1, 1],  # 'S H1 +': 25 -> Sulfur atomic number is 16
    26: [6, -1, 1],  # 'C H1 -': 26 -> Carbon atomic number is 6
    27: [15, 1, 1],  # 'P H1 +': 27 -> Phosphorus atomic number is 15
}


class ZincWithRDKit(ZINC):
    def __init__(
        self,
        root: str,
        subset: bool = False,
        split: str = "train",
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
    ):
        super().__init__(root, subset, split, transform, pre_transform, pre_filter)

    def to_rdkit_molecule_list(self):
        """Converts the dataset to a list of RDKit molecules.

        Returns:
            List[rdkit.Chem.rdchem.Mol]: List of RDKit molecules.
        """

        mols = []
        RDLogger.DisableLog("rdApp.warning")
        for data in tqdm(self):
            mol = Chem.RWMol()

            # Add atoms to the molecule
            for features in data.x:
                atom_features = zinc_mapping[int(features[0])]
                atom = Chem.Atom(atom_features[0])
                atom.SetFormalCharge(atom_features[1])
                if atom_features[2] != -1:
                    atom.SetNumExplicitHs(atom_features[2])
                mol.AddAtom(atom)

            # Add bonds to the molecule
            for idx, edge in enumerate(data.edge_index.T):
                i, j = edge[0].item(), edge[1].item()
                if (
                    i < j
                ):  # because the edge_index is undirected, we only need to add the bond once.
                    continue
                bond_type = Chem.rdchem.BondType.values[
                    data.edge_attr[idx].item()
                ]  # Assuming the bond type is in the first element of edge_attr
                mol.AddBond(i, j, order=bond_type)

            # Convert the RDKit molecule to a non-editable molecule

            mol.UpdatePropertyCache(strict=False)
            Chem.GetSSSR(mol)
            try:
                Chem.SanitizeMol(mol)
            except Chem.rdchem.AtomValenceException:
                print("Sanitization failed. Skipping molecule.")
                return None

            AllChem.EmbedMolecule(mol)

            # Optimize the generated conformer
            AllChem.MMFFOptimizeMolecule(mol)

            mols.append(mol)

        RDLogger.EnableLog("rdApp.warning")
        return mols
