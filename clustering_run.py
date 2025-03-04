import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.ML.Cluster import Butina
import src.bitbirch as bitbirch
from rdkit.Chem import Draw
#from rdkit.Chem.Draw import IPythonConsole
import os
import argparse

# Argument parsing for input file
parser = argparse.ArgumentParser(description="Clustering of compounds")
parser.add_argument('-i', '--input', type=str, required=True, help="Input CSV file containing compounds")
args = parser.parse_args()

# Load dataset
df = pd.read_csv(args.input, on_bad_lines='skip')

# Fingerprint generator setup
fpgen = rdFingerprintGenerator.GetMorganGenerator()

def smi2numpyarr(smi):
    mol = Chem.MolFromSmiles(smi)
    fp = fpgen.GetFingerprintAsNumPy(mol)
    return fp

# Generate fingerprints for all compounds
X = np.array([smi2numpyarr(smi) for smi in df.smiles.to_list()])

Xfp = [DataStructs.CreateFromBitString("".join(x.astype(str).tolist())) for x in X]

# Tanimoto distance matrix
def tanimoto_distance_matrix(fp_list):
    dissimilarity_matrix = []
    for i in range(1, len(fp_list)):
        similarities = DataStructs.BulkTanimotoSimilarity(fp_list[i], fp_list[:i])
        dissimilarity_matrix.extend([1 - x for x in similarities])
    return dissimilarity_matrix

# Clustering using Butina
def cluster_fingerprints(fingerprints, cutoff=0.2):
    distance_matrix = tanimoto_distance_matrix(fingerprints)
    clusters = Butina.ClusterData(distance_matrix, len(fingerprints), cutoff, isDistData=True)
    clusters = sorted(clusters, key=len, reverse=True)
    return clusters

clusters = cluster_fingerprints(Xfp, cutoff=0.4)

# Define the output directory
output_dir = "output_cluster"

# Check if the output directory exists; if not, create it
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save Butina clustering results to CSV in the 'output' directory
butina_cluster_data = []
for cluster_id, cluster in enumerate(clusters):
    for mol_idx in cluster:
        butina_cluster_data.append({'Compound': df.smiles[mol_idx], 'Cluster_ID': cluster_id})

butina_df = pd.DataFrame(butina_cluster_data)
butina_df.to_csv(os.path.join(output_dir, 'butina_clusters.csv'), index=False)
print(f"Butina clustering results saved as '{os.path.join(output_dir, 'butina_clusters.csv')}'")

# BitBirch clustering
bb = bitbirch.BitBirch(threshold=0.3)
res = bb.fit(X)

# Save BitBirch clustering results to CSV in the 'output' directory
bitbirch_cluster_data = []
for cluster_id, cluster in enumerate(res.get_cluster_mol_ids()):
    for mol_idx in cluster:
        bitbirch_cluster_data.append({'Compound': df.smiles[mol_idx], 'Cluster_ID': cluster_id})

bitbirch_df = pd.DataFrame(bitbirch_cluster_data)
bitbirch_df.to_csv(os.path.join(output_dir, 'bitbirch_clusters.csv'), index=False)
print(f"BitBirch clustering results saved as '{os.path.join(output_dir, 'bitbirch_clusters.csv')}'")

############ Visualization of clusters with more than 5 compounds ####################
#bt_gt5 = [c for c in clusters if len(c) > 5]
#bb_gt5 = [c for c in res.get_cluster_mol_ids() if len(c) > 5]

#mols = [Chem.MolFromSmiles(smi) for smi in df.smiles.to_list()]

#Draw.MolsToGridImage([mols[i] for i in bt_gt5[1]][:10], molsPerRow=5, subImgSize=(250, 150))
#Draw.MolsToGridImage([mols[i] for i in bb_gt5[1]][:10], molsPerRow=5, subImgSize=(300, 200))

