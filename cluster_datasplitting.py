import os
import pandas as pd
import src.split_utils as uru
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, AllChem
from rdkit.DataStructs import BulkTanimotoSimilarity, ExplicitBitVect
from rdkit.ML.Cluster import Butina
import numpy as np
from tqdm.auto import tqdm
import seaborn as sns
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
from src.lgbm_wrapper import LGBMMorganCountWrapper, LGBMPropWrapper
from sklearn.metrics import r2_score, mean_absolute_error
import itertools
import time
import src.bitbirch as bb
import argparse  # Import argparse for argument parsing

# Set up argument parser
parser = argparse.ArgumentParser(description="Data Splitting and Model Training")
parser.add_argument('-i', '--input', type=str, required=True, help="Path to input CSV file")
args = parser.parse_args()

# Get the input file path from the argument
input_file_path = args.input

# Check and create 'output' directory if it doesn't exist
output_dir = "output_datasplitting"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load dataset from the input file
df = pd.read_csv(input_file_path)
print(df)

# Add cluster columns
df['random_cluster'] = uru.get_random_clusters(df.SMILES)
df['scaffold_cluster'] = uru.get_bemis_murcko_clusters(df.SMILES)
df['butina_cluster'] = uru.get_butina_clusters(df.SMILES)
df['umap_cluster'] = uru.get_umap_clusters(df.SMILES, n_clusters=7)

# Function to get BitBirch clusters
def get_bitbirch_clusters(smiles_list):
    mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
    fps = np.array([Chem.RDKFingerprint(mol) for mol in mols])
    bitbirch = bb.BitBirch(branching_factor=50, threshold=0.65)  
    bitbirch.fit(fps)
    cluster_list = bitbirch.get_cluster_mol_ids()
    n_molecules = len(fps)
    cluster_labels = [0] * n_molecules
    for cluster_id, indices in enumerate(cluster_list):
        for idx in indices:
            cluster_labels[idx] = cluster_id
    return cluster_labels

df['bitbirch_cluster'] = get_bitbirch_clusters(df.SMILES)

##############################  Comparing Dataset Sizes ######################################
# Perform 5x5 cross-validation and examine the dataset sizes produced by the five splitting strategies.

size_df = df
size_df['mol'] = size_df.SMILES.apply(Chem.MolFromSmiles)
fpgen = rdFingerprintGenerator.GetMorganGenerator()
size_df['fp'] = size_df.mol.apply(fpgen.GetCountFingerprintAsNumPy)
size_df['binary_fps'] = size_df.mol.apply(Chem.RDKFingerprint)

split_list = ["random_cluster", "butina_cluster", "umap_cluster", "scaffold_cluster", "bitbirch_cluster"]
split_dict = {
    "random_cluster": uru.get_random_clusters,
    "butina_cluster": uru.get_butina_clusters,
    "umap_cluster": uru.get_umap_clusters,
    "scaffold_cluster": uru.get_bemis_murcko_clusters,
    "bitbirch_cluster": get_bitbirch_clusters
}

result_list = []
for split in split_list:
    for i in tqdm(range(0, 5), desc=split):
        cluster_list = split_dict[split](size_df.SMILES)
        group_kfold_shuffle = uru.GroupKFoldShuffle(n_splits=5, shuffle=True)
        if split == 'bitbirch_cluster':
            for train, test in group_kfold_shuffle.split(np.stack(size_df.binary_fps), size_df.logS, cluster_list):
                result_list.append([split, len(test)])
        else:
            for train, test in group_kfold_shuffle.split(np.stack(size_df.fp), size_df.logS, cluster_list):
                result_list.append([split, len(test)])

result_df = pd.DataFrame(result_list, columns=["split", "num_test"])

print("splitting done")

# Save the plot
sns.set_style('whitegrid')
plt.figure(figsize=(8, 5))
ax = sns.boxplot(x="split", y="num_test", data=result_df)
ax.set_xlabel("Dataset Splitting Strategy")
ax.set_ylabel("Test Set Size")
plt.savefig(os.path.join(output_dir, 'splitting_boxplot.png'), dpi=300)
plt.close()

print("Cluster plot for test set molecules saved")

########################### Examining the impact of the number of clusters ############################
print("Examining the impact of number of clusters on the size of the test sets")

urc_result_list = []
for num_clus in tqdm(range(5, 76, 5)):
    for i in range(0, 5):
        cluster_list = uru.get_umap_clusters(size_df.SMILES, n_clusters=num_clus)
        group_kfold_shuffle = uru.GroupKFoldShuffle(n_splits=5, shuffle=True)
        for train, test in group_kfold_shuffle.split(np.stack(size_df.fp), size_df.logS, cluster_list):
            urc_result_list.append([num_clus, len(test)])

urc_result_df = pd.DataFrame(urc_result_list, columns=["num_clus", "num_test"])

# Save the plot
sns.set_style('whitegrid')
plt.figure(figsize=(8, 5))
ax = sns.boxplot(x="num_clus", y="num_test", data=urc_result_df)
ax.set_xlabel("Number of Clusters")
ax.set_ylabel("Number of Test Set Molecules")
plt.savefig(os.path.join(output_dir, 'cluster_boxplot.png'), dpi=300)
plt.close()

print("Examining Done!")
print("Cluster plot for test set molecules saved")

############################# tSNE Plots for Comparing the Data Splits ############################

def get_tsne_coords(smiles_list):
    fp_gen = rdFingerprintGenerator.GetMorganGenerator()
    mol_list = [Chem.MolFromSmiles(x) for x in smiles_list]
    fp_list = [fp_gen.GetFingerprintAsNumPy(x) for x in mol_list]
    pca = PCA(n_components=50)
    pcs = pca.fit_transform(fp_list)
    tsne = TSNE(n_components=2, init='pca', learning_rate='auto')
    res = tsne.fit_transform(pcs)
    tsne_x = res[:, 0]
    tsne_y = res[:, 1]
    return tsne_x, tsne_y

# Use GroupKFoldShuffle to split the dataset using each method we defined above
split_list = ["random_cluster", "butina_cluster", "umap_cluster", "scaffold_cluster", "bitbirch_cluster"]
split_dict = {}
for split in split_list:
    kf = uru.GroupKFoldShuffle(n_splits=5, shuffle=True)
    for train_idx, test_idx in kf.split(df, groups=df[split]):
        split_dict[split] = [train_idx, test_idx]
        break

# Get tSNE 2D coordinates for the molecules and add them to the dataframe.
tsne_x, tsne_y = get_tsne_coords(df.SMILES)

df['tsne_x'] = tsne_x
df['tsne_y'] = tsne_y

tmp_df = df[['tsne_x', 'tsne_y']].copy()
tmp_df.reset_index(inplace=True)

# Add training and test set labels to tmp_df using the splitting methods
for split in split_list:
    tmp_df[split] = "train"
    _, test_idx = split_dict[split]
    for t in test_idx:
        tmp_df[split].at[t] = "test"

# Plot the tSNE results
sns.set_style('white')
sns.set_context('talk')
figure, axes = plt.subplots(1, 5, figsize=(15, 5), sharey=True)

for i, split in enumerate(split_list):
    scatter_ax = sns.scatterplot(x="tsne_x", y="tsne_y", data=tmp_df.query(f"{split} == 'train'"), ax=axes[i], color="lightblue", alpha=0.3, legend=False)
    sns.scatterplot(x="tsne_x", y="tsne_y", data=tmp_df.query(f"{split} == 'test'"), ax=axes[i], color="red", alpha=0.5, legend=False)
    scatter_ax.set_title(split)

# Save the plot
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'scatterplot.png'), dpi=300)
plt.close()

print("All cluster maps saved")

########################## Cross-validation with Model ##########################
# Cross-validation with LGBM and cluster methods
model_list = [("lgbm_morgan", LGBMMorganCountWrapper)]
group_list = [("butina", uru.get_butina_clusters), ("random", uru.get_random_clusters),
              ("scaffold", uru.get_bemis_murcko_clusters), ("umap", uru.get_umap_clusters),
              ("bitbirch", get_bitbirch_clusters)]
y = "logS"

result_df = uru.cross_validate(df, model_list, y, group_list, 5, 5)
outfile_name = os.path.join(output_dir, "all_clusters_results.csv")
result_df.to_csv(outfile_name, index=False)

# Display the first few rows of the result_df to understand its structure
print(result_df.head())

# Get the unique values from the 'group' column
unique_groups = result_df['group'].unique()

# Ensure the output directory exists
output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Split the DataFrame based on the 'group' column and save each group to a separate CSV file
for group_value in unique_groups:
    # Filter rows corresponding to the current group
    group_df = result_df[result_df['group'] == group_value]
    
    # Create the output file name in the specified format
    outfile_name = f"{output_dir}/{group_value}_group.csv"
    
    # Save the group DataFrame to a CSV file
    group_df.to_csv(outfile_name, index=False)
    print(f"Group '{group_value}'  to {outfile_name}")

# All groups have been saved
print("All groups have been saved to the 'output' directory.")

print("**********************************************************************")
print("Data Splitting Job Completed")

