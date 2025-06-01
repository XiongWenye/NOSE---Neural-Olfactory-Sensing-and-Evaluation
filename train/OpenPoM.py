import sys
import warnings
warnings.filterwarnings('ignore')

import deepchem as dc
import os
os.environ['TF_ENABLE_MLIR_OPTIMIZATIONS'] = '1'
from openpom.feat.graph_featurizer import GraphFeaturizer, GraphConvConstants
from openpom.utils.data_utils import get_class_imbalance_ratio
from openpom.models.mpnn_pom import MPNNPOMModel
import numpy as np
import random
import torch
import pandas as pd
from constants import *

seed = 2024
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed_all(seed)
# set_seeds(2024)

base_path = 'datasets'

def convert_todf_openpom(embeddings_dataset,cids,subjects=None,y=None):
    embeddings_dataset = pd.DataFrame(embeddings_dataset)
    embeddings_dataset['embeddings'] = embeddings_dataset.loc[:, 0:768].values.tolist()
    embeddings_dataset['CID'] = cids
    if subjects is not None:
        embeddings_dataset['subject'] = subjects
    if y is not None:
        y_dataset = pd.DataFrame(y)
        y_dataset['y'] = y_dataset.loc[:, 0:256].values.tolist()
    
        df = pd.concat([embeddings_dataset, y_dataset], axis=1)
        return df
    else:
        return embeddings_dataset
    
def embed_mols(input_file):
    # get dataset
    # print(os.getcwd())
    featurizer = GraphFeaturizer()
    smiles_field = 'nonStereoSMILES'
    loader = dc.data.CSVLoader(tasks=[],
                       feature_field=smiles_field,
                       featurizer=featurizer)
    dataset = loader.create_dataset(inputs=[input_file])
    
    embeddings=model.predict_embedding(dataset)
    return embeddings,dataset

def postproce_molembeddings(embeddings,index):
    # molecules_embeddings_penultimate = torch.cat(embeddings)
    df_molecules_embeddings = pd.DataFrame(embeddings, index=index)
    df_molecules_embeddings['Combined'] = df_molecules_embeddings.loc[:, '0':'767'].values.tolist()
    df_molecules_embeddings=df_molecules_embeddings.reset_index()
    return(df_molecules_embeddings)

def prepare_mols_helper(input_file,tasks,mol_type="nonStereoSMILES",index="cid"):
    featurizer = GraphFeaturizer()
    # smiles_field = 'nonStereoSMILES'
    loader = dc.data.CSVLoader(tasks=tasks,
                   feature_field=mol_type,
                   featurizer=featurizer
                          )
    dataset = loader.create_dataset(inputs=[input_file])
    df_mols = pd.read_csv(input_file)
    print(df_mols.columns)

    df_mols_embeddings_original=model.predict_embedding(dataset)
    return df_mols_embeddings_original,dataset

input_file = 'dataset/curated_GS_LF_merged_4983.csv'
df_gslf = pd.read_csv(input_file)

# get dataset
print(os.getcwd())
featurizer = GraphFeaturizer()
smiles_field = 'nonStereoSMILES'
loader = dc.data.CSVLoader(tasks=gs_lf_tasks,
                   feature_field=smiles_field,
                   featurizer=featurizer)
dataset = loader.create_dataset(inputs=[input_file])
n_tasks = len(dataset.tasks)
# get train valid test splits
randomstratifiedsplitter = dc.splits.RandomStratifiedSplitter()
train_dataset, valid_dataset, test_dataset = randomstratifiedsplitter.train_valid_test_split(dataset, frac_train = 0.8, frac_valid = 0.1, frac_test = 0.1, seed = seed)

train,valid,test=randomstratifiedsplitter.split(dataset, frac_train = 0.8, frac_valid = 0.1, frac_test = 0.1, seed = seed)

df_train_valid_test = pd.DataFrame({'main_idx': train + valid + test,
                   'split': ['train'] * len(train) + ['valid'] * len(valid) + ['test'] * len(test)})

for i in range(len(train)):
    if not np.array_equal(train_dataset.y[i],dataset.y[train[i]]):
        print(i)

for i in range(len(valid)):
    if not np.array_equal(valid_dataset.y[i],dataset.y[valid[i]]):
        print(i)

for i in range(len(test)):
    if not np.array_equal(test_dataset.y[i],dataset.y[test[i]]):
        print(i)


for i in range(len(train)):
    if not np.array_equal(train_dataset.y[i],df_gslf.iloc[train[i]].values[2:].tolist()):
        print(i)

for i in range(len(valid)):
    if not np.array_equal(valid_dataset.y[i],df_gslf.iloc[valid[i]].values[2:].tolist()):
        print(i)

for i in range(len(test)):
    if not np.array_equal(test_dataset.y[i],df_gslf.iloc[test[i]].values[2:].tolist()):
        print(i)
        
train_ratios = get_class_imbalance_ratio(train_dataset)
assert len(train_ratios) == n_tasks
learning_rate = 0.001
nb_epoch = 150

# initialize model
device_name = 'cuda'
model = MPNNPOMModel(n_tasks = n_tasks,
                            batch_size=128,
                            learning_rate=learning_rate,
                            class_imbalance_ratio = train_ratios,
                            loss_aggr_type = 'sum',
                            node_out_feats = 100,
                            edge_hidden_feats = 75,
                            edge_out_feats = 100,
                            num_step_message_passing = 5,
                            mpnn_residual = True,
                            message_aggregator_type = 'sum',
                            mode = 'classification',
                            number_atom_features = GraphConvConstants.ATOM_FDIM,
                            number_bond_features = GraphConvConstants.BOND_FDIM,
                            n_classes = 1,
                            readout_type = 'set2set',
                            num_step_set2set = 3,
                            num_layer_set2set = 2,
                            ffn_hidden_list= [392, 392],
                            ffn_embeddings = 256,
                            ffn_activation = 'relu',
                            ffn_dropout_p = 0.12,
                            ffn_dropout_at_input_no_act = False,
                            weight_decay = 1e-5,
                            self_loop = False,
                            optimizer_name = 'adam',
                            log_frequency = 32,
                            model_dir = '../examples/experiments',
                            device_name=device_name)

metric = dc.metrics.Metric(dc.metrics.roc_auc_score)
model.model_dir
model.load_from_pretrained(model)
embeddings_dataset=model.predict_embedding(dataset)
cids_gslf= df_gslf.index.values.tolist()
df_embeddings = convert_todf_openpom(embeddings_dataset,cids_gslf,None,dataset.y)
df_embeddings.to_csv('gslf_pom_embeddings.csv', index=False)
input_file_keller= 'dataset/keller2016_binarized.csv'
df_keller_temp=pd.read_csv(input_file_keller)
keller_tasks= df_keller_temp.columns.to_list()[5:]
cids_keller= df_keller_temp['CID'].values.tolist()
subjects_keller= df_keller_temp['Subject'].values.tolist()
df_mols_embeddings_original_keller,keller_dataset=prepare_mols_helper(input_file_keller,keller_tasks,index="CID")

df_embeddings_keller = convert_todf_openpom(df_mols_embeddings_original_keller,cids_keller,subjects_keller,keller_dataset.y)
df_embeddings_keller.to_csv('keller_pom_embeddings.csv', index=False)

