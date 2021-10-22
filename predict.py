import pandas as pd
import numpy as np
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, DataStructs

from models.net import Net

# 预测各种数据结果

col = 'fruity'
filename = 'smiles_list_A_to_B_1'
smiles_file = 'data/211004/sts_v1/%s.csv' % filename
model_path = 'model/sts_stat.pkl'
proba_res = 'data/211004/sts_v1/%s_%s.csv' % (filename, col)


model = Net()
model.load_state_dict(torch.load(model_path))
model.to('cuda:0')

smiles = pd.read_csv(smiles_file, usecols=['smiles']).drop_duplicates().dropna()
# print(smiles)
# smiles = (smiles.loc[:49999, ['mols']])
# smiles.columns = ['smiles']
# print(smiles.shape)
def get_fp(smiles):
    mol = smiles.applymap(lambda x: Chem.MolFromSmiles(x))
    mol_clean = mol.applymap(lambda x: x if x else 0)
    mol_clean = mol_clean[~mol_clean['smiles'].isin([0])]
    mol_clean.index = [i for i in range(len(mol_clean))]
    mol_clean_smiles = mol_clean.applymap(lambda x: Chem.MolToSmiles(x))
    fp = mol_clean.applymap(lambda x: AllChem.GetMorganFingerprintAsBitVect(x, 2))
    # print(fp.shape)
    fp_bit_arr = []
    for i in fp.index:
        arr = np.zeros((0, ), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp.at[i, 'smiles'], arr)
        fp_bit_arr.append(arr)
    fp_bit_array = np.array(fp_bit_arr)

    X_fp_bit = pd.DataFrame(fp_bit_array)
    return mol_clean_smiles, X_fp_bit

def predict_res(X_fp_bit):
# 预测 filled fragment
    mols_input = torch.FloatTensor(X_fp_bit.values)
    mols_input = mols_input.cuda()
    pred_mols = model(mols_input)
    pred_mols.sigmoid_()
    pred_mols = pred_mols.cpu()


    pred_mols_arr = pred_mols.detach().numpy()
    pred_mols_df = pd.DataFrame(pred_mols_arr)
    pred_mols_df = pred_mols_df.applymap(lambda x: np.around(x, 4))

    return pred_mols_df

def output_csv(smiles, pred_mols_df, proba_res):
    smiles.index = [i for i in range(len(smiles))]
    res_df = pd.concat([smiles, pred_mols_df], axis=1)
    #print(smiles)
    #print(pred_mols_df)
    #print(res_df.shape)
    res_df.columns = ['smiles', col]
    # ref_df = res_df
    ref_df = res_df.sort_values(by=col, ascending=False)
    ref_df.index = [i for i in range(len(smiles))]

    ref_df.to_csv(proba_res, index=None)
    print('ok!')

'''prior, later = 0, 50000
for i in range(4):
    smiles_p = smiles[prior + i * 50000: later + i * 50000]
    mol_clean_smiles, X_fp_bit = get_fp(smiles_p)
    pred_mols_df = predict_res(X_fp_bit)
    proba_res = 'zinc_p%d.csv' % i
    output_csv(smiles_p, pred_mols_df, proba_res)
'''
mol_clean_smiles, X_fp_bit = get_fp(smiles)
pred_mols_df = predict_res(X_fp_bit)
output_csv(smiles, pred_mols_df, proba_res)

print('success!')
