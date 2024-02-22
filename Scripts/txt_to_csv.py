import pandas as pd

df = pd.read_table('Data/chembl_24_1_chemreps.txt.gz')
smiles = df['canonical_smiles'].values
to_drop = []
for i, sm in enumerate(smiles):
    if len(sm) > 100:
        to_drop.append(i)
    if df['chembl_id'][i] == 'CHEMBL1201364':
        to_drop.append(i)

df_dropped = df.drop(to_drop)
df_dropped = df_dropped.drop(['standard_inchi', 'standard_inchi_key'], axis=1)
L = len(df_dropped)
df_dropped.head()

df_dropped.to_csv('Data/chembl_24.csv', index=False)