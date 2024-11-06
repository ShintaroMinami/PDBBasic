from pdbecif.mmcif_tools import MMCIF2Dict
import numpy as np
import pandas as pd

# mmcif module
mmcif2dict = MMCIF2Dict()

def mmcif2dataframe(filepath, atoms=['N', 'CA', 'C', 'O'], models=None):
    models = [] if models==None else models
    models = [models] if type(models)!=list else models
    # read mmcif file
    cif_dict = mmcif2dict.parse(filepath)
    pdbid = list(cif_dict.keys())[0]
    # check polypeptides
    pp_entity = cif_dict[pdbid]['_entity_poly']
    if len(pp_entity['entity_id']) == 1:
        polypep, sequence = [pp_entity['entity_id']], [pp_entity['pdbx_seq_one_letter_code_can']]
    else:
        polypep, sequence = zip(*[(i,s) for t,i,s in zip(pp_entity['type'], pp_entity['entity_id'], pp_entity['pdbx_seq_one_letter_code_can']) if t=='polypeptide(L)'])
    # get data
    data = zip(
        cif_dict[pdbid]['_atom_site']['label_entity_id'],
        cif_dict[pdbid]['_atom_site']['pdbx_PDB_model_num'],
        cif_dict[pdbid]['_atom_site']['label_asym_id'],
        cif_dict[pdbid]['_atom_site']['label_seq_id'],
        cif_dict[pdbid]['_atom_site']['label_atom_id'],
        cif_dict[pdbid]['_atom_site']['label_comp_id'],
        cif_dict[pdbid]['_atom_site']['Cartn_x'],
        cif_dict[pdbid]['_atom_site']['Cartn_y'],
        cif_dict[pdbid]['_atom_site']['Cartn_z'],
        cif_dict[pdbid]['_atom_site']['occupancy'],
        cif_dict[pdbid]['_atom_site']['B_iso_or_equiv'],
    )
    # model, polypeptide
    if models:
        data = [d for d in data if int(d[1]) in models]
    data = [d for d in data if d[0] in polypep]
    data = [d[1:] for d in data if d[4] in atoms]
    # to Dataframe
    df = pd.DataFrame(data, columns=['model', 'chain', 'iaa_org', 'atom', 'resname', 'x', 'y', 'z', 'occupancy', 'bfactor'])
    df['coord'] = df.apply(lambda x: np.array([x['x'], x['y'], x['z']], dtype=float), axis=1)
    df = df[['model', 'chain', 'iaa_org', 'atom', 'resname', 'coord', 'occupancy', 'bfactor']]
    df['model'] = df['model'].astype(np.int)
    df['iaa_org'] = df['iaa_org'].astype(np.int)
    df['occupancy'] = df['occupancy'].astype(float)
    df['bfactor'] = df['bfactor'].astype(float)
    # return
    return df
