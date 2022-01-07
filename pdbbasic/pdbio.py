import numpy as np
import torch
import pandas as pd
import urllib
import gzip
import tempfile
from .mmcif_utils import mmcif2dataframe


ONE_LETTER_AAS = "ARNDCQEGHILKMFPSTWYV"
THREE_LETTER_AAS = [
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS',
    'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO',
    'SER', 'THR', 'TRP', 'TYR', 'VAL'
    ]
one2id = {a:i for i,a in enumerate(ONE_LETTER_AAS)}
three2id = {a:i for i,a in enumerate(THREE_LETTER_AAS)}
one2three = {o:t for o,t in zip(ONE_LETTER_AAS, THREE_LETTER_AAS)}
three2one = {t:o for t,o in zip(THREE_LETTER_AAS, ONE_LETTER_AAS)}
BACKBONE_ATOMS = ['N', 'CA', 'C', 'O']
DOWNLOAD_URL = 'https://files.rcsb.org/download/'


def readpdb(file, with_info=False, CA_only=False, atoms=BACKBONE_ATOMS, model_id=1):
    atoms = ['CA'] if CA_only == True else atoms
    data = _read_pdb_file(file, atoms=atoms, models=model_id)
    xyz, info = _get_information(data, atoms=atoms)
    xyz = xyz.squeeze()
    return (xyz, info) if with_info == True else xyz


def readmmcif(file, with_info=False, CA_only=False, atoms=BACKBONE_ATOMS, model_id=1):
    atoms = ['CA'] if CA_only == True else atoms
    data = mmcif2dataframe(file, atoms=atoms, models=model_id)
    xyz, info = _get_information(data, atoms=atoms)
    xyz = xyz.squeeze()
    return (xyz, info) if with_info == True else xyz


def download(pdbid, with_info=False, CA_only=False, atoms=BACKBONE_ATOMS, model_id=1, url=DOWNLOAD_URL):
    fh_gzip = urllib.request.urlopen(url+pdbid+'.cif.gz')
    fh_unzip = gzip.GzipFile(mode='rb', fileobj=fh_gzip)
    with tempfile.NamedTemporaryFile(mode='w') as tmpf:
        tmpf.write(fh_unzip.read().decode().strip())
        output = readmmcif(tmpf.name, with_info=with_info, CA_only=CA_only, atoms=atoms, model_id=model_id)
    return output


def writepdb(backbone, info=None, sidechain=None, use_original_resnum=False, ignore_MODEL=False):
    backbone = np.array(backbone) if torch.is_tensor(backbone) else backbone
    info = [info] if len(backbone.shape) == 3 else info
    backbone = np.expand_dims(backbone, axis=0) if len(backbone.shape) == 3 else backbone
    length = backbone.shape[1]
    ignore_MODEL = True if backbone.shape[0] == 1 else ignore_MODEL
    if info:
        winfo = [_get_writeinfo(length,info=i,original=use_original_resnum) for i in info]
    else:
        winfo = [_get_writeinfo(length,original=use_original_resnum) for _ in range(backbone.shape[0])]
    line = ""
    for imodel, (bb, wi) in enumerate(list(zip(backbone, winfo))):
        if ignore_MODEL == False:
            line = line + "MODEL    {:4d}\n".format(imodel)
        resnum, aa3, chain = wi
        count = 0
        for atoms, resnum, aa3, chain in zip(bb, resnum, aa3, chain):
            for iatom, atom in enumerate(atoms):
                count += 1
                line_header = "ATOM"
                line_atomname = "{:6d}  {:2s}  {:3s}".format(count, BACKBONE_ATOMS[iatom], aa3)
                line_resnum = "{:s}{:4d}   ".format(chain, resnum)
                line_coord = "{:8.3f}{:8.3f}{:8.3f}".format(*atom)
                line = line + "{} {} {} {} \n".format(line_header, line_atomname, line_resnum, line_coord)
        if ignore_MODEL == False:
            line = line + "ENDMDL\n"
    return line


def _get_writeinfo(length, info=None, original=False):
    if info:
        resnum = info['resnum']
        aa3 = info['aa3']
        chain = info['chain']
    else:
        resnum = list(range(1, length+1))
        aa3 = ['GLY'] * length
        chain = ['A'] * length
    if original == False:
        resnum = list(range(1, length+1))
    return resnum, aa3, chain


def _get_information(data, atoms=BACKBONE_ATOMS):
    df = data
    df['id'] = df.apply(lambda x: f"{x['model']:03d}-{x['chain']}-{x['iaa_org']:05d}", axis=1)
    dfg = df.groupby('id')
    # format
    chain, res3, xyz, iorg, occu, bfac, mod = [], [], [], [], [], [], []
    for _,d in dfg:
        if len(d.coord.values)==len(atoms):
            chain.append(d.chain.values[0])
            res3.append(d.resname.values[0])
            xyz.append(np.stack(d.coord.values))
            iorg.append(d.iaa_org.values[0])
            occu.append(d.occupancy.values[0])
            bfac.append(d.bfactor.values[0])
            mod.append(d.model.values[0])
        else:
            continue
    # reshape
    xyz = np.stack(xyz).astype(np.float)
    res3 = np.array(res3)
    res1 = np.array([three2one.get(t,'X') for t in res3])
    iorg = np.array(iorg)
    occu = np.array(occu)
    bfac = np.array(bfac)
    model = np.array(mod)
    sequence = ''.join(res1)
    return xyz, {'model':model, 'chain':chain, 'aa1':res1, 'aa3':res3, 'resnum':iorg, 'sequence':sequence, 'occupancy':occu, 'bfactor':bfac}


def _read_pdb_file(file, atoms=BACKBONE_ATOMS, models=[1]):
    models = [] if models==None else models
    models = [models] if type(models)!=list else models    
    with open(file, "r") as fh:
        lines = fh.read().splitlines()
    # exists protein length
    data, data_now = [], []
    imodel_now = 1
    for l in lines:
        header = l[0:6]
        if "MODEL " in header:
            imodel = np.int(l.split()[-1])
            if imodel_now != imodel:
                if (imodel_now in models) | (len(models)==0):
                    data = data + data_now
            data_now = []
            imodel_now = imodel
        # main
        atomtype  = ''.join(l[12:16].split())
        if not atomtype in atoms: continue
        resname   = l[17:20]
        chain     = l[21:22]
        iaa_org   = l[22:27]
        coord     = [l[30:38], l[38:46], l[46:54]]
        occupancy = l[54:60] if len(l) > 60 else 0.0
        bfactor   = l[60:66] if len(l) > 66 else 0.0
        data_now.append({
            'model': imodel_now,
            'chain': chain.strip(),
            'iaa_org': np.int(iaa_org),
            'atom': atomtype.strip(),
            'resname': resname.strip(),
            'coord': np.array([np.float(c) for c in coord]),
            'occupancy': np.float(occupancy),
            'bfactor':  np.float(bfactor)
            })
    if (imodel_now in models) | (len(models)==0):
        data = data + data_now
    return pd.DataFrame(data)
