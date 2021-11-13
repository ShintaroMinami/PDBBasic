import numpy as np
import torch
import pandas as pd


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


def readpdb(file, backbone_only=False):
    data = _read_file(file)
    _check_atomnum(data)
    backbone = _get_backbone(data)
    # sidechain is on construction
    info = _get_information(data)
    return backbone if backbone_only==True else (backbone, info)


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


def _get_information(data):
    chain = np.array(data[data['atom'] == 'CA'].chain.values)
    res3 = np.array(data[data['atom'] == 'CA'].resname.values)
    res1 = np.array([three2one.get(t,'X') for t in res3])
    iorg = np.array(data[data['atom'] == 'CA'].iaa_org.values)
    sequence = ''.join(res1)
    return {'chain':chain, 'aa1':res1, 'aa3':res3, 'resnum':iorg, 'sequence':sequence}


def _get_backbone(data):
    backbone = [
        data[data['atom']=='N'].coord.values,
        data[data['atom']=='CA'].coord.values,
        data[data['atom']=='C'].coord.values,
        data[data['atom']=='O'].coord.values
        ]
    return np.array(list(zip(*backbone)))


def _read_file(file):
    with open(file, "r") as fh:
        lines = fh.read().splitlines()
    # exists protein length
    data = []
    for l in lines:
        header   = l[0:4]
        if not header == "ATOM": continue
        atomtype = l[12:16]
        resname  = l[17:20]
        chain    = l[21:22]
        iaa_org  = l[22:27]
        coord    = [l[30:38], l[38:46], l[46:54]]
        data.append({
            'header': header.strip(),
            'atom': atomtype.strip(),
            'resname': resname.strip(),
            'chain': chain.strip(),
            'iaa_org': np.int(iaa_org),
            'coord': np.array([np.float(c) for c in coord])
            })
    return pd.DataFrame(data)


def _check_atomnum(data):
    n  = len(data[data['atom']=='N']) 
    ca = len(data[data['atom']=='CA'])
    c  = len(data[data['atom']=='C'])
    o  = len(data[data['atom']=='O'])
    assert n == ca, 'different # of atoms: N ({}) != CA ({})'.format(n, ca)
    assert c == ca, 'different # of atoms: C ({}) != CA ({})'.format(c, ca)
    assert o == ca, 'different # of atoms: O ({}) != CA ({})'.format(o, ca)
    return

