# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import random
from functools import partial
from itertools import accumulate

import dgl
import numpy as np
import numpy.random as nrd
import torch
from dgl.data.utils import Subset
from dgllife.data import PDBBind
from dgllife.model import ACNN, PotentialNet
from dgllife.utils import (PN_graph_construction_and_featurization,
                           RandomSplitter, ScaffoldSplitter,
                           SingleTaskStratifiedSplitter)


def rand_hyperparams():
    """ Randomly generate a set of hyperparameters.
    Returns a dictionary of randomized hyperparameters.
    """
    hyper_params = {}
    hyper_params['f_bond'] = nrd.randint(70,120)
    hyper_params['f_gather'] = nrd.randint(80,129)
    hyper_params['f_spatial'] = nrd.randint(hyper_params['f_gather'], 129)
    hyper_params['n_bond_conv_steps'] = nrd.randint(1,3)
    hyper_params['n_spatial_conv_steps'] = nrd.randint(1,2)
    hyper_params['wd'] = nrd.choice([1e-7, 1e-5])
    hyper_params['dropouts'] = [nrd.choice([0, 0.25, 0.4]) for i in range(3)]
    hyper_params['n_rows_fc'] = [nrd.choice([16])]
    hyper_params['max_num_neighbors'] = nrd.randint(3, 13)
    return hyper_params

def set_random_seed(seed=0):
    """Set random seed.
    Parameters
    ----------
    seed : int
        Random seed to use. Default to 0.
    """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def load_dataset(args):
    """Load the dataset.
    Parameters
    ----------
    args : dict
        Input arguments.
    Returns
    -------
    train_set
        Train subset of the dataset.
    val_set
        Validation subset of the dataset.
    test_set
        Test subset of the dataset.
    """
    assert args['dataset'] in ['PDBBind'], 'Unexpected dataset {}'.format(args['dataset'])
    if args['dataset'] == 'PDBBind':
        if not args['pdb_path']:
            args['pdb_path'] = None
        if args['model'] == 'PotentialNet': 
            dataset = PDBBind(subset=args['subset'], pdb_version=args['version'], local_path=args['pdb_path'],
                                remove_coreset_from_refinedset=args['remove_coreset_from_refinedset'],
                                load_binding_pocket=args['load_binding_pocket'],
                                num_processes=args['num_workers'],
                                construct_graph_and_featurize=partial(PN_graph_construction_and_featurization, 
                                                                    distance_bins=args['distance_bins'],))
        elif args['model'] == 'ACNN':
            dataset = PDBBind(subset=args['subset'], pdb_version=args['version'], load_binding_pocket=args['load_binding_pocket'], local_path=args['pdb_path'])

        if args['split'] == 'sequence':
            train_set, val_set, test_set = [Subset(dataset, indices) for indices in dataset.agg_sequence_split]
        elif args['split'] == 'structure':
            train_set, val_set, test_set = [Subset(dataset, indices) for indices in dataset.agg_structure_split]

        elif args['split'] == 'random':
            train_set, val_set, test_set = RandomSplitter.train_val_test_split(
                dataset,
                frac_train=args['frac_train'],
                frac_val=args['frac_val'],
                frac_test=args['frac_test'],
                random_state=args['random_seed'])

        elif args['split'] == 'scaffold':
            train_set, val_set, test_set = ScaffoldSplitter.train_val_test_split(
                dataset,
                mols=dataset.ligand_mols,
                sanitize=False,
                frac_train=args['frac_train'],
                frac_val=args['frac_val'],
                frac_test=args['frac_test'])

        elif args['split'] == 'stratified':
            train_set, val_set, test_set = SingleTaskStratifiedSplitter.train_val_test_split(
                dataset,
                labels=dataset.labels,
                task_id=0,
                frac_train=args['frac_train'],
                frac_val=args['frac_val'],
                frac_test=args['frac_test'],
                random_state=args['random_seed'])

        elif args['split'] == 'temporal':
            years = dataset.df['release_year'].values.astype(np.float32)
            indices = np.argsort(years).tolist()
            frac_list = np.array([args['frac_train'], args['frac_val'], args['frac_test']])
            num_data = len(dataset)
            lengths = (num_data * frac_list).astype(int)
            lengths[-1] = num_data - np.sum(lengths[:-1])
            train_set, val_set, test_set = [
                Subset(dataset, list(indices[offset - length:offset]))
                for offset, length in zip(accumulate(lengths), lengths)]

        else:
            raise ValueError('Expect the splitting method '
                             'to be "random", "scaffold", "stratified" or "temporal", got {}'.format(args['split']))
        if args['frac_train'] > 0:
            train_labels = torch.stack([train_set.dataset.labels[i] for i in train_set.indices])
            train_set.labels_mean = train_labels.mean(dim=0)
            train_set.labels_std = train_labels.std(dim=0)

    return train_set, val_set, test_set

def collate(data):
    indices, ligand_mols, protein_mols, graphs, labels = map(list, zip(*data))
    if (type(graphs[0]) == tuple):
        bg1 = dgl.batch([g[0] for g in graphs])
        bg2 = dgl.batch([g[1] for g in graphs])
        bg = (bg1, bg2) # return a tuple for PotentialNet
    else:
        bg = dgl.batch(graphs)
        for nty in bg.ntypes:
            bg.set_n_initializer(dgl.init.zero_initializer, ntype=nty)
        for ety in bg.canonical_etypes:
            bg.set_e_initializer(dgl.init.zero_initializer, etype=ety)

    labels = torch.stack(labels, dim=0)
    return indices, ligand_mols, protein_mols, bg, labels

def load_model(args):
    assert args['model'] in ['ACNN', 'PotentialNet'], 'Unexpected model {}'.format(args['model'])
    if args['model'] == 'ACNN':
        model = ACNN(hidden_sizes=args['hidden_sizes'],
                     weight_init_stddevs=args['weight_init_stddevs'],
                     dropouts=args['dropouts'],
                     features_to_use=args['atomic_numbers_considered'],
                     radial=args['radial'])
    if args['model'] == 'PotentialNet': 
        model = PotentialNet(n_etypes=(len(args['distance_bins'])+ 5),
                             f_in=args['f_in'],
                             f_bond=args['f_bond'],
                             f_spatial=args['f_spatial'],
                             f_gather=args['f_gather'],
                             n_rows_fc=args['n_rows_fc'],
                             n_bond_conv_steps=args['n_bond_conv_steps'],
                             n_spatial_conv_steps=args['n_spatial_conv_steps'],
                             dropouts=args['dropouts'])
    return model
