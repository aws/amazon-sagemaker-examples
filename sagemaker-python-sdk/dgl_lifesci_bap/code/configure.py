# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch

PotentialNet_PDBBind_core_pocket_random = { 
    'dataset': 'PDBBind',
    'subset': 'core',
    'load_binding_pocket': True,
    'random_seed': 123,
    'frac_train': 0.6,
    'frac_val': 0.2,
    'frac_test': 0.2,
    'batch_size': 200,
    'shuffle': False,
    'max_num_neighbors': 5,
    'distance_bins': [1.5, 2.5, 3.5, 4.5],
    'f_in': 44,
    'f_bond': 48, # has to be larger than f_in
    'f_gather': 48,
    'f_spatial': 48, # better to be the same as f_gather
    'n_rows_fc': [48, 24],
    'n_bond_conv_steps': 2,
    'n_spatial_conv_steps': 1,
    'dropouts': [0.25, 0.25, 0.25],
    'lr': 0.01,
    'num_epochs': 100,
    'wd': 1e-05,
    'metrics': ['r2', 'mae'],
    'split': 'random'
}

PotentialNet_PDBBind_core_pocket_scaffold = { 
    'dataset': 'PDBBind',
    'subset': 'core',
    'load_binding_pocket': True,
    'random_seed': 123,
    'frac_train': 0.6,
    'frac_val': 0.2,
    'frac_test': 0.2,
    'batch_size': 200,
    'shuffle': False,
    'max_num_neighbors': 5,
    'distance_bins': [1.5, 2.5, 3.5, 4.5],
    'f_in': 44,
    'f_bond': 48, # has to be larger than f_in
    'f_gather': 48,
    'f_spatial': 48, # better to be the same as f_gather
    'n_rows_fc': [48, 24],
    'n_bond_conv_steps': 2,
    'n_spatial_conv_steps': 1,
    'dropouts': [0.25, 0.25, 0.25],
    'lr': 0.01,
    'num_epochs': 100,
    'wd': 1e-05,
    'metrics': ['r2', 'mae'],
    'split': 'scaffold'
}

PotentialNet_PDBBind_core_pocket_stratified = { 
    'dataset': 'PDBBind',
    'subset': 'core',
    'load_binding_pocket': True,
    'random_seed': 123,
    'frac_train': 0.6,
    'frac_val': 0.2,
    'frac_test': 0.2,
    'batch_size': 200,
    'shuffle': False,
    'max_num_neighbors': 5,
    'distance_bins': [1.5, 2.5, 3.5, 4.5],
    'f_in': 44,
    'f_bond': 48, # has to be larger than f_in
    'f_gather': 48,
    'f_spatial': 48, # better to be the same as f_gather
    'n_rows_fc': [48, 24],
    'n_bond_conv_steps': 2,
    'n_spatial_conv_steps': 1,
    'dropouts': [0.25, 0.25, 0.25],
    'lr': 0.01,
    'num_epochs': 100,
    'wd': 1e-05,
    'metrics': ['r2', 'mae'],
    'split': 'stratified'
}

PotentialNet_PDBBind_refined_pocket_random = { 
    'dataset': 'PDBBind',
    'subset': 'refined',
    'load_binding_pocket': True,
    'remove_coreset_from_refinedset': True,
    'random_seed': 123,
    'frac_train': 0.85,
    'frac_val': 0.15,
    'frac_test': 0.,
    'batch_size': 200,
    'shuffle': False,
    'max_num_neighbors': 5,
    'distance_bins': [1.5, 2.5, 3.5, 4.5],
    'f_in': 44,
    'f_bond': 48, # has to be larger than f_in
    'f_gather': 48,
    'f_spatial': 48, # better to be the same as f_gather
    'n_rows_fc': [48, 24],
    'n_bond_conv_steps': 2,
    'n_spatial_conv_steps': 1,
    'dropouts': [0.25, 0.25, 0.25],
    'lr': 0.01,
    'num_epochs': 150,
    'wd': 1e-05,
    'metrics': ['r2', 'mae'],
    'split': 'random'
}

PotentialNet_PDBBind_refined_pocket_sequence = { 
    'dataset': 'PDBBind',
    'subset': 'refined',
    'load_binding_pocket': True,
    'random_seed': 123,
    'frac_train': 0.8,
    'frac_val': 0.2,
    'frac_test': 0.,
    'batch_size': 200,
    'shuffle': False,
    'max_num_neighbors': 5,
    'distance_bins': [1.5, 2.5, 3.5, 4.5],
    'f_in': 44,
    'f_bond': 48, # has to be larger than f_in
    'f_gather': 48,
    'f_spatial': 48, # better to be the same as f_gather
    'n_rows_fc': [48, 24],
    'n_bond_conv_steps': 2,
    'n_spatial_conv_steps': 1,
    'dropouts': [0.25, 0.25, 0.25],
    'lr': 0.01,
    'num_epochs': 200,
    'wd': 1e-05,
    'metrics': ['r2', 'mae'],
    'split': 'sequence'
}

PotentialNet_PDBBind_refined_pocket_structure = { 
    'dataset': 'PDBBind',
    'subset': 'refined',
    'load_binding_pocket': True,
    'random_seed': 123,
    'frac_train': 0.8,
    'frac_val': 0.2,
    'frac_test': 0.,
    'batch_size': 200,
    'shuffle': False,
    'max_num_neighbors': 5,
    'distance_bins': [1.5, 2.5, 3.5, 4.5],
    'f_in': 44,
    'f_bond': 48, # has to be larger than f_in
    'f_gather': 48,
    'f_spatial': 48, # better to be the same as f_gather
    'n_rows_fc': [48, 24],
    'n_bond_conv_steps': 2,
    'n_spatial_conv_steps': 1,
    'dropouts': [0.25, 0.25, 0.25],
    'lr': 0.01,
    'num_epochs': 200,
    'wd': 1e-05,
    'metrics': ['r2', 'mae'],
    'split': 'structure'
}

PotentialNet_PDBBind_refined_pocket_scaffold = { 
    'dataset': 'PDBBind',
    'subset': 'refined',
    'load_binding_pocket': True,
    'remove_coreset_from_refinedset': True,
    'random_seed': 123,
    'frac_train': 0.8,
    'frac_val': 0.2,
    'frac_test': 0.,
    'batch_size': 200,
    'shuffle': False,
    'max_num_neighbors': 5,
    'distance_bins': [1.5, 2.5, 3.5, 4.5],
    'f_in': 44,
    'f_bond': 48, # has to be larger than f_in
    'f_gather': 48,
    'f_spatial': 48, # better to be the same as f_gather
    'n_rows_fc': [48, 24],
    'n_bond_conv_steps':2,
    'n_spatial_conv_steps':1,
    'dropouts': [0.25, 0.25, 0.25],
    'lr': 0.01,
    'num_epochs': 200,
    'wd': 1e-05,
    'metrics': ['r2', 'mae'],
    'split': 'scaffold'
}

PotentialNet_PDBBind_refined_pocket_stratified = { 
    'dataset': 'PDBBind',
    'subset': 'refined',
    'load_binding_pocket': True,
    'remove_coreset_from_refinedset': True,
    'random_seed': 123,
    'frac_train': 0.8,
    'frac_val': 0.2,
    'frac_test': 0.,
    'batch_size': 200,
    'shuffle': False,
    'max_num_neighbors': 5,
    'distance_bins': [1.5, 2.5, 3.5, 4.5],
    'f_in': 44,
    'f_bond': 48, # has to be larger than f_in
    'f_gather': 48,
    'f_spatial': 48, # better to be the same as f_gather
    'n_rows_fc': [48, 24],
    'n_bond_conv_steps': 2,
    'n_spatial_conv_steps': 1,
    'dropouts': [0.25, 0.25, 0.25],
    'lr': 0.01,
    'num_epochs': 150,
    'wd': 1e-05,
    'metrics': ['r2', 'mae'],
    'split': 'stratified'
}

ACNN_PDBBind_core_pocket_random = {
    'dataset': 'PDBBind',
    'subset': 'core',
    'load_binding_pocket': True,
    'random_seed': 123,
    'frac_train': 0.8,
    'frac_val': 0.,
    'frac_test': 0.2,
    'batch_size': 24,
    'shuffle': False,
    'hidden_sizes': [32, 32, 16],
    'weight_init_stddevs': [1. / float(np.sqrt(32)), 1. / float(np.sqrt(32)),
                            1. / float(np.sqrt(16)), 0.01],
    'dropouts': [0., 0., 0.],
    'atomic_numbers_considered': torch.tensor([
        1., 6., 7., 8., 9., 11., 12., 15., 16., 17., 20., 25., 30., 35., 53.]),
    'radial': [[12.0], [0.0, 4.0, 8.0], [4.0]],
    'lr': 0.001,
    'wd': 0,
    'num_epochs': 120,
    'metrics': ['r2', 'mae'],
    'split': 'random'
}


ACNN_PDBBind_core_pocket_scaffold = {
    'dataset': 'PDBBind',
    'subset': 'core',
    'load_binding_pocket': True,
    'random_seed': 123,
    'frac_train': 0.8,
    'frac_val': 0.,
    'frac_test': 0.2,
    'batch_size': 24,
    'shuffle': False,
    'hidden_sizes': [32, 32, 16],
    'weight_init_stddevs': [1. / float(np.sqrt(32)), 1. / float(np.sqrt(32)),
                            1. / float(np.sqrt(16)), 0.01],
    'dropouts': [0., 0., 0.],
    'atomic_numbers_considered': torch.tensor([
        1., 6., 7., 8., 9., 11., 12., 15., 16., 17., 20., 25., 30., 35., 53.]),
    'radial': [[12.0], [0.0, 4.0, 8.0], [4.0]],
    'lr': 0.001,
    'wd': 0,
    'num_epochs': 170,
    'metrics': ['r2', 'mae'],
    'split': 'scaffold'
}

ACNN_PDBBind_core_pocket_stratified = {
    'dataset': 'PDBBind',
    'subset': 'core',
    'load_binding_pocket': True,
    'random_seed': 123,
    'frac_train': 0.8,
    'frac_val': 0.,
    'frac_test': 0.2,
    'batch_size': 24,
    'shuffle': False,
    'hidden_sizes': [32, 32, 16],
    'weight_init_stddevs': [1. / float(np.sqrt(32)), 1. / float(np.sqrt(32)),
                            1. / float(np.sqrt(16)), 0.01],
    'dropouts': [0., 0., 0.],
    'atomic_numbers_considered': torch.tensor([
        1., 6., 7., 8., 9., 11., 12., 15., 16., 17., 20., 25., 30., 35., 53.]),
    'radial': [[12.0], [0.0, 4.0, 8.0], [4.0]],
    'lr': 0.001,
    'wd': 0,
    'num_epochs': 110,
    'metrics': ['r2', 'mae'],
    'split': 'stratified'
}

ACNN_PDBBind_core_pocket_temporal = {
    'dataset': 'PDBBind',
    'subset': 'core',
    'load_binding_pocket': True,
    'random_seed': 123,
    'frac_train': 0.8,
    'frac_val': 0.,
    'frac_test': 0.2,
    'batch_size': 24,
    'shuffle': False,
    'hidden_sizes': [32, 32, 16],
    'weight_init_stddevs': [1. / float(np.sqrt(32)), 1. / float(np.sqrt(32)),
                            1. / float(np.sqrt(16)), 0.01],
    'dropouts': [0., 0., 0.],
    'atomic_numbers_considered': torch.tensor([
        1., 6., 7., 8., 9., 11., 12., 15., 16., 17., 20., 25., 30., 35., 53.]),
    'radial': [[12.0], [0.0, 4.0, 8.0], [4.0]],
    'lr': 0.001,
    'wd': 0,
    'num_epochs': 80,
    'metrics': ['r2', 'mae'],
    'split': 'temporal'
}

ACNN_PDBBind_refined_pocket_random = {
    'dataset': 'PDBBind',
    'subset': 'refined',
    'load_binding_pocket': True,
    'random_seed': 123,
    'frac_train': 0.8,
    'frac_val': 0.,
    'frac_test': 0.2,
    'batch_size': 24,
    'shuffle': False,
    'hidden_sizes': [128, 128, 64],
    'weight_init_stddevs': [0.125, 0.125, 0.177, 0.01],
    'dropouts': [0.4, 0.4, 0.],
    'atomic_numbers_considered': torch.tensor([
        1., 6., 7., 8., 9., 11., 12., 15., 16., 17., 19., 20., 25., 26., 27., 28.,
        29., 30., 34., 35., 38., 48., 53., 55., 80.]),
    'radial': [[12.0], [0.0, 2.0, 4.0, 6.0, 8.0], [4.0]],
    'lr': 0.001,
    'wd': 0,
    'num_epochs': 200,
    'metrics': ['r2', 'mae'],
    'split': 'random'
}

ACNN_PDBBind_refined_pocket_scaffold = {
    'dataset': 'PDBBind',
    'subset': 'refined',
    'load_binding_pocket': True,
    'random_seed': 123,
    'frac_train': 0.8,
    'frac_val': 0.,
    'frac_test': 0.2,
    'batch_size': 24,
    'shuffle': False,
    'hidden_sizes': [128, 128, 64],
    'weight_init_stddevs': [0.125, 0.125, 0.177, 0.01],
    'dropouts': [0.4, 0.4, 0.],
    'atomic_numbers_considered': torch.tensor([
        1., 6., 7., 8., 9., 11., 12., 15., 16., 17., 19., 20., 25., 26., 27., 28.,
        29., 30., 34., 35., 38., 48., 53., 55., 80.]),
    'radial': [[12.0], [0.0, 2.0, 4.0, 6.0, 8.0], [4.0]],
    'lr': 0.001,
    'wd': 0,
    'num_epochs': 350,
    'metrics': ['r2', 'mae'],
    'split': 'scaffold'
}

ACNN_PDBBind_refined_pocket_stratified = {
    'dataset': 'PDBBind',
    'subset': 'refined',
    'load_binding_pocket': True,
    'random_seed': 123,
    'frac_train': 0.8,
    'frac_val': 0.,
    'frac_test': 0.2,
    'batch_size': 24,
    'shuffle': False,
    'hidden_sizes': [128, 128, 64],
    'weight_init_stddevs': [0.125, 0.125, 0.177, 0.01],
    'dropouts': [0.4, 0.4, 0.],
    'atomic_numbers_considered': torch.tensor([
        1., 6., 7., 8., 9., 11., 12., 15., 16., 17., 19., 20., 25., 26., 27., 28.,
        29., 30., 34., 35., 38., 48., 53., 55., 80.]),
    'radial': [[12.0], [0.0, 2.0, 4.0, 6.0, 8.0], [4.0]],
    'lr': 0.001,
    'wd': 0,
    'num_epochs': 400,
    'metrics': ['r2', 'mae'],
    'split': 'stratified'
}

ACNN_PDBBind_refined_pocket_temporal = {
    'dataset': 'PDBBind',
    'subset': 'refined',
    'load_binding_pocket': True,
    'random_seed': 123,
    'frac_train': 0.8,
    'frac_val': 0.,
    'frac_test': 0.2,
    'batch_size': 24,
    'shuffle': False,
    'hidden_sizes': [128, 128, 64],
    'weight_init_stddevs': [0.125, 0.125, 0.177, 0.01],
    'dropouts': [0.4, 0.4, 0.],
    'atomic_numbers_considered': torch.tensor([
        1., 6., 7., 8., 9., 11., 12., 15., 16., 17., 19., 20., 25., 26., 27., 28.,
        29., 30., 34., 35., 38., 48., 53., 55., 80.]),
    'radial': [[12.0], [0.0, 2.0, 4.0, 6.0, 8.0], [4.0]],
    'lr': 0.001,
    'wd': 0,
    'num_epochs': 350,
    'metrics': ['r2', 'mae'],
    'split': 'temporal'
}

experiment_configures = {
    'ACNN_PDBBind_core_pocket_random': ACNN_PDBBind_core_pocket_random,
    'ACNN_PDBBind_core_pocket_scaffold': ACNN_PDBBind_core_pocket_scaffold,
    'ACNN_PDBBind_core_pocket_stratified': ACNN_PDBBind_core_pocket_stratified,
    'ACNN_PDBBind_core_pocket_temporal': ACNN_PDBBind_core_pocket_temporal,
    'ACNN_PDBBind_refined_pocket_random': ACNN_PDBBind_refined_pocket_random,
    'ACNN_PDBBind_refined_pocket_scaffold': ACNN_PDBBind_refined_pocket_scaffold,
    'ACNN_PDBBind_refined_pocket_stratified': ACNN_PDBBind_refined_pocket_stratified,
    'ACNN_PDBBind_refined_pocket_temporal': ACNN_PDBBind_refined_pocket_temporal,
    'PotentialNet_PDBBind_core_pocket_random' : PotentialNet_PDBBind_core_pocket_random,
    'PotentialNet_PDBBind_core_pocket_scaffold': PotentialNet_PDBBind_core_pocket_scaffold,
    'PotentialNet_PDBBind_core_pocket_stratified' : PotentialNet_PDBBind_core_pocket_stratified,
    'PotentialNet_PDBBind_refined_pocket_random' : PotentialNet_PDBBind_refined_pocket_random,
    'PotentialNet_PDBBind_refined_pocket_structure': PotentialNet_PDBBind_refined_pocket_structure,
    'PotentialNet_PDBBind_refined_pocket_sequence': PotentialNet_PDBBind_refined_pocket_sequence,
    'PotentialNet_PDBBind_refined_pocket_scaffold': PotentialNet_PDBBind_refined_pocket_scaffold,
    'PotentialNet_PDBBind_refined_pocket_stratified': PotentialNet_PDBBind_refined_pocket_stratified,
}

def get_exp_configure(exp_name):
    return experiment_configures[exp_name]
