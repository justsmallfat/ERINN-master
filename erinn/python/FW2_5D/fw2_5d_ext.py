from __future__ import division, absolute_import, print_function

import os
import shutil
import datetime
from functools import partial
from itertools import combinations

import numpy as np
import multiprocessing as mp
from tqdm import tqdm

from .fw2_5d import dcfw2_5D
from .fw2_5d import get_2_5Dpara
from .rand_synth_model import get_rand_model
from ..utils.io_utils import read_pkl, read_urf, read_config_file, write_pkl



def prepare_for_get_2_5d_para(config_file, return_urf=False):
    config = read_config_file(config_file)

    urf = config['geometry_urf']
    Tx_id, Rx_id, RxP2_id, coord, data = read_urf(urf)
    # Collect pairs id
    if np.all(np.isnan(data)):
        C_pair = [set(i) for i in combinations(Tx_id.flatten().tolist(), 2)]  # list of C_pair set
        P_pair = [set(i) for i in combinations(Rx_id.flatten().tolist(), 2)]  # list of P_pair set
        CP_pair = []
        for i in range(len(C_pair)):
            for j in range(len(P_pair)):
                if C_pair[i].isdisjoint(P_pair[j]):  # Return True if two sets have a null intersection.
                    CP_pair.append(sorted(C_pair[i]) + sorted(P_pair[j]))  # use sorted to convert set to list
        CP_pair = np.array(CP_pair, dtype=np.int64)
    else:
        CP_pair = data[:, :4].astype(np.int64)

    # Convert id to coordinate
    recloc = np.hstack((coord[CP_pair[:, 2] - 1, 1:4:2],
                        coord[CP_pair[:, 3] - 1, 1:4:2]))
    recloc[:, 1:4:2] = np.abs(recloc[:, 1:4:2])  # In urf, z is positive up. In fw25d, z is positive down.
    SRCLOC = np.hstack((coord[CP_pair[:, 0] - 1, 1:4:2],
                        coord[CP_pair[:, 1] - 1, 1:4:2]))
    SRCLOC[:, 1:4:2] = np.abs(SRCLOC[:, 1:4:2])  # In urf, z is positive up. In fw25d, z is positive down.

    # Collect pairs that fit the array configuration
    if config['array_type'] != 'all_combination':
        # Check if the electrode is on the ground
        at_ground = np.logical_and(np.logical_and(SRCLOC[:, 1] == 0, SRCLOC[:, 3] == 0),
                                   np.logical_and(recloc[:, 1] == 0, recloc[:, 3] == 0))
        SRCLOC = SRCLOC[at_ground, :]
        recloc = recloc[at_ground, :]
        AM = recloc[:, 0] - SRCLOC[:, 0]
        MN = recloc[:, 2] - recloc[:, 0]
        NB = SRCLOC[:, 2] - recloc[:, 2]
        # Check that the electrode arrangement is correct
        positive_idx = np.logical_and(np.logical_and(AM > 0, MN > 0), NB > 0)
        SRCLOC = SRCLOC[positive_idx, :]
        recloc = recloc[positive_idx, :]
        AM = AM[positive_idx]
        MN = MN[positive_idx]
        NB = NB[positive_idx]
        if config['array_type'] == 'Wenner_Schlumberger':
            # Must be an integer multiple?
            row_idx = np.logical_and(AM == NB, AM % MN == 0)
            SRCLOC = SRCLOC[row_idx, :]
            recloc = recloc[row_idx, :]
        elif config['array_type'] == 'Wenner':
            row_idx = np.logical_and(AM == MN, MN == NB)
            SRCLOC = SRCLOC[row_idx, :]
            recloc = recloc[row_idx, :]
        elif config['array_type'] == 'Wenner_Schlumberger_NonInt':
            row_idx = np.logical_and(AM == NB, AM >= MN)
            SRCLOC = SRCLOC[row_idx, :]
            recloc = recloc[row_idx, :]

    srcloc, srcnum = np.unique(SRCLOC, return_inverse=True, axis=0)
    srcnum = np.reshape(srcnum, (-1, 1))  # matlab index starts from 1, python index starts from 0

    array_len = max(coord[:, 1]) - min(coord[:, 1])
    srcloc[:, [0, 2]] = srcloc[:, [0, 2]] - array_len / 2
    recloc[:, [0, 2]] = recloc[:, [0, 2]] - array_len / 2
    dx = np.ones((config['nx'], 1))
    dz = np.ones((config['nz'], 1))

    if return_urf:
        return [[srcloc, dx, dz, recloc, srcnum],
                [Tx_id, Rx_id, RxP2_id, coord, data]]
    else:
        return srcloc, dx, dz, recloc, srcnum


def get_forward_para(config_file):
    config = read_config_file(config_file)
    srcloc, dx, dz, recloc, srcnum = prepare_for_get_2_5d_para(config)
    para_pkl = config['Para_pkl']
    num_k_g = config['num_k_g']

    if not os.path.isfile(para_pkl):
        print('Create Para for FW2_5D.')
        s = np.ones((config['nx'], config['nz']))
        Para = get_2_5Dpara(srcloc, dx, dz, s, num_k_g, recloc, srcnum)
        write_pkl(Para, para_pkl)
        config['Para'] = Para
    else:
        print('Load Para pickle file')
        Para = read_pkl(para_pkl)
        # Check if Para is in accordance with current configuration
        if 'Q' not in Para:
            print('No Q matrix in `Para` dictionary, creating a new one.')
            s = np.ones((config['nx'], config['nz']))
            Para = get_2_5Dpara(srcloc, dx, dz, s, num_k_g, recloc, srcnum)
            write_pkl(Para, para_pkl)
            config['Para'] = Para
        elif Para['Q'].shape[0] != srcnum.shape[0] \
                or Para['Q'].shape[1] != dx.size * dz.size \
                or Para['b'].shape[1] != srcloc.shape[0]:
            print('Size of Q matrix is wrong, creating a new one.')
            s = np.ones((config['nx'], config['nz']))
            Para = get_2_5Dpara(srcloc, dx, dz, s, num_k_g, recloc, srcnum)
            write_pkl(Para, para_pkl)
            config['Para'] = Para
        else:
            config['Para'] = Para

    config['srcloc'] = srcloc
    config['dx'] = dx
    config['dz'] = dz
    config['recloc'] = recloc
    config['srcnum'] = srcnum

    return config


def forward_simulation(sigma, config_file):

    config = read_config_file(config_file)
    if 'Para' not in config:
        config = get_forward_para(config)
    Para = config['Para']
    dx = config['dx']
    dz = config['dz']

    # Inputs: delta V/I (potential)
    sigma_size = (dx.size, dz.size)
    s = np.reshape(sigma, sigma_size)
    dobs, _ = dcfw2_5D(s, Para)

    return dobs.flatten()


def next_path(path_pattern, only_num=False):
    i = 1
    while os.path.exists(path_pattern % i):
        i = i * 2

    # Result lies somewhere in the interval (i/2..i]
    # We call this interval (a..b] and narrow it down until a + 1 = b
    a, b = (i // 2, i)
    while a + 1 < b:
        c = (a + b) // 2  # interval midpoint
        a, b = (c, b) if os.path.exists(path_pattern % c) else (a, c)

    if only_num:
        return b
    else:
        return path_pattern % b


def make_dataset(config_file, progressData):
    print('config_file : '+config_file)
    # parse config
    config = read_config_file(config_file)
    num_samples_train = int(config['num_samples'] * config['train_ratio'])
    num_samples_valid = int(config['num_samples']
                            * (config['train_ratio'] + config['valid_ratio'])
                            - num_samples_train)
    num_samples_test = config['num_samples'] - num_samples_train - num_samples_valid

    config = get_forward_para(config)
    userStop = False
    for dir_name, num_samples in (('train', num_samples_train),
                                  ('valid', num_samples_valid),
                                  ('test', num_samples_test)):
        dir = os.path.join(config['dataset_dir'], dir_name)
        if num_samples == 0:
            pass
        else:
            os.makedirs(dir, exist_ok=True)
            suffix_num = next_path(os.path.join(dir, 'raw_data_%s.pkl'), only_num=True)
            par = partial(_make_dataset, config=config, dir_name=dir, config_file=config_file)
            sigma_generator = get_rand_model(config, num_samples=num_samples)
            suffix_generator = range(suffix_num, suffix_num + num_samples)
            pool = mp.Pool(processes=mp.cpu_count(), maxtasksperchild=1)
            i = 1
            results = pool.imap_unordered(par, zip(sigma_generator, suffix_generator))
            for result in results:
                # print(f'result : {result} {i} ')
                progressData['generateData']['value'] = f'{dir_name} {i}/{num_samples} '
                progressData['generateData']['message'] = ''
                if 'true' == result:
                    progressData['generateData']['value'] = 'User Stop !'
                    progressData['generateData']['message'] = ''
                    progressData['log']['name'] = f'{datetime.datetime.now().strftime("%y-%m-%d %X")}'
                    progressData['log']['value'] = 'generateData'
                    progressData['log']['message'] = f'User Stop'
                    shutil.rmtree(dir)
                    shutil.rmtree(config['dataset_dir'])
                    userStop = True
                    break
                i=i+1
            pool.close()
            pool.join()
        if userStop:
            break
    if userStop:
        progressData['generateData']['value'] = 'User Stop !'
        progressData['generateData']['message'] = ''
    else:
        progressData['generateData']['value'] = 'Finish!'
        progressData['generateData']['message'] = ''

    return progressData
                # if is_favourable(result):
                #     break  # Stop loop and exit Pool context.
            #
            # for _ in tqdm(results = pool.imap_unordered(par, zip(sigma_generator, suffix_generator)),
            #               total=num_samples, desc=os.path.basename(dir)):

            #     i=i+1
            #     print(f'iiiii : {i}')
            #     config = read_config_file(config_file)
            #     if i>=100 :
            #
            #         pool.close()
            #         pool.join()
            #     pass


            # Serial version
            # suffix_num = next_path(os.path.join(dir_name, 'raw_data_%s.pkl'), only_num=True)
            # for sigma in tqdm(get_rand_model(config, num_samples=num_samples),
            #                   total=num_samples, desc=os.path.basename(dir_name)):
            #     dobs = forward_simulation(sigma, config)
            #     # pickle dump/load is faster than numpy savez_compressed(or save)/load
            #     pkl_name = os.path.join(dir_name, f'raw_data_{suffix_num}.pkl')
            #     write_pkl({'inputs': dobs, 'targets': 1 / sigma}, pkl_name)
            #     suffix_num += 1


def _make_dataset(zip_item, config, dir_name, config_file):
    sigma, suffix_num = zip_item
    dobs = forward_simulation(sigma, config)
    # pickle dump/load is faster than numpy savez_compressed(or save)/load
    pkl_name = os.path.join(dir_name, f'raw_data_{suffix_num:0>6}.pkl')
    write_pkl({'inputs': dobs, 'targets': np.log10(1 / sigma)}, pkl_name)
    tempConfig = read_config_file(config_file)
    return tempConfig['generateDataStop']
