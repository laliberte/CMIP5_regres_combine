from __future__ import division, absolute_import, print_function

import logging
import tempfile
import shutil
import netCDF4

import numpy as np
import random

import multiprocessing
from numba import jit
import datetime
import copy
import pandas as pd
import dask.dataframe as dd
import dask
import fractions

import scipy.stats as stats
#import matplotlib.pyplot as plt

fill_value=np.iinfo(np.int32).max


def exec_in_subprocess(func, *args):
    def worker(q):
        q.put(func(*args))
    q = multiprocessing.Queue()
    proc = multiprocessing.Process(target=worker,
                                   args=(q,))
    proc.start()
    proc.join()
    proc.terminate()
    return q.get()


def combine_from_input(input_file, extractor, field, simulations_desc,
                       num_procs=1, sample_size=1000, noise=True,
                       error=True):

    temp_dir = tempfile.mkdtemp(prefix='/dev/shm/')
    try:
        levels_names = exec_in_subprocess(load_normalize_and_dump_data,
                                          input_file, extractor,
                                          simulations_desc,
                                          num_procs, temp_dir)
        df_dask = dd.read_parquet(temp_dir)
        return combine_sorted(df_dask, field, simulations_desc, levels_names,
                              num_procs=num_procs, sample_size=sample_size,
                              noise=noise, error=True)
    finally:
        shutil.rmtree(temp_dir)


def load_normalize_and_dump_data(input_file, extractor, simulations_desc,
                                 num_procs, temp_dir):
    with netCDF4.Dataset(input_file) as dataset:
        df = extractor(dataset, simulations_desc)
    levels_names, df = normalize_data(df)
    chunks = get_dask_chunks(df, simulations_desc, num_procs=num_procs)
    dask_dataframe_though_parquet(temp_dir, df, chunks)
    return levels_names


def get_dask_chunks(df, simulations_desc, num_procs=1):
    npartitions = df.groupby(simulations_desc).count().max().max()
    chunks = int(df.shape[0] / npartitions)
    chunks_factor = min(df.shape[0] // (chunks * num_procs) + 1, 10000)
    return chunks*chunks_factor


def combine(df, field, simulations_desc,
            num_procs=1, sample_size=1000,
            noise=True, error=True):
    #Set the process pool size (should be less than or equal to number of compute cores):
    levels_names, df = normalize_data(df)
    return combine_sorted(df, field, simulaions_desc, levels_names,
                          num_procs=num_procs, sample_size=sample_size, noise=noise,
                          error=error)


def normalize_data(df):
    levels_names = df.index.names

    df = df.sort_index().reset_index()
    df['time'] = df['time'].apply(pd.tslib.normalize_date)
    return levels_names, df
                          

def combine_sorted(df, field, simulations_desc, levels_names,
                   num_procs=1, sample_size=1000, noise=True,
                   error=True):
    """
    Takes a lst of regressions obtained from regression_array  corresponding to a list
    of models of the form [(institute,model, ensemble),..] and a list of years axes [[1979,1980,...,2014],...]

    :param df: dataframe
    :param num_procs: number of processes to spawn
    :param sample_size: sample size to use
    :param noise: boolean to determine whether to use a noise model
    :param error: boolean to determine whether to use trend computation error
    :rtype: array-like
    """

    meta = {'hist': np.int, 'bin_edge_right': np.float, 'bin_edge_left': np.float,
            'p-value': np.float, 'slope': np.float, 'r-value': np.float,
            'xmean': np.float, 'intercept': np.float, 'nmod': np.int}

    if isinstance(df, pd.DataFrame):
        df_dask = dd.from_pandas(
                        df, chunks=get_dask_chunks(df, simulations_desc, num_procs=num_procs))
    else:
        df_dask = df

    with dask.set_options(get=dask.multiprocessing.get):
        df_out =(df_dask
                 .groupby(levels_names)
                 .apply(lambda x: combine_one_dim(x.dropna(), sample_size, field, simulations_desc,
                                                  levels_names, noise, error),
                        meta=meta)
                 .compute(num_workers=num_procs))

    df_out.columns = list(meta.keys())
    df_out.index.names = [name if name is not None else 'bins'
                          for name in df_out.index.names]
    return df_out


def dask_dataframe_though_parquet(temp_dir, df, chunks):
        df_dask = dd.from_pandas(df, chunksize=chunks)
        dd.to_parquet(temp_dir, df_dask,
                      compression='GZIP')
        return


def rvs_trends(npoints, loc, scale, size, error):
    try:
        if error:
            return stats.t.rvs(npoints - 2, loc=loc, scale=scale,
                               size=size)
        else:
            return np.array([loc for id in range(size)])
    except ValueError:
        return np.array([np.nan])


def rvs_pearsoncorr(npoints, loc, scale, size, error):
        #Fisher trasform:
    try:
        fisher_loc = np.arctanh(loc)
        fisher_scale = 1/np.sqrt(npoints - 3)
        if error:
            return stats.norm.rvs(loc=fisher_loc,
                                  scale=fisher_scale,
                                  size=size)
        else:
            return np.array([fisher_loc for id in range(size)])
    except RuntimeError:
        return np.array([np.nan])


def noise_weight_safe(x, y):
    def noise_weight(arg):
        return np.sqrt(np.maximum(1.0 - x / arg, 0.0))

    return np.piecewise(y, [y <= 0, y > 0],
                        [0.0, noise_weight])


def natural_meta(df):
    return {key: df[key].dtype for key in df.columns}


def combine_one_dim(df, sample_size, field, simulations_desc, levels_names, noise,
                    error):
    """
    Combines trends along one dimension.
    """
    df.drop(levels_names, axis=1, inplace=True)

    subsampling_size = sample_size / 10

    if noise:
        # Get the noise model:
        df_noise = compute_noise_model(df, sample_size, field, simulations_desc, subsampling_size=subsampling_size)
        if not df_noise.empty:
            df_noise.index = range(df_noise.shape[0])
            noise_variance = df_noise[field].var()
        else:
            noise_variance = 0.0

    if field == 'slope':
        rvs_func = rvs_trends
    elif field == 'r-value':
        rvs_func = rvs_pearsoncorr
    
    def generate_rvs(sub_df):
        series = sub_df.squeeze()
        stats_model = rvs_func(series['npoints'],
                               series[field],
                               series['stderr'],
                               subsampling_size,
                               error)
        sub_df = sub_df.sample(len(stats_model), replace=True)
        sub_df[field] = stats_model
        return sub_df.dropna()

    def maybe_add_noise_weight_and_subsample(sub_df):
        sub_variance = sub_df[field].var()
        if noise and noise_variance > 0.0:
            weight = noise_weight_safe(sub_variance, noise_variance)
            sub_df.loc[:, field + '_noise_weight'] = weight*np.ones(sub_df.shape[0])
        # Find rvs from each simulation:
        sub_df = (sub_df
                  .groupby(simulations_desc, group_keys=False)
                  .apply(generate_rvs))
        if sub_df.shape[0] > 0:
            # Each model ends up with the same sample size:
            sub_df = sub_df.sample(subsampling_size, replace=True)
        return sub_df

    df = (df
          .groupby(simulations_desc[:-1], group_keys=False)
          .apply(maybe_add_noise_weight_and_subsample))

    if df.empty:
        return create_output(df, field, sample_size, simulations_desc)
    else:
        df = df.sample(sample_size, replace=True)

    df.index = range(df.shape[0])

    if noise and noise_variance > 0.0:
        # At this point df and df_noise have the same shape:
        df[field] += df[field + '_noise_weight'] * df_noise[field]

    return create_output(df, field, sample_size, simulations_desc)


def create_output(df, field, sample_size, simulations_desc):
    nbins = int(np.ceil((sample_size)**(1.0/3.0)))
    df_out = pd.DataFrame(index=range(nbins))
    if df.empty:
        df_out['hist'] = np.nan
        df_out['bin_edge_right'] = [np.nan] * nbins
        df_out['bin_edge_left'] = [np.nan] * nbins
        df_out['p-value'] = np.nan
        df_out[field] = np.nan
        for name in ['slope', 'r-value', 'xmean', 'intercept']:
            df_out[name] = np.nan
        df_out['nmod'] = 0
        return df_out
    else:
        hist, bin_edges = np.histogram(df[field], bins=nbins)
        df_out['hist'] = hist
        df_out['bin_edge_right'] = bin_edges[1:]
        df_out['bin_edge_left'] = bin_edges[:-1]
        # compute p-value:
        p_value = stats.percentileofscore(df[field], 0.0, kind='weak') / 100.0
        if p_value > 0.5: p_value = 1.0 - p_value
        df_out['p-value'] = p_value
        df_out[field] = df[field].mean()
        for name in ['slope', 'r-value', 'xmean', 'intercept']:
            df_out[name] = df[name].mean()
        df_out['nmod'] = df.groupby(simulations_desc[:-1]).count().shape[0]
        return df_out


def compute_noise_model(df, sample_size, field, simulations_desc, subsampling_size=1000):
    def noise_model(sub_df):
        sub_df.loc[:,field] -= sub_df[field].mean()*np.ones(sub_df.shape[0])
        n = len(sub_df.index)
        if n > 1:
            sub_df.loc[:, field] *= np.sqrt(n/(n - 1))
        else:
            sub_df.loc[:, field] = np.nan
        sub_df = sub_df.dropna()
        if sub_df.shape[0] > 0:
            return sub_df.sample(subsampling_size, replace=True)
        else:
            return sub_df

    # Model variance
    df_noise = (df
                .groupby(simulations_desc[:-1], group_keys=False)
                .apply(noise_model))
    if df_noise.empty:
        return df_noise
    else:
        return (df_noise
                .sample(sample_size, replace=True))
