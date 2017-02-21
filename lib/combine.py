from __future__ import division, absolute_import, print_function

import logging

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

def combine_trends(df, num_procs=1, sample_size=1000):
    """
    Takes a lst of regressions obtained from regression_array  corresponding to a list
    of models of the form [(institute,model, ensemble),..] and a list of years axes [[1979,1980,...,2014],...]

    :param df: dataframe
    :param num_procs: number of processes to spawn
    :param sample_size: sample size to use
    :rtype: array-like
    """
    #Set the process pool size (should be less than or equal to number of compute cores):
    field = 'slope'
    simulations_desc = ['institute', 'model', 'ensemble']

    levels_names = df.index.names

    df = df.sort_index().reset_index()
    df['time'] = df['time'].apply(pd.tslib.normalize_date)

    npartitions = df.groupby(simulations_desc).count().max().max()
    chunks = int(df.shape[0] / npartitions)
    chunks_factor = df.shape[0] // (chunks * num_procs)

    df_dask = dd.from_pandas(df, chunksize=chunks*chunks_factor)
    #df_dask = df

    meta = {'hist': np.int, 'bin_edge_right': np.float, 'bin_edge_left': np.float,
            'p-value': np.float, 'slope': np.float, 'r-value': np.float,
            'xmean': np.float, 'intercept': np.float, 'nmod': np.int}
    with dask.set_options(get=dask.multiprocessing.get):
        df_out =(df_dask
                 .groupby(levels_names)
                 .apply(lambda x: combine_one_dim(x.dropna(), sample_size, field, simulations_desc,
                                                  levels_names),
                        meta=meta)
                 .compute(num_workers=num_procs))
    df_out.columns = list(meta.keys())
    df_out.index.names = [name if name is not None else 'bins'
                          for name in df_out.index.names]
    return df_out


def combine_pearsoncorr(df, num_procs=1, sample_size=1000):
    """
    Takes a lst of regressions obtained from regression_array  corresponding to a list
    of models of the form [(institute,model, ensemble),..] and a list of years axes [[1979,1980,...,2014],...]

    :param df: dataframe
    :param num_procs: number of processes to spawn
    :param sample_size: sample size to use
    :rtype: array-like
    """
    #Set the process pool size (should be less than or equal to number of compute cores):
    field = 'r-value'
    simulations_desc = ['institute', 'model', 'ensemble']

    return (df
            .groupby(level=['lon', 'lat', 'time'], group_keys=False)
            .apply(lambda x: combine_one_dim(x.dropna(), sample_size, field, simulations_desc)))


def rvs_trends(npoints, loc, scale, size):
    try:
        return stats.t.rvs(npoints - 2, loc=loc, scale=scale,
                           size=size)
    except ValueError:
        return np.array([np.nan])

def rvs_pearsoncorr(npoints, loc, scale, size):
        #Fisher trasform:
    try:
        return stats.norm.rvs(loc=np.arctanh(loc),
                              scale=1/np.sqrt(npoints - 3),
                              size=size)
    except RuntimeError:
        return np.array([np.nan])

def noise_weight_safe(x, y):
    def noise_weight(arg):
        return np.sqrt(np.maximum(1.0 - x / arg, 0.0))

    return np.piecewise(y, [y <= 0, y > 0],
                        [0.0, noise_weight])

def natural_meta(df):
    return {key: df[key].dtype for key in df.columns}

def combine_one_dim(df, sample_size, field, simulations_desc, levels_names,
                    subsampling_size=1000):
    """
    Combines trends along one dimension.
    """
    df.drop(levels_names, axis=1, inplace=True)

    # Get the noise model:
    df_noise = compute_noise_model(df, sample_size, field, simulations_desc, subsampling_size=1000)
    df_noise.index = range(df_noise.shape[0])
    noise_variance = df_noise[field].var()

    if field == 'slope':
        rvs_func = rvs_trends
    elif field == 'r-value':
        rvs_func = rvs_pearsoncorr
    
    def generate_rvs(sub_df):
        series = sub_df.squeeze()
        stats_model = rvs_func(series['npoints'],
                               series[field],
                               series['stderr'],
                               subsampling_size)
        sub_df = sub_df.sample(len(stats_model), replace=True)
        sub_df[field] = stats_model
        return sub_df.dropna()

    def add_noise_weight_and_subsample(sub_df):
        sub_variance = sub_df[field].var()
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
          .apply(add_noise_weight_and_subsample)
          .sample(sample_size, replace=True))
    df.index = range(df.shape[0])

    # At this point df and df_noise have the same shape:
    df[field] += df[field + '_noise_weight'] * df_noise[field]

    nbins = int(np.ceil((sample_size)**(1.0/3.0)))
    df_out = pd.DataFrame(index=range(nbins))
    hist, bin_edges = np.histogram(df[field], bins=nbins)
    df_out['hist'] = hist
    df_out['bin_edge_right'] = bin_edges[1:]
    df_out['bin_edge_left'] = bin_edges[:-1]
    # compute p-value:
    p_value = stats.percentileofscore(df[field], 0.0, kind='weak')/100.0
    if p_value>0.5: p_value=1.0-p_value
    # Make two-sided:
    p_value = 2*p_value
    df_out['p-value'] = p_value
    df_out[field] = df[field].mean()
    for name in ['slope', 'r-value', 'xmean', 'intercept']:
        df_out[name] = df[name].mean()
    df_out['nmod'] = df.groupby(simulations_desc).count().shape[0]
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
    return (df
            .groupby(simulations_desc[:-1], group_keys=False)
            .apply(noise_model)
            .sample(sample_size, replace=True))
