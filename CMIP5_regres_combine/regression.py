from __future__ import division, absolute_import, print_function

import numpy as np
import random

import multiprocessing
import datetime

import scipy.stats as stats
import matplotlib.pyplot as plt

import click
import netCDF4

import netcdf4_soft_links.remote_netcdf.timeaxis_mod as timeaxis_mod

import dask.array as da
import dask.multiprocessing 
import dask.async
from reduce_along_axis_n_arrays import reduce_along_axis_n_chunked_arrays

_dtype=[('slope',np.float),('intercept',np.float),('r-value',np.float),('p-value',np.float),('stderr',np.float),('npoints',np.int),('xmean',np.float)]

def trend_dataset_mp(ds,response_variable,num_procs=1):
    """
    Computes the (least-square) trends of array along the first dimension.
    len(time) must equal array.shape[0]
    """
    return regression_dataset_mp(ds,'time',response_variable,num_procs=num_procs)

def regression_dataset_mp(ds,predictor_variable,response_variable,num_procs=1):
    ds = regression_dataset(ds,response_variable,predictor_variable,num_chunks=num_procs)
    if num_procs==1:
        with dask.set_options(get=dask.async.get_sync):
            ds.load()
    else:
        pool=multiprocessing.Pool(num_procs)
        try:
            with dask.set_options(get=dask.multiprocessing.get, pool=pool):
                ds.load()
        finally:
            pool.terminate()
            pool.join()
    return ds

def regression_dataset(ds,variable,dim_to_regres,num_chunks=1):
    #Chunk only the related dimensions:
    size=np.prod([len(ds[variable].coords[dim]) if dim != dim_to_regres else 1 for dim in ds[variable].dims])
    chunk=np.ceil((size/np.float(num_chunks))**(1.0/(len(ds[variable].dims)-1)))
    chunking={dim:chunk for dim in ds[variable].dims if dim != dim_to_regres}
    chunking.update({dim_to_regres:len(ds[variable].coords[dim_to_regres])})

    return (ds[variable].chunk(chunking)
                          .reduce(regression_array,
                                 dim=dim_to_regres, allow_lazy=True,
                                 x=ds[dim_to_regres].values)
                          .to_dataset(name=variable+'_regres_'+dim_to_regres))

def regression_array(y,x=[],axis=0):
    """
    Compute the (least-square) regression of y against x along the first dimension.
    If len(x) equals y.shape[0], broadcasts x 

    :param time: array-like (len(time) must equal array.shape[0])
    :param array: array-like
    :rtype: array-like (shape is (1,)+array.shape[1:])
    """
    return reduce_along_axis_n_chunked_arrays(linregres,(x,y),axis=axis)

def linregres(x,y):
    """
    Apply stats.linregresss and organizes the output.
    """
    #return assign_to_struct(stats.linregress(x,y),dtype=dtype)
    return assign_to_struct(nanlinregress(x,y),dtype=_dtype)

def nanlinregress(x,y):
    mask = ~np.logical_or(np.isnan(x),np.isnan(y))
    npoints = np.count_nonzero(mask)
    if npoints>3:
        return stats.linregress(x[mask],y[mask])+(npoints,np.mean(x[mask]))
    else:
        return tuple([ np.nan for item in stats.linregress([0.0,0.5,1.0],[0.0,0.5,1.0])])+(npoints,np.mean(x[mask]))

def assign_to_struct(out,dtype=[('x',np.float),]):
    out_struct=np.empty((1,),dtype=dtype)
    for out_field_id,out_field in enumerate(dtype):
        out_struct[out_field[0]]=out[out_field_id]
    return out_struct
