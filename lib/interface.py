from __future__ import division, absolute_import, print_function

import numpy as np
import random

import multiprocessing
from numba import jit
import datetime
import copy

import scipy.stats as stats
import matplotlib.pyplot as plt

import click
import netCDF4

import netcdf4_soft_links.netcdf_utils as netcdf_utils
import netcdf4_soft_links.timeaxis_mod as timeaxis_mod
import netcdf4_soft_links.subset_utils as subset_utils

import xarray as xr

import dask.array as da
import dask.multiprocessing 
import dask.async
from reduce_along_axis_n_arrays import reduce_along_axis_n_chunked_arrays

#External
from . import regression, combine

@click.group()
def regres_and_combine():
    return

default_num_procs=1
@click.option('--num_procs',default=default_num_procs,help='')
@click.argument('output_file')
@click.argument('input_file')
@click.argument('variable')
@regres_and_combine.command()
def trend(variable,input_file,output_file,num_procs=default_num_procs):
    with netCDF4.Dataset(input_file) as dataset:
        with netCDF4.Dataset(output_file,'w') as output:
            #convert time to years axis:
            years_axis=get_years_axis_and_output_single_time(dataset,output)
            #compute trends:
            ds=xr.Dataset({variable:(dataset.variables[variable].dimensions,
                                        np.ma.filled(dataset.variables[variable][:],fill_value=np.nan))},
                            coords={dim:(dataset.variables[dim].dimensions,dataset.variables[dim][:]) for dim in dataset.variables[variable].dimensions})
            ds['time']=years_axis
            netcdf_utils.replicate_netcdf_file(dataset,output)
            netcdf_utils.replicate_netcdf_var_dimensions(dataset,output,variable)

            ds_trend=regression.trend_dataset_mp(ds,variable,num_procs=num_procs)
            write_structured_array(dataset,output,variable,ds_trend[variable+'_regres_time'].values)
    return

def get_years_axis_and_output_single_time(dataset,output):
    date_axis=netcdf_utils.get_time(dataset)
    min_year=np.min([date.year for date in date_axis])
    units='years since {0}-01-01 00:00:00'.format(min_year)
    years_axis=timeaxis_mod.Date2num(date_axis,units,netcdf_utils.netcdf_calendar(dataset))

    output.createDimension('time',size=1)
    temp_time=output.createVariable('time','d',('time',))
    units='days since {0}-01-01 00:00:00'.format(min_year)
    temp_time[:]=netCDF4.date2num(date_axis[0],units,netcdf_utils.netcdf_calendar(dataset))
    temp_time.units=units
    temp_time.calendar=netcdf_utils.netcdf_calendar(dataset)
    return years_axis


@click.option('--num_procs',default=default_num_procs,help='')
@click.argument('output_file')
@click.argument('input_file')
@regres_and_combine.command()
def combine_trend(input_file,output_file,num_procs=default_num_procs):
    with netCDF4.Dataset(input_file) as dataset:
        with netCDF4.Dataset(output_file,'w') as output:
            dataset_grp, regression_array_list, simulations_list = extract_regression_array(dataset)
            
            first_var_name=regression._dtype[0][0]
            output_grp = create_model_mean_tree(output)
            netcdf_utils.replicate_netcdf_var_dimensions(dataset_grp,output_grp,first_var_name)

            combined_trends=combine.combine_trends(regression_array_list,simulations_list,num_procs=num_procs)
            write_structured_array_combined(dataset_grp,output_grp,first_var_name,combined_trends)
    return

@click.option('--num_procs',default=default_num_procs,help='')
@click.argument('output_file')
@click.argument('input_file')
@regres_and_combine.command()
def combine_pearsoncorr(input_file,output_file,num_procs=default_num_procs):
    with netCDF4.Dataset(input_file) as dataset:
        with netCDF4.Dataset(output_file,'w') as output:
            dataset_grp, regression_array_list, simulations_list = extract_regression_array(dataset)

            first_var_name=regression._dtype[0][0]
            output_grp = create_model_mean_tree(output)
            netcdf_utils.replicate_netcdf_var_dimensions(dataset_grp,output_grp,first_var_name)

            combined_pearsoncorr=combine.combine_pearsoncorr(regression_array_list,simulations_list,num_procs=num_procs)
            write_structured_array_combined(dataset_grp,output_grp,first_var_name,combined_pearsoncorr)
    return

def create_model_mean_tree(output):
    grp_ins = output.createGroup('ALL')
    grp_ins.setncattr('level_name','institute')
    grp_mod = grp_ins.createGroup('MODEL-MEAN')
    grp_mod.setncattr('level_name','model')
    grp_ens = grp_mod.createGroup('r1i1p1')
    grp_ens.setncattr('level_name','ensemble')
    return grp_ens

def write_structured_array_combined(dataset,output,variable,struct_array):
    fill_value=1e20
    output.createDimension('bins',size=struct_array.shape[0])
    for var_name in struct_array.dtype.names:
        temp = np.ma.fix_invalid(struct_array[var_name], fill_value=fill_value)

        if var_name in ['slope','r','p-value']:
            temp = temp[0,...]
            dimensions = dataset.variables[variable].dimensions
        else:
            temp = np.rollaxis(temp,0,len(temp.shape))
            dimensions = dataset.variables[variable].dimensions + ('bins',)

        output.createVariable(var_name,'f',
                              dimensions,
                              fill_value=fill_value)[:]=temp
    return

def write_structured_array(dataset,output,variable,struct_array):
    fill_value=1e20
    for var_name in struct_array.dtype.names:
        temp = np.ma.fix_invalid(struct_array[var_name], fill_value=fill_value)
        output.createVariable(var_name,'f',
                            dataset.variables[variable].dimensions,
                            fill_value=fill_value)[:]=temp
    return

def extract_regression_array(dataset):
    simulations_list = []
    regression_array_list = []
    for institute in dataset.groups:
        grp_ins = dataset.groups[institute]
        for model in grp_ins.groups:
            grp_mod = grp_ins.groups[model]
            for ensemble in grp_mod.groups:
                grp_ens =  grp_mod.groups[ensemble]
                simulations_list.append(copy.copy((institute, model, ensemble)))
                regression_array = np.empty(grp_ens.variables[regression._dtype[0][0]].shape,dtype=regression._dtype)
                for var_name, dtype in regression._dtype:
                    regression_array[var_name] = grp_ens.variables[var_name][...]
                regression_array_list.append(copy.copy(regression_array))
    return grp_ens, regression_array_list, simulations_list
