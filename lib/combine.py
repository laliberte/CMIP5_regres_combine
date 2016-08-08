from __future__ import division, absolute_import, print_function

import numpy as np
import random

import multiprocessing
from numba import jit
import datetime

import scipy.stats as stats
import matplotlib.pyplot as plt

def combine_trends(regression_struct_list,model_list,num_procs=1):
    """
    Takes a lst of regressions obtained from regression_array  corresponding to a list
    of models of the form [(institute,model, ensemble),..] and a list of years axes [[1979,1980,...,2014],...]

    :param regression_struct_list: list of array-like
    :param model_list: list of tuples
    :param num_procs: number of processes to spawn
    :rtype: array-like
    """
    #Set the process pool size (should be less than or equal to number of compute cores):
    if num_procs==1:
        pool_map=map
    else:
        pool=multiprocessing.Pool(num_procs)
        pool_map=pool.map
    try:
        combined_regression_struct=np.concatenate(regression_struct_list,axis=0)

        axis_split=np.argmax(combined_regression_struct.shape[1:])+1
        return np.concatenate(pool_map( combine_trends_apply_vec,[ ( regression_struct, model_list) for 
                                                        regression_struct in np.array_split(np.concatenate(regression_struct_list,axis=0), num_procs,axis=axis_split)]),
                                                        axis=axis_split)
    finally:
        if num_procs>1:
            pool.close()
            pool.terminate()

def combine_trends_apply_vec(x):
    return combine_trends_apply(*x)

def combine_trends_apply(regression_struct,model_list):
    return np.apply_along_axis(combine_trends_one_dim,0,regression_struct,model_list)

def combine_pearsoncorr(regression_struct_list,model_list,num_procs=1):
    """
    Takes a lst of regressions obtained from regression_array  corresponding to a list
    of models of the form [(institute,model, ensemble),..] and a list of years axes [[1979,1980,...,2014],...]

    :param regression_struct_list: list of array-like
    :param model_list: list of tuples
    :param num_procs: number of processes to spawn
    :rtype: array-like
    """
    #Set the process pool size (should be less than or equal to number of compute cores):
    if num_procs==1:
        pool_map=map
    else:
        pool=multiprocessing.Pool(num_procs)
        pool_map=pool.map
    try:
        combined_regression_struct=np.concatenate(regression_struct_list,axis=0)
        axis_split=np.argmax(combined_regression_struct.shape[1:])+1
        return np.concatenate(pool_map( combine_pearsoncorr_apply_vec,[ ( regression_struct, model_list) for 
                                                        regression_struct in np.array_split(np.concatenate(regression_struct_list,axis=0), num_procs,axis=axis_split)]),
                                                        axis=axis_split)
    finally:
        if num_procs>1:
            pool.close()
            pool.terminate()

def combine_pearsoncorr_apply_vec(x):
    return combine_pearsoncorr_apply(*x)

def combine_pearsoncorr_apply(regression_struct,model_list):
    return np.apply_along_axis(combine_pearsoncorr_one_dim,0,regression_struct,model_list)

def combine_pearsoncorr_one_dim(regression_struct,model_list,sample_size=1000,sample_size_ensemble=1):
    """
    Combines pearson correlation along one dimension.
    """
    #Find the list of models after discarding the ensemble designator:
    diff_model_list=sorted(list(set([model_desc[:-1] for model_desc in model_list])))

    ensemble_list=[]
    for model_desc in diff_model_list:
        model_indices=[model_id for model_id, model_desc_full in enumerate(model_list) if model_desc==model_desc_full[:-1]]
        realization_ensemble=[]
        if regression_struct['p-value'][model_indices[0]]!=1.0: 
            for id in model_indices:
                correlation_model=np.reshape(
                                        stats.norm.rvs(
                                        #Fisher trasform:
                                         loc=np.arctanh(regression_struct[id]['r']),
                                         scale=1/np.sqrt(regression_struct[id]['npoints']-3),
                                         size=sample_size),(sample_size,1))
                realization_ensemble.append(correlation_model)
            ensemble_list.append(np.concatenate(realization_ensemble,axis=1))

    out_struct=np.zeros((1,),dtype=regression_struct.dtype)
    if len(ensemble_list)>0:
        out_struct['r'], out_struct['p-value']=additive_noise_model(ensemble_list,sample_size_ensemble,sample_size)
        #out_struct['r']=sum([ensemble.mean(-1).mean(-1) for ensemble in ensemble_list])/len(ensemble_list)
        #invert Fishert trasform:
        out_struct['r']=np.tanh(out_struct['r'])
    else:
        out_struct['r'], out_struct['p-value']=(0.0,1.0)
    return out_struct

def combine_trends_one_dim(regression_struct,model_list,sample_size=1000,sample_size_ensemble=1,plot=False):
    """
    Combines trends along one dimension.
    """
    #Find the list of models after discarding the ensemble designator:
    diff_model_list=sorted(list(set([model_desc[:-1] for model_desc in model_list])))
    
    #time_start=datetime.datetime.now()
    ensemble_list=[]
    for model_desc in diff_model_list:
        #Find indices corresponding to the model:
        model_indices=[model_id for model_id, model_desc_full in enumerate(model_list) if model_desc==model_desc_full[:-1]]
        realization_ensemble=[]
        if regression_struct['p-value'][model_indices[0]]!=1.0: 
            #if regressiohas some signifcance (e.g. it is not over land):
            for id in model_indices:
                #Use the t-distribution to find the probable trends: 
                trends_model=np.reshape(stats.t.rvs(regression_struct[id]['npoints']-2,
                                         loc=regression_struct[id]['slope'],
                                         scale=regression_struct[id]['stderr'],
                                         size=sample_size),(sample_size,1))
                realization_ensemble.append(trends_model)
            ensemble_list.append(np.concatenate(realization_ensemble,axis=1))

    out_struct=np.zeros((1,),dtype=regression_struct.dtype)
    if len(ensemble_list)>0:
        #apply additive noise model:
        out_struct['slope'], out_struct['p-value']=additive_noise_model(ensemble_list,sample_size_ensemble,sample_size,plot=plot)
        #out_struct['slope']=sum([ensemble.mean(-1).mean(-1) for ensemble in ensemble_list])/len(ensemble_list)
    else:
        #If ensemble_list is empty (happens when all values are missing (e.g. over land). slope is 0.0, with no confidence. 
        out_struct['slope'], out_struct['p-value']=(0.0,1.0)
    return out_struct

def additive_noise_model(ensemble_list,sample_size_ensemble,sample_size,plot=False):
    #Describe the noise model:
    noise_sample_size=1000
    noise_model_input=[None]*len(ensemble_list)
    for id,item in enumerate(ensemble_list):
        noise_model_input[id]=np.ma.mean(item,axis=0)
    noise_variance=stats.describe(generate_noise_model(noise_model_input,np.split(np.random.random(size=noise_sample_size),2)))[3]

    if plot:
        plt.figure()
        ax=plt.subplot(121)
        ax.hist(generate_noise_model(noise_model_input,np.split(np.random.random(size=noise_sample_size),2)),alpha=0.5)
        ax.hist(stats.norm.rvs(scale=np.sqrt(noise_variance),size=1000),color='r',alpha=0.5)

    #Select a random list of ensemble ids. There can be duplicates:
    if sample_size_ensemble==1:
        ensemble_list_ids=range(len(ensemble_list))
    else:
        ensemble_list_ids=choice_with_replacement(len(ensemble_list),np.random.random(sample_size_ensemble*len(ensemble_list)))
    #Create one argument for each:
    args_list=[(ensemble_list[ensemble_id],ensemble_list,noise_variance,np.random.random(size=4*sample_size)) for ensemble_id in ensemble_list_ids]
    
    #Compute the monte-carlo simulations. Subdivide into len(ensemble_list) chunks and average:
    mc_simulation=np.ravel(np.reshape(np.concatenate(map(additive_noise_model_single_model_vec,args_list),axis=1),
                                (sample_size,sample_size_ensemble,len(ensemble_list))).mean(-1))
    #Find p-value of null-hypothesis: 0.0 trends
    p_value=stats.percentileofscore(mc_simulation,0.0,kind='weak')/100.0
    if p_value>0.5: p_value=1.0-p_value

    if plot:
        ax=plt.subplot(122)
        ax.hist(mc_simulation,alpha=0.5)
        ax.axvline(np.ma.mean(mc_simulation),color='g',linestyle='--')
        ax.axvline(0.0,color='r')
        ax.axvline(np.mean([np.mean(ni) for ni in noise_model_input]),color='k',linestyle=':')
        plt.show()

    #Factor 2 to make two-sided:
    return np.ma.mean(mc_simulation),2*p_value

@jit
def additive_noise_model_single_model_vec(x):
    #Alias function for easy multiprocessing:
    return  additive_noise_model_single_model(*x)

@jit
def additive_noise_model_single_model(ensemble,ensemble_list,noise_variance,sample):
    ensemble_variance=np.ma.mean(ensemble,axis=0).var()
    #Laliberte, Howell, Kushner 2016 criterion on variance:
    noise_model_input=[None]*len(ensemble_list)
    for id,item in enumerate(ensemble_list):
        noise_model_input[id]=np.ma.mean(item,axis=0)
    #Split the sample in three:
    mr_sample,r_sample,nme_sample,nmr_sample=np.split(sample,4)
    if noise_variance>ensemble_variance:
        #scale the Swart et al. noise model by the variance deficit:
        mc_simulation=(
                       ( multiple_realization_distribution(ensemble,[mr_sample,r_sample])+
                         np.sqrt(1.0-ensemble_variance/noise_variance)*generate_noise_model(noise_model_input,[nme_sample,nmr_sample])) )
    else:
        #do not add noise model because ensemble is noisy enough:
        mc_simulation=(
                       multiple_realization_distribution(ensemble,[mr_sample,r_sample]))
    sample_size=len(mr_sample)
    return np.reshape(mc_simulation,(sample_size,1))

@jit
def multiple_realization_distribution(ensemble,sample):
    #Chose size number of realizations in the ensemble:
    realization_list_id=choice_with_replacement(ensemble.shape[1],sample[0])
    temp=np.zeros((len(sample[1]),))
    for id, (realization_id, sample_item) in enumerate(zip(realization_list_id,sample[1])):
        temp[id]=single_realization_distribution(ensemble[:,realization_id],[sample_item,]) 
    return temp

@jit
def single_realization_distribution(realization,sample):
    #Choose one element in the realization:
    return np.array(sample_with_replacement(np.squeeze(realization),sample))

@jit
def generate_noise_model(ensemble_list,sample):
    #Find the limited list of ensembles:
    ensemble_size=np.zeros((len(ensemble_list),))
    for id,ensemble in enumerate(ensemble_list):
        ensemble_size[id]=len(ensemble)

    ids_gt_one=np.arange(len(ensemble_list))[ensemble_size>1]
    if len(ids_gt_one)>0:
        #Split sample in two:
        ensemble_sample,realization_sample=sample

        ensemble_list_id=ids_gt_one[choice_with_replacement(len(ids_gt_one),ensemble_sample)]

        temp=np.zeros((len(realization_sample),))
        for id, (ens_id, sample_item) in enumerate(zip(ensemble_list_id,realization_sample)):
            temp[id]=noise_model(ensemble_list[ens_id],[sample_item,])
        return temp
    else:
        return np.zeros((len(sample[0]),))

@jit
def noise_model(ensemble,sample):
    realization_id=choice_with_replacement(len(ensemble),sample)[0]
    return np.sqrt(np.float(len(ensemble))/np.float((len(ensemble)-1)))*(ensemble[realization_id]-ensemble.mean())

#http://code.activestate.com/recipes/273085-sample-with-replacement/
@jit
def choice_with_replacement(length,sample):
    "Chooses size random indices (with replacement) from 0 to length"
    _int = int  # speed hack 
    result = [None] * len(sample)
    for i in xrange(len(sample)):
        j = _int(sample[i] * length)
        result[i] = xrange(length)[j]
    return result   

#http://code.activestate.com/recipes/273085-sample-with-replacement/
@jit
def sample_with_replacement(population,sample):
    "Chooses size random indices (with replacement) from 0 to length"
    _int = int  # speed hack 
    result = [None] * len(sample)
    for i in xrange(len(sample)):
        j = _int(sample[i] * len(population))
        result[i] = population[j]
    return result   
