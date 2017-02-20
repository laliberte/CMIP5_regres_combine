from __future__ import division, absolute_import, print_function

import logging
_logger = logging.getLogger(__name__)

import numpy as np
import random

import multiprocessing
from numba import jit
import datetime
import copy

import scipy.stats as stats
#import matplotlib.pyplot as plt

fill_value=np.iinfo(np.int32).max

class _SharedRandomState():
    def __init__(self):
        self.str = 'MT19937'
        self.array = multiprocessing.Array('I',624)
        self.pos = multiprocessing.Value('i')
        self.has_gauss = multiprocessing.Value('i')
        self.cached_gaussian = multiprocessing.Value('f')
        self._lock = multiprocessing.Lock()

        self._states_list = ['str', 'array', 'pos', 'has_gauss', 'cached_gaussian'] 
        self.broadcast_state()

    def broadcast_state(self):
        val = np.random.get_state()
        for id, state in enumerate(self._states_list[1:]): 
            if 'value' in dir(getattr(self,state)):
                getattr(self,state).value = val[id+1]
            else:
                getattr(self,state)[:] = val[id+1]

    def __enter__(self):
        self._lock.acquire()
        val = [None] * len(self._states_list)
        val[0] = self.str
        for id, state in enumerate(self._states_list[1:]): 
            if 'value' in dir(getattr(self,state)):
                val[id+1] = getattr(self,state).value
            else:
                val[id+1] = getattr(self,state)[:]
        return val

    def __exit__(self, type, value, traceback):
        self.broadcast_state()
        self._lock.release()
        return 

class _PhonySharedRandomState():
    def __init__(self):
        return
    def __enter__(self):
        return self
    def __exit__(self, type, value, traceback):
        return

_random_state = _SharedRandomState()
#_random_state = _PhonySharedRandomState()

def combine_trends(regression_struct_list,model_list,num_procs=1,sample_size=1000):
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
        combined_regression_struct=np.concatenate(list(map(lambda x:x[np.newaxis,...], regression_struct_list)),axis=0)

        axis_split=np.argmax(combined_regression_struct.shape[1:])+1
        _logger.info('Begin combination')
        return np.concatenate(list(pool_map( combine_trends_apply_vec,[ ( regression_struct, model_list, sample_size) for 
                                                        regression_struct in np.array_split(combined_regression_struct,
                                                                                            np.minimum(num_procs,combined_regression_struct.shape[axis_split]),
                                                                                            axis =axis_split)])),
                                                        axis=axis_split)
    finally:
        if num_procs>1:
            pool.close()
            pool.terminate()

def combine_trends_apply_vec(x):
    return combine_trends_apply(*x)

def combine_trends_apply(regression_struct,model_list,sample_size):
    _logger.info('Applying along axis '+str(np.prod(regression_struct.shape[1:])))
    return np.apply_along_axis(combine_trends_one_dim, 0, regression_struct, model_list, sample_size)

def combine_pearsoncorr(regression_struct_list,model_list,num_procs=1,sample_size=1000):
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
        #combined_regression_struct=np.concatenate(regression_struct_list,axis=0)
        #combined_regression_struct=np.concatenate(map(lambda x:x[np.newaxis,...], regression_struct_list),axis=0)
        combined_regression_struct=np.concatenate(list(map(lambda x:x[np.newaxis,...], regression_struct_list),axis=0))

        axis_split=np.argmax(combined_regression_struct.shape[1:])+1
        return np.concatenate(list(pool_map( combine_pearsoncorr_apply_vec,[ ( regression_struct, model_list, sample_size) for 
                                                        regression_struct in np.array_split(combined_regression_struct, num_procs,axis=axis_split)])),
                                                        axis=axis_split)
    finally:
        if num_procs>1:
            pool.close()
            pool.terminate()

def combine_pearsoncorr_apply_vec(x):
    return combine_pearsoncorr_apply(*x)

def combine_pearsoncorr_apply(regression_struct,model_list,sample_size):
    _logger.info('Applying along axis')
    return np.apply_along_axis(combine_pearsoncorr_one_dim,0,regression_struct,model_list,sample_size)

def combine_pearsoncorr_one_dim(regression_struct,model_list,sample_size,sample_size_ensemble=1):
    """
    Combines pearson correlation along one dimension.
    """
    #Find the list of models after discarding the ensemble designator:
    diff_model_list=sorted(list(set([model_desc[:-1] for model_desc in model_list])))

    ensemble_list=[]
    with _random_state as s:
        np.random.set_state(s)
        for model_desc in diff_model_list:
            model_indices=[model_id for model_id, model_desc_full in enumerate(model_list) if model_desc==model_desc_full[:-1]]
            valid_model_indices = []
            for id in model_indices:
                if not ( np.isnan(regression_struct['p-value'][id]) or
                         regression_struct['p-value'][id] == 1.0 ): 
                    valid_model_indices.append(id)

            realization_ensemble=[]
            for id in valid_model_indices:
                if regression_struct[id]['npoints'] > 3:
                    correlation_model=np.reshape(
                                            stats.norm.rvs(
                                            #Fisher trasform:
                                             loc=np.arctanh(regression_struct[id]['r']),
                                             scale=1/np.sqrt(regression_struct[id]['npoints']-3),
                                             size=sample_size),(sample_size,1))
                    realization_ensemble.append(correlation_model)
            if len(realization_ensemble) > 0:
                ensemble_list.append(np.concatenate(realization_ensemble,axis=1))

    out_dtype=[('r',np.float),('p-value',np.float),
               ('bin_edge_left',np.float),('bin_edge_right',np.float),('hist',np.float)]
    nbins = np.ceil((sample_size*sample_size_ensemble)**(1.0/3.0))
    out_struct=np.zeros((nbins,),dtype=out_dtype)

    if len(ensemble_list)>0:
        #apply additive noise model:
        out_tuple=additive_noise_model(ensemble_list,sample_size_ensemble,sample_size,plot=plot)
    else:
        #If ensemble_list is empty (happens when all values are missing (e.g. over land). slope is 0.0, with no confidence. 
        out_tuple=(0.0,1.0,np.full((nbins,),np.nan),np.full((nbins,),np.nan),np.full((nbins,),np.nan))

    for name_id, name in enumerate(out_struct.dtype.names):
        out_struct[name][:] = out_tuple[name_id]
    return out_struct

def combine_trends_one_dim(regression_struct,model_list,sample_size, sample_size_ensemble=1,plot=False):
    """
    Combines trends along one dimension.
    """
    #t_start = datetime.datetime.now()
    #Find the list of models after discarding the ensemble designator:
    diff_model_list=sorted(list(set([model_desc[:-1] for model_desc in model_list])))
    
    #time_start=datetime.datetime.now()
    ensemble_list=[]
    intercept_list=[]
    xmean_list=[]
    with _random_state as s:
        _logger.debug('Setting random state '+str(s[1][0]))
        np.random.set_state(s)
        for model_desc in diff_model_list:

            #Find indices corresponding to the model:
            model_indices=[model_id for model_id, model_desc_full in enumerate(model_list) if model_desc==model_desc_full[:-1]]
            valid_model_indices = []
            for id in model_indices:
                if not ( np.isnan(regression_struct['p-value'][id])  or
                         regression_struct['p-value'][id] == 1.0 ): 
                    valid_model_indices.append(id)
                
            realization_ensemble=[]
            intercept_realization_ensemble=[]
            xmean_realization_ensemble=[]
            #if regression has some signifcance (e.g. it is not over land):
            for id in valid_model_indices:
                #Use the t-distribution to find the probable trends: 
                if regression_struct[id]['npoints'] > 2:
                    trends_model = np.reshape(stats.t.rvs(regression_struct[id]['npoints']-2,
                                             loc=regression_struct[id]['slope'],
                                             scale=regression_struct[id]['stderr'],
                                             size=sample_size),(sample_size,1))
                    realization_ensemble.append(trends_model)
                    intercept_realization_ensemble.append(regression_struct[id]['intercept'])
                    xmean_realization_ensemble.append(regression_struct[id]['xmean'])
            if len(realization_ensemble)>0:
                ensemble_list.append(np.concatenate(realization_ensemble,axis=1))
                intercept_list.append(np.mean(intercept_realization_ensemble))
                xmean_list.append(np.mean(xmean_realization_ensemble))
    
    out_dtype=[('slope',np.float),('p-value',np.float),
               ('bin_edge_left',np.float),('bin_edge_right',np.float),('hist',np.float),
               ('intercept',np.float),('xmean',np.float),('nmod',np.int)]

    nbins = np.ceil((sample_size*sample_size_ensemble)**(1.0/3.0))
    out_struct=np.zeros((nbins,),dtype=out_dtype)
    if len(ensemble_list)>0:
        #apply additive noise model:
        out_tuple = additive_noise_model(ensemble_list,sample_size_ensemble,sample_size,plot=plot)
        out_tuple+=(np.mean(intercept_list),np.mean(xmean_list),len(ensemble_list))
    else:
        #If ensemble_list is empty (happens when all values are missing (e.g. over land). slope is 0.0, with no confidence. 
        out_tuple = (0.0,1.0,np.full((nbins,),np.nan),np.full((nbins,),np.nan),np.full((nbins,),np.nan))
        out_tuple+=(np.nan,np.nan,0)

    for name_id, name in enumerate(out_struct.dtype.names):
        out_struct[name][:] = out_tuple[name_id]
    #_logger.info('One application '+str(datetime.datetime.now() - t_start))
    return out_struct

def additive_noise_model(ensemble_list,sample_size_ensemble,sample_size,plot=False):
    #Describe the noise model:
    noise_sample_size=1000
    noise_model_input=[None]*len(ensemble_list)
    for id,item in enumerate(ensemble_list):
        noise_model_input[id]=np.mean(item,axis=0)
    #with _random_state as s:
    #    _logger.debug('Setting random state '+str(s[1][0]))
    #np.random.set_state(s)
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
        #with _random_state as s:
        #    _logger.debug('Setting random state '+str(s[1][0]))
        #    np.random.set_state(s)
        ensemble_list_ids=choice_with_replacement(len(ensemble_list),np.random.random(sample_size_ensemble*len(ensemble_list)))
    #Create one argument for each but generate noise model only once:
    noise_model_input = list(map(lambda x: np.mean(x,axis=0),ensemble_list))
    #with _random_state as s:
    #    np.random.set_state(s)
    noise_model_instance = generate_noise_model(noise_model_input,np.split(np.random.random(size=2*sample_size),2))
    args_list=[(ensemble_list[ensemble_id],ensemble_list,noise_variance,np.random.random(size=2*sample_size),noise_model_instance) for ensemble_id in ensemble_list_ids]
    #args_list=[(ensemble_list[ensemble_id],ensemble_list,noise_variance,np.random.random(size=4*sample_size)) for ensemble_id in ensemble_list_ids]
    
    #Compute the monte-carlo simulations. Subdivide into len(ensemble_list) chunks and average:
    mc_simulation = np.ravel(np.reshape(np.concatenate(additive_noise_model_single_model_map(args_list),axis=1),
                                (sample_size,sample_size_ensemble,len(ensemble_list))).mean(-1))
    #Find p-value of null-hypothesis: 0.0 trends
    p_value = stats.percentileofscore(mc_simulation,0.0,kind='weak')/100.0
    if p_value>0.5: p_value=1.0-p_value

    if plot:
        ax=plt.subplot(122,sharex=ax,sharey=ax)
        ax.hist(mc_simulation,alpha=0.5)
        ax.axvline(np.mean(mc_simulation),color='g',linestyle='--')
        ax.axvline(0.0,color='r')
        ax.axvline(np.mean([np.mean(ni) for ni in noise_model_input]),color='k',linestyle=':')
        plt.show()

    #Create histogram:
    nbins = np.ceil((sample_size*sample_size_ensemble)**(1.0/3.0))
    #hist, bin_edges = np.histogram(mc_simulation,bins='fd')
    hist, bin_edges = np.histogram(mc_simulation,bins=nbins)

    #Factor 2 to make two-sided:
    return (np.mean(mc_simulation),2*p_value,
           _mk_cst_len(bin_edges[:-1],nbins), _mk_cst_len(bin_edges[1:],nbins), _mk_cst_len(hist,nbins))

@jit
def _mk_cst_len(x,n):
    y = np.full((n,),np.nan)
    y[:min(len(y),len(x))] = x[:min(len(y),len(x))]
    return y

@jit
def additive_noise_model_single_model_map(x):
    result = [None] * len(x)
    for id, item in enumerate(x):
        result[id] = additive_noise_model_single_model(*item)
    return result

@jit
def additive_noise_model_single_model(ensemble, ensemble_list, noise_variance, sample, noise_model_instance):
    ensemble_variance = np.mean(ensemble,axis=0).var()
    #noise_model_input = [None]*len(ensemble_list)
    #for id, item in enumerate(ensemble_list):
    #    noise_model_input[id] = np.mean(item,axis=0)
    #Split the sample in four:
    #mr_sample, r_sample, nme_sample, nmr_sample=np.split(sample,4)

    mr_sample, r_sample=np.split(sample,2)
    #Laliberte, Howell, Kushner 2016 criterion on variance:
    if noise_variance > ensemble_variance:
        #scale the Swart et al. noise model by the variance deficit:
        mc_simulation=(
                       ( multiple_realization_distribution(ensemble,[mr_sample,r_sample])+
                         np.sqrt(1.0-ensemble_variance/noise_variance)*noise_model_instance) )
                         #np.sqrt(1.0-ensemble_variance/noise_variance)*generate_noise_model(noise_model_input,[nme_sample,nmr_sample])) )
    else:
        #do not add noise model because ensemble is noisy enough:
        mc_simulation=(
                       multiple_realization_distribution(ensemble,[mr_sample,r_sample]))
    sample_size=len(mr_sample)
    return np.reshape(mc_simulation,(sample_size,1))

@jit
def multiple_realization_distribution(ensemble,sample):
    #Chose size number of realizations in the ensemble:
    realization_list_id = choice_with_replacement(ensemble.shape[1],sample[0])
    element_list_id = choice_with_replacement(ensemble.shape[0],sample[1])
    out = ensemble[np.array(element_list_id),np.array(realization_list_id)]
    return out

@jit
def generate_noise_model(ensemble_list,sample):
    #Find the limited list of ensembles:
    ensemble_size=np.zeros((len(ensemble_list),))
    for id,ensemble in enumerate(ensemble_list):
        ensemble_size[id]=len(ensemble)

    ids_gt_one=np.arange(len(ensemble_list))[ensemble_size>1]
    if len(ids_gt_one)>0:
        #Split sample in two:
        ensemble_sample, realization_sample=sample

        ensemble_list_id = ids_gt_one[choice_with_replacement(len(ids_gt_one),ensemble_sample)]

        temp=np.zeros((len(realization_sample),))
        for id, (ens_id, sample_item) in enumerate(zip(ensemble_list_id,realization_sample)):
            temp[id] = noise_model(ensemble_list[ens_id],[sample_item,])
        return temp
    else:
        return np.zeros((len(sample[0]),))

@jit
def noise_model(ensemble,sample):
    realization_id = choice_with_replacement(len(ensemble),sample)[0]
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
