import dask.array as da
import numpy as np
import datetime
import inspect
from . import apply_along_n_axes


def reduce_along_axis_n_arrays(func,array_tuple,axis=0,num_chunks=1):
    """
    This function first chunks the arrays so that the reduction axis is
    entirely available in each chunk
    """

    #Chunk only the related dimensions:
    shape = array_tuple[max_array_id(array_tuple)].shape
    size = np.prod([shape[axis_id] if axis_id!=axis else 1 for axis_id in range(len(shape))])
    optimal_chunks = [np.ceil((size/np.float(num_chunks))**(1.0/(len(shape)-1))) 
                        if axis_id!=axis else shape[axis] for axis_id in range(len(shape))]

    return reduce_along_axis_n_chunked_arrays(func, broadcast_and_rechunk(array_tuple, chunks=optimal_chunks), axis=axis)

def max_array_id(array_tuple):
    dims_list=np.array([x.ndim for x in array_tuple])
    size_list=np.array([x.size for x in array_tuple])

    return np.arange(len(dims_list))[dims_list==np.max(dims_list)][np.argmax(size_list[dims_list==np.max(dims_list)])]

def reduce_along_axis_n_chunked_arrays(func,array_tuple,axis=0):
    return reduce_along_axis(func,structure_along_axis(array_tuple),axis=axis)

def reduce_along_axis(func,struct_array,axis=0):
    if 'dask' in inspect.getfile(make_like_reduction(func)):
        raise InputError('For some weird reason the reduction function must not contain the string "dask".')

    shape=struct_array.shape
    squeezed_shape=np.asarray(shape)[np.arange(len(shape))!=axis]
    return da.coarsen(make_like_reduction(func),
                                struct_array,
                                mapping_reduction_along_axes(struct_array,axes=axis)).reshape(squeezed_shape)

def structure_along_axis(array_tuple,axis=0):
    return da.map_blocks(structure,*broadcast_and_rechunk(array_tuple) )

def broadcast_and_rechunk(array_tuple, chunks=None):
    dest_array_id = max_array_id(array_tuple)
    dest_shape = array_tuple[dest_array_id].shape

    if chunks is None:
        chunks = array_tuple[dest_array_id].chunks
        
    broadcast_list=[None]*len(array_tuple)
    for array_id, array in enumerate(array_tuple):
        try:
            arr_chunks = array.chunks
            if arr_chunks != chunks:
                broadcast_list[array_id] = (da.broadcast_to(add_dims_to_broadcast(array,
                                                                               dest_shape),
                                                                               dest_shape)
                                                                               .rechunk(chunks))
            elif dest_shape != array.shape:
                broadcast_list[array_id] = (da.broadcast_to(add_dims_to_broadcast(array,
                                                                               dest_shape),
                                                                               dest_shape)
                                                                                )
            else:
                broadcast_list[array_id] = array
        except AttributeError:
            broadcast_list[array_id] = da.from_array(array, chunks=remove_chunks_from_shape(array, dest_shape, chunks))
            if dest_shape != array.shape:
                broadcast_list[array_id] = (da.broadcast_to(add_dims_to_broadcast(broadcast_list[array_id],
                                                                               dest_shape),
                                                                               dest_shape)
                                                                               .rechunk(chunks))
    return broadcast_list


def add_dims_to_broadcast(array,shape):
    src_shape=np.squeeze(array).shape
    dst_shape=[None]*len(shape)
    src_id=0
    for id, length in enumerate(shape):
        if (src_id < len(src_shape) and 
            length==src_shape[src_id]):
            dst_shape[id]=length
            src_id+=1
        else:
            dst_shape[id]=1
    if shape!=tuple(dst_shape):
        return array.reshape(tuple(dst_shape))
    else:
        return array

def remove_chunks_from_shape(array,shape,chunks):
    src_shape=np.squeeze(array).shape
    src_chunks=[None]*len(src_shape)
    src_id=0
    for length, chunk in zip(shape, chunks):
        if (src_id < len(src_shape) and 
            length==src_shape[src_id]):
            src_chunks[src_id] = chunk
            src_id+=1
    return src_chunks

def broadcast_shape_reduction_along_axes(x,axes=0):
    axes_mapping=mapping_reduction_along_axes(x,axes=axes)
    shape=np.ones((len(axes_mapping.keys(),)),dtype=np.int)
    for id in axes_mapping.keys():
        shape[id]=axes_mapping[id]
    return tuple(shape)

def mapping_reduction_along_axes(x,axes=0):
    axes_mapping={id:1 for id,shape in enumerate(x.shape) if not id in np.asarray(axes)}
    axes_mapping.update({id:shape for id,shape in enumerate(x.shape) if id in np.asarray(axes)})
    return axes_mapping

def make_like_reduction(func):
    def reduction_function(x,axis=0):
        return apply_along_n_axes.apply_1d_func_along_n_axes(func,axis,unstructure(x))
    return reduction_function

def unstructure(x):
    out=[None] * len(x.dtype.names)
    for name_id,name in enumerate(x.dtype.names):  
        out[name_id]=x[name]
    return out

def structure(*args):
    new_struct=np.empty(args[0].shape,dtype=','.join([get_dtype_string(x) for x in args]))
    for id,arg in enumerate(args):
        new_struct['f'+str(id)]=arg
    return new_struct

def get_dtype_string(array):
    return array.dtype.str
