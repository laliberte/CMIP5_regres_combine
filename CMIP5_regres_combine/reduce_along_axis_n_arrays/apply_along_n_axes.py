import numpy as np

def apply_1d_func_along_n_axes(func,axes,array_tuple):
    return apply_along_n_axes(pass_flattened_inputs(func),axes,array_tuple)

def pass_flattened_inputs(func):
    def function_wrapper(*args,**kwargs):
        return func(*[np.ravel(x) for x in args],**kwargs)
    return function_wrapper

#@jit(nogil=True)
def apply_along_n_axes(funcnd, axes, arr_tuple):
    """
    Adapted from np.apply_along_axis. Takes two arrays and apply a function along axes.
    """
    if np.isscalar(axes):
        axes=(axes,)

    arr1 = np.asarray(arr_tuple[0])
    nd = arr1.ndim
    for axis in axes:
        if axis < 0:
            axis += nd
        if (axis >= nd):
            raise ValueError("axis must be less than arr.ndim; axis=%d, rank=%d."% (axis, nd))
    ind = [0]*(nd-len(axes))
    i = np.zeros(nd, 'O')
    indlist = list(range(nd))
    for axis in axes:
        indlist.remove(axis)
        i[axis] = slice(None, None)
    i.put(indlist, ind)

    arr_tuple_selection=[None]*len(arr_tuple)
    getitem=tuple(i.tolist())
    for arr_id,arr in enumerate(arr_tuple):
        arr_tuple_selection[arr_id]=arr[getitem]
    res = funcnd(*arr_tuple_selection)

    #  if res is a number, then we have a smaller output array
    if np.isscalar(res) or len(res)==1:
        outshape = np.asarray(arr1.shape).take(indlist)
        outarr = np.zeros(outshape, np.asarray(res).dtype)
        outarr[tuple(ind)] = res

        Ntot = np.product(outshape)
        k = 1
        while k < Ntot:
            # increment the index
            ind[-1] += 1
            n = -1
            while (ind[n] >= outshape[n]) and (n > (1-nd)):
                ind[n-1] += 1
                ind[n] = 0
                n -= 1
            i.put(indlist, ind)
            getitem=tuple(i.tolist())
            for arr_id,arr in enumerate(arr_tuple):
                arr_tuple_selection[arr_id]=arr[getitem]
            #res = funcnd(*(tuple(arr_tuple_selection)+args), **kwargs)
            res = funcnd(*arr_tuple_selection)
            outarr[tuple(ind)] = res
            k += 1
        return outarr
    else:
        outshape = np.asarray(arr1.shape).take(indlist)
        Ntot = np.product(outshape)
        holdshape = outshape

        outshape = np.asarray(arr1.shape)
        out_indlist=range(len(outshape))
        for id,axis in enumerate(axes):
            if res.shape[id]>1:
                outshape[axis] = res.shape[id]
            else:
                #remove remaining axes:
                out_indlist.remove(axis)
        outshape=outshape[np.asarray(out_indlist)]

        outarr = np.zeros(outshape, np.asarray(res).dtype)
        out_indlist_specific=[None] * len(out_indlist)
        for id, idx in enumerate(out_indlist):
            out_indlist_specific[id]=i.tolist()[idx]
        outarr[tuple(out_indlist_specific)] = res
        k = 1
        while k < Ntot:
            # increment the index
            ind[-1] += 1
            n = -1
            while (ind[n] >= holdshape[n]) and (n > (1-nd)):
                ind[n-1] += 1
                ind[n] = 0
                n -= 1
            i.put(indlist, ind)
            getitem=tuple(i.tolist())
            for arr_id,arr in enumerate(arr_tuple):
                arr_tuple_selection[arr_id]=arr[getitem]
            #res = funcnd(*(tuple(arr_tuple_selection)+args), **kwargs)
            res = funcnd(*(tuple(arr_tuple_selection)))
            for id, idx in enumerate(out_indlist):
                out_indlist_specific[id]=i.tolist()[idx]
            outarr[tuple(out_indlist_specific)] = res
            k += 1
        return outarr 
