import cv2
import numpy as np


def process(img, filters, dtype=np.float16):
    #accum = np.zeros_like(img)
    #accum = np.ones_like(img) * -np.inf
    stack = list()
    #accum_ind = np.zeros(shape=img.shape, dtype=np.uint16)
    for kern_ind, kern in enumerate(filters):
        fimg = cv2.filter2D(img, -1, kern).astype(dtype)
        #cond = np.where(fimg > accum)
        #accum_ind[cond] = kern_ind + 1
        #np.maximum(accum, fimg, accum)
        stack.append(fimg)
    #return np.stack(stack)
    return stack


def process_pyr(img, filters, layers_nb=3, dtype=np.float16):
    accum_ind_arr = list()
    shape_arr = list()
    for layer_ind in range(layers_nb):
        stack = process(img, filters, dtype)
        shape_arr.append(img.shape)
        img = cv2.pyrDown(img, dstsize=(img.shape[1] // 2, img.shape[0] // 2))
        accum_ind_arr.append(stack)
    for layer_ind in range(1, layers_nb):
        for layer_ind2 in range(layer_ind, layers_nb):
            for i in range(len(accum_ind_arr[layer_ind2])):
                accum_ind_arr[layer_ind2][i] = cv2.pyrUp(accum_ind_arr[layer_ind2][i].astype(np.float32),
                                                         (accum_ind_arr[layer_ind2][i].shape[1]*2,
                                                          accum_ind_arr[layer_ind2][i].shape[0]*2)).astype(dtype)
            ind = layer_ind2 - layer_ind
            if accum_ind_arr[layer_ind2][0].shape != shape_arr[ind]:
                for i in range(len(accum_ind_arr[layer_ind2])):
                    accum_ind_arr[layer_ind2][i] = cv2.copyMakeBorder(accum_ind_arr[layer_ind2][i].astype(np.float32), 0,
                                                shape_arr[ind][0] - accum_ind_arr[layer_ind2][i].shape[0], 0,
                                                shape_arr[ind][1] - accum_ind_arr[layer_ind2][i].shape[1], cv2.BORDER_REPLICATE).astype(dtype)
    accum_ind_arr = np.stack(accum_ind_arr)
    accum_ind_arr = accum_ind_arr.reshape((-1,) + accum_ind_arr.shape[2:])
    accum_ind_arr = np.rollaxis(accum_ind_arr, 0, 3)

    """
    for i in range(accum_ind_arr.shape[-1]):
        aaa = accum_ind_arr[..., i]
        cv2.imwrite(f'img{i}.png', (aaa/np.nanmax(aaa)*255).astype(np.uint8))
    """
    return accum_ind_arr
