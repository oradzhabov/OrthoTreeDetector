import cv2
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from skimage.feature import BRIEF
from skimage.transform import pyramid_gaussian, pyramid_expand


def apply_filter(img, filters, dtype=np.float16):
    stack = list()
    for kern_ind, kern in enumerate(filters):
        fimg = cv2.filter2D(img, -1, kern).astype(dtype)
        stack.append(fimg)
    return stack


def get_mean_std(img, ksize):
    # Input could be np.uint8, which will corrupt data by following logic
    img = img.astype(np.float32)

    mu = gaussian_filter(img, sigma=ksize/3)
    mu2 = gaussian_filter(img * img, sigma=ksize/3)

    sigma = mu2 - mu * mu
    sigma[np.where(sigma < 0.0)] = 0.0
    sigma = np.sqrt(sigma)
    return list([mu, sigma])


def get_lbp_space(img):
    from skimage.feature import daisy
    from skimage.feature import local_binary_pattern

    radius = 1
    n_points = 8 * radius
    #n_points = 24
    #radius = 8
    METHOD = 'uniform'
    lbp = local_binary_pattern(img, n_points, radius, METHOD)
    res = np.sum(np.unpackbits(lbp.astype(np.uint8)).reshape(lbp.shape[0], lbp.shape[1], 8), axis=-1)
    return [res]


def get_brief_space(img, descriptor_size, get_bit_counts_only=False):
    # https://stats.stackexchange.com/questions/89914/building-a-classification-model-for-strictly-binary-data
    descriptor_extractor = BRIEF(descriptor_size=descriptor_size)

    indices = np.meshgrid(np.arange(img.shape[0]), np.arange(img.shape[1]), sparse=False)
    keypoints = np.stack(indices).T.reshape(-1, 2).astype(np.float32)
    descriptor_extractor.extract(img, keypoints)
    result = np.zeros(shape=(len(keypoints), descriptor_size), dtype=bool)
    result[descriptor_extractor.mask] = descriptor_extractor.descriptors
    result = result.reshape(img.shape[0], img.shape[1], descriptor_size)

    if get_bit_counts_only:
        result = np.count_nonzero(result, axis=-1)
        return [result]

    result = np.moveaxis(result, -1, 0)  # channels first
    result = [i for i in result]
    return result


def process_pyr(img, generator, layers_nb=3, dtype=np.float16):
    accum_ind_arr = list()
    shape_arr = list()
    img_scaled = img.copy()
    for layer_ind in range(layers_nb):
        stack = generator(img_scaled)
        shape_arr.append(img_scaled.shape)
        scale = 2**(layer_ind + 1)
        img_scaled = cv2.resize(img, (img.shape[1] // scale, img.shape[0] // scale), interpolation=cv2.INTER_NEAREST)
        accum_ind_arr.append(stack)
    for layer_ind in range(1, layers_nb):
        for layer_ind2 in range(layer_ind, layers_nb):
            for i in range(len(accum_ind_arr[layer_ind2])):
                accum_ind_arr[layer_ind2][i] = cv2.resize(accum_ind_arr[layer_ind2][i].astype(np.float32),
                                                          (accum_ind_arr[layer_ind2][i].shape[1]*2,
                                                           accum_ind_arr[layer_ind2][i].shape[0]*2),
                                                          interpolation=cv2.INTER_CUBIC).astype(dtype)  # ATTENTION: Features approximation
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
