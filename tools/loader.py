import os.path
import json
import cv2
import gc
import numpy as np
import rasterio as rio
from rasterio.windows import Window
import rasterio.mask
from skimage.exposure import equalize_hist
from .gabor import build_filters
from .ml import process_pyr
from .ml import apply_filter, get_mean_std, get_brief_space, get_lbp_space


def get_ndvi(ortho, walls_cosine, sloped_is_ground_or_tree):
    ortho = ortho/np.float32(255)
    ndvi = (2 * (ortho[..., 1] ** 2) - np.max(ortho[..., [0, 2]], axis=-1) * (ortho[..., 0] + ortho[..., 2])) / \
           (2 * (ortho[..., 1] ** 2) + np.max(ortho[..., [0, 2]], axis=-1) * (ortho[..., 0] + ortho[..., 2]))
    ndvi[np.isnan(ndvi)] = -255
    ndvi[np.isinf(ndvi)] = -255
    ndvi[np.isnan(walls_cosine)] = -255

    result = np.zeros(shape=ndvi.shape, dtype=np.uint8)
    cond_any_green = ndvi > 0.05
    cond_sloped = walls_cosine < np.cos(np.radians(45))

    # 0: background
    # 1: usual bench
    # 2: green bench
    # 3: flat part of trees
    # 4: sloped part of trees
    result[cond_sloped] = 1
    result[cond_any_green] = 3
    result[cond_any_green & cond_sloped] = 2 if sloped_is_ground_or_tree else 4
    conde = np.where(ndvi == -255)        # error-noise
    return result, conde


def get_xy(proj_dir, wnd, sloped_is_ground_or_tree, filters_nb, layers_nb, is_inference, mask_geojson=None, mask_class_ind=-1):
    # note: np.float16 do not provide convergence in solvers with default iters
    dtype = np.float32

    print(f'Process data from folder {proj_dir}', flush=True)
    ortho_fpath = f'{proj_dir}/tmp_ortho.tif'
    cosine_fpath = f'{proj_dir}/walls_cosine.tif'
    mask_fpath = f'{proj_dir}/{mask_geojson}'
    with rio.open(cosine_fpath) as src:
        w_cos = src.read(None, window=Window(*wnd))
        if src.nodata is not None:
            w_cos[w_cos == src.nodata] = np.nan
    w_cos = np.moveaxis(w_cos, 0, -1).squeeze()
    if is_inference:
        w_cos[np.isnan(w_cos)] = 1
    cv2.imwrite('w_cos.png', w_cos * 255)
    with rio.open(ortho_fpath) as src:
        bgr = src.read(None, window=Window(*wnd))
    bgr = np.moveaxis(bgr, 0, -1)[..., [2, 1, 0]]
    hsv = cv2.cvtColor(bgr.astype(np.uint8), cv2.COLOR_BGR2HSV)
    cv2.imwrite('w.png', bgr)
    mask, mask_err = get_ndvi(bgr, w_cos, sloped_is_ground_or_tree)
    cv2.imwrite('w_m.png', mask / 4 * 255)

    if os.path.exists(mask_fpath):
        print(f'Use file {mask_fpath} for labeling. CRS should be EPSG:4326', flush=True)
        with open(mask_fpath, 'r') as fin:
            tree_dict = json.load(fin)
            shapes = [feature["geometry"] for feature in tree_dict['features']]
        with rio.open(cosine_fpath) as src:
            for i in range(len(shapes)):
                shapes[i] = rio.warp.transform_geom("EPSG:4326", src.crs, shapes[i])
            out_image, transformed = rasterio.mask.mask(src, shapes, nodata=np.nan, filled=True)
            out_image = out_image[0]
            mask[~np.isnan(out_image)] = mask_class_ind
            mask[(w_cos < np.cos(np.radians(45))) & (~np.isnan(out_image))] = 4  # ATTENTION: expecting only tree mask so sloped only trees

    if filters_nb > 0:
        features = np.empty(shape=(w_cos.shape[0], w_cos.shape[1], layers_nb * filters_nb + 4), dtype=dtype)
        filters = build_filters(filters_nb)
        features[..., :layers_nb * filters_nb] = process_pyr(w_cos, lambda x: apply_filter(x, filters, dtype), layers_nb, dtype)
    elif filters_nb == 0:
        features = np.empty(shape=(w_cos.shape[0], w_cos.shape[1], 1 + 4), dtype=dtype)
        features[..., :1] = w_cos
    else:
        features = np.empty(shape=(w_cos.shape[0], w_cos.shape[1], layers_nb * 4 + 4), dtype=dtype)
        features[..., :layers_nb * 2] = process_pyr(w_cos, lambda x: get_mean_std(x, -filters_nb), layers_nb, dtype)
        features[..., layers_nb * 2:layers_nb * 3] = process_pyr(w_cos, lambda x: [get_mean_std(get_mean_std(x, -filters_nb)[1], -filters_nb)[1]], layers_nb, dtype)
        features[..., layers_nb * 3:layers_nb * 4] = process_pyr(w_cos, lambda x: [get_mean_std(get_mean_std(get_mean_std(x, -filters_nb)[1], -filters_nb)[1], -filters_nb)[1]], layers_nb, dtype)
        if False:
            f4 = process_pyr(hsv[..., 0], lambda x: get_mean_std(x, -filters_nb), layers_nb, dtype)
            f5 = process_pyr(hsv[..., 0], lambda x: [get_mean_std(get_mean_std(x, -filters_nb)[1], -filters_nb)[1]], layers_nb, dtype)
            f6 = process_pyr(hsv[..., 0], lambda x: [get_mean_std(get_mean_std(get_mean_std(x, -filters_nb)[1], -filters_nb)[1], -filters_nb)[1]], layers_nb, dtype)
            features = np.dstack([f1, f2, f3, f4, f5, f6])

    save_channels = False
    if save_channels:
        for i in range(features.shape[-1]):
            feature = features[..., i]
            cv2.imwrite(f'feature_{i}.png', (feature/np.nanmax(feature)*255).astype(np.uint8))

    mask[mask_err] = 255
    features[..., features.shape[-1] - 4:features.shape[-1] - 1] = bgr
    features[..., features.shape[-1] - 1] = hsv[..., 0]  # seems with bgr, hsv has only [0]-channel valuable. todo: Maybe [0,1]-channels have better accuracy? [2]-ch definetely defined earlier as mean in pyramid.
    X = features.reshape(-1, features.shape[-1])
    Y = mask.reshape(-1)

    # cond = np.where(np.all(X!=0, axis=-1))
    cond = np.where(~np.any(np.isnan(X), axis=-1) & (Y != 255))[0]
    X = X[cond]
    Y = Y[cond]

    gc.collect()

    return X, Y, mask.shape, cond
