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


def get_ndvi(ortho, walls_cosine, use_old_approach):
    hsv = cv2.cvtColor(ortho.astype(np.uint8), cv2.COLOR_BGR2HSV)
    ortho = ortho/np.float32(255)
    ndvi = (2 * (ortho[..., 1] ** 2) - np.max(ortho[..., [0, 2]], axis=-1) * (ortho[..., 0] + ortho[..., 2])) / \
           (2 * (ortho[..., 1] ** 2) + np.max(ortho[..., [0, 2]], axis=-1) * (ortho[..., 0] + ortho[..., 2]))
    #ndvi = (ortho[..., 1] - ortho[..., 0] + ortho[..., 1] - ortho[..., 2]) / \
    #       (ortho[..., 1] + ortho[..., 0] + ortho[..., 1] + ortho[..., 2])
    ndvi[np.isnan(ndvi)] = -255
    ndvi[np.isinf(ndvi)] = -255
    ndvi[np.isnan(walls_cosine)] = -255
    #
    # Cases:
    # 1. Strong NDVI could have any brightens. Light-green(almost white) trees, sunshine.
    # 2. Not strong NDVI could be applied with low brightens only. To avoid suppressing green ground-surface (copper
    #    slag), take in account the slope.
    """
    result = np.where((ndvi > 0.2) |
                      ((ndvi > 0.05) & (hsv[..., 2] <= 158) & (walls_cosine > np.cos(np.radians(45)))),
                      1, -1)
    """
    result = np.zeros(shape=ndvi.shape, dtype=np.uint8)
    relax = 1.0/1.0
    if use_old_approach:
        cond1 = np.where((ndvi > 0.05 * relax) & (hsv[..., 2] <= 158))  # old version
    else:
        cond1 = np.where((ndvi > 0.05 * relax) & (hsv[..., 2] <= 158) & (walls_cosine > np.cos(np.radians(45*relax)))) # soft green
    #cond2 = np.where((ndvi > 0.12 * relax) & (hsv[..., 2] <= 50) & (walls_cosine <= np.cos(np.radians(45*relax))))  # ?
    cond3 = np.where(ndvi > 0.2 * relax)  # strong green
    conde = np.where(ndvi == -255)        # error-noise
    result[cond1] = 1
    #result[cond2] = 2
    result[cond3] = 3
    return result, conde


def get_xy(proj_dir, wnd, use_old_approach, filters_nb, layers_nb, is_inference, mask_geojson=None, mask_class_ind=-1):
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
    cos_mask = ~np.isnan(w_cos)
    if is_inference:
        w_cos[np.isnan(w_cos)] = 1
    cv2.imwrite('w_cos.png', w_cos * 255)
    with rio.open(ortho_fpath) as src:
        bgr = src.read(None, window=Window(*wnd))
    bgr = np.moveaxis(bgr, 0, -1)[..., [2, 1, 0]]
    # bgr = (equalize_hist(bgr, mask=cos_mask) * 255).astype(np.uint8)
    hsv = cv2.cvtColor(bgr.astype(np.uint8), cv2.COLOR_BGR2HSV)
    cv2.imwrite('w.png', bgr)
    mask, mask_err = get_ndvi(bgr, w_cos, use_old_approach)
    cv2.imwrite('w_m.png', mask / 3 * 255)

    if os.path.exists(mask_fpath):
        print(f'Use file {mask_fpath} for labeling. CRS should be EPSG:4326', flush=True)
        with open(mask_fpath, 'r') as fin:
            tree_dict = json.load(fin)
            shapes = [feature["geometry"] for feature in tree_dict['features']]
        with rio.open(cosine_fpath) as src:
            for i in range(len(shapes)):
                shapes[i] = rio.warp.transform_geom("EPSG:32719", src.crs, shapes[i])
            out_image, transformed = rasterio.mask.mask(src, shapes, nodata=np.nan, filled=True)
            out_image = out_image[0]
            mask.fill(255)  # Do not take in account everything except this filter.
            mask[~np.isnan(out_image)] = mask_class_ind

    if filters_nb > 0:
        filters = build_filters(filters_nb)
        features = process_pyr(w_cos, lambda x: apply_filter(x, filters, dtype), layers_nb, dtype)
    elif filters_nb == 0:
        features = w_cos
    else:
        features = process_pyr(w_cos, lambda x: get_mean_std(x, -filters_nb), layers_nb, dtype)

    mask[mask_err] = 255
    features = np.dstack([features, bgr, hsv])
    # features = features[..., [0, 4, 5]]  # RFE suggests using: Cos,Hue,Saturation
    X = features.reshape(-1, features.shape[-1])
    Y = mask.reshape(-1)

    # cond = np.where(np.all(X!=0, axis=-1))
    cond = np.where(~np.any(np.isnan(X), axis=-1) & (Y != 255))
    X = X[cond]
    Y = Y[cond]

    gc.collect()

    return X, Y, mask.shape, cond
