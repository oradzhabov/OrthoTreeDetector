import cv2
import numpy as np
import rasterio as rio
from rasterio.windows import Window
from .gabor import build_filters
from .ml import process_pyr
from .ml import apply_filter, get_mean_std


def get_ndvi(ortho, walls_cosine, use_old_approach):
    hsv = cv2.cvtColor(ortho.astype(np.uint8), cv2.COLOR_BGR2HSV)
    ortho = ortho/np.float32(255)
    ndvi = (2 * (ortho[..., 1] ** 2) - np.max(ortho[..., [0, 2]], axis=-1) * (ortho[..., 0] + ortho[..., 2])) / \
           (2 * (ortho[..., 1] ** 2) + np.max(ortho[..., [0, 2]], axis=-1) * (ortho[..., 0] + ortho[..., 2]))
    #ndvi = (ortho[..., 1] - ortho[..., 0] + ortho[..., 1] - ortho[..., 2]) / \
    #       (ortho[..., 1] + ortho[..., 0] + ortho[..., 1] + ortho[..., 2])
    ndvi[np.isnan(ndvi)] = -1
    ndvi[ndvi == np.inf] = -1
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
    conde = np.where(ndvi < 0)          # error-noise
    result[cond1] = 1
    #result[cond2] = 2
    result[cond3] = 3
    # result[conde] = 255
    return result, conde


def get_xy(proj_dir, wnd, use_old_approach, filters_nb, layers_nb):
    # note: np.float16 do not provide convergence in solvers with default iters
    dtype = np.float32

    ortho_fpath = f'{proj_dir}/tmp_ortho.tif'
    cosine_fpath = f'{proj_dir}/walls_cosine.tif'
    with rio.open(cosine_fpath) as src:
        w_cos = src.read(None, window=Window(*wnd))
    w_cos = np.moveaxis(w_cos, 0, -1).squeeze()
    cv2.imwrite('w_cos.png', w_cos * 255)
    with rio.open(ortho_fpath) as src:
        bgr = src.read(None, window=Window(*wnd))
    bgr = np.moveaxis(bgr, 0, -1)[..., [2, 1, 0]]
    hsv = cv2.cvtColor(bgr.astype(np.uint8), cv2.COLOR_BGR2HSV)
    cv2.imwrite('w.png', bgr)
    mask, mask_err = get_ndvi(bgr, w_cos, use_old_approach)
    cv2.imwrite('w_m.png', mask / 3 * 255)

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

    return X, Y, mask.shape, cond
