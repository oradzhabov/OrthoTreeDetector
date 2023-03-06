import os.path
import json
import cv2
import gc
import numpy as np
import rasterio as rio
from rasterio.windows import Window
from rasterio import Affine
from rasterio.enums import Resampling
import rasterio.mask
from skimage.exposure import equalize_hist
from .gabor import build_filters
from .ml import process_pyr
from .ml import apply_filter, get_mean_std, get_brief_space, get_lbp_space, apply_accum_filter


def count_dist_peaks(series, bins, prominence, width):
    from scipy.signal import find_peaks
    count, division = np.histogram(series, bins=bins)
    peaks, props = find_peaks(count, prominence=prominence, width=width)
    return peaks


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


def get_ndvi_new(ortho, walls_cosine):
    hsv = cv2.cvtColor(ortho.astype(np.uint8), cv2.COLOR_BGR2HSV)
    ortho = ortho/np.float32(255)
    """
    ndvi = (2 * (ortho[..., 1] ** 2) - np.max(ortho[..., [0, 2]], axis=-1) * (ortho[..., 0] + ortho[..., 2])) / \
           (2 * (ortho[..., 1] ** 2) + np.max(ortho[..., [0, 2]], axis=-1) * (ortho[..., 0] + ortho[..., 2]))
    """
    ndvi = (ortho[..., 1] - ortho[..., 2] + ortho[..., 1] - ortho[..., 0]) / \
           (ortho[..., 1] + ortho[..., 2] + ortho[..., 1] + ortho[..., 0])
    ndvi[np.isnan(ndvi)] = -255
    ndvi[np.isinf(ndvi)] = -255
    ndvi[np.isnan(walls_cosine)] = -255

    result = np.zeros(shape=ndvi.shape, dtype=np.uint8)
    cond_any_green = ndvi > 0  # 0.0
    cond_sloped = walls_cosine < np.cos(np.radians(45))

    # 0: background
    # 1: usual bench
    # 2: green bench
    # 3: flat part of trees
    # 4: sloped part of trees
    result[cond_sloped] = 1
    result[cond_any_green] = 3
    if True:
        from sklearn.cluster import KMeans
        from sklearn import preprocessing

        # h = hsv[..., 2] / 255
        h = hsv[..., 0] / 180
        # data = np.dstack([ndvi, h])
        data = ndvi[..., np.newaxis]
        # scaler = preprocessing.StandardScaler()  # seems this better for logistic-reg rather MinMaxScaler(even shifted)
        # data = scaler.fit_transform(data.reshape(-1, data.shape[-1])).reshape(data.shape)

        mean_green_ndvi = np.mean(data[~(ndvi == -255.0) & cond_any_green], axis=0)

        kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(data[cond_sloped & ~(ndvi == -255.0)])
        ndvi_sloped_centers = kmeans.cluster_centers_
        print(f'ndvi_sloped_centers: {ndvi_sloped_centers}')
        labels = kmeans.labels_
        if True:
            ndvi_sloped_centers = ndvi_sloped_centers[..., 0]
            map = [2, 4] if ndvi_sloped_centers[0] < ndvi_sloped_centers[1] else [4, 2]
            result[cond_sloped & ~(ndvi == -255.0)] = np.array(map)[labels]
        else:
            if False:
                ndvi_sloped_centers = ndvi_sloped_centers[..., 0]
                result[(ndvi > np.max(ndvi_sloped_centers)) & cond_sloped & ~(ndvi == -255.0)] = 4
                result[(ndvi < np.min(ndvi_sloped_centers)) & cond_sloped & ~(ndvi == -255.0)] = 2
                result[(ndvi >= np.min(ndvi_sloped_centers)) & (ndvi <= np.max(ndvi_sloped_centers)) & cond_sloped & ~(ndvi == -255.0)] = 1
            else:
                from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
                t4 = data[(ndvi > np.max(ndvi_sloped_centers)) & cond_sloped & ~(ndvi == -255.0)]
                t2 = data[(ndvi < np.min(ndvi_sloped_centers)) & cond_sloped & ~(ndvi == -255.0)]
                l4 = np.ones(shape=len(t4))
                l2 = np.zeros(shape=len(t2))
                model = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=0.0)
                if False:
                    from sklearn.model_selection import GridSearchCV
                    from sklearn.model_selection import RepeatedStratifiedKFold
                    # define model evaluation method
                    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
                    # define grid
                    grid = dict()
                    grid['solver'] = ['svd', 'lsqr', 'eigen']
                    grid['shrinkage'] = np.arange(0, 1, 0.01)
                    # define search
                    search = GridSearchCV(model, grid, scoring='accuracy', cv=cv, n_jobs=-1)
                    # perform the search
                    results = search.fit(np.vstack([t4, t2]), np.hstack([l4, l2]))
                    # summarize
                    print('Mean Accuracy: %.3f' % results.best_score_)
                    print('Config: %s' % results.best_params_)
                model.fit(np.vstack([t4, t2]), np.hstack([l4, l2]))
                labels = model.predict(data[cond_sloped & ~(ndvi == -255.0)]).astype(np.uint8)
                result[cond_sloped & ~(ndvi == -255.0)] = np.array([2, 4])[labels]

        if True:
            import matplotlib.pyplot as plt

            peaks = count_dist_peaks(ndvi[cond_sloped & ~(ndvi == -255.0)], bins=255, prominence=75, width=None)

            plt.rcParams.update({'figure.figsize': (7, 5), 'figure.dpi': 100})
            plt.hist(ndvi[cond_sloped & ~(ndvi == -255.0)], bins=500)
            plt.gca().set(title='Frequency Histogram', ylabel='Frequency')
            plt.show()
    else:
        result[cond_any_green & cond_sloped] = 2 if sloped_is_ground_or_tree else 4
    conde = np.where(ndvi == -255)        # error-noise
    return result, conde


def read_scaled(fpath, wnd, target_mppx=1.0, out_fname=None):
    with rio.open(fpath) as src:
        scale = target_mppx / src.transform.a
        if scale != 1.0:
            print(f'Data from file \"{os.path.basename(fpath)}\" has mppx: {src.transform.a:.3f} and scaled to working mppx: {target_mppx}', flush=True)
            t = src.transform

            # rescale the metadata
            transform = Affine(t.a * scale, t.b, t.c, t.d, t.e * scale, t.f)
            height = int(src.height // scale)
            width = int(src.width // scale)

            profile = src.profile
            profile.update(transform=transform, driver='GTiff', height=height, width=width, crs=src.crs)

            # data = src.read(window=Window(*wnd),
            data = src.read(out_shape=(int(src.count), int(height), int(width)),
                            resampling=Resampling.cubic)
        else:
            # data = src.read(None, window=Window(*wnd))
            data = src.read(None)
            profile = src.profile
        if src.nodata is not None:
            data[data == src.nodata] = np.nan
    if out_fname is not None:
        with rasterio.open(out_fname, 'w', **profile) as dst:
            dst.write(data)
    data = data[:, wnd[1]:max(wnd[1] + wnd[3], data.shape[0]), wnd[0]:max(wnd[0] + wnd[2], data.shape[1])]
    return data


def get_xy(proj_dir, wnd, sloped_is_ground_or_tree, filters_nb, layers_nb, is_inference, working_scale=1.0, skip_ground_class=True, mask_geojson=None, mask_class_ind=-1):
    # note: np.float16 do not provide convergence in solvers with default iters
    dtype = np.float32

    print(f'>>>>>> Process data from folder {proj_dir}', flush=True)
    ortho_fpath = f'{proj_dir}/tmp_ortho.tif'
    cosine_fpath = f'{proj_dir}/walls_cosine.tif'
    cosine_scaled_fpath = f'{proj_dir}/walls_cosine_scaled.tif'
    mask_fpath = f'{proj_dir}/{mask_geojson}'
    w_cos = read_scaled(cosine_fpath, wnd, working_scale, cosine_scaled_fpath)
    w_cos = np.moveaxis(w_cos, 0, -1).squeeze()
    if is_inference:
        w_cos[np.isnan(w_cos)] = 1
    cv2.imwrite('w_cos.png', w_cos * 255)
    bgr = read_scaled(ortho_fpath, wnd, working_scale)
    bgr = np.moveaxis(bgr, 0, -1)[..., [2, 1, 0]]
    hsv = cv2.cvtColor(bgr.astype(np.uint8), cv2.COLOR_BGR2HSV)
    cv2.imwrite('w.png', bgr)
    mask, mask_err = get_ndvi(bgr, w_cos, sloped_is_ground_or_tree)

    if os.path.exists(mask_fpath):
        print(f'Use file {mask_fpath} for labeling. CRS should be EPSG:4326', flush=True)
        with open(mask_fpath, 'r') as fin:
            tree_dict = json.load(fin)
            shapes = [feature["geometry"] for feature in tree_dict['features']]
        with rio.open(cosine_scaled_fpath) as src:
            for i in range(len(shapes)):
                shapes[i] = rio.warp.transform_geom("EPSG:4326", src.crs, shapes[i])
            out_image, transformed = rasterio.mask.mask(src, shapes, nodata=np.nan, filled=True)
            out_image = out_image[0]
            mask[~np.isnan(out_image)] = mask_class_ind
            mask[(w_cos < np.cos(np.radians(45))) & (~np.isnan(out_image))] = 4  # ATTENTION: expecting only tree mask so sloped only trees
    cv2.imwrite('w_m.png', mask / 4 * 255)
    # exit(0)
    if skip_ground_class:
        mask[mask == 0] = 255

    if filters_nb > 0:
        features = np.empty(shape=(w_cos.shape[0], w_cos.shape[1], layers_nb * filters_nb + 4), dtype=dtype)
        filters = build_filters(filters_nb)
        features[..., :layers_nb * filters_nb] = process_pyr(w_cos, lambda x: apply_filter(x, filters, dtype), layers_nb, dtype)
    elif filters_nb == 0:
        features = np.empty(shape=(w_cos.shape[0], w_cos.shape[1], 1 + 4), dtype=dtype)
        features[..., :1] = w_cos
    else:
        ksize = -filters_nb / working_scale
        features = np.empty(shape=(w_cos.shape[0], w_cos.shape[1], layers_nb * 5 + 4), dtype=dtype)

        features[..., :layers_nb * 2] = process_pyr(w_cos, lambda x: get_mean_std(x, ksize), layers_nb, dtype)
        features[..., layers_nb * 2:layers_nb * 3] = process_pyr(w_cos, lambda x: [get_mean_std(get_mean_std(x, ksize)[1], ksize)[1]], layers_nb, dtype)
        features[..., layers_nb * 3:layers_nb * 4] = process_pyr(w_cos, lambda x: [get_mean_std(get_mean_std(get_mean_std(x, ksize)[1], ksize)[1], ksize)[1]], layers_nb, dtype)
        features[..., layers_nb * 4:layers_nb * 5] = process_pyr(hsv[..., 0], lambda x: [get_mean_std(get_mean_std(x, ksize)[1], ksize)[1]], layers_nb, dtype)
        if False:
            # features[..., layers_nb * 4:layers_nb * 6] = process_pyr(hsv[..., 0], lambda x: get_mean_std(x, ksize), layers_nb, dtype)
            filters = build_filters(64)
            ## features[..., layers_nb * 4:layers_nb * 6] = process_pyr(hsv[..., 0], lambda x: apply_filter(x, filters, dtype), layers_nb, dtype)

            #features[..., layers_nb * 6:layers_nb * 7] = process_pyr(hsv[..., 0], lambda x: [get_mean_std(get_mean_std(x, ksize)[1], ksize)[1]], layers_nb, dtype)
            #features[..., layers_nb * 7:layers_nb * 8] = process_pyr(hsv[..., 0], lambda x: [get_mean_std(get_mean_std(get_mean_std(x, ksize)[1], ksize)[1], ksize)[1]], layers_nb, dtype)
            #features[..., layers_nb * 7:layers_nb * 8] = process_pyr(hsv[..., 0], lambda x: get_lbp_space(get_mean_std(x, ksize)[1]), layers_nb, dtype)
            #features[..., layers_nb * 7:layers_nb * 8] = process_pyr(hsv[..., 0], lambda x: apply_accum_filter(x, build_filters(8), dtype), layers_nb, dtype)
            #features[..., layers_nb * 6:layers_nb * 8] = process_pyr(w_cos,lambda x: get_mean_std(apply_accum_filter(x, build_filters(64), dtype)[1], ksize), layers_nb, dtype)
            features[..., layers_nb * 4:layers_nb * 5] = process_pyr(w_cos, lambda x: [apply_accum_filter(x, filters, dtype)[1]], layers_nb, dtype)
            features[..., layers_nb * 5:layers_nb * 6] = process_pyr(bgr[..., 1], lambda x: [apply_accum_filter(x, filters, dtype)[1]], layers_nb, dtype)
            features[..., layers_nb * 6:layers_nb * 7] = process_pyr(hsv[..., 0], lambda x: [apply_accum_filter(x, filters, dtype)[1]], layers_nb, dtype)
            features[..., layers_nb * 7:layers_nb * 8] = process_pyr(hsv[..., 1], lambda x: [apply_accum_filter(x, filters, dtype)[1]], layers_nb, dtype)

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
