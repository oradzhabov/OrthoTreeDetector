import cv2
import numpy as np


def build_filters(theta_nb=32):
    filters = []
    ksize = 11  # 11, 31
    #
    # sigma, lambd, gamma, psi = 1, 1, 0.02, 1
    sigma, lambd, gamma, psi = 1, 1, 0.02, 0
    # sigma, lambd, gamma, psi = 4, 10, 0.5, 0
    if True:
        for theta in np.arange(0, np.pi, np.pi / theta_nb):
            kern = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F)
            kern /= 1.0 * kern.sum()  # Brightness normalization. Applying, make tiny effect but looks like improvement
            filters.append(kern)
    else:  # tests show that here is no principal improvement
        for theta in np.arange(0, np.pi, np.pi / theta_nb):
            for sigma in range(1, 3):
                #for gamma in np.arange(0.02, 0.5, 0.2):
                for gamma in [0.02, 0.5]:
                    kern = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F)
                    # kern /= 1.0 * kern.sum()  # Brightness normalization
                    filters.append(kern)
    return filters


