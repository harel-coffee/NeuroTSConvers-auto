import numpy as np
import random
from rpy2.robjects.packages import importr
import rpy2
import rpy2.robjects as ro

from  sklearn.metrics import mutual_info_score as mi
from  sklearn.metrics import normalized_mutual_info_score as nmi

from sklearn.feature_selection import mutual_info_regression as mi_r

ro.r ("library (NlinTS)")

#======================================================================#
# A wrapper of the continuous mutual information of the NLinTS package
#======================================================================#
def normalized_discr_mu_nlints (x, y, normalize):

    x_ = rpy2.robjects. IntVector (x)
    y_ = rpy2.robjects. IntVector (y)

    ro.globalenv ['x'] = x_
    ro.globalenv ['y'] = y_
    ro.globalenv ['normalize'] = normalize

    mi = ro.r ("mi_disc_bi (x, y, log = 'log2', normalize = normalize)")[0]

    return mi

#======================================================================#
# A wrapper of the continuous mutual information of the NLinTS package
#======================================================================#
def normalized_cont_mu_nlints (x, y, k, normalize):

    x_ = rpy2.robjects. FloatVector (x)
    y_ = rpy2.robjects. FloatVector (y)

    ro.globalenv ['x'] = x_
    ro.globalenv ['y'] = y_
    ro.globalenv ['k'] = k
    ro.globalenv ['normalize'] = normalize

    mi = ro.r ("mi_cont (x, y, k = k, alg = 'ksg1', normalize = TRUE)")[0]

    return mi

#======================================================================#
# Univariate feature selection based on mutual information
#======================================================================#
#def select_mi

#======================================================================#
if __name__ == '__main__':
    #x = [1, 1, 3, 3, 2, 2]
    #y = [1, 1, 3, 5, 2, 2]

    x = RandomI_ListOfIntegers = [random.randrange(1, 15) for iter in range (100)]
    y = RandomI_ListOfIntegers = [random.randrange(1, 15) for iter in range (100)]

    noise = noise = np.random.normal(0,0.1,100)
    max_x = max (x)
    max_y = max (y)


    print ("Discrete MI")
    print ("    Sikit learn: ", nmi (x, y))
    print ("    NlinTS: ", normalized_discr_mu_nlints (x, y, 1))

    # normalization
    x = [(a / max_x) + u for a, u in zip (x, noise)]
    y = [(a / max_y) + u for a, u in zip (y, noise)]
    #y = [a / max_y for a in y]
    #print (x, "\n", y, "\n", 18 * '-')

    print ("Continuous MI")
    print ("    Sikit learn: ", mi_r (np. array (x). reshape (-1,1), y, n_neighbors = 3))
    print ("    NlinTS: ", normalized_cont_mu_nlints (x, y, 3, 0))
