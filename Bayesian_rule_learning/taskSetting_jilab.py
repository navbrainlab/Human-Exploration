import numpy as np
import pandas as pd
import scipy.stats


allChoices = []

rewardSetting = [
                [0.2, 0.4, 0.6, 0.8],
                [1/7, 2/7, 3/7, 4/7, 5/7, 6/7],
                [1/8, 2/8, 3/8, 4/8, 5/8, 6/8, 7/8]
                 ] 

hypotheses = [
    # #1d relevant
    [[1, np.nan ,np.nan],[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]],
    [[np.nan, 1 ,np.nan],[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]],
    [[np.nan, np.nan, 1],[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]],
    [[np.nan, np.nan, np.nan],[1, np.nan, np.nan], [np.nan, np.nan, np.nan]],
    [[np.nan, np.nan, np.nan],[np.nan, 1, np.nan], [np.nan, np.nan, np.nan]],
    [[np.nan, np.nan, np.nan],[np.nan, np.nan, 1], [np.nan, np.nan, np.nan]],
    [[np.nan, np.nan, np.nan],[np.nan, np.nan, np.nan], [1, np.nan, np.nan]],
    [[np.nan, np.nan, np.nan],[np.nan, np.nan, np.nan], [np.nan, 1, np.nan]],
    [[np.nan, np.nan, np.nan],[np.nan, np.nan, np.nan], [np.nan, np.nan, 1]],

    # # 2d relevant
    [[1, np.nan, np.nan],[1, np.nan, np.nan], [np.nan, np.nan, np.nan]],
    [[1, np.nan, np.nan],[np.nan, 1, np.nan], [np.nan, np.nan, np.nan]],
    [[1, np.nan, np.nan],[np.nan, np.nan, 1], [np.nan, np.nan, np.nan]],
    [[np.nan, 1, np.nan],[1, np.nan, np.nan], [np.nan, np.nan, np.nan]],
    [[np.nan, 1, np.nan],[np.nan, 1, np.nan], [np.nan, np.nan, np.nan]],
    [[np.nan, 1, np.nan],[np.nan, np.nan, 1], [np.nan, np.nan, np.nan]],
    [[np.nan, np.nan, 1],[1, np.nan, np.nan], [np.nan, np.nan, np.nan]],
    [[np.nan, np.nan, 1],[np.nan, 1, np.nan], [np.nan, np.nan, np.nan]],
    [[np.nan, np.nan, 1],[np.nan, np.nan, 1], [np.nan, np.nan, np.nan]],

    [[1, np.nan, np.nan],[np.nan, np.nan, np.nan], [1, np.nan, np.nan]],
    [[1, np.nan, np.nan],[np.nan, np.nan, np.nan], [np.nan, 1, np.nan]],
    [[1, np.nan, np.nan],[np.nan, np.nan, np.nan], [np.nan, np.nan, 1]],
    [[np.nan, 1, np.nan],[np.nan, np.nan, np.nan], [1, np.nan, np.nan]],
    [[np.nan, 1, np.nan],[np.nan, np.nan, np.nan], [np.nan, 1, np.nan]],
    [[np.nan, 1, np.nan],[np.nan, np.nan, np.nan], [np.nan, np.nan, 1]],
    [[np.nan, np.nan, 1],[np.nan, np.nan, np.nan], [1, np.nan, np.nan]],
    [[np.nan, np.nan, 1],[np.nan, np.nan, np.nan], [np.nan, 1, np.nan]],
    [[np.nan, np.nan, 1],[np.nan, np.nan, np.nan], [np.nan, np.nan, 1]],

    [[np.nan, np.nan, np.nan],[1, np.nan, np.nan], [1, np.nan, np.nan]],
    [[np.nan, np.nan, np.nan],[1, np.nan, np.nan], [np.nan, 1, np.nan]],
    [[np.nan, np.nan, np.nan],[1, np.nan, np.nan], [np.nan, np.nan, 1]],
    [[np.nan, np.nan, np.nan],[np.nan, 1, np.nan], [1, np.nan, np.nan]],
    [[np.nan, np.nan, np.nan],[np.nan, 1, np.nan], [np.nan, 1, np.nan]],
    [[np.nan, np.nan, np.nan],[np.nan, 1, np.nan], [np.nan, np.nan, 1]],
    [[np.nan, np.nan, np.nan],[np.nan, np.nan, 1], [1, np.nan, np.nan]],
    [[np.nan, np.nan, np.nan],[np.nan, np.nan, 1], [np.nan, 1, np.nan]],
    [[np.nan, np.nan, np.nan],[np.nan, np.nan, 1], [np.nan, np.nan, 1]], 
    

    # # #3d relevant
    [[1, np.nan, np.nan],[1, np.nan, np.nan], [1, np.nan, np.nan]],
    [[1, np.nan, np.nan],[1, np.nan, np.nan], [np.nan, 1, np.nan]],
    [[1, np.nan, np.nan],[1, np.nan, np.nan], [np.nan, np.nan, 1]],

    [[1, np.nan, np.nan],[np.nan, 1, np.nan], [1, np.nan, np.nan]],
    [[1, np.nan, np.nan],[np.nan, 1, np.nan], [np.nan, 1, np.nan]],
    [[1, np.nan, np.nan],[np.nan, 1, np.nan], [np.nan, np.nan, 1]],

    [[1, np.nan, np.nan],[np.nan, np.nan, 1], [1, np.nan, np.nan]],
    [[1, np.nan, np.nan],[np.nan, np.nan, 1], [np.nan, 1, np.nan]],
    [[1, np.nan, np.nan],[np.nan, np.nan, 1], [np.nan, np.nan, 1]],

    [[np.nan, 1, np.nan],[1, np.nan, np.nan], [1, np.nan, np.nan]],
    [[np.nan, 1, np.nan],[1, np.nan, np.nan], [np.nan, 1, np.nan]],
    [[np.nan, 1, np.nan],[1, np.nan, np.nan], [np.nan, np.nan, 1]],

    [[np.nan, 1, np.nan],[np.nan, 1, np.nan], [1, np.nan, np.nan]],
    [[np.nan, 1, np.nan],[np.nan, 1, np.nan], [np.nan, 1, np.nan]],
    [[np.nan, 1, np.nan],[np.nan, 1, np.nan], [np.nan, np.nan, 1]],

    [[np.nan, 1, np.nan],[np.nan, np.nan, 1], [1, np.nan, np.nan]],
    [[np.nan, 1, np.nan],[np.nan, np.nan, 1], [np.nan, 1, np.nan]],
    [[np.nan, 1, np.nan],[np.nan, np.nan, 1], [np.nan, np.nan, 1]],

    [[np.nan, np.nan, 1],[1, np.nan, np.nan], [1, np.nan, np.nan]],
    [[np.nan, np.nan, 1],[1, np.nan, np.nan], [np.nan, 1, np.nan]],
    [[np.nan, np.nan, 1],[1, np.nan, np.nan], [np.nan, np.nan, 1]],

    [[np.nan, np.nan, 1],[np.nan, 1, np.nan], [1, np.nan, np.nan]],
    [[np.nan, np.nan, 1],[np.nan, 1, np.nan], [np.nan, 1, np.nan]],
    [[np.nan, np.nan, 1],[np.nan, 1, np.nan], [np.nan, np.nan, 1]],

    [[np.nan, np.nan, 1],[np.nan, np.nan, 1], [1, np.nan, np.nan]],
    [[np.nan, np.nan, 1],[np.nan, np.nan, 1], [np.nan, 1, np.nan]],
    [[np.nan, np.nan, 1],[np.nan, np.nan, 1], [np.nan,np.nan, 1 ]]
]


# hypotheses space for 3d
hypotheses_original = [
    #1d relevant
    [[1, np.nan ,np.nan],[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]],
    [[np.nan, 1 ,np.nan],[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]],
    [[np.nan, np.nan, 1],[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]],
    [[np.nan, np.nan, np.nan],[1, np.nan, np.nan], [np.nan, np.nan, np.nan]],
    [[np.nan, np.nan, np.nan],[np.nan, 1, np.nan], [np.nan, np.nan, np.nan]],
    [[np.nan, np.nan, np.nan],[np.nan, np.nan, 1], [np.nan, np.nan, np.nan]],
    [[np.nan, np.nan, np.nan],[np.nan, np.nan, np.nan], [1, np.nan, np.nan]],
    [[np.nan, np.nan, np.nan],[np.nan, np.nan, np.nan], [np.nan, 1, np.nan]],
    [[np.nan, np.nan, np.nan],[np.nan, np.nan, np.nan], [np.nan, np.nan, 1]],

    # 2d relevant
    [[1, np.nan, np.nan],[1, np.nan, np.nan], [np.nan, np.nan, np.nan]],
    [[1, np.nan, np.nan],[np.nan, 1, np.nan], [np.nan, np.nan, np.nan]],
    [[1, np.nan, np.nan],[np.nan, np.nan, 1], [np.nan, np.nan, np.nan]],
    [[np.nan, 1, np.nan],[1, np.nan, np.nan], [np.nan, np.nan, np.nan]],
    [[np.nan, 1, np.nan],[np.nan, 1, np.nan], [np.nan, np.nan, np.nan]],
    [[np.nan, 1, np.nan],[np.nan, np.nan, 1], [np.nan, np.nan, np.nan]],
    [[np.nan, np.nan, 1],[1, np.nan, np.nan], [np.nan, np.nan, np.nan]],
    [[np.nan, np.nan, 1],[np.nan, 1, np.nan], [np.nan, np.nan, np.nan]],
    [[np.nan, np.nan, 1],[np.nan, np.nan, 1], [np.nan, np.nan, np.nan]],

    [[1, np.nan, np.nan],[np.nan, np.nan, np.nan], [1, np.nan, np.nan]],
    [[1, np.nan, np.nan],[np.nan, np.nan, np.nan], [np.nan, 1, np.nan]],
    [[1, np.nan, np.nan],[np.nan, np.nan, np.nan], [np.nan, np.nan, 1]],
    [[np.nan, 1, np.nan],[np.nan, np.nan, np.nan], [1, np.nan, np.nan]],
    [[np.nan, 1, np.nan],[np.nan, np.nan, np.nan], [np.nan, 1, np.nan]],
    [[np.nan, 1, np.nan],[np.nan, np.nan, np.nan], [np.nan, np.nan, 1]],
    [[np.nan, np.nan, 1],[np.nan, np.nan, np.nan], [1, np.nan, np.nan]],
    [[np.nan, np.nan, 1],[np.nan, np.nan, np.nan], [np.nan, 1, np.nan]],
    [[np.nan, np.nan, 1],[np.nan, np.nan, np.nan], [np.nan, np.nan, 1]],

    [[np.nan, np.nan, np.nan],[1, np.nan, np.nan], [1, np.nan, np.nan]],
    [[np.nan, np.nan, np.nan],[1, np.nan, np.nan], [np.nan, 1, np.nan]],
    [[np.nan, np.nan, np.nan],[1, np.nan, np.nan], [np.nan, np.nan, 1]],
    [[np.nan, np.nan, np.nan],[np.nan, 1, np.nan], [1, np.nan, np.nan]],
    [[np.nan, np.nan, np.nan],[np.nan, 1, np.nan], [np.nan, 1, np.nan]],
    [[np.nan, np.nan, np.nan],[np.nan, 1, np.nan], [np.nan, np.nan, 1]],
    [[np.nan, np.nan, np.nan],[np.nan, np.nan, 1], [1, np.nan, np.nan]],
    [[np.nan, np.nan, np.nan],[np.nan, np.nan, 1], [np.nan, 1, np.nan]],
    [[np.nan, np.nan, np.nan],[np.nan, np.nan, 1], [np.nan, np.nan, 1]], 

    #3d relevant
    [[1, np.nan, np.nan],[1, np.nan, np.nan], [1, np.nan, np.nan]],
    [[1, np.nan, np.nan],[1, np.nan, np.nan], [np.nan, 1, np.nan]],
    [[1, np.nan, np.nan],[1, np.nan, np.nan], [np.nan, np.nan, 1]],

    [[1, np.nan, np.nan],[np.nan, 1, np.nan], [1, np.nan, np.nan]],
    [[1, np.nan, np.nan],[np.nan, 1, np.nan], [np.nan, 1, np.nan]],
    [[1, np.nan, np.nan],[np.nan, 1, np.nan], [np.nan, np.nan, 1]],

    [[1, np.nan, np.nan],[np.nan, np.nan, 1], [1, np.nan, np.nan]],
    [[1, np.nan, np.nan],[np.nan, np.nan, 1], [np.nan, 1, np.nan]],
    [[1, np.nan, np.nan],[np.nan, np.nan, 1], [np.nan, np.nan, 1]],

    [[np.nan, 1, np.nan],[1, np.nan, np.nan], [1, np.nan, np.nan]],
    [[np.nan, 1, np.nan],[1, np.nan, np.nan], [np.nan, 1, np.nan]],
    [[np.nan, 1, np.nan],[1, np.nan, np.nan], [np.nan, np.nan, 1]],

    [[np.nan, 1, np.nan],[np.nan, 1, np.nan], [1, np.nan, np.nan]],
    [[np.nan, 1, np.nan],[np.nan, 1, np.nan], [np.nan, 1, np.nan]],
    [[np.nan, 1, np.nan],[np.nan, 1, np.nan], [np.nan, np.nan, 1]],

    [[np.nan, 1, np.nan],[np.nan, np.nan, 1], [1, np.nan, np.nan]],
    [[np.nan, 1, np.nan],[np.nan, np.nan, 1], [np.nan, 1, np.nan]],
    [[np.nan, 1, np.nan],[np.nan, np.nan, 1], [np.nan, np.nan, 1]],

    [[np.nan, np.nan, 1],[1, np.nan, np.nan], [1, np.nan, np.nan]],
    [[np.nan, np.nan, 1],[1, np.nan, np.nan], [np.nan, 1, np.nan]],
    [[np.nan, np.nan, 1],[1, np.nan, np.nan], [np.nan, np.nan, 1]],

    [[np.nan, np.nan, 1],[np.nan, 1, np.nan], [1, np.nan, np.nan]],
    [[np.nan, np.nan, 1],[np.nan, 1, np.nan], [np.nan, 1, np.nan]],
    [[np.nan, np.nan, 1],[np.nan, 1, np.nan], [np.nan, np.nan, 1]],

    [[np.nan, np.nan, 1],[np.nan, np.nan, 1], [1, np.nan, np.nan]],
    [[np.nan, np.nan, 1],[np.nan, np.nan, 1], [np.nan, 1, np.nan]],
    [[np.nan, np.nan, 1],[np.nan, np.nan, 1], [np.nan, 1, np.nan]]
]