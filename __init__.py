# ---------------------------------------------------------------
# Initial Settings for Training and Testing YOLO Person Search
#
# Author: Liangqi Li
# Creating Date: May 17, 2018
# Latest rectified: May 17, 2018
# ---------------------------------------------------------------
import time
import functools


def clock_non_return(func):
    @functools.wraps(func)
    def clocked(*args, **kwargs):
        t0 = time.time()
        func(*args, **kwargs)
        elapsed = time.time() - t0
        if elapsed < 60:
            trans_elap = elapsed
            unit = 'seconds'
        elif elapsed < 3600:
            trans_elap = elapsed / 60
            unit = 'minutes'
        else:
            trans_elap = elapsed / 3600
            unit = 'hours'
        print('\n' + '*' * 40)
        print('Entire process costs {:.2f} {:s}.'.format(trans_elap, unit))
    return clocked
