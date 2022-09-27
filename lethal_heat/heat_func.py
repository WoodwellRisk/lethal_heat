import scipy.interpolate as interpolate
import matplotlib.pyplot as plt
import numpy as np
import dask.array as da

class Vecellio22():
    
    interp_td = [36,   38,   40,   44,   48,    50]
    interp_rh = [66.3, 66.8, 50.3, 28.8, 20.14, 12.7]
    
    def __init__(self, **kwargs):
        
        interp_func = interpolate.interp1d(self.interp_td, 
                                           self.interp_rh, 
                                           bounds_error=False, 
                                           **kwargs)
        self.interp_func = interp_func
        
    def islethal(self, tdb, rh, lazy=False):
        if lazy:
            return self._map_to_data(tdb, rh)
        else:
            return rh > self.interp_func(tdb)
    
    def plot(self, tdb=None, rh=None):
        
        td_min = np.min(self.interp_td)
        td_max = np.max(self.interp_td)
        rh_min = np.min(self.interp_rh)
        rh_max = np.max(self.interp_rh)
        rh_range = rh_max - rh_min
        
        f,a = plt.subplots(1,1, figsize=(7,7))
        
        x = np.arange(td_min, td_max, 0.01)
        y = self.interp_func(x)
        
        a.fill_between(x, y, np.ones(len(x))*100, color='r', alpha=.25)
        a.plot(x, y, linewidth=3, linestyle='--', c='r', alpha=0.5)
        a.scatter(self.interp_td, self.interp_rh, marker='s', s=100, c='r')
        a.set_xlim(td_min, td_max)
        a.set_ylim(rh_min - 0.2*rh_range, rh_max + 0.2*rh_range)
        a.grid()
        a.set_xlabel('Dry Bulb Temperature', fontsize=15)
        a.set_ylabel('Relative Humidity', fontsize=15)
        
        if tdb is not None and rh is not None:
            a.scatter(tdb, rh, marker='x', s=75, c='k')
        
    def _map_to_data(self, tdb, rh):
        mapped = da.map_blocks(self.islethal, tdb.data, rh.data)
        return mapped
        