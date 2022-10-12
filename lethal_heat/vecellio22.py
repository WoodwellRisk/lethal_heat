import scipy.interpolate as interpolate
import matplotlib.pyplot as plt
import numpy as np
import dask.array as da
import xarray as xr
import multiprocessing as mp
from dask.distributed import Client, LocalCluster

class Vecellio22():
    '''
    Calculates lethal heat according to the Vecellio22 study.
    By default, fits a linear polynomial to the data.
    You 
    
    Example useage:
        
        # Create Vecellio22 object
        v22 = Vecellio22(degree=1)
        
        # Test if a set of temperatures and relative humidities are lethal
        temperature = [35, 40, 41, 45]
        rel_humidity = [60, 70, 80, 90]
        v22.isLethal(temperature, rel_humidity)
        
        # Plot lethal region with temperature, humidity pairs
        v22.plot( tdb = temperature,
                  rh = rel_humidity)
                  
        # Calculate booleans lazily, in chunks and in parallel
        temperature = xr.open_dataset(filename_temp, chunks={'time':10})
        rel_humidity = xr.open_dataset(filename_rh, chunks={'time',:10})
        uncomputed = v22.map_to_data(temperature, rel_humidity)
        uncomputed.to_netcdf(fp_out)
        
        # Or straight from file
        Vecellio22.calculate_from_files( fp_temperature, fp_humidity, fp_out )
        
    See individual method docstrings for more detailed information in some cases.
    '''
    
    # These are the values to interpolate from the paper.
    interp_td = [36,   38,   40,   44,   48,    50]
    interp_rh = [66.3, 66.8, 50.3, 28.8, 20.14, 12.7]
    
    def __init__(self, polyfit = True, degree = 1, **kwargs):
        
        # Determine interpolation function
        if polyfit:
            z = np.polyfit(self.interp_td, self.interp_rh, degree)
            self.interp_func = np.poly1d(z)
        else:
            self.interp_func = interpolate.interp1d(self.interp_td, 
                                               self.interp_rh, 
                                               bounds_error=False, 
                                               **kwargs)
            
    def isLethal(self, tdb, rh):
        ''' Returns whether heat is lethal for given tdb (temperature)
        and rh (relative humidity) arrays or scalars. '''
        return rh > self.interp_func(tdb)
    
    def plot(self, tdb=None, rh=None, figsize = (7,7)):
        ''' Plots the lethal region on a 2D plot.
        Optionally, you can add temperature, humidity pairs to see where
        they lie'''
        td_min = np.min(self.interp_td)
        td_max = np.max(self.interp_td)
        rh_min = np.min(self.interp_rh)
        rh_max = np.max(self.interp_rh)
        rh_range = rh_max - rh_min
        
        f,a = plt.subplots(1,1, figsize=figsize)
        
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
            a.scatter(tdb, rh, marker='x', s=25, c='k', alpha=0.4)
        
    def map_to_data(self, tdb, rh):
        ''' Lazily calculates lethal heat over two chunked xarray datasets '''
        ds_out = xr.Dataset(tdb.coords)
        mapped = da.map_blocks(self.islethal, tdb.data, rh.data)
        ds_out['lethal_heat'] = (tdb.dims, mapped) 
        return ds_out
    
    @classmethod
    def calculate_from_files(cls, fp_td, fp_rh, fp_out = None,
                             td_name = 'tasmax', rh_name = 'hursmin', 
                             chunks = {'time':100}, input_units='celcius',
                             overwrite = True):
        ''' 
        Calculates lethal heat from file paths -- in parallel chunks.
        Either returns an uncomputed data array or, optionally,
        saves to a new file.
        
        INPUTS
            fp_td (path)       : Path to temperature file (.nc)
            fp_rh (path)       : Path to relative humidity file (.nc)
            fp_out (path)      : Path to output file. Optional.
            td_name (str)      : String name of temperature in input file
            rh_name (str)      : String name of Rel. Hum. in input file
            chunks (dict)      : Dictionary of {'dimension' : size}. Dictates
                                 how the data should be chunked using Dask
            inputs_units (str) : Temeprature units of input. Will be converted to
                                 Celcius before analysis. Can be ['celcius', 'kelvin']
            overwrite (bool)   : Whether or not to overwrite output file.
                                 Default = True.
        '''

        # Define temperature conversion dictionaries
        scaling_dict = {'celcius':1, 'kelvin':1}
        offset_dict = {'celcius':0, 'kelvin':-273.15}

        # Open datasets
        td = xr.open_dataset(fp_td, chunks=chunks)[td_name]
        rh = xr.open_dataset(fp_rh, chunks=chunks)[rh_name]

        # Adjust temperature if needed
        alpha = offset_dict[input_units]
        beta = scaling_dict[input_units]
        ds_td = ds_td * beta + alpha
        ds_rh = ds_rh * beta + alpha

        # Create lethal heat function object and map
        v22 = Vecellio22()
        lethalheat = v22.map_to_data(td, rh)

        # Write to output file (maybe)
        if fp_out is not None:
            lethalheat.to_netcdf(fp_out)
        else:
            return lethalheat
