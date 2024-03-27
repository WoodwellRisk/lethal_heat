import scipy.interpolate as interpolate
import matplotlib.pyplot as plt
import numpy as np
import dask.array as da
import xarray as xr
import multiprocessing as mp
from dask.distributed import Client, LocalCluster
import shapely.geometry as geom

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
        v22.is_lethal(temperature, rel_humidity)
        
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
    interp_td = np.array([36,   38,   40,   44,   48,    50])
    interp_rh = np.array([66.3, 66.8, 50.3, 28.8, 20.14, 12.7])
    
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
            
    def is_lethal(self, tdb, rh):
        ''' Returns whether heat is lethal for given tdb (temperature)
        and rh (relative humidity) arrays or scalars. '''
        return rh > self.interp_func(tdb)
    
    def distance_from_function(self, tdb, rh):
        ''' Returns a numerical estimate of the minimum distance from the
        function used by this lethal heat instance. Output is signed according
        to whether the point is lethal or not'''
        
        # Check type of inputs are array. If not, make it so
        if not hasattr(tdb, "__len__"):
            tdb = np.array([tdb])
        if not hasattr(rh, "__len__"):
            rh = np.array([rh])
        n_pts = len(tdb)
        
        # Create curve coordinates as a shapely LineString
        x_vals = np.arange(0, 60, 0.1)
        y_vals = self.interp_func(x_vals)
        coords = [ [x_vals[ii], y_vals[ii]] for ii in range(len(x_vals)) ]
        curve = geom.LineString(coords)
        
        # Create list of input points and use to calculate distances
        point_list = [ geom.Point( tdb[ii], rh[ii] ) for ii in range(n_pts) ]
        distances = np.array( [ pt.distance(curve) for pt in point_list] )
        
        # Apply a maximum distance
        distances[distances > 1000] = 1000
        
        # Work out signs of output
        islethal = self.is_lethal(tdb, rh).astype(int)
        islethal[islethal==0] = -1
        distances = distances*islethal
        
        # Filter NaNs
        distances[np.isnan(tdb)] = np.nan
        distances[np.isnan(rh)] = np.nan
        
        # If only one input, then return a scalar
        if n_pts == 1:
            return distances[0]
        else:
            return distances
    
    def plot(self, tdb=None, rh=None, figsize = (7,7), 
             tbounds = None, rbounds = None,
             convert_to_f = False):
        ''' Plots the lethal region on a 2D plot.
        Optionally, you can add temperature, humidity pairs to see where
        they lie'''
        
        f,a = plt.subplots(1,1, figsize=figsize)
        
        x = np.arange(0, 70, 0.01)
        y = self.interp_func(x)
        x_scat = self.interp_td
        y_scat = self.interp_rh

        if convert_to_f:
            x = x * (9/5) + 32
            x_scat = x_scat * (9/5) + 32

        if tbounds is None:
            td_min = np.min(x_scat)
            td_max = np.max(x_scat)
        else: 
            td_min = tbounds[0]
            td_max = tbounds[1]
            
        if rbounds is None:
            rh_min = np.min(y_scat)
            rh_max = np.max(y_scat)
        else:
            rh_min = rbounds[0]
            rh_max = rbounds[1]
        rh_range = rh_max - rh_min
        
        a.fill_between(x, y, np.ones(len(x))*100, color='r', alpha=.25)
        a.plot(x, y, linewidth=3, linestyle='--', c='r', alpha=0.5)
        a.scatter(x_scat, y_scat, marker='s', s=100, c='r')
        a.set_xlim(td_min, td_max)
        a.set_ylim(rh_min, rh_max)
        a.grid()
        a.set_xlabel('Temperature ($^{\circ} C$)', fontsize=15)
        if convert_to_f:
            a.set_xlabel('Temperature ($^{\circ} F$)', fontsize=15)
        a.set_ylabel('Relative Humidity (%)', fontsize=15)
        
        if tdb is not None and rh is not None:
            a.scatter(tdb, rh, marker='x', s=25, c='k', alpha=0.4)
        return f,a
        
    def map_to_data(self, tdb, rh):
        ''' Lazily calculates lethal heat over two chunked xarray datasets '''
        #ds_out = xr.Dataset(tdb.coords)
        mapped = xr.map_blocks(self.is_lethal, tdb, [rh])
        #ds_out['lethal_heat'] = (list(tdb.dims), mapped) 
        #return ds_out
        return mapped
    
    def time_over_lh(self, tmean, rmean, tamp, ramp):
        ''' Get number of minutes over lethal heat for a given day.
            tmean - temperature mean (daily)
            rmean - RH mean (daily)
            tamp - temperature amplitude (daily) (half the range)
            ramp - RH amplituce
            v22 Vecellio22 lethal heat object.'''

        if rmean <= 100:

            # Determine time frequency (1 minute)
            x = np.linspace(0, 2*np.pi, 24*60)

            # Create time series of t and rh, and get lethal heat
            ts_t = tamp * np.sin(x) + tmean
            ts_rh = np.clip(-ramp * np.sin(x) + rmean, 0, 100)
            ts_lh = self.is_lethal(ts_t, ts_rh)

            # Return number of minutes for this day over threshold
            return np.nansum(ts_lh)
        else:
            return np.nan
        
    def create_lookup_table( self, tmean, rmean, tamp, ramp, fp_out=None ):
        ''' Creates a lookup table for quickly referencing the number of hours
            per day that are lethal, according to this object. Input arrays should
            have a constant increment, e.g. as created by numpy.arange().
            
            args
             tmean :: array of temperature means
             rmean :: array of RH means
             tamp :: array of temperature amplitudes
             ramp :: array of RH amplitudes
             fp_out :: output filename (string). Default = no output file.
             
            output
             4-dimensional lookup table with dimension (tmean, rmean, tamp, ramp)
             '''
        
        # Get sizes of inputs and create empty output array
        n_tmean = len(tmean)
        n_rmean = len(rmean)
        n_tamp = len(tamp)
        n_ramp = len(ramp)
        lookup = np.zeros((n_tmean, n_rmean, n_tamp, n_ramp))
        
        # Nasty nested loops for allocating time_over_lh one at a time
        for ii in range(n_tmean):
            print(f'{100*ii/n_tmean}%', end='\r')
            for jj in range(n_rmean):
                for kk in range(n_tamp):
                    for ll in range(n_ramp):
                        lookup[ii,jj,kk,ll] = self.time_over_lh(tmean[ii], rmean[jj], 
                                                               tamp[kk], ramp[ll])
                        
        # Place into output dataset (divide by 60 to convert minutes to hours)
        ds_out = xr.Dataset()
        ds_out['tmean'] = tmean
        ds_out['tamp'] = tamp
        ds_out['rmean'] = rmean
        ds_out['ramp'] = ramp
        ds_out['hours_over_lh'] = (['tmean','rmean','tamp','ramp'], lookup/60)

        if fp_out is not None:
            ds_out.to_netcdf(fp_out)
        return ds_out

    def lethal_heat_hours_from_lookup(self, lookup, 
                                      tmean, rmean, 
                                      tamp, ramp):
        ''' 
        
        '''

        # Create numpy arrays from lookup dimensions
        lk_tmean = lookup.tmean.values
        lk_rmean = lookup.rmean.values
        lk_tamp = lookup.tamp.values
        lk_ramp = lookup.ramp.values

        # Get bounds of input arrays
        tmean_min = np.min(lk_tmean)
        rmean_min = np.min(lk_rmean)
        tamp_min = np.min(lk_tamp)
        ramp_min = np.min(lk_ramp)

        # Get increments of input arrays
        tmean_inc = lk_tmean[1] - lk_tmean[0]
        rmean_inc = lk_rmean[1] - lk_rmean[0]
        tamp_inc = lk_tamp[1] - lk_tamp[0]
        ramp_inc = lk_ramp[1] - lk_ramp[0]

        # Convert input arrays into indices using bounds and increments
        lk_index_tmean = ( (tmean - tmean_min) / tmean_inc ).round().astype(int).clip(0)
        lk_index_rmean = ( (rmean - rmean_min) / rmean_inc ).round().astype(int).clip(0)
        lk_index_tamp = ( (tamp - tamp_min) / tamp_inc ).round().astype(int).clip(0)
        lk_index_ramp = ( (ramp - ramp_min) / ramp_inc ).round().astype(int).clip(0)

        # Flatten indexing arrays
        original_shape = lk_index_tmean.shape
        lk_index_tmean = xr.DataArray( lk_index_tmean.flatten() )
        lk_index_rmean = xr.DataArray( lk_index_rmean.flatten() )
        lk_index_tamp = xr.DataArray( lk_index_tamp.flatten() )
        lk_index_ramp = xr.DataArray( lk_index_ramp.flatten() )
        print( lk_index_tmean, lk_index_rmean, lk_index_tamp, lk_index_ramp )

        # Index lookup table and reshape
        lookup_indexed = lookup.hours_over_lh.isel( tmean = lk_index_tmean,
                                                    rmean = lk_index_rmean,
                                                    tamp = lk_index_tamp,
                                                    ramp = lk_index_ramp )
        lookup_indexed = lookup_indexed.values.reshape(original_shape)
        return lookup_indexed
    
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
