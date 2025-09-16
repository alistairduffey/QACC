import os
import glob
import pandas as pd
import numpy as np
import xarray as xr
from xmip.preprocessing import rename_cmip6
import cftime


def get_ds_from_ceda(model, centre, scenario_group, scenario, 
                     variable, members, table='Amon', grid='gn'):
    ds_list = []
    for es in members:
        path = '/badc/cmip6/data/CMIP6/{sg}/{c}/{m}/{s}/{e}/{t}/{v}/{g}/latest/'.format(
            sg=scenario_group, s=scenario, c=centre, m=model, e=es, t=table,v=variable, g=grid)
        #files = os.listdir(path)
        ds = rename_cmip6(xr.open_mfdataset(path+'*.nc'))
        ds['Ensemble_member']=es
        ds_list.append(ds)
    
    DS = xr.concat(ds_list, dim='Ensemble_member')
    return DS

# note, in general we have to account for different month lengths
def weighted_annual_resample(ds, var):
    """
    weight by days in each month
    adapted from NCAR docs 
    https://ncar.github.io/esds/posts/2021/yearly-averages-xarray/
    """
    # Determine the month length
    month_length = ds.time.dt.days_in_month

    # Calculate the weights
    wgts = (month_length.groupby("time.year") / month_length.groupby("time.year").sum()).load()

    # Make sure the weights in each year add up to 1
    np.testing.assert_allclose(wgts.groupby("time.year").sum(xr.ALL_DIMS), 1.0)

    numerator = (ds[var] * wgts).resample(time="YS").sum(dim="time")
    denominator = wgts.resample(time="YS").sum(dim="time")
    ds_out = (numerator/denominator).to_dataset(name=var)
    ds_out['year'] = ds_out.time.dt.year
    return ds_out

