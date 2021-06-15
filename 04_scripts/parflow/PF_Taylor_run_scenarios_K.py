# %%
import os
import pandas as pd
from parflow import Run
from parflow.tools.fs import cp, mkdir
from pathlib import Path
import numpy as np
import sys
import shutil
import glob
import calendar

# %%

year_run = int(sys.argv[1])
if calendar.isleap(year_run):
  no_day = 366
else:
  no_day = 365

year_training = [1988,1990,2002,2015,2016,2017,2018]
year_validation = [2000,2012,2013]

# Run ensemble
for idx in range(n_run):
  
  name_run = "name_of_this_run_which_helps_recognising_it"
  path_folder = '/home/quinnfolder/'
  
  run_dir = f'{path_folder}{name_run}'
  #creating directory for the run
  os.makedirs(run_dir)
  
  #cd into the run directory
  os.chdir(run_dir)
  
  #setting all file paths to copy required input files
  path_slope_x = f'/hydrodata/PFCLM/Taylor/Simulations/inputs/slope_x.pfb'
  path_slope_y = f'/hydrodata/PFCLM/Taylor/Simulations/inputs/slope_y.pfb'
  path_drv_clmin = f'/hydrodata/PFCLM/Taylor/Simulations/inputs/drv_clmin.dat'
  path_drv_vegm = f'/hydrodata/PFCLM/Taylor/Simulations/inputs/drv_vegm_v2.Taylor.dat'
  path_drv_vegp = f'/hydrodata/PFCLM/Taylor/Simulations/inputs/drv_vegp.dat'
  path_indicator = f'/hydrodata/PFCLM/Taylor/Simulations/inputs/Taylor.IndicatorFile_v2.pfb'
  path_pfsol = f'/hydrodata/PFCLM/Taylor/Simulations/inputs/Taylor.pfsol'
  
  indicator = f'Taylor.IndicatorFile_v2.pfb'
  
  #copy all the input files in current directory
  shutil.copy(f'{path_slope_x}', run_dir)
  shutil.copy(f'{path_slope_y}', run_dir)
  shutil.copy(f'{path_drv_clmin}', run_dir)
  shutil.copy(f'{path_drv_vegm}', run_dir)
  shutil.copy(f'{path_drv_vegp}', run_dir)
  shutil.copy(f'{path_pfsol}', run_dir)
  shutil.copy(f'{path_indicator}', run_dir)
    
  #extra step to change the drv_clmin so that it has the correct start and end date:
  a_file = open(f'drv_clmin.dat', "r")
  list_of_lines = a_file.readlines()
  #changing syr and eyr
  list_of_lines[35]=f'syr {year_run-1} Starting Year\n'
  list_of_lines[42]=f'eyr {year_run} Ending Year\n'
  #changing startcodes: starting new run from scratch
  list_of_lines[29]=f'startcode      2\n'
  list_of_lines[46]=f'clm_ic         2\n'
  a_file = open(f'drv_clmin.dat', "w")
  a_file.writelines(list_of_lines)
  a_file.close()
  
  #copying initial pressure
  ip0 = f'/hydrodata/PFCLM/Taylor/Simulations/{year_run}/Taylor_{year_run}.out.press.00000.pfb'
  ip = 'initial_pressure.pfb'
  shutil.copy(ip0, f'{run_dir}/{ip}')
  
  
  met_path = f'{run_dir}/NLDAS/'
  os.mkdir(met_path)
  #copy the correct forcing
  met_path_to_copy = f'/hydrodata/PFCLM/Taylor/Simulations/{year_run}/NLDAS/'
    
  
  for filename in glob.glob(os.path.join(met_path_to_copy, '*.*')):
    shutil.copy(filename, met_path)
  
  shutil.copy(f'{reference_run_path}{reference_run_name}', run_dir)
  #Read in the run
  run = Run.from_definition(f'{reference_run_name}')
  run.set_name(f'Taylor_{year_run}')
  
  #updating the directory where the forcing is
  run.Solver.CLM.MetFilePath = met_path
  run.TimingInfo.StopTime = 24*no_day
  
  print(f'running from {run.TimingInfo.StartCount} or {run.TimingInfo.StartTime} to {run.TimingInfo.StopTime}')
  
  #THIS IS THE PERMEABILITY VALUE TO CHANGE: Dominant soil type in Taylor surface&subsurface layer(s)
  run.Geom.s3.Perm.Value = SET_THIS_TO_WHATEVER_K_VALUE_YOU_WANT #in m/s

  
  print("Starting Distribution Inputs/Forcing")
  run.dist('slope_x.pfb')
  run.dist('slope_y.pfb')
  for filename_forcing in os.listdir(met_path):
    run.dist(f'{met_path}{filename_forcing}')
  print('Done distributing forcing')
  
  run.dist(indicator)
  
  run.dist(ip)
  
  print(f'Starting run')
  run.write()
  run.write(file_format='yaml')
  run.write(file_format='json')
  run.run()
  

# %%
