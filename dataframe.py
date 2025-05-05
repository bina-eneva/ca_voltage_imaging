# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 13:29:39 2025

@author: binae
"""

import sys
import os

#import functions as canf
from pathlib import Path
import datetime
import numpy as np
import re
import pandas as pd

def get_tif_smr(
    topdir, savefile, min_date, max_date, cell_line,
):
    if min_date is None:
        min_date = "20000101"
    if max_date is None:
        max_date = "21000101"

    home = Path.home()
    local_home = "/home/yilin"
    hpc_home = "/rds/general/user/ys5320/home"
    if str(home) == hpc_home:
        HPC = True
    else:
        HPC = False

    files = Path(topdir).glob("./**/*.ome.tif")
    tif_files = []

    for f in files:

        parts = f.parts
        #print(parts)
        day_idx = parts.index(f"{cell_line}") + 1
        day = re.sub("\D", "", parts[day_idx])

        # reject non-date experiment (test etc.)
        try:
            int(day)
        except ValueError:
            continue

        if not select_daterange(day, min_date, max_date):
            continue
        
        if "brightfield" in str(f) or "beightfield" in str(f) or "breightfield" in str(f) or "brigthfield" in str(f) :
            continue
       
        tif_files.append(str(f))


    df = pd.DataFrame()

    df["tif_file"] = tif_files


    # now consolidate files that were split (i.e. two/three tif files in same directory, one has _1/_2 at end,
    # due to file size limits on tif file size)
    remove = []

    for data in df.itertuples():
        fname = data.tif_file
        base_name = fname.split(".ome.tif")[0]  # Extract base name before ".ome.tif"

        all_files = os.listdir(Path(fname).parent)
 
        matching_files = [f for f in all_files if f.endswith(".ome.tif")]
        
        # Count the number of split files
        num_splits = len(matching_files)
        
        # Update multi_tif column
        df.loc[data.Index, "multi_tif"] = num_splits
        
        # If there are split files, add them to remove list
        if (base_name.endswith('_1')) or (base_name.endswith('_2') ):
            remove.append(data.Index)
    
    df = df.drop(labels=remove)
    

    #df.to_csv(savefile)

    return df
def select_daterange(str_date, str_mindate, str_maxdate):
    if (
        (
            datetime.date(*strdate2int(str_date))
            - datetime.date(*strdate2int(str_mindate))
        ).days
        >= 0
    ) and (
        (
            datetime.date(*strdate2int(str_date))
            - datetime.date(*strdate2int(str_maxdate))
        ).days
        <= 0
    ):
        return True
    else:
        return False

home = Path.home()
cell_line = 'MDA_MB_468'
min_date="20250127" #could be None
max_date=None #could be None

if "ys5320" in str(home):
    HPC = True
    top_dir = Path(home, "firefly_link/Calcium_Voltage_Imaging",f'{cell_line}')
    savestr = "_HPC"
    
elif "be320" in str(home):
    HPC = True
    top_dir = Path( "/rds/general/user/be320/projects/thefarm2/live/Firefly/Calcium_Voltage_Imaging",f'{cell_line}')
    savestr = ""

elif os.name == "nt":
    HPC = False
    top_dir = Path( "/rds/general/user/be320/projects/thefarm2/live/Firefly/Calcium_Voltage_Imaging",f'{cell_line}')
    savestr = ""
else:
    HPC = False
    top_dir = Path(home, f"data/Firefly/Calcium_Voltage_Imaging/{cell_line}")
    savestr = ""


save_file = Path(
    top_dir, "analysis",
    "dataframes",
    f"long_acqs_{datetime.datetime.now().year}{datetime.datetime.now().month:02}{datetime.datetime.now().day:02}{savestr}.csv",
)


df = get_tif_smr(
    top_dir, save_file,min_date , max_date , cell_line = cell_line,
)

dates = []
slips = []
areas = []
expt = []
trial_string = []
folder = []

for data in df.itertuples():
    s = data.tif_file

    par = Path(s).parts

    dates.append(par[par.index(f"{cell_line}") + 1][-8:])
    #print(dates[-1])
    slips.append(s[s.find("slip") + len("slip") : s.find("slip") + len("slip") + 1])

    if "area" in s:
        areas.append(s[s.find("area") + len("area") : s.find("area") + len("area") + 1])
    else:
        areas.append(s[s.find("cell") + len("cell") : s.find("cell") + len("cell") + 1])

    folder.append(Path(s).parent)
    last_part = par[-1]  # Get the actual filename

    segments = last_part.split("_")  # Split filename by '_'
    
    trial_string.append("_".join(segments[:3]))
    
    extracted_parts = []  # List to store extracted substrings

    # Loop through segments and find the part between "areaX" and "_X"
    for i in range(len(segments)):
        if segments[i].startswith("area") and i < len(segments) - 1:  
            # Take all parts until the next '_X' (which usually indicates MMStack or numbering)
            for j in range(i + 1, len(segments)):
                if segments[j].isdigit():  # Stop if it's a number (like _1)
                    break
                extracted_parts.append(segments[j])  # Store the valid part
            
            break  # Stop processing once we found and extracted the experiment name

    expt.append("_".join(extracted_parts) if extracted_parts else None ) # Join extracted parts with '_'



df['folder'] = folder
df["date"] = dates
df["slip"] = slips
df["area"] = areas
df["trial_string"] = trial_string
df['expt'] = expt
df['microscopy_setup']=[]
df['use']=[]


# drop bad goes
df = df[df["multi_tif"] != 0]

df = df.sort_values(by=["date", "slip", "area"])

df.to_csv(
    Path(
        top_dir,
        "analysis", 'dataframes',
        f"long_acqs_{cell_line}_{datetime.datetime.now().year}{datetime.datetime.now().month:02}{datetime.datetime.now().day:02}{savestr}_labelled.csv",
    )
)