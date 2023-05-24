#%%
from utils import data as data_utils

data_utils.sdf_to_arrow("data/pcqm4mv2/raw/pcqm4m-v2-train.sdf", to_disk_location="data/pcqm4mv2/processed/arrow")