import numpy as np
import os
import matplotlib.ticker as ptick
import seaborn as sns
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
import cmath
import time
from scipy.interpolate import lagrange
import scipy.interpolate as scipl
from scipy import signal
import SA_func as sa

light_speed = sa.light_speed
df = sa.df
dr = sa.dr
ad_samp_point = sa.ad_samp_point
chirp_rate = sa.chirp_rate

f_array = np.arange(0, df * ad_samp_point, df)
tau_array = f_array / chirp_rate
r_array = tau_array * light_speed / 2  #往復なので半分
print("距離分解能: " + str(dr * 2) + "\n")

start = time.time()
dir_name = "../中間報告/Tamami Hata/07-14-12-33-47 (提案手法用)/"
filename = dir_name + "fft_data"
log_name = dir_name + "flight_log"

fft_data = sa.read_fft_data(filename)
data = sa.code_V_convert(fft_data)
raw_data = sa.get_raw_data(data)

conv_az_n = 50
spline_d_array = np.load(dir_name + "spline_d_array.npy")

### 時間領域で畳み込み処理を行い、合成開口 ###
index = [0, sa.az_n, 0, 50] #[az_start, az_end, rg_start, rg_end]
# index = [500, 1000, 1, 50]
all_sar_data = np.zeros((sa.ch, (index[1] - index[0]), (index[3] - index[2])), dtype = np.complex64)
for TRX in range(sa.ch):
    all_sar_data[TRX] = sa.back_projection(raw_data[TRX], index, conv_az_n, spline_d_array)

### アジマス全体での合成開口後の配列保存、画像化 ###
np.save(dir_name + "all_sar_data_conv" + str(conv_az_n), all_sar_data)
for TRX in range(sa.ch):
    save_name = ["", "range [m]", "azimuth [m]", dir_name + str(TRX) + "_bpconv" + str(conv_az_n) + "_amp_full"]
    sa.sar_imaging("amp", all_sar_data[TRX], index, spline_d_array, save_name)
    save_name = ["", "range [m]", "azimuth [m]", dir_name + str(TRX) + "_bpconv" + str(conv_az_n) + "_phase_full"]
    sa.sar_imaging("phase", all_sar_data[TRX], index, spline_d_array, save_name)
    
end = time.time()
print("実行時間: " + str(end - start))