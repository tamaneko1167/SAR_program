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
import sys
import statistics
from scipy.interpolate import lagrange
import scipy.interpolate as scipl
from scipy import signal
import SA_func_revise as sa

light_speed = sa.light_speed
df = sa.df
dr = sa.dr
ad_samp_point = sa.ad_samp_point
chirp_rate = sa.chirp_rate

f_array = np.arange(0, df * ad_samp_point, df)
tau_array = f_array / chirp_rate
r_array = tau_array * light_speed / 2  #往復なので半分
print("レンジ理論分解能 [m]: " + str(dr * 2) + "\n")


start = time.time()
dir_name = "../send/2024-06-20/06-20-16-44-49/"
filename = dir_name + "fft_data"
log_name = dir_name + "flight_log"
add_name = "part"

# fft_data = sa.read_fft_datda(filename)
# data = sa.code_V_convert(fft_data)
# raw_data = sa.get_raw_data(data)

conv_az_n = 100
spline_d_array = np.load(dir_name + "spline_d_array.npy")
all_sar_data = np.load(dir_name + "all_sar_data_conv" + str(conv_az_n) +  ".npy")
raw_data = np.load(dir_name + "raw_data" + ".npy")

### 部分的に合成開口画像を表示 ###
az_s_index = 1058
az_e_index = 1108
az_s_index = 610
az_e_index = 650
index = [az_s_index, az_e_index,0, 50]
print("azimuth start: " + str(spline_d_array[az_s_index]) + " m")
print("azimuth end: " + str(spline_d_array[az_e_index]) + " m")
#sa.part_sar_imaging(dir_name, all_sar_data, conv_az_n, index, spline_d_array, add_name)
### 各8ch間の合成開口後の位相差を取得、画像化 ###
#sa.insar_imaging(dir_name, all_sar_data, index, spline_d_array, add_name)

# for range_i in range(11, 17): # 観察したいrangeインデックス
    # sa.range_1dplot(all_sar_data[4], range_i, 750, 850, spline_d_array, dir_name + "range1d_" + str(range_i))

### 部分的に拡大（3dB落ちから分解能を調べたい） ###
index = [1745,1765, 40, 44]
#index = [600,650, 14, 20]
#print(sa.fft2d_expand(dir_name, all_sar_data, index, spline_d_array, conv_az_n, 16)) #これが本物
print(sa.fft2d_expand(dir_name, raw_data, index, spline_d_array, conv_az_n, 16)) #どれくらい拡大した出力画像を得るか
### ある部分（ターゲットの反応部分）についてのチャネルごとの位相値を比較
#sa.channel_phase_compare(all_sar_data, spline_d_array, dir_name + "phase_compare_smallcr", 20 * 2 * np.pi / 360, True)

##限界の合成開口長を出す
# D = 0.076 ##レーダの実開口長
# L = [D] 
# az_t_res, az_m_res, rg_t_res, rg_m_res,length = sa.fft2d_expand(dir_name, raw_data, index, spline_d_array, D, 16)
# az_theori = [az_t_res]
# az_measured = [rg_t_res]

# # L = [] 
# # az_theori = []
# # az_measured = []
# for i in [50,100,150,200]:
#     sar_data = np.load(dir_name + "no_fit_all_sar_data_conv" + str(i) +  ".npy")
#     #sar_data = np.load(dir_name + "fit_all_sar_data_conv" + str(i) +  ".npy")
#     az_t_res, az_m_res, rg_t_res, rg_m_res,length = sa.fft2d_expand(dir_name, sar_data, index, spline_d_array, i, 16)
#     L.append(length)
#     az_theori.append(az_t_res)
#     az_measured.append(az_m_res)
# print()

# plt.plot(L,az_theori,c='b',label='theoritical')
# plt.plot(L,az_measured,c='orange',label='measured')
# plt.rcParams["font.size"] = 8
# plt.xlabel("L")    
# plt.ylabel("Resolution")
# plt.tick_params(labelsize=15)
# plt.legend()
# plt.savefig(dir_name + "no_fit_limit.png", format = "png", bbox_inches = 'tight')
# plt.clf()
# plt.close()

### 複数回観測した干渉位相についての統計処理 ###
filename =[]
for i in range(1, 8):
    # if(i == 5): continue
    filename.append("../1206_data/range_box_" + str(i) + "/phase_compare_bifcr_cut.npy")
# sa.inter_phase_statistics(filename, "../1206_data/range_box_cut_")

end = time.time()
print("実行時間: " + str(end - start))
sys.exit()
