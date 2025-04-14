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
import SA_func as sa
import glob

light_speed = sa.light_speed
df = sa.df
dr = sa.dr
ad_samp_point = sa.ad_samp_point
chirp_rate = sa.chirp_rate

f_array = np.arange(0, df * ad_samp_point, df)
tau_array = f_array / chirp_rate
r_array = tau_array * light_speed / 2  #往復なので半分
print("レンジ理論分解能: " + str(dr * 2) + "¥n")


start = time.time()
folders = os.listdir('/Users/hatatamami/Library/CloudStorage/OneDrive-TheUniversityofTokyo/ドローンSAR/中間報告/Tamami Hata')
sorted_folders = sorted(folders, reverse=True)
#最新のフォルダを取得する
#dir_name = '/Users/hatatamami/Library/CloudStorage/OneDrive-TheUniversityofTokyo/ドローンSAR/send/' + sorted_folders[0] + '/'
dir_name = "../send/2024-06-20/06-20-16-44-49/"
print(dir_name)
filename = dir_name + "fft_data"
log_name = dir_name + "flight_log_v.npy"
add_name = ""

fft_data = sa.read_fft_data(filename)
data = sa.code_V_convert(fft_data)
raw_data = sa.get_raw_data(data)
np.save(dir_name + "raw_data" , raw_data)

# 合成開口前の任意の画像範囲における振幅最大値を確認（壁の反応位置を特定したい）
# sa.wall_check(raw_data[4], [0, sa.az_n, 51, 100])
# 取得データの状態で、あるレンジ位置の位相回転を確認
# sa.range_1dplot(raw_data[4], 13, 560, 700, [-1], "test_range_check")

### 合成開口前の時間領域の生データ画像を保存 ###
#index = [0, sa.az_n, 0, 50] #[az_start, az_end, rg_start, rg_end]
index = [0, sa.az_n, 25, 50]
title_tag = ["TX2RX1", "TX2RX2", "TX2RX3", "TX2RX4","TX1RX1", "TX1RX2","TX1RX3", "TX1RX4"]
for TRX in range(sa.ch):
    save_name = ["Amplitude of " + title_tag[TRX], "range [m]", "azimuth [s]", dir_name + str(TRX) + "_amp" + add_name]
    sa.heatmap_imaging("amp", raw_data[TRX], index, dr, sa.az_dt, save_name)
    save_name = ["Phase of " + title_tag[TRX], "range [m]", "azimuth [s]", dir_name + str(TRX) + "_phase" + add_name]
    sa.heatmap_imaging("phase", raw_data[TRX], index, dr, sa.az_dt, save_name)

### 各8ch間の合成開口前の位相差を取得、画像化
tag = ["TX2RX1", "TX2RX2", "TX2RX3", "TX2RX4", "TX1RX1", "TX1RX2", "TX1RX3", "TX1RX4"]
for rx_first in range(8):
    for rx_second in range(rx_first + 1, (int(rx_first / 4 + 1)) * 4):
        phase_dif_data = raw_data[rx_first] * raw_data[rx_second].conjugate()
        save_name = ["phase difference " + tag[rx_first] + "-" + tag[rx_second], "range [m]", "azimuth [s]", dir_name + tag[rx_first] + "-" + tag[rx_second] + "_phasedif" + add_name]
        #sa.heatmap_imaging("phase", phase_dif_data, index, dr, sa.az_dt, save_name)
for tx_first in range(4):
    phase_dif_data = raw_data[tx_first] * raw_data[tx_first + 4].conjugate()
    save_name = ["", "range [m]", "azimuth [s]", dir_name + tag[tx_first] + "-" + tag[tx_first + 4] + "_phasedif" + add_name]
    #sa.heatmap_imaging("phase", phase_dif_data, index, dr, sa.az_dt, save_name)

### スプライン補間により参照関数を作成し、生データの位相値と比較し、部分的にグラフ表示 ###
# 参照関数を作るために注目するピクセルは、合成開口前に最も強いピークを有していた場所にしておく
check_az_index = 1220
check_rg_index = 29
print("target pixel")
print("azimuth [s] ... " + str(check_az_index * sa.az_dt))
print("range [m] ... " + str(check_rg_index * dr))
print("¥n")
conv_az_n = 100
spline_d_array = np.zeros(sa.az_n, dtype = np.float64)
v_platform = sa.spline_interpolation(dir_name, log_name)
v_platform += 0 #位相回転の速度を合わせるための大雑把な補正
for i in range(1, sa.az_n):
    spline_d_array[i] = spline_d_array[i - 1] + sa.az_dt * v_platform[i]

## スプライン補間後の飛行パスグラフ化
#sa.make_path_graph(spline_d_array[:100], sa.az_dt, 10, 101, dir_name + "spline_d_array")
sa.make_path_graph(spline_d_array[:2000], sa.az_dt, 100, 2001, dir_name + "full_spline_d_array")
print("azimuth [m] ... " + str(spline_d_array[check_az_index]))
compare_data = sa.get_compare_data(spline_d_array, raw_data[4], check_az_index, check_rg_index, conv_az_n)
sa.compare_imaging(compare_data, dir_name + "test_compare_conv" + str(conv_az_n), check_az_index, check_rg_index, conv_az_n)
np.save(dir_name + "spline_d_array", spline_d_array)


end = time.time()
print("実行時間: " + str(end - start))
sys.exit()

### 生データと参照関数の相互相関を確認 ###
corr_window = 50
max_index = np.zeros(conv_az_n - corr_window, dtype = np.int64) #相互相関における最大ピークのインデックスを格納
for i in range(0, conv_az_n - corr_window):
    corr = np.correlate(np.angle(compare_data[0, i : i + corr_window]), np.angle(compare_data[1, i : i + corr_window]), "full")
    save_name = "corr_image/test_corr_" + str(i)
    #sa.corr_imaging(abs(corr), save_name)
    max_index[i] = np.argmax(abs(corr))
    #maxid = signal.argrelmax(abs(corr), order=1) #ピークを持つ複数のインデックスを取得
'''
# 相互相関値を用いて、スプライン補間した飛行経路を補正（？）
zero_pad = np.zeros(corr_window, dtype = np.float64)
corr_array = np.concatenate([zero_pad, max_index - corr_window, zero_pad])
gaus_window = signal.gaussian(corr_window, std = 0.45) #ガウス窓、標準偏差はガウス値の平均が0.1くらいになるように決めている
for n in range(conv_az_n):
    az = int(check_az_index - conv_az_n / 2 + n)
    #print(np.sum(corr_array[n : n + corr_window] * gaus_window) * sa.az_dt * v_platform[az])
    spline_d_array[az] -= np.sum(corr_array[n : n + corr_window] * gaus_window) * sa.az_dt * v_platform[az]
compare_data = sa.get_compare_data(spline_d_array, raw_data, check_az_index, check_rg_index, conv_az_n)
sa.compare_imaging(compare_data, "test_compare_conv_re" + str(conv_az_n), check_az_index, check_rg_index, conv_az_n)

np.save(dir_name + "spline_d_array_re", spline_d_array)
'''
