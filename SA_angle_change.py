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
import statistics as stat
from scipy.interpolate import lagrange
import scipy.interpolate as scipl
from scipy import signal
import SA_func as sa

## ターゲットの標高の変化前(Prime)と変化後(Second)のデータセットを用意し，レーダの角度を変えながら
start = time.time()
dir_date = "../1206_data/"
dir_name_p = dir_date + "range_4/"
dir_name_s = dir_date + "range_box_4/"
H_change = True #高さを変えたデータ間ならTrue，同じ環境データならFalse

conv_az_n = 50
spline_d_array_p = np.load(dir_name_p + "spline_d_array.npy")
all_sar_data_p = np.load(dir_name_p + "all_sar_data_conv" + str(conv_az_n) +  ".npy")
spline_d_array_s = np.load(dir_name_s + "spline_d_array.npy")
all_sar_data_s = np.load(dir_name_s + "all_sar_data_conv" + str(conv_az_n) +  ".npy")

tx_tag = ["TX1", "TX2"]
rx_tag = ["RX1-RX2", "RX1-RX3", "RX1-RX4", "RX2-RX3", "RX2-RX4", "RX3-RX4"]

## レーダの傾け角度を変えたときに，地形縞の値（高さ変更前後の干渉位相の差分）がどう変わるか，理論値と一番誤差が小さいときを最小二乗誤差で見つける
min_error_sum = 10000
min_theta_p = 0
min_theta_s = 0
min_true_matrix = np.zeros((2, 6), dtype = np.float64)
for de_theta_p in range(10, 30): # 1°ずつ動かしてみる
    for de_theta_s in range(10, 30):
        theta_p = de_theta_p * 2 * np.pi / 360
        theta_s = de_theta_s * 2 * np.pi / 360
        
        phase_matrix_p = np.zeros((2, 6), dtype = np.float64)
        phase_matrix_s = np.zeros((2, 6), dtype = np.float64)
        for tx in range(2):
            if(tx == 0): Tx = 1
            else: Tx = 0
            Rx = 0
            for rx_p in range(0,3):
                for rx_s in range(rx_p + 1, 4):
                    insar_data_p = all_sar_data_p[tx * 4 + rx_p] * all_sar_data_p[tx * 4 + rx_s].conjugate()
                    insar_data_p = sa.orbital_phase_cut(insar_data_p, rx_p, rx_s, [0, sa.az_n, 0, 50], 5.0, theta_p)
                    phase_matrix_p[Tx, Rx] = sa.calc_average_phase(insar_data_p, spline_d_array_p)
                    insar_data_s = all_sar_data_s[tx * 4 + rx_p] * all_sar_data_s[tx * 4 + rx_s].conjugate()
                    insar_data_s = sa.orbital_phase_cut(insar_data_s, rx_p, rx_s, [0, sa.az_n, 0, 50], 5.0, theta_s)
                    phase_matrix_s[Tx, Rx] = sa.calc_average_phase(insar_data_s, spline_d_array_s)
                    Rx += 1
        
        dif_matrix = phase_matrix_s - phase_matrix_p

        true_phase = np.zeros((2, 6), dtype = np.float64)
        if(H_change):
            #理論的な地形縞
            theta_array = np.zeros(6, dtype = np.float64)
            index = 0
            for rx_second in range(3):
                for rx_first in range(rx_second + 1, 4):
                    height = 5.0
                    Rg = 8.5 #グランドレンジ距離
                    H = 0.33 #ターゲットの標高
                    phi = np.arctan((Rg - sa.l_d * np.cos(theta_s) * (3 - rx_first)) / (height - H + sa.l_d * np.sin(theta_s) * (3 - rx_first)))
                    B_perp = (rx_first - rx_second) * sa.l_d * np.sin(theta_s + phi) #アンテナの傾きに応じて本来は少し短くなるが、近似的にそのまま
                    
                    theta_array[index] = 2 * np.pi / sa.wl * B_perp * H / (Rg - sa.l_d * np.cos(theta_s) * (3 - rx_first))
                    index += 1
            ## ターゲットの標高を変えた際の理論的な位相変化
            for i in range(6):
                true_phase[:, i] = theta_array[i]
        else:
            ave = np.average(dif_matrix)
            for i in range(6):
                true_phase[:, i] = ave

        error_sum = np.sum((dif_matrix - true_phase)**2)
        
        if(min_error_sum > error_sum):
            plt.figure(figsize = (9,2))
            sns.heatmap(dif_matrix, cmap = "hsv", annot = True, vmax=np.pi, vmin=-np.pi, xticklabels = rx_tag, yticklabels = tx_tag, fmt='.3f')
            # plt.savefig(dir_date + "in_phase_min.pdf", format = "pdf", bbox_inches = 'tight')
            plt.clf()
            plt.close()
            min_theta_p, min_theta_s = de_theta_p, de_theta_s
            min_error_sum = error_sum
            min_true_matrix = true_phase
print(min_theta_p, min_theta_s)
print(min_true_matrix)
print(min_error_sum)
plt.figure(figsize = (9,2))
min_true_matrix=[[0.112, 0.224, 0.336, 0.112, 0.224, 0.112], [0.112, 0.224, 0.336, 0.112, 0.224, 0.112]]
sns.heatmap(min_true_matrix, cmap = "hsv", annot = True, vmax=np.pi, vmin=-np.pi, xticklabels = rx_tag, yticklabels = tx_tag, fmt='.3f')
plt.savefig(dir_date + "true_phase_matrix.pdf", format = "pdf", bbox_inches = 'tight')
plt.clf()
plt.close()
