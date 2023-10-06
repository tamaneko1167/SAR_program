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

### 2地点の干渉位相差から標高差を求める
h = 20.0
B_perp = sa.l_d
theta_topo_1 = -0.822 # -1.355 -0.328
theta_topo_2 = -0.631 # -0.991 -0.781
theta_topo_3 = -0.517
R_1 = 49.4628
R_2 = 52.1856
R_3 = 54.0007

d_H = sa.wl / (2 * np.pi * B_perp) * (theta_topo_1 * np.sqrt(R_1**2 - h**2) - theta_topo_2 * np.sqrt(R_2**2 - h**2))
print(d_H)

### 地形縞の具体値の確認
for rx_second in range(3):
    for rx_first in range(rx_second + 1, 4):
        height = 5.0
        # Rg = 8.5 #グランドレンジ距離
        Rp = 10.45
        Rg = np.sqrt(10.45**2 - height**2)
        H = 0.33 #ターゲットの標高
        theta = 20 * 2 * np.pi / 360
        phi = np.arccos((height - H + sa.l_d * np.sin(theta) * (3 - rx_first)) / Rp)
        B_perp = (rx_first - rx_second) * sa.l_d * np.sin(theta + phi) #アンテナの傾きに応じて本来は少し短くなるが、近似的にそのまま
        print(B_perp, sa.l_d)
        theta = 2 * np.pi / sa.wl * B_perp * H / (Rp * np.sin(phi))
        print(theta)

sys.exit()

### レンジ方向にアンテナが並んでいる際の、ターゲットとアンテナ間の経路差と位相差
l_d = sa.wl / 2
height = 5.0
H = 0
R_g = 8.5
theta = 20 * 2 * np.pi / 360
print("range方向にアンテナ並び\n")
for i in range(3):
    R_p = np.sqrt((height - H + l_d * np.sin(theta) * i)**2 + (R_g - l_d * np.cos(theta) * i)**2)
    R_s = np.sqrt((height - H + l_d * np.sin(theta) * 3)**2 + (R_g - l_d * np.cos(theta) * 3)**2)
    dl = abs(R_p - R_s)
    print("経路差: " + str(dl))
    in_phase = dl * 2 * np.pi / sa.wl
    print("位相差: " + str(in_phase) + "\n")


### 軌道縞の具体値の確認
l_d = sa.wl / 2
height = 5.0
theta = 20 * 2 * np.pi / 360
orb_image = np.zeros((200, 50), dtype = np.complex64)
for i in range(3):
    for r in range(50):
        R_g = r * sa.dr
        R_p = np.sqrt((height + l_d * np.sin(theta) * i)**2 + (R_g - l_d * np.cos(theta) * i)**2)
        R_s = np.sqrt((height + l_d * np.sin(theta) * 3)**2 + (R_g - l_d * np.cos(theta) * 3)**2)
        dR = abs(R_p - R_s)
        in_phase = dR * 2 * np.pi / sa.wl
        orb_image[:, r] = np.exp(1j * in_phase)
    save_name = ["", "grand range [m]", "azimuth [s]", "orb" + str(i)]
    sa.heatmap_imaging("phase", orb_image, [0, 200, 0, 50], sa.dr, 0.1, save_name)

### アジマス方向にアンテナが並んでいる際の、ターゲットとアンテナ間の経路差と位相差
R_g = 2.5
height = 5.0
print("azimuth方向にアンテナ並び\n")
l_array = [sa.wl / 2, sa.wl, 3 * sa.wl / 2]
for l in l_array:
    R_p = np.sqrt(height**2 + R_g**2)
    R_s = np.sqrt(R_p**2 + l**2)
    dl = abs(R_p - R_s)
    print("経路差: " + str(dl))
    in_phase = dl * 2 * np.pi / sa.wl
    print("位相差: " + str(in_phase) + "\n")

### 生データの確認
raw_data = sa.read_raw_data("../1226_data/east_1/raw_data")
raw_data = sa.code_V_convert(raw_data)
print(raw_data[0, :, 100])

for i in range (sa.ch):
    fig = plt.figure(figsize = (8,6))
    plt.rcParams["font.size"] = 16
    plt.plot(raw_data[i, 1000, :256])
    x_step = 16
    plt.xticks(np.arange(0, 256, step = x_step), np.round(np.arange(0, 256 * sa.rg_dt * 10**6, x_step * sa.rg_dt * 10**6), 2), fontsize = 14, rotation = 90)
    plt.xlabel("time [μs]")
    # plt.title("range =" + str(np.round(range_i * dr, 2)) + " m")
    # plt.ylabel("amp [dB]")
    plt.savefig("raw_data_" + str(i) + ".pdf", format = "pdf", bbox_inches = 'tight')
    plt.clf()
    plt.close()

### ドローンの高さ、ターゲットとのグランドレンジ距離にどれほどの誤差が生まれると、ターゲットまでのスラントレンジ距離にどれほど影響するか
height = 5.0
R_g = 2.5
delta_array = np.arange(-0.1, 0.1, 0.005, dtype = np.float64) #誤差の範囲と誤差の間隔
n = np.shape(delta_array)[0]
slant_range_matrix = np.zeros((n, n), dtype = np.float64)
wl_matrix = np.zeros((n, n), dtype = np.float64)

R_t = np.sqrt(height**2 + R_g**2)
for i in range(n):
    for j in range(n):
        dh = delta_array[i]
        drg = delta_array[j]
        R = np.sqrt((height + dh)**2 + (R_g + drg)**2)
        slant_range_matrix[i, j] = R - R_t
        if(abs(R - R_t) < sa.wl / 2): wl_matrix[i, j] = 1
        else: wl_matrix[i, j] = 0
# 横軸グランドレンジ距離、縦軸高さで、スラントレンジ距離の変化を確認
plt.figure(figsize = (12,8))
sns.heatmap(slant_range_matrix, cmap = "jet")
plt.ylabel("height")
plt.xlabel("grand_range")
plt.yticks(np.arange(0, n, step = 5), np.round(delta_array[0 : n : 5], 3), fontsize = 14)
plt.xticks(np.arange(0, n, step = 5), np.round(delta_array[0 : n : 5], 3), fontsize = 14)
save_name = "slant_range"
# plt.savefig(save_name + ".pdf", format = "pdf", bbox_inches = 'tight')

# スラントレンジ距離の変化量が半波長以下の部分だけ見られるように
plt.figure(figsize = (12,8))
sns.heatmap(wl_matrix)
plt.ylabel("height")
plt.xlabel("grand_range")
plt.yticks(np.arange(0, n, step = 5), np.round(delta_array[0 : n : 5], 3), fontsize = 14)
plt.xticks(np.arange(0, n, step = 5), np.round(delta_array[0 : n : 5], 3), fontsize = 14)
save_name = "wl_compare"
# plt.savefig(save_name + ".pdf", format = "pdf", bbox_inches = 'tight')