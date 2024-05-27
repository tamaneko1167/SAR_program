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
import SA_func_revise as sa
import scipy.signal as signal

light_speed = sa.light_speed
df = sa.df
dr = sa.dr
ad_samp_point = sa.ad_samp_point
chirp_rate = sa.chirp_rate

f_array = np.arange(0, df * ad_samp_point, df)
tau_array = f_array / chirp_rate
r_array = tau_array * light_speed / 2  #往復なので半分
print("レンジ理論分解能: " + str(dr * 2) + "\n")


start = time.time()
dir_name = "../Optimization/07-14-12-33-47/"
print(dir_name)
filename = dir_name + "fft_data"
log_name = dir_name + "flight_log_v.npy"
add_name = ""

fft_data = sa.read_fft_data(filename)
data = sa.code_V_convert(fft_data)
raw_data = sa.get_raw_data(data)
np.save(dir_name + "raw_data" , raw_data)

### 合成開口前の時間領域の生データ画像を保存 ###
#index = [0, sa.az_n, 0, 50] #[az_start, az_end, rg_start, rg_end]
index = [0, sa.az_n, 0, 50]
title_tag = ["TX2RX1", "TX2RX2", "TX2RX3", "TX2RX4","TX1RX1", "TX1RX2","TX1RX3", "TX1RX4"]
for TRX in range(sa.ch):
    save_name = ["Amplitude of " + title_tag[TRX], "range [m]", "azimuth [s]", dir_name + str(TRX) + "_amp" + add_name]
    #sa.heatmap_imaging("amp", raw_data[TRX], index, dr, sa.az_dt, save_name)
    save_name = ["Phase of " + title_tag[TRX], "range [m]", "azimuth [s]", dir_name + str(TRX) + "_phase" + add_name]
    #sa.heatmap_imaging("phase", raw_data[TRX], index, dr, sa.az_dt, save_name)

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

### raw_dataは8つの受信機、アジマス方向に2000px, レンジ方向に512px

###とりあえずレンジ方向に17px目の列の中の極大行列をゲットしたい
##まずはレンジ方向の特定, 位相の回転が1番早いpixelを取得する
all_phase = np.angle(raw_data[4])

all_phase_sum = []
phase_sum_max = 0
# 最終的には反射体ごとにレンジ方向の位置も違うはずなのでそれを特定できるようにする
for i in range(len(all_phase[0])):
    #位相の変化の大きさの和をとる
    phase_sum = 0
    for j in range(1,len(all_phase)):
        phase_sum += (abs(all_phase[j,i]-all_phase[j-1,i]))
    all_phase_sum.append(phase_sum)
    if( all_phase_sum[phase_sum_max] < all_phase_sum[i]):
        phase_sum_max = i

#check_rg_index = phase_sum_max
print(phase_sum_max)

## 続いて, アジマス方向に最適な位置を探す
check_data = raw_data[4,:,phase_sum_max]
#check_phase = np.angle(raw_data[4,:,phase_sum_max])
check_amp = np.abs(raw_data[4,:,phase_sum_max])
print(check_amp)

##まず、リフレクタがいる区間を特定する, リフレクタの数分取得できるはず
rif_sec_s = [] #区間の始まり
rif_sec_e = [] #区間の終わり
for i in range(1,len(check_amp)):
    if check_amp[i-1] < 20 and check_amp[i] > 20: 
        ##この25適当なのでここも評価したい
        start = i
    if check_amp[i-1] > 20 and check_amp[i] < 20 and i>start+10: ##要検討
        rif_sec_s.append(start)
        rif_sec_e.append(i)

print(rif_sec_s)
print(rif_sec_e)

## その中で、対称性のあるところを見つける
###左右のΔtの位置の値の差の二乗の和が最小になるように
dt = 1 #Δtの幅 ##理想的なdtはどうやって生成するか
rif_ex = []
for i in range(len(rif_sec_s)):
    part_data = check_data[rif_sec_s[i]:rif_sec_e[i]]
    diff = np.zeros((len(part_data)-dt) , dtype = np.float64)
    num = np.zeros((len(part_data)-dt) , dtype = np.float64)
    dif_min = dt
    for j in range(1,len(part_data)-2):
        diff[j] = np.angle((part_data[j-dt]*np.conj(part_data[j])))**2+np.angle((part_data[j+dt]*np.conj(part_data[j])))**2#これもっと精度いいの考える
        num[j] = rif_sec_s[i]+(j-1)
        if diff[dif_min]>diff[j]:
            dif_min = j+1
    rif_ex.append(rif_sec_s[i] + dif_min)
    plt.scatter(num,diff,s=10)
    plt.rcParams["font.size"] = 8
    plt.xlabel("t")
    plt.ylabel("diff")
    plt.savefig(dir_name + "diff"+str(i)+".png", format = "png", bbox_inches = 'tight')
    #print("diff" + " image was saved\n")
    plt.clf()
    plt.close()

print(rif_ex)

##複数個の補正を行いたい
#checkポイント自動認識プログラム作成したい
check_az_index = [261, 630, 1083]
check_az_index = rif_ex
#check_rg_index = [17, 17, 17]
check_rg_index = [phase_sum_max, phase_sum_max, phase_sum_max]
# print("target pixel")
# print("azimuth [s] ... " + str(check_az_index[1] * sa.az_dt))
# print("range [m] ... " + str(check_rg_index[1] * dr))
# print("\n")

conv_az_n =  150

###測定したままの速度データによる経路
v_lis = np.load(log_name) #測定した速度データ
spline_d_array = sa.integrate_velocity(v_lis)

## スプライン補間後の飛行パスグラフ化
#sa.make_path_graph(spline_d_array[:1000], sa.az_dt, 10, 1001, dir_name + "spline_d_array")
# print("azimuth [m] ... " + str(spline_d_array[check_az_index]))
compare_data = sa.get_compare_data(spline_d_array, raw_data[4], check_az_index[2], check_rg_index[2], conv_az_n)
sa.compare_imaging(compare_data, dir_name + "test_compare_conv" + str(conv_az_n), check_az_index[2], check_rg_index[2], conv_az_n)
np.save(dir_name + "spline_d_array", spline_d_array)

no_fit_v_lis = [v_lis[i] for i in range(len(v_lis))]
fit_v_lis = [v_lis[i] for i in range(len(v_lis))]
### 軌道補正を行う
#横軸v, 縦軸(理論と実測の位相の)差分二乗の和のグラフを生成し、その中での極小値を割り出す
for i in range(len(check_rg_index)):
    v = np.arange(0,2,0.02) #考慮するvの範囲(要検討)
    origin_v_lis = [v_lis[i] for i in range(len(v_lis))]

    # 回転速度を合わせたいアジマス方向の幅を考えない場合
    no_ex_min_dv = sa.fit_dv(i,v,dir_name,origin_v_lis,raw_data[4], check_az_index[i], check_rg_index[i], conv_az_n)
    no_fit_v_lis = sa.revise_v_lis(v[no_ex_min_dv], no_fit_v_lis, check_az_index[i], conv_az_n)
    print("no_velocity="+str(v[no_ex_min_dv]))
    no_fit_spline_d_array = sa.integrate_velocity(no_fit_v_lis)

    # 回転速度を合わせたいアジマス方向の幅, 位相一回転分になるように自動取得したい
    fit_range = 2
    for j in range(2,len(raw_data[4])-1,2): 
        #print(raw_data[4,j,17])
        if (np.angle(raw_data[4][check_az_index[i]-int(j/2)][17]) < 0 and np.angle(raw_data[4][check_az_index[i]-int((j-2)/2)][17]) > 0) or \
            (np.angle(raw_data[4][check_az_index[i]+int(j/2)][17]) < 0 and np.angle(raw_data[4][check_az_index[i]+int((j-2)/2)][17]) > 0):
            fit_range = j
            print(fit_range)
            break
    print(check_az_index[i]-int(fit_range/2), check_az_index[i]+int(fit_range/2))
    ex_min_dv = sa.fit_dv(i,v,dir_name,origin_v_lis,raw_data[4], check_az_index[i], check_rg_index[i], fit_range)
    
    #ex_min_dv = sa.fit_dv(i,v,dir_name,origin_v_lis,raw_data[4], check_az_index[i], check_rg_index[i], conv_az_n)
    ##修正した速度データを使う
    #fit_v_lis = sa.revise_v_lis(0.48, fit_v_lis, check_az_index[i], fit_range)
    ##何番目を取り出してくるか考える
    print("velocity="+str(v[ex_min_dv]))
    fit_v_lis = sa.revise_v_lis(v[ex_min_dv], fit_v_lis, check_az_index[i], fit_range)
    fit_spline_d_array = sa.integrate_velocity(fit_v_lis)

    ## スプライン補間後の飛行パスグラフ化
    sa.make_path_graph([[spline_d_array[:1500],no_fit_spline_d_array[:1500],fit_spline_d_array[:1500]],['no revise','no fitting','optimize']], sa.az_dt, 10, 1501, dir_name + "fit_spline_d_array")
    print("azimuth [m] ... " + str(fit_spline_d_array[check_az_index]))
    fit_compare_data = sa.get_compare_data(fit_spline_d_array, raw_data[4], check_az_index[i], check_rg_index[i], conv_az_n)
    fit_compare_data[0]-fit_compare_data[1]
    sa.compare_imaging(fit_compare_data, dir_name + "no_revise_test_compare_conv" + str(conv_az_n)+"_"+str(i), check_az_index[i], check_rg_index[i], conv_az_n)

np.save(dir_name + "fit_spline_d_array", fit_spline_d_array)

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
