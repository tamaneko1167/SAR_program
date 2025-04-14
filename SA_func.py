from pickle import FALSE
import numpy as np
# import matplotlib.ticker as ptick
import seaborn as sns
# import pandas as pd
import matplotlib 
import matplotlib.pyplot as plt
matplotlib.use("Agg") 
# import math
import cmath
# import time
# import os
# import decimal
import statistics as stat
from scipy.interpolate import lagrange
import scipy.interpolate as scipl


light_speed = 299792458
f_c = 24.15e9 #中心周波数 24.06~24.24?
wl = light_speed / f_c
ad_samp_rate = 1.8e6  #レーダのレンジ方向サンプリングレート
rg_dt = 1 / ad_samp_rate
ad_samp_point = 512 #レーダのレンジ方向サンプリング数
df = ad_samp_rate / ad_samp_point
up_sweep_time = 310e-6  #チャープのスイープ時間
band_width = 180e6  #レーダの帯域幅
chirp_rate = band_width / up_sweep_time
d_tau = df / chirp_rate
dr = d_tau * light_speed / 2 / 2  #ゼロパディングのため、見かけの分解能はさらに半分
ch = 8
az_dt =0.01
az_n = 2000 #アジマス方向のピクセル数
#az_n = 14160
az_time = az_dt * az_n
l_d = wl / 2
l_array = [wl / 2, wl / 2 * 2, wl / 2 * 3]

# レーダソフトが出力してくれるraw_dataを読み込む
def read_raw_data(filename):
    raw_data = np.genfromtxt(filename + ".csv", delimiter = ',')  # 区切り文字としてカンマを指定
    data = np.zeros(raw_data.shape[0] * ch, dtype = np.complex64)
    data = np.reshape(data, (-1, ch))
    for TRX in range(ch):
        for n in range(raw_data.shape[0]):
            data[n, TRX] = raw_data[n, TRX]
    raw_data = np.zeros(data.shape[0] * ch, dtype = np.complex64)
    raw_data = np.reshape(raw_data, (ch, -1, ad_samp_point))
    #print(raw_data)
    for TRX in range(ch):
        raw_data[TRX] = np.reshape(data[:, TRX], (-1, ad_samp_point))
    return raw_data

# レーダソフトが出力してくれるfft_dataを読み込む
def read_fft_data(filename):
    fft_data = np.genfromtxt(filename + ".csv", delimiter = ',')  # 区切り文字としてカンマを指定
    data = np.zeros(fft_data.shape[0] * 8, dtype = np.complex64)
    data = np.reshape(data, (-1, 8))
    for TRX in range(8):
        for n in range(fft_data.shape[0]):
            data[n, TRX] = cmath.rect(fft_data[n, TRX * 2], fft_data[n, TRX * 2 + 1])
    return data

# コード値？で出力されている値を電力値に変換？
def code_V_convert(data):
    V_os = 0
    V_ref = 1.5 #[V]
    V_ge = 0
    bit = 16
    return V_os + (data + 2**(bit - 1)) * (V_ref - V_ge) / 2**(bit)

# 一列になっているデータを、チャネル、レンジ、アジマス、で多次元配列に変換
def get_raw_data(data):
    raw_data = np.zeros(data.shape[0] * 8, dtype = np.complex64)
    raw_data = np.reshape(raw_data, (8, -1, ad_samp_point))
    for TRX in range(ch):
        raw_data[TRX] = np.reshape(data[:, TRX], (-1, ad_samp_point))
    return raw_data

# [合成開口前の画像について] 振幅や位相を、レンジアジマス平面に表示。indexとnameは配列
def heatmap_imaging(cmd, data, index, dx, dy, name):
    all_font = 20
    plt.rcParams["font.size"] = all_font
    az_s_index = index[0]
    az_e_index = index[1]
    az_len = az_e_index - az_s_index
    rg_s_index = index[2]
    rg_e_index = index[3]
    rg_len = rg_e_index - rg_s_index
    plt.figure(figsize = (12,8))
    if(cmd == "amp"):
        sns.heatmap(20 * np.log10(abs(data[az_s_index:az_e_index, rg_s_index:rg_e_index])), cmap = "jet", vmin = -30, vmax = 30)
        plt.text(rg_len + 9, -30, "[dB]", fontsize = all_font) # カラーバーの単位を手動でテキスト表示
    if(cmd == "phase"):
        sns.heatmap(np.angle(data[az_s_index:az_e_index, rg_s_index:rg_e_index]), cmap = "hsv", vmin = -np.pi, vmax = np.pi)
        plt.text(rg_len + 8, -30, "[rad]", fontsize = all_font) # カラーバーの単位を手動でテキスト表示
    if(cmd == "ref"):
        sns.heatmap(abs(data[az_s_index:az_e_index, rg_s_index:rg_e_index]), cbar = False)
    x_step = int(rg_len / 25) + 1
    y_step = int(az_len / 20)
    # x_step = 6
    # y_step = 300
    plt.xticks(np.arange(0, rg_e_index - rg_s_index, step = x_step), np.round(np.arange(rg_s_index * dx, rg_e_index * dx, step = dx * x_step), 2), fontsize = all_font, rotation = 90)
    plt.yticks(np.arange(0, az_e_index - az_s_index, step = y_step), np.round(np.arange(az_s_index * dy, az_e_index * dy, step = dy * y_step), 2), fontsize = all_font)
    plt.title(name[0])
    plt.xlabel(name[1], fontsize = all_font)
    plt.ylabel(name[2], fontsize = all_font)
    plt.tight_layout()
    plt.savefig(name[3] + ".png", format = "png", bbox_inches = 'tight')
    print(name[3] + " PDFfile was saved\n")
    plt.clf()
    plt.close()

import matplotlib.pyplot as plt
import numpy as np

def plot_radar_and_acceleration(cmd, data, index, dx, dy, name, acceleration_data, selected_range_idx):
    """
    レーダーデータの一レンジ方向のデータを折れ線グラフとしてプロットし、
    加速度データと並べて表示する。

    Parameters:
        cmd: str - "amp" (振幅) または "phase" (位相)
        data: ndarray - レーダーデータ
        index: list - [az_s_index, az_e_index, rg_s_index, rg_e_index]
        dx, dy: float - レーダーデータの解像度
        name: list - [タイトル, x軸ラベル, y軸ラベル, 出力ファイル名]
        acceleration_data: ndarray - 加速度データ
        selected_range_idx: int - どのレンジ方向のデータを取得するかのインデックス
    """
    all_font = 20
    plt.rcParams["font.size"] = all_font

    az_s_index = index[0]
    az_e_index = index[1]
    az_len = az_e_index - az_s_index

    # 📌 **cmd に応じてデータを選択**
    if cmd == "amp":
        radar_data = 20 * np.log10(abs(data[az_s_index:az_e_index, selected_range_idx]))
        radar_label = "Radar Intensity (dB)"
    elif cmd == "phase":
        radar_data = np.angle(data[az_s_index:az_e_index, selected_range_idx])
        radar_label = "Radar Phase (radian)"
    else:
        raise ValueError("Invalid cmd. Use 'amp' or 'phase'.")

    # 時間軸の作成
    time_indices = np.arange(az_len) * dy  # `dy` を時間軸として適用
    time_acceleration = np.arange(len(acceleration_data)) * dy * 10  # **10倍スケールの時間軸**

    # 📌 **レーダーデータと加速度データを並べてプロット**
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # 📌 **レーダーデータの折れ線グラフ**
    axes[0].plot(time_indices, radar_data, color='b', marker='o', linestyle='-', label=radar_label, linewidth=2)
    axes[0].set_ylabel(radar_label, fontsize=all_font)
    axes[0].set_title(f"{name[0]} - Range Index {selected_range_idx}", fontsize=all_font)
    axes[0].legend()
    axes[0].grid(True)

    # 📌 **加速度データの折れ線グラフ (10倍の時間スケール)**
    axes[1].plot(time_acceleration, acceleration_data, color='r', marker='o', linestyle='-', label="Acceleration Data", linewidth=2)
    axes[1].set_xlabel("Time (s)", fontsize=all_font)
    axes[1].set_ylabel("Acceleration (m/s²)", fontsize=all_font)
    axes[1].set_title("Acceleration Over Time (Scaled 10x)", fontsize=all_font)
    axes[1].legend()
    axes[1].grid(True)

    # 📌 **間隔を調整**
    plt.subplots_adjust(hspace=0.3)

    # 📌 **画像として保存**
    plt.savefig(name[3] + f"_range_{selected_range_idx}_{cmd}.png", format="png", bbox_inches="tight")

    # 📌 **明示的に表示**
    plt.show()

    print(name[3] + f" Range {selected_range_idx} {cmd} Overlay Image was saved\n")

# [合成開口後の画像について] 振幅や位相を、レンジアジマス平面に表示。indexとnameは配列
def sar_imaging(cmd, sar_data, index, az_d_array, name):
    all_font = 20
    plt.rcParams["font.size"] = all_font
    az_s_index = index[0]
    az_e_index = index[1]
    az_len = az_e_index - az_s_index
    rg_s_index = index[2]
    rg_e_index = index[3]
    rg_len = rg_e_index - rg_s_index
    
    plt.figure(figsize = (12,8))
    if(cmd == "amp"):
        sns.heatmap(20 * np.log10(abs(sar_data[az_s_index : az_e_index, rg_s_index : rg_e_index])), cmap = "jet", vmin = -30, vmax = 30)
        plt.text(rg_len + 9, -30, "[dB]") # カラーバーの単位を手動でテキスト表示
    if(cmd == "phase"):
        sns.heatmap(np.angle(sar_data[az_s_index : az_e_index, rg_s_index : rg_e_index]), cmap = "hsv")
        plt.text(rg_len + 8, -30, "[rad]") # カラーバーの単位を手動でテキスト表示
    x_step = int(rg_len / 25) + 1
    y_step = int(az_len / 20) + 1
    # x_step = 6
    # y_step = 80
    
    #plt.yticks(np.arange(0, az_e_index - az_s_index, step = y_step), np.round(np.arange(az_s_index * az_dt, az_e_index * az_dt, step = az_dt * y_step), 2), fontsize = all_font)
    plt.yticks(np.arange(0, az_len, step = y_step), np.round(az_d_array[az_s_index : az_e_index : y_step], 2), fontsize = all_font)
    plt.xticks(np.arange(0, rg_len, step = x_step), np.round(np.arange(rg_s_index * dr, rg_e_index * dr, step = dr * x_step), 2), fontsize = all_font, rotation = 90)
    plt.title(name[0])
    plt.xlabel(name[1], fontsize = all_font)
    plt.ylabel(name[2], fontsize = all_font)
    plt.savefig(name[3] + ".png", format = "png", bbox_inches = 'tight')
    print(name[3] + " PDFfile was saved\n")
    plt.clf()
    plt.close()

# 補間した飛行経路をアジマス時間軸上に図示するための関数
def make_path_graph(data, dt, x_step, x_max, save_name):
    fig = plt.figure(figsize = (8,6))
    plt.rcParams["font.size"] = 16
    plt.plot(data, marker = '.', linestyle='')
    #plt.scatter(data, marker = '.', linestyle='', s=0.5)
    #plt.xticks(np.arange(0, x_max, step = x_step), np.round(np.arange(0, x_max * dt, step = dt * x_step), 2), fontsize = 14)
    plt.xticks(np.arange(0, x_max, step = x_step), np.round(np.arange(0, x_max * dt, step = dt * x_step), 2), fontsize = 12)
    plt.xlabel("azimuth [t]")
    plt.ylabel("path length [m]")
    plt.tight_layout()
    plt.savefig(save_name + ".png", format = "png", bbox_inches = 'tight')
    print(save_name + " PDFfile was saved\n")
    plt.clf()
    plt.close()

# スプライン補間によって、フライトログで測定仕切れない部分の速度を補間
def spline_interpolation(dir_name, log_name):
    v_lis = np.load(log_name)
    az_d_array = np.zeros(int(az_n / 10), dtype = np.float64) # スプライン補間前の飛行パスグラフ化
    az_d_array[0] = 0
    for i in range(58,68):
        v_lis[i] += 0.8
    for i in range(1, int(az_n / 10)):
        az_d_array[i] = az_d_array[i - 1] + az_dt * v_lis[i] * 10

    ## スプライン補間前の飛行パスグラフ化
    make_path_graph(az_d_array[:10], az_dt * 10, 1, 11, dir_name + "az_d_array")
    
    t_lis = np.arange(0, az_time, 0.1, dtype = np.float64)
    v_sci = scipl.CubicSpline(t_lis, v_lis)
    t_array = np.arange(0, az_time, 0.01)

    return v_sci(t_array)

# 1次元データへの平均化フィルタ
def ave_filter(data, window_n):
    N = data.shape[0]
    ave_data = np.zeros(N, dtype = np.complex64)
    for i in range(N):
        if(i + window_n >= N):
            window_n -= 1
        ave_data[i] = np.average(data[i : i + window_n])
    return ave_data

# 参照関数と測定データを部分的に取得
def get_compare_data(az_d_array, raw_data, az_index, rg_index, conv_az_n):
    compare_data = np.zeros(2 * conv_az_n, np.complex64)
    compare_data = np.reshape(compare_data, (2, -1))

    R_0 = dr * rg_index
    mig_matrix = np.zeros(az_n * ad_samp_point)
    mig_matrix = np.reshape(mig_matrix, (az_n, -1))
    for n in range(conv_az_n):
        az = int(az_index - conv_az_n / 2 + n)
        if(az < 0 or az >= az_n): continue
        az_d_n = az_d_array[az_index] - az_d_array[az]
        R_n = np.sqrt(R_0 ** 2 + az_d_n ** 2)
        theta = (R_n * 2 / wl)* 2 * np.pi  #参照値となる位相
        mig_rg = int((R_n - R_0 + dr / 2) / dr) + rg_index  #マイグレーションを考慮した参照値のレンジインデックス
        if mig_rg < ad_samp_point: mig_matrix[az, mig_rg] = 1
        corr = 1.8 #定数補正
        compare_data[0, n] = raw_data[az, mig_rg]
        compare_data[1, n] = np.exp(1j * (theta+corr))

    ## 参照関数のどの部分を比較しているかを画像表示
    index = [0, az_n, 1, 50]
    az_time = round(az_index * az_dt, 2)
    rg_distance = round(rg_index * dr, 2)
    save_name = ["ref_func az:" + str(az_time) + ",rg:" + str(rg_distance), "range [m]", "azimuth [s]", "ref_func_az" + str(az_time) + "_rg" + str(rg_distance)]
    #heatmap_imaging("ref", mig_matrix, index, dr, az_dt, save_name)

    return compare_data

# 生データと参照関数の位相値を比較するために、重ねて表示
def compare_imaging(compare_data, save_name, az_index, rg_index, conv_az_n):
    fig = plt.figure(figsize = (8,8))
    plt.rcParams["font.size"] = 16
    ax = fig.add_subplot(2, 1, 1)
    title = "range = " + str(np.round(dr * rg_index, 2)) + " [m]"
    plt.title(title)
    plt.plot(20 * np.log10(abs(compare_data[0])))
    #plt.plot(abs(compare_data[0]))
    x_step = int(conv_az_n / 10) + 1
    plt.xticks(np.arange(0, conv_az_n, step = x_step), np.round(np.arange((az_index - conv_az_n / 2) * az_dt, (az_index + conv_az_n / 2) * az_dt, step = az_dt * x_step), 2), fontsize = 14)
    ax.set_ylabel("amp [dB]", fontsize = 20)

    ax = fig.add_subplot(2, 1, 2)
    plt.plot(np.angle(compare_data[0]), label = "measured_data")
    plt.plot(np.angle(compare_data[1]), label = "reference")
    plt.xticks(np.arange(0, conv_az_n, step = x_step), np.round(np.arange((az_index - conv_az_n / 2) * az_dt, (az_index + conv_az_n / 2) * az_dt, step = az_dt * x_step), 2), fontsize = 14)
    ax.set_ylabel("phase [rad]", fontsize = 20)
    plt.legend(bbox_to_anchor=(1, 0), loc='lower right', borderaxespad=1)
    
    # d_compare_data = np.zeros((2, conv_az_n), np.complex64)
    # for i in range(1, conv_az_n):
    #     d_compare_data[0, i] = (np.angle(compare_data[0, i] * compare_data[0, i - 1].conjugate())) / az_dt
    #     # d_compare_data[0, i] = (np.angle(compare_data[0, i]) - np.angle(compare_data[0, i - 1])) / az_dt
    #     d_compare_data[1, i] = (np.angle(compare_data[1, i] * compare_data[1, i - 1].conjugate())) / az_dt
    #     # d_compare_data[1, i] = (np.angle(compare_data[1, i]) - np.angle(compare_data[1, i - 1])) / az_dt
    # ## 取得データに対して線形近似
    # x= np.linspace((az_index - conv_az_n / 2) * az_dt, (az_index + conv_az_n / 2) * az_dt, 50)
    # res = np.polyfit(x, d_compare_data[0], 1)
    # poly_data = np.poly1d(res)(x)

    # ax = fig.add_subplot(3, 1, 3)
    # # plt.plot(ave_filter(d_compare_data[0], 10), label = "raw_data") #生データに対して平均化フィルタをかける
    # plt.plot(d_compare_data[0], label = "measured_data")
    # plt.plot(poly_data, label = "linear approximation", color = 'c')
    # plt.plot(d_compare_data[1], label = "reference")
    # plt.xticks(np.arange(0, conv_az_n, step = x_step), np.round(np.arange((az_index - conv_az_n / 2) * az_dt, (az_index + conv_az_n / 2) * az_dt, step = az_dt * x_step), 2), fontsize = 14)
    ax.set_xlabel("azimuth [s]", fontsize = 20)
    # ax.set_ylabel("phase change ratio [rad/s]", fontsize = 16)
    # plt.legend(bbox_to_anchor=(1, 0), loc='lower right', borderaxespad=1, fontsize = 16)
    # plt.tight_layout()
    plt.savefig(save_name + ".png", format = "png", bbox_inches = 'tight')
    print(save_name + " PDFfile was saved\n")
    plt.clf()
    plt.close()

# 相互相関をグラフ表示
def corr_imaging(corr, save_name):
    fig = plt.figure(figsize = (8,6))
    plt.rcParams["font.size"] = 16
    plt.plot(corr)
    plt.savefig(save_name + ".pdf", format = "pdf", bbox_inches = 'tight')
    print(save_name + " PDFfile was saved\n")
    plt.clf()
    plt.close()

# 時間領域でのアジマス方向畳み込み（back projection）によって合成開口（しているつもり）
def back_projection(data, index, conv_az_n, az_d_array):
    az_s_index = index[0]
    az_e_index = index[1]
    az_len = az_e_index - az_s_index
    rg_s_index = index[2]
    rg_e_index = index[3]
    rg_len = rg_e_index - rg_s_index
    
    sar_data = np.zeros((az_len, rg_len), dtype = np.complex64)

    for Rg in range(rg_s_index, rg_e_index):
        R_0 = dr * Rg #スラントレンジ距離
        for Az in range(az_s_index, az_e_index):
            conv_count = 0
            for n in range(conv_az_n):
                az = int(Az - conv_az_n/2 + n)
                if(az < 0 or az >= az_n): continue
                az_d_n = az_d_array[Az] - az_d_array[az] #アジマス方向の距離
                R_n = np.sqrt(R_0 ** 2 + az_d_n ** 2)
                
                theta = R_n * 2 / wl * 2 * np.pi #参照値となる位相
                rg = int((R_n - R_0 + dr/2) / dr) + Rg

                # 取得データと参照関数を畳み込み
                #if(rg - 1 >= rg_s_index): sar_data[Az - az_s_index, Rg - rg_s_index] += data[az, rg - 1] * np.exp(1j * theta).conjugate()
                sar_data[Az - az_s_index, Rg - rg_s_index] += data[az, rg] * np.exp(1j * theta).conjugate()
                #if(rg + 1 <= rg_e_index): sar_data[Az - az_s_index, Rg - rg_s_index] += data[az, rg + 1] * np.exp(1j * theta).conjugate()
                conv_count += 1
            sar_data[Az - az_s_index, Rg - rg_e_index] /= conv_count

    return sar_data

# 合成開口前後の画像において、あるレンジでの振幅、位相の様子を1次元表示
# 合成開口前の画像を扱う場合（スプライン補間を行う前）は、az_d_array = [-1]にする（横軸が秒になる）
def range_1dplot(data, range_i, az_s_index, az_e_index, az_d_array, save_name):
    az_len = az_e_index - az_s_index
    x_step = int((az_e_index - az_s_index) / 25)
    fig = plt.figure(figsize = (8,6))
    plt.rcParams["font.size"] = 16
    plt.plot(20 * np.log10(abs(data[az_s_index : az_e_index, range_i])))
    if az_d_array.shape[0] == 1 and az_d_array == [-1]:
        plt.xticks(np.arange(0, az_len, step = x_step), np.round(np.arange(az_s_index * az_dt, az_e_index * az_dt, x_step * az_dt), 2), fontsize = 14, rotation = 90)
        plt.xlabel("azimuth [s]")
    else:
        plt.xticks(np.arange(0, az_len, step = x_step), np.round(az_d_array[az_s_index : az_e_index : x_step], 2), fontsize = 14, rotation = 90)
        plt.xlabel("azimuth [m]")
    plt.title("range =" + str(np.round(range_i * dr, 2)) + " m")
    plt.ylabel("amp [dB]")
    plt.savefig(save_name + "_amp.pdf", format = "pdf", bbox_inches = 'tight')
    plt.clf()
    plt.close()

    fig = plt.figure(figsize = (8,6))
    plt.rcParams["font.size"] = 16
    plt.plot(np.angle(data[az_s_index : az_e_index, range_i]))
    if az_d_array.shape[0] == 1 and az_d_array == [-1]:
        plt.xticks(np.arange(0, az_len, step = x_step), np.round(np.arange(az_s_index * az_dt, az_e_index * az_dt, x_step * az_dt), 2), fontsize = 14, rotation = 90)
        plt.xlabel("azimuth [s]")
    else:
        plt.xticks(np.arange(0, az_len, step = x_step), np.round(az_d_array[az_s_index : az_e_index : x_step], 2), fontsize = 14, rotation = 90)
        plt.xlabel("azimuth [m]")
    plt.title("range =" + str(np.round(range_i * dr, 2)) + " m")
    plt.ylabel("phase")
    plt.savefig(save_name + "_phase.pdf", format = "pdf", bbox_inches = 'tight')
    print(save_name + " PDFfile was saved\n")
    plt.clf()
    plt.close()

# 合成開口（時間領域の畳み込み）後のデータのch同士の位相差を画像化
def phase_difference(all_sar_data, index, az_d_array, save_name):
    az_s_index = index[0]
    az_e_index = index[1]
    az_len = az_e_index - az_s_index
    rg_s_index = index[2]
    rg_e_index = index[3]
    rg_len = rg_e_index - rg_s_index
    
    # TX1-RX1とTX1-RX2の位相差
    data = all_sar_data[4] * all_sar_data[5].conjugate()
    
    x_step = int(rg_len / 25)
    y_step = int(az_len / 20) + 1
    plt.figure(figsize = (9,6))
    sns.heatmap(np.angle(data[az_s_index : az_e_index, rg_s_index : rg_e_index]), cmap = "hsv")
    plt.yticks(np.arange(0, az_len, step = y_step), np.round(az_d_array[az_s_index : az_e_index : y_step], 2), fontsize = 14)
    plt.xticks(np.arange(0, rg_len, step = x_step), np.round(np.arange(rg_s_index * dr, rg_e_index * dr, step = dr * x_step), 2), fontsize = 14, rotation = 90)
    #plt.title("az_t - rg_t(Amp)", fontsize = 20)
    plt.xlabel("range [m]", fontsize = 20)
    plt.ylabel("azimuth [m]", fontsize = 20)
    plt.tight_layout()
    plt.savefig(save_name + "_phasedif.pdf", format="pdf", bbox_inches='tight')
    print(save_name + " PDFfile was saved\n")
    plt.clf()
    plt.close()

# 合成開口前のデータから、壁の反応インデックスを確認する
def wall_check(data, index):
    az_s_index = index[0]
    az_e_index = index[1]
    rg_s_index = index[2]
    rg_e_index = index[3]

    flag_matrix = np.zeros((az_n, ad_samp_point), dtype = np.int64)
    for az in range(az_s_index, az_e_index):
        thres = sorted(abs(data[az, rg_s_index:rg_e_index]))[-6]
        for rg in range(rg_s_index, rg_e_index):
            if abs(data[az, rg]) > thres: flag_matrix[az, rg] = 1
    index[2] = 0
    save_name = ["flag", "range [m]", "azimuth [s]", "wall_flag"]
    heatmap_imaging("ref", flag_matrix, index, dr, az_dt, save_name)

# 2次元配列の中で最も大きな値を持つインデックスを取得
def argmax_2d(data_2d):
    max_value = -1
    y_max_index = -1
    x_max_index = -1
    for i in range(data_2d.shape[0]):
        for j in range(data_2d.shape[1]):
            if max_value < abs(data_2d[i, j]):
                max_value = abs(data_2d[i, j])
                y_max_index = i
                x_max_index = j
    return (y_max_index, x_max_index)

# アジマス方向の理論分解能を計算（Initial Evaluation ... の論文参照）
def cal_az_resolution(az_index, rg_index, d_array, conv_az_n):
    squint_theta = 0 * 2 * np.pi / 360
    height = 5.0
    Rs = rg_index * dr #スラントレンジ距離
    print(Rs)
    #Rs = 9.86 #5.59 
    Rg = np.sqrt(Rs**2 - height**2)
    #print(d_array[int(az_index + conv_az_n / 2)] - d_array[int(az_index - conv_az_n / 2)])
    # return wl * R / (np.cos(squint_theta)**2 * 2 * (d_array[int(az_index + conv_az_n / 2)] - d_array[int(az_index - conv_az_n / 2)]))
    return wl * Rs /((np.cos(squint_theta)**2 * 2 * (d_array[int(az_index + conv_az_n / 2)] - d_array[int(az_index - conv_az_n / 2)]))),(d_array[int(az_index + conv_az_n / 2)] - d_array[int(az_index - conv_az_n / 2)])

# 3dB落ちのインデックス幅を調べる
def check_resolution(dB_data):
    max_index = np.argmax(dB_data)
    max_value = dB_data[max_index]
    s_index = max_index
    e_index = max_index
    while(dB_data[s_index] > max_value - 3):
        s_index -= 1
    while(dB_data[e_index] > max_value - 3):
        e_index += 1
    return e_index - s_index

#　合成開口後の画像の一部を詳細に確認するために、2dfft→高周波をゼロ埋め→逆FFT
def fft2d_expand(dir_name, data, index, d_array, conv_az_n, scope):
    all_font = 20
    az_s_index = index[0]
    az_e_index = index[1]
    rg_s_index = index[2]
    rg_e_index = index[3]
    az_len = (az_e_index - az_s_index) * scope
    rg_len = (rg_e_index - rg_s_index) * scope
    TRX = 1

    # フーリエ変換前に最大値インデックスを取得し，合成開口長などから分解能を計算
    (az_max_index, rg_max_index) = argmax_2d(data[TRX, az_s_index : az_e_index, rg_s_index : rg_e_index])
    ##L:
    az_t_res,L = cal_az_resolution(az_max_index + az_s_index, rg_max_index + rg_s_index, d_array, conv_az_n)

    ## 画像の切り出した部分を表示
    plt.figure(figsize = (8,6))
    sns.heatmap(20 * np.log10(abs(data[TRX, az_s_index : az_e_index, rg_s_index : rg_e_index])), cmap = "jet", cbar = False)
    #plt.title("az_t - rg_t(Amp)", fontsize = all_font)
    plt.xlabel("range", fontsize = all_font)
    plt.ylabel("azimuth", fontsize = all_font)
    plt.tick_params(labelbottom = False, bottom = False, labelleft = False, left = False)
    plt.tight_layout()
    save_name = dir_name + "test_part_image"
    plt.savefig(save_name + ".png", format="png", bbox_inches='tight')
    print(save_name + " PDFfile was saved\n")
    plt.clf()
    plt.close()
    
    ## フーリエ変換→ゼロパディング→逆フーリエ変換
    part_az = az_e_index - az_s_index
    part_rg = rg_e_index - rg_s_index
    f = np.fft.fft2(data[TRX, az_s_index : az_e_index, rg_s_index : rg_e_index])
    f_shift = np.fft.fftshift(f)
    zero_pad_data = np.zeros((az_len, rg_len), dtype = np.complex64)
    for az in range(part_az):
        for rg in range(part_rg):
            zero_pad_data[int((az_len - part_az) / 2) + az][int((rg_len - part_rg) / 2) + rg] = f_shift[az][rg]
    f_back_shift = np.fft.ifftshift(zero_pad_data)
    back_data = np.fft.ifft2(f_back_shift)

    ## 拡大後の画像
    plt.figure(figsize = (8,6))
    sns.heatmap(20 * np.log10(abs(back_data)), cmap = "jet", cbar = False)
    #plt.title("az_t - rg_t(Amp)", fontsize = 20)
    plt.xlabel("range", fontsize = 20)
    plt.ylabel("azimuth", fontsize = 20)
    plt.tick_params(labelbottom = False, bottom = False, labelleft = False, left = False)
    plt.tight_layout()
    save_name = dir_name + "test_2dfft"
    plt.savefig(save_name + ".png", format="png", bbox_inches='tight')
    print(save_name + " PDFfile was saved\n")
    plt.clf()
    plt.close()

    ##　画像中の最大ピークを取得
    (az_max_index, rg_max_index) = argmax_2d(back_data)

    ## 拡大範囲内の速度は等速だと仮定するため，1pxの速度を使う
    v_s = d_array[az_s_index + az_max_index+1] - d_array[az_s_index + az_max_index] 
    #print(az_max_index)
    ## アジマス固定，レンジ方向の2dplot
    x_step = 7
    plt.figure(figsize = (8,6))
    plt.rcParams["font.size"] = all_font
    plt.plot(20 * np.log10(abs(back_data[az_max_index, :])))
    plt.xticks(np.arange(0, rg_len, step = x_step), np.round(np.arange(rg_s_index * dr, rg_e_index * dr, x_step * (dr / scope)), 2), fontsize = all_font, rotation = 90)
    plt.xlabel("range [m]")
    plt.ylabel("amp [dB]")
    save_name = dir_name + "range_amp"
    plt.savefig(save_name + ".png", format = "png", bbox_inches = 'tight')
    print(save_name + " PDFfile was saved\n")
    plt.clf()
    plt.close()

    ## レンジ固定，アジマス方向の2dplot
    x_step = 21
    s_index = 0
    e_index = az_len
    plt.figure(figsize = (8,6))
    plt.rcParams["font.size"] = all_font
    plt.plot(20 * np.log10(abs(back_data[s_index:e_index, rg_max_index])))
    x_1 = d_array[az_s_index] + v_s * s_index / scope
    x_2 = d_array[az_s_index] + v_s * e_index / scope
    x_3 = x_step * v_s / scope
    plt.xticks(np.arange(0, e_index - s_index, step = x_step*5), np.round(np.arange(x_1 ,x_2, step = x_3*5), 2), fontsize = 15, rotation = 90)
    #plt.xticks(np.arange(0, e_index - s_index, step = x_step), np.round(np.arange(d_array[az_s_index] + v_s * s_index / scope , d_array[az_s_index] + v_s * e_index / scope, x_step * v_s / scope), 2), fontsize = all_font, rotation = 90)
    plt.xlabel("azimuth [m]")
    plt.ylabel("amp [dB]")
    save_name = dir_name + "azimuth_amp"
    plt.savefig(save_name + ".png", format = "png", bbox_inches = 'tight')
    print(save_name + " PDFfile was saved\n")
    plt.clf()
    plt.close()

    rg_m_res = dr / scope * check_resolution(20 * np.log10(abs(back_data[az_max_index, :])))
    az_m_res = v_s / scope * check_resolution(20 * np.log10(abs(back_data[:, rg_max_index])))
    print(d_array[az_s_index + az_max_index+1] - d_array[az_s_index + az_max_index] )
    rg_t_res = dr * 2
    #print(d_array[az_e_index + 1] - d_array[az_e_index])

    return (az_t_res, az_m_res, rg_t_res, rg_m_res,L)

# 合成開口処理後、部分的に合成開口画像を表示
def part_sar_imaging(dir_name, data, conv_az_n, index, spline_d_array, add_name):
    for TRX in range(ch):
        save_name = ["", "range [m]", "azimuth [m]", dir_name + str(TRX) + "_bpconv" + str(conv_az_n) + "_amp" + add_name]
        sar_imaging("amp", data[TRX], index, spline_d_array, save_name)
        save_name = ["", "range [m]", "azimuth [m]", dir_name + str(TRX) + "_bpconv" + str(conv_az_n) + "_phase" + add_name]
        sar_imaging("phase", data[TRX], index, spline_d_array, save_name)

# 干渉画像から、軌道縞の成分をカット
def orbital_phase_cut(insar_data, rx_first, rx_second, index, height, theta):
    az_s_index = index[0]
    az_e_index = index[1]
    rg_s_index = index[2]
    rg_e_index = index[3]
    rx_first %= 4
    rx_second %= 4
    
    # for az in range(az_s_index, az_e_index):
    for rg in range(rg_s_index, rg_e_index):
        R_p = dr * rg
        R_g = np.sqrt(abs(R_p**2 - (height + l_d * np.sin(theta) * (3 - rx_first))**2)) + l_d * np.cos(theta) * (3 - rx_first)
        R_s = np.sqrt((height + l_d * np.sin(theta) * (3 - rx_second))**2 + (R_g - l_d * np.cos(theta) * (3 - rx_second))**2)
        dR = R_s - R_p
        in_phase = dR * 2.0 * np.pi / wl
        insar_data[:, rg] = insar_data[:, rg] * np.exp(1j * in_phase).conjugate()
    return insar_data

# 各チャネル間の組み合わせの干渉SAR画像をを作成
def insar_imaging(dir_name, data, index, spline_d_array, add_name):
    tag = ["TX2RX1", "TX2RX2", "TX2RX3", "TX2RX4", "TX1RX1", "TX1RX2", "TX1RX3", "TX1RX4"]
    for rx_first in range(8):
        for rx_second in range(rx_first + 1, (int(rx_first / 4 + 1)) * 4):
            # そのまま干渉をとって画像表示
            insar_data = data[rx_first] * data[rx_second].conjugate()
            save_name = ["", "range [m]", "azimuth [m]", dir_name + tag[rx_first] + "-" + tag[rx_second] + "_insar" + add_name]
            sar_imaging("phase", insar_data, index, spline_d_array, save_name)
            # 軌道縞の成分の補正を行った上で再び干渉画像表示
            insar_data = orbital_phase_cut(insar_data, rx_first, rx_second, [0, az_n, 0, 50], 5.0, 20 * 2 * np.pi / 360)
            save_name = ["", "range [m]", "azimuth [m]", dir_name + tag[rx_first] + "-" + tag[rx_second] + "_insar_cut" + add_name]
            sar_imaging("phase", insar_data, index, spline_d_array, save_name)
    for tx_first in range(4):
        insar_data = data[tx_first] * data[tx_first + 4].conjugate()
        save_name = ["", "range [m]", "azimuth [m]", dir_name + tag[tx_first] + "-" + tag[tx_first + 4] + "_insar" + add_name]
        sar_imaging("phase", insar_data, index, spline_d_array, save_name)

# 部分的に位相の相加平均を取得
def calc_average_phase(data, d_array):
    target_az, target_rg = argmax_2d(data[:, 20:30]) #合成開口後の注目したい振幅ピークを持つ値を探す（ために探す範囲を調整）
    target_az += 0   #探す範囲に応じてインデックスの調整
    target_rg += 20  #探す範囲に応じてインデックスの調整
    print(target_az, np.round(d_array[target_az], 3), target_rg, target_rg * dr) # 位相を取得する部分が本当に自分の取得したい部分かを確認
    ## ピークを中心とする周囲25ピクセルの位相平均
    d_az = [-3, -2, -1, 0, 1, 2, 3]
    d_rg = [-1, 0, 1]
    phase_sum = 0
    for az in d_az:
        for rg in d_rg:
            phase_sum += data[target_az + az, target_rg + rg]
    return np.angle(phase_sum / 21.0)

# 送信アンテナ同じ、受信アンテナ異なる場合の干渉画像における、部分的な位相平均をチャネル間で比較（cutflagで軌道縞を引くかどうかを選択）
def channel_phase_compare(data, d_array, save_name, theta, cutflag):
    plt.rcParams["font.size"] = 12
    phase_matrix = np.zeros((2, 6), dtype = np.float64)
    for tx in range(2):
        if(tx == 0): Tx = 1
        else: Tx = 0
        Rx = 0
        for rx_p in range(0,3):
            for rx_s in range(rx_p + 1, 4):
                insar_data = data[tx * 4 + rx_p] * data[tx * 4 + rx_s].conjugate()
                if(cutflag): insar_data = orbital_phase_cut(insar_data, rx_p, rx_s, [0, az_n, 0, 50], 5.0, theta)
                phase_matrix[Tx, Rx] = calc_average_phase(insar_data, d_array)
                Rx += 1
    tx_tag = ["TX1", "TX2"]
    rx_tag = ["RX1-RX2", "RX1-RX3", "RX1-RX4", "RX2-RX3", "RX2-RX4", "RX3-RX4"]
    print(phase_matrix)
    plt.figure(figsize = (9,2))
    sns.heatmap(phase_matrix, cmap = "hsv", annot = True, vmax=np.pi, vmin=-np.pi, xticklabels = rx_tag, yticklabels = tx_tag, fmt='.3f')
    if(cutflag):
        np.save(save_name + "_cut", phase_matrix)
        plt.savefig(save_name + "_cut.pdf", format = "pdf", bbox_inches = 'tight')
    else:
        np.save(save_name, phase_matrix)
        plt.savefig(save_name + ".pdf", format = "pdf", bbox_inches = 'tight')

# 複数データ（同じ実験条件）の各チャネルごとの平均値や標準偏差を計算し、散布図に表示（どれくらい値にばらつきがあるか等の確認）
def inter_phase_statistics(filename, save_name):
    n = len(filename)
    inter_phase = np.zeros((n, 2, 6), dtype = np.float64) #各データごとの12組の干渉位相値を格納する配列
    index = 0
    for f_name in filename:
        inter_phase[index] = np.load(f_name)
        index += 1
    tx_tag = ["TX1", "TX2"]
    rx_tag = ["RX1-RX2", "RX1-RX3", "RX1-RX4", "RX2-RX3", "RX2-RX4", "RX3-RX4"]
    average = np.zeros((2, 6), dtype = np.float64) #標本平均
    variance = np.zeros((2, 6), dtype = np.float64) #不偏分散
    stdev = np.zeros((2, 6), dtype = np.float64) #標本標準偏差
    for i in range(2):
        for j in range(6):
            average[i, j] = stat.mean(inter_phase[:, i, j])
            variance[i, j] = stat.variance(inter_phase[:, i, j])
            stdev[i, j] = stat.stdev(inter_phase[:, i, j])
    plt.figure(figsize = (9,2))
    sns.heatmap(average, cmap = "hsv", annot = True, vmax=np.pi, vmin=-np.pi, xticklabels = rx_tag, yticklabels = tx_tag, fmt='.3f')
    plt.savefig(save_name + "_average.pdf", format = "pdf", bbox_inches = 'tight')
    plt.figure(figsize = (9,2))
    sns.heatmap(stdev, annot = True, cmap = "gray", xticklabels = rx_tag, yticklabels = tx_tag, fmt='.3f', vmax=0.5)
    plt.savefig(save_name + "_stdev.pdf", format = "pdf", bbox_inches = 'tight')
    print(average)
    print(stdev)

    tx_tag = ["1", "2"]
    rx_tag = ["1-2", "1-3", "1-4", "2-3", "2-4", "3-4"]
    trx_tag = []
    for i in range(12):
        trx_tag.append(tx_tag[int(i / 6)] + ":" + rx_tag[i % 6])
    plt.figure(figsize = (12,8))
    plt.rcParams["font.size"] = 18
    x_num = np.arange(0, 12)
    # 標準偏差に基づいてエラーバーを表示
    plt.errorbar(x_num, average.reshape(-1), stdev.reshape(-1), fmt = 'x', markersize = 10, color = "black", ecolor = "black", elinewidth = 0.5, capsize = 8)
    for i in range(n):
        plt.scatter(x_num, inter_phase[i].reshape(-1), marker = 'o')
    plt.ylim(-np.pi, np.pi)
    plt.ylabel("phase")
    plt.xlabel("TX:RX-RX")
    plt.xticks(x_num, trx_tag, rotation = 0)
    plt.grid(linestyle='dotted', linewidth = 0.5)
    plt.text(-1.3, 3.3, "[rad]") # カラーバーの単位を手動でテキスト表示
    plt.tight_layout()
    plt.savefig(save_name + "_inphase_scatter_plot.pdf", format = "pdf", bbox_inches = 'tight')
    plt.clf()
    plt.close()
