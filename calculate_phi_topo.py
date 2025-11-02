## calculate_topographic_fring.py
### 干渉SARの地形縞を求める

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import SA_func as sa
from theory_phi_topo import generate_all_phi_maps

## データの取得
dir_name = "../send_sensor/10-14-10-53-18/"
filename = dir_name + "fft_data"
log_name = dir_name + "flight_log"
add_name = "part"
fft_data = sa.read_fft_data(filename)
data = sa.code_V_convert(fft_data)
raw_data = sa.get_raw_data(data)

## BNO055から取得したドローンの姿勢角データを読み込み
df = pd.read_csv("../send_sensor/10-14-10-53-18/bno055_data_20251014_105220.csv")
theta_series = df["Euler_Pitch"].iloc[4:191].to_numpy()

### Rx1を基準に、Rx2〜Rx4の位相差から地形縞を算出
# 前提: raw_data は [ch, chirp, range] の形
#       sa.wl : 波長, sa.l_d : 素子間隔, sa.ch : 受信チャネル数
#       ドローン姿勢角 angle は別途取得済み [rad]

### Tx1-Rx1 を基準に、Rx2〜Rx4との位相差を計算
target_bin = 24  # 解析したいレンジを指定
base_bin = 20    #(軌道縞を求めるための)基準レンジ
base_azi = 647  # 基準とするアジマスインデックス
ref_idx = 4       # "TX1RX1"
azi_idx = 800  # 解析したいアジマスインデックス

ch = 4  # "TX1RX1"
amp_values = np.abs(raw_data[ch, :, base_bin])   # 振幅を計算（複素数なら絶対値）
azi_idx = np.argmax(amp_values)


# 振幅を計算（複素数対応）
amp_values = np.abs(raw_data[ch, :, base_bin])
s_ch = raw_data

# レーダのangleデータをBNO055から取得
n_chirps = s_ch.shape[1]  # =2000
imu_idx = np.arange(len(theta_series))
chirp_idx = np.linspace(0, len(theta_series)-1, n_chirps) # SAR側のインデックス（0〜1999）
theta_interp = np.interp(chirp_idx, imu_idx, theta_series) # --- 線形補間で拡張 ---
theta_rad = np.deg2rad(theta_interp) # ラジアン変換
np.save("theta_rad.npy", theta_rad)
print("Saved θ_rad.npy")

l_d = sa.wl/2
height = 0.6 #ピークの高さを目視で取ってくる

phi_topo_list = []
dh_list = []

# --- 理論 φ_topo マップをあらかじめ計算 ---
phi_maps, H_vals, theta_array = generate_all_phi_maps()

for m in [5, 6, 7]:  # "TX1RX2", "TX1RX3", "TX1RX4"
    # --- 観測位相差（軌道縞+地形縞）---
    phi_obs = np.angle(s_ch[m, :,:] * np.conj(s_ch[ref_idx, :,:]))  # shape=(2000, 512)
    print("phi_obs size ="+str(phi_obs.shape))

    # --- 軌道縞の理論値 ---
    ## アンテナの位置の違いによって生まれる位相差
    delta_rx = m - ref_idx
    phi_orb = np.zeros(phi_obs.shape, dtype=complex)  # shape=(2000, 512)
    for r in range(512):
        R_g = r * sa.dr
        R_p = np.sqrt((height + l_d * np.sin(theta_rad) * delta_rx)**2 + (R_g - l_d * np.cos(theta_rad) * delta_rx)**2) #primaryの伝搬距離
        R_s = np.sqrt(height**2 + R_g **2) #secondary(TX1RX1)の伝搬距離
        dR = abs(R_p - R_s) #高さがない場合の伝搬経路差
        in_phase = dR * 2 * np.pi / sa.wl #伝搬経路差を位相差
        phi_orb[:,r] = np.array(list(np.exp(1j * in_phase)))

    # --- offsetの計算 ---
    # 対応するペア
    pair_key = {1: "RX1-RX2", 2: "RX1-RX3", 3: "RX1-RX4"}[delta_rx]

    # 理論マップからデータ取得
    phi_theory_map = phi_maps[pair_key]

    # 高さ0mに最も近いHインデックスを取得
    H_index = np.argmin(np.abs(H_vals - 0.0))

    # θ(観測値, IMU)に最も近い列を取得
    theta_val = theta_rad[base_azi]
    theta_index = np.argmin(np.abs(theta_array - theta_val))

    # 理論 φ_topo(高さ0, θ_obs)
    phi_theory_0 = phi_theory_map[H_index, theta_index]

    # 実測 φ_topo(高さ0mの観測点)
    phi_obs_0 = phi_obs[base_azi, base_bin]
    offset = phi_obs_0 - phi_theory_0
    offset = np.angle(np.exp(1j * offset))
    print("offset: "+ str(offset))

    # --- 地形縞 = 観測 - オフセット - 軌道 ---
    #phi_topo = np.angle(np.exp(1j * (phi_obs - np.angle(phi_orb)- offset)))
    # phi_topo = (phi_obs - np.angle(phi_orb) - offset + np.pi) % (2 * np.pi) - np.pi
    phi_topo = phi_obs - np.angle(phi_orb) - offset
    phi_topo = np.angle(np.exp(1j * phi_topo))
    # if m == 6:
    #     phi_topo *= -1
    ## なぜかRX3だけ符号が逆になる、、
    phi_topo_list.append(phi_topo)
    print("phi_topo(H=0) ="+ str(phi_topo[base_azi,base_bin]))
 
    # φ_topo 計算完了後に保存
    np.save(f"phi_topo_ch{m}.npy", phi_topo)
    print(f"Saved φ_topo_ch{m}.npy")

    # --- 標高に変換 ---
    R_g = 10
    theta_2d = np.tile(theta_rad[:, np.newaxis], (1, phi_topo.shape[1]))  # shape=(2000, 512) theta_radをレンジ方向に拡大したやつ
    dH = (1/(2*height-sa.wl*np.sin(theta_2d)))*(-(phi_topo*sa.wl/2*np.pi)**2+(phi_topo*sa.wl*R_g/np.pi)+height**2+(sa.wl/2)**2)
    dh_list.append(dH)

    ## オブジェクトの高さの推定結果を表示

    target_1_pos = [20, 647]#ターゲット1の位置　
    target_2_pos = [29, 920]#ターゲット2の位置
    target_3_pos = [34, 1400]#ターゲット3の位置
    ch = 4  # "TX1RX1"
    amp_values = np.abs(raw_data[ch, :, target_3_pos[0]])   # 振幅を計算（複素数なら絶対値）
    azi_idx = np.argmax(amp_values)
    #print(f"チャンネル {ch} の base_bin {target_3_pos[0]} における最大振幅のアジマスインデックス: {azi_idx}")
    
    dH_target_1 = dH[target_1_pos[1], target_1_pos[0]]
    dH_target_2 = dH[target_2_pos[1], target_2_pos[0]]
    dH_target_3 = dH[target_3_pos[1], target_3_pos[0]]
    # print(f"Channel {m} - Target 1 Estimated Height Difference dH: {dH_target_1:.3f} m")
    # print(f"Channel {m} - Target 2 Estimated Height Difference dH: {dH_target_2:.3f} m")
    # print(f"Channel {m} - Target 3 Estimated Height Difference dH: {dH_target_3:.3f} m")  

    ## 実際の高さから実際のthetaを計算
    wl = sa.wl

    # 各ターゲットの位置・実際高さ・推定高さをまとめる
    targets = [
        {"pos": target_1_pos, "true_h": 0.0,  "measured_h": dH_target_1, "name": "Target 1"},
        {"pos": target_2_pos, "true_h": 0.3,  "measured_h": dH_target_2, "name": "Target 2"},
        {"pos": target_3_pos, "true_h": 0.45, "measured_h": dH_target_3, "name": "Target 3"},
    ]

    # for t in targets:
    #     r = t["pos"][0]   # range方向
    #     azi = t["pos"][1] # azimuth方向
    #     R = (r + 1) * sa.dr  # スラント距離

    #     phi_topo_val = phi_topo[azi, r]
    #     dH_true = t["true_h"]
    #     dH_measured = t["measured_h"]

    #     # --- 入射角 φ（Rと高さから算出）---
    #     limited_phi = np.clip(height / R, -1.0, 1.0)
    #     phi_val = np.arccos(limited_phi)

    #     A = (-(phi_topo_val*sa.wl/2*np.pi)**2+(phi_topo_val*sa.wl*R_g/np.pi)+height**2+(sa.wl/2)**2)
    #     #A = (-(phi_topo_val*sa.wl*R_g/np.pi)+height**2)
    #     print("Phi_topo_val =", phi_topo_val)

        # #theta_true = np.arcsin((2 * height * dH_true - A) / (dH_true * sa.wl))
        # val = (2 * height - (A/dH_true)) / (sa.wl)
        # print("dH_true =", dH_true )
        # print("lamda =", sa.wl )
        # print("A =", A )
        # print("sin(theta) =", val)
        # val = np.clip(val, -1.0, 1.0)
        # theta_true = np.arcsin(val)

        # # --- 出力 ---
        # print(f"\n=== {t['name']} ===")
        # print(f"高さ（実際）: {dH_true:.3f} m, 高さ（推定）: {dH_measured:.3f} m")
        # print(f"角度(IMU) θ_imu          = {np.degrees(theta_2d[azi, r]):.2f}°")
        # #print(f"角度(再計算) θ_calculate  = {np.degrees(theta_measured):.2f}°")
        # print(f"角度(理想値) θ_true       = {np.degrees(theta_true).item():.2f}°")
        # # print(f"→ 差分 = {np.degrees(theta_true - theta_measured):.2f}°")

    ## 画像の生成
    # --- phi_obsとして保存 ---
    # plt.figure(figsize=(12, 6))
    # sns.heatmap(np.angle(phi_orb), cmap="jet", cbar=True)
    # plt.xlabel("Range bin")
    # plt.ylabel("Azimuth (chirp index)")
    # plt.title(f"Observed Phase φ_obs [rad] (Channel {m})")
    # plt.tight_layout()
    # plt.savefig(f"phi_obs{m}.png", dpi=300, bbox_inches="tight")
    # plt.close()

    # --- phi_topoとして保存 ---
    plt.figure(figsize=(12, 6))
    sns.heatmap(np.angle(phi_topo), cmap="jet", cbar=True)
    plt.xlabel("Range bin")
    plt.ylabel("Azimuth (chirp index)")
    plt.title(f"Estimated Topographic Phase φ_topo [rad] (Channel {m})")
    plt.tight_layout()
    plt.savefig(f"phi_topo{m}.png", dpi=300, bbox_inches="tight")
    plt.close()

    # --- 軌道縞をヒートマップとして保存 ---
    plt.figure(figsize=(12, 6))
    sns.heatmap(np.angle(phi_orb), cmap="jet", cbar=True)
    plt.xlabel("Range bin")
    plt.ylabel("Azimuth (chirp index)")
    plt.title(f"Estimated Orbital Phase φ_orb [rad] (Channel {m})")
    plt.tight_layout()
    plt.savefig(f"phi_orb{m}.png", dpi=300, bbox_inches="tight")
    plt.close()

    # --- 標高をヒートマップとして保存 ---
    index = [0, 2000, 0, 50]
    rg_e_index = index[3]
    rg_s_index = index[2]
    time_interval = 0.01  # [s] 各chirpの時間間隔
    time_axis = np.arange(n_chirps) * time_interval  # 0.00, 0.01, 0.02, ... 秒

    plt.figure(figsize=(12, 6))
    ax = sns.heatmap(dH[:, rg_s_index:rg_e_index], cmap="jet", cbar=True, vmin=0, vmax=0.5)

    # --- y軸の位置とラベルを対応させる ---
    step = 100  # 100 chirpごとに1ラベル（=1秒刻み）
    yticks = np.arange(0, len(time_axis), step)
    yticklabels = np.round(time_axis[yticks], 1)

    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontsize=10)
    ax.set_ylabel("Time [s]")
    ax.set_xlabel("Range bin")
    rg_len = rg_e_index - rg_s_index 
    x_step = int(rg_len / 25) + 1 
    dx = sa.dr 
    plt.xticks(np.arange(0, rg_e_index - rg_s_index, step = x_step), np.round(np.arange(rg_s_index * dx, rg_e_index * dx, step = dx * x_step), 2), fontsize = 10, rotation = 90)
    plt.title(f"Estimated Height Difference dH [m] (Channel {m})")
    plt.tight_layout()
    plt.savefig(f"dH_heatmap_ch{m}.png", dpi=300, bbox_inches="tight")
    plt.savefig(dir_name+f"dH_heatmap_ch{m}.png", dpi=300, bbox_inches="tight")
    plt.close()
