# estimate_height.py
## 実測 φ_topo と理論 φ_theory(H, θ) の照合から高さ候補を推定

import numpy as np
import matplotlib.pyplot as plt
from theory_phi_topo import compute_phi_topo
import SA_func as sa

# ===== パラメータ =====
lam = sa.wl
R1 = 10.0
h = 0.6
H_vals = np.linspace(0, 1.0, 200)
theta_range = np.deg2rad(np.linspace(-90, 90, 181))
theta_offsets = np.deg2rad(np.linspace(-10, 10, 21)) 

# ===== ターゲット位置 =====
targets = {
    "Target 1": [20, 647],   # [range, azimuth]
    "Target 2": [29, 920],
    "Target 3": [34, 1400],
}

# ===== ベースライン長（対応チャンネル） =====
baselines = {
    5: lam / 2,    # RX1-RX2
    6: lam,        # RX1-RX3
    7: 1.5 * lam,  # RX1-RX4
}

# ===== 実測 φ_topo と θ_rad を読み込み =====
phi_topo_ch5 = np.load("phi_topo_ch5.npy")
phi_topo_ch6 = np.load("phi_topo_ch6.npy")
phi_topo_ch7 = np.load("phi_topo_ch7.npy")
theta_rad = np.load("theta_rad.npy")

# ===== チャンネル対応 =====
phi_topo_dict = {5: phi_topo_ch5, 6: phi_topo_ch6, 7: phi_topo_ch7}

# ===== 高さ候補を保存する辞書 =====
H_candidates = {5: {}, 6: {}, 7: {}}


# ===== 高さ推定 =====
for name, (r_idx, azi_idx) in targets.items():
    print(f"\n===== {name} =====")

    best_Hs = []
    for m in [5, 6, 7]:
        d = baselines[m]
        phi_obs = phi_topo_dict[m][azi_idx, r_idx] #観測位相
        theta_measured = theta_rad[azi_idx] #観測角度

        # 高さ0m点での観測位相（基準点）
        phi_obs_ref = phi_topo_dict[m][targets["Target 1"][1], targets["Target 1"][0]]

        best_H, best_theta, min_diff = None, None, 1e9

        # --- θを±5°で動かしながら探索 ---
        for dtheta in theta_offsets:
            theta_test = theta_measured + dtheta

            # φ_theory(H=0)
            phi_theory_0 = compute_phi_topo(lam, R1, h, d, [0], [theta_test])[0, 0]

            # offset補正
            offset = (phi_obs_ref - phi_theory_0 + np.pi) % (2*np.pi) - np.pi

            # φ_theory(H, θ) 全体を計算
            phi_theory_all = compute_phi_topo(lam, R1, h, d, H_vals, [theta_test]).flatten()

            # 実測と理論値phi_topoの差を最小化
            diff = np.abs((phi_obs - phi_theory_all - offset + np.pi) % (2*np.pi) - np.pi)
            idx = np.argmin(diff)

            if diff[idx] < min_diff:
                min_diff = diff[idx]
                best_H = H_vals[idx]
                best_theta = theta_test

        best_Hs.append(best_H)
        print(f"  Channel {m}: H = {best_H:.3f} m (θ = {np.degrees(best_theta):.2f}°, diff = {min_diff:.3f})")

    # --- 3チャネルの整合性を確認 ---
    H_mean = np.mean(best_Hs)
    H_std = np.std(best_Hs)
    print(f"  → 推定高さ H_est = {H_mean:.3f} m ± {H_std:.3f}")

