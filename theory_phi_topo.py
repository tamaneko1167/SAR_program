## theory_phi_topo.py  
# θ-H 関係から理論 φ_topo を計算する関数


import numpy as np
import matplotlib.pyplot as plt
import SA_func as sa

# ===== φ_topo 計算関数 =====
def compute_phi_topo(lam, R1, h, d, H_vals, theta):
    """
    φ_topo(H, θ) マップを計算して2次元配列として返す。
    lam : 波長 [m]
    R1  : アンテナ–ターゲットの基準距離 [m]
    h   : レーダ高さ [m]
    d   : 基線長 [m] (λ/2, λ, 1.5λなど)
    H_vals : 高さの配列 [m]
    theta  : 角度の配列 [rad]
    """
    phi_topo = np.zeros((len(H_vals), len(theta)))
    for i, H in enumerate(H_vals):
        for j, th in enumerate(theta):
            # 幾何モデル
            term1 = (np.sqrt(R1**2 - (H - h)**2) - d * np.cos(th))**2
            term2 = (H - d * np.sin(th))**2
            R2 = np.sqrt(term1 + term2)
            phi = (2 * np.pi / lam) * (R2 - R1)
            # wrap to [-π, π]
            phi_topo[i, j] = (phi + np.pi) % (2 * np.pi) - np.pi
    return phi_topo


def generate_all_phi_maps(lam=None, R1=10.0, h=0.6, H_vals=None, theta_deg=None):
    """
    RX1-RX2, RX1-RX3, RX1-RX4 の φ_topo マップをまとめて計算。
    戻り値: dict {"RX1-RX2": φmap2D, ...}, H_vals, theta(rad)
    """
    if lam is None:
        lam = sa.wl
    if H_vals is None:
        H_vals = np.linspace(0, 1.0, 100)
    if theta_deg is None:
        theta_deg = np.linspace(-90, 90, 181)

    theta = np.deg2rad(theta_deg)
    baselines = {
        "RX1-RX2": lam / 2,
        "RX1-RX3": lam,
        "RX1-RX4": 1.5 * lam
    }

    phi_maps = {}
    for pair, d in baselines.items():
        phi_maps[pair] = compute_phi_topo(lam, R1, h, d, H_vals, theta)
        print(f"[✓] {pair}: done")

    return phi_maps, H_vals, theta


def plot_phi_maps(phi_maps, H_vals, theta):
    """
    φ_topo マップを3ペア分可視化
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, (pair, phi_map) in zip(axes, phi_maps.items()):
        im = ax.imshow(
            phi_map,
            extent=[np.degrees(theta[0]), np.degrees(theta[-1]), H_vals[0], H_vals[-1]],
            origin="lower",
            aspect="auto",
            cmap="turbo",
            vmin=-np.pi, vmax=np.pi
        )
        ax.set_title(pair)
        ax.set_xlabel("θ [deg]")
    axes[0].set_ylabel("Object height H [m]")
    fig.colorbar(im, ax=axes, label="φ_topo [rad]", fraction=0.02, pad=0.04)
    plt.tight_layout(rect=[0, 0, 0.95, 1])
    plt.savefig("phi_topo_map.png", dpi=300, bbox_inches='tight')
    # plt.show()


if __name__ == "__main__":
    phi_maps, H_vals, theta = generate_all_phi_maps()
    plot_phi_maps(phi_maps, H_vals, theta)
