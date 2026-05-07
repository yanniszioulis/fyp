"""Plot top-3 PCs of the SPX IV surface in level and logdiff space.

Reads SPX_surfaces.csv, computes cross-sectional PCA (demean over time),
and writes pca_level.png and pca_logdiff.png with each PC drawn as a
(tau x moneyness) heatmap titled by the variance share it explains.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_grid(iv_cols):
    """Return (moneyness, tau) sorted floats and (n_tau, n_moneyness) given the
    CSV's `iv_{m}_{tau}` columns. Layout: col k = i_t * n_moneyness + i_m,
    so a (T, n_iv) reshape (T, n_tau, n_moneyness) puts tau on axis -2."""
    m_seen, t_seen = [], []
    for c in iv_cols:
        _, rest = c.split("_", 1)
        m_str, t_str = rest.rsplit("_", 1)
        if m_str not in m_seen:
            m_seen.append(m_str)
        if t_str not in t_seen:
            t_seen.append(t_str)
    n_m, n_t = len(m_seen), len(t_seen)
    if n_m * n_t != len(iv_cols):
        raise ValueError(f"grid is not complete: {n_m} m x {n_t} tau != {len(iv_cols)}")
    return np.array([float(s) for s in m_seen]), np.array([float(s) for s in t_seen]), n_t, n_m


def top_k_pcs(X, k=3):
    """X: (T, F). Returns (pcs, var_share) where pcs is (k, F) and var_share is (k,)."""
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    var = S ** 2 / (X.shape[0] - 1)
    return Vt[:k], var[:k] / var.sum()


def plot_pcs(pcs, var_share, n_tau, n_m, m_grid, t_grid, title_prefix, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    vmax = float(np.abs(pcs).max())
    for k, ax in enumerate(axes):
        # pc shape (n_iv,); reshape to (n_tau, n_moneyness) per CSV layout
        Z = pcs[k].reshape(n_tau, n_m)
        # sign-flip so the "dominant" direction is positive — purely cosmetic
        if Z[Z.shape[0] // 2, Z.shape[1] // 2] < 0:
            Z = -Z
        im = ax.imshow(Z, aspect="auto", origin="lower",
                       cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                       extent=[m_grid.min(), m_grid.max(),
                               t_grid.min(), t_grid.max()])
        ax.set_xlabel("log-moneyness")
        if k == 0:
            ax.set_ylabel("tau (yrs)")
        ax.set_title(f"PC{k+1}: {var_share[k]*100:.2f}% variance")
    fig.suptitle(title_prefix, fontsize=13)
    fig.colorbar(im, ax=axes, shrink=0.8, label="loading")
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"wrote {out_path}  (cum top-3 = {var_share.sum()*100:.2f}%)")


def main():
    df = pd.read_csv("SPX_surfaces.csv")
    iv_cols = [c for c in df.columns if c.startswith("iv_")]
    m_grid, t_grid, n_tau, n_m = parse_grid(iv_cols)
    iv = df[iv_cols].to_numpy(dtype=np.float64)
    print(f"loaded {iv.shape[0]} dates, {iv.shape[1]} cells "
          f"({n_m} moneyness x {n_tau} tau)")

    # Level
    pcs_l, var_l = top_k_pcs(iv, k=3)
    plot_pcs(pcs_l, var_l, n_tau, n_m, m_grid, t_grid,
             "Top-3 PCs of IV LEVEL surface", "pca_level.png")

    # Logdiff
    log_iv = np.log(iv)
    diffs  = log_iv[1:] - log_iv[:-1]
    pcs_d, var_d = top_k_pcs(diffs, k=3)
    plot_pcs(pcs_d, var_d, n_tau, n_m, m_grid, t_grid,
             "Top-3 PCs of IV LOGDIFF (daily) surface", "pca_logdiff.png")


if __name__ == "__main__":
    main()
