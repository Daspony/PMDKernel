"""Evaluación de v5_deepsets_pinn — predice (Bx, By, Bz) y reporta RMSE/R² mT."""
import numpy as np
import torch
from torchmetrics.regression import MeanSquaredError, R2Score

from .data import load_full_grid_for_sample


def _predict_full_grid_mT(lit_model, sensors_J, query_xyz_J, device, *, chunk=8192):
    """Forward chunked sobre los J puntos de una sample. Devuelve (J, 3) numpy mT."""
    lit_model.eval()
    outs = []
    with torch.no_grad():
        for start in range(0, sensors_J.shape[0], chunk):
            end = start + chunk
            s = sensors_J[start:end].to(device)
            q = query_xyz_J[start:end].to(device)
            b_mt = lit_model.predict_mT(s, q)                # (chunk, 3)
            outs.append(b_mt.cpu().numpy())
    return np.concatenate(outs, axis=0)                       # (J, 3)


def evaluate(lit_model, h5_path, sample_indices, device, *,
             rmse_per_component=True, predict_chunk=8192):
    lit_model.eval().to(device)

    yt_chunks, yp_chunks = [], []
    for sample_i in sample_indices:
        sensors_J, query_xyz_J, b_true_mt = load_full_grid_for_sample(h5_path, int(sample_i))
        b_pred_mt = _predict_full_grid_mT(lit_model, sensors_J, query_xyz_J, device,
                                           chunk=predict_chunk)
        yt_chunks.append(b_true_mt.numpy())
        yp_chunks.append(b_pred_mt)
    yt = np.concatenate(yt_chunks, axis=0).astype(np.float32)
    yp = np.concatenate(yp_chunks, axis=0).astype(np.float32)

    yt_t = torch.from_numpy(yt).to(device)
    yp_t = torch.from_numpy(yp).to(device)

    rmse = MeanSquaredError(squared=False).to(device)(yp_t, yt_t).item()
    r2   = R2Score(multioutput="uniform_average").to(device)(yp_t, yt_t).item()

    out = {"rmse_mt": float(rmse), "r2": float(r2), "n": int(len(yt))}

    if rmse_per_component:
        per = {}
        for c, name in enumerate(["Bx", "By", "Bz"]):
            err = yp[:, c] - yt[:, c]
            per[name] = float(np.sqrt(np.mean(err ** 2)))
        out["rmse_per_component"] = per

    return out


def report(name, m):
    line = (f"[{name:>5}]  RMSE={m['rmse_mt']:.4e} mT   "
            f"R²={m['r2']:+.4f}   (N={m['n']})")
    print(line)
    if "rmse_per_component" in m:
        per = m["rmse_per_component"]
        print(f"          RMSE Bx={per['Bx']:.4e}  By={per['By']:.4e}  Bz={per['Bz']:.4e}  mT")
