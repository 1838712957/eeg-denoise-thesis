import os
import glob
import time

import numpy as np
import pandas as pd
import mne
from scipy import stats
from scipy.signal import welch, resample
from sklearn.metrics import confusion_matrix, accuracy_score
from tensorflow.keras.models import load_model
import tensorflow as tf
import sys
from pathlib import Path


CORE_CODE_DIR = Path(__file__).resolve().parents[2] / "02_核心代码"
if str(CORE_CODE_DIR) not in sys.path:
    sys.path.append(str(CORE_CODE_DIR))
from bootstrap_paths import PROJECT_ROOT
BASE_DIR = str(PROJECT_ROOT)
RAW_EDF_DIR = os.path.join(BASE_DIR, "Raw_edf 2")

V4_MODEL_PATH = os.path.join(BASE_DIR, "V4_Best_Model.h5")
REFEREE_MODEL_PATH = os.path.join(BASE_DIR, "DeepSleepNet_Referee_Model.h5")

# ================= Subject Range =================
# 只处理 subjectX 之后的受试者（例如 subject8 后面 -> 从 subject9 开始）
SUBJECT_START = 8
INCLUDE_SUBJECT_START = False  # False: subject_start 后面 (subject9..); True: 包含 subject8

# ------------------------ V4 Custom Layers ------------------------
def se_block(input_tensor, ratio=16):
    # V4 推理脚本在原仓库里使用的自定义 SE block 定义
    filters = input_tensor.shape[-1]
    se = tf.keras.layers.GlobalAveragePooling1D()(input_tensor)
    se = tf.keras.layers.Reshape((1, filters))(se)
    se = tf.keras.layers.Dense(filters // ratio, activation="relu", use_bias=False)(se)
    se = tf.keras.layers.Dense(filters, activation="sigmoid", use_bias=False)(se)
    return tf.keras.layers.Multiply()([input_tensor, se])


def res_block(input_tensor, filters, kernel_size=3):
    x = tf.keras.layers.Conv1D(filters, kernel_size, padding="same")(input_tensor)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Conv1D(filters, kernel_size, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = se_block(x)
    x = tf.keras.layers.Add()([x, input_tensor])
    x = tf.keras.layers.Activation("relu")(x)
    return x


def load_models():
    # 只做推理
    v4_model = load_model(
        V4_MODEL_PATH,
        compile=False,
        custom_objects={"se_block": se_block, "res_block": res_block},
    )
    referee_model = load_model(REFEREE_MODEL_PATH, compile=False)
    return v4_model, referee_model


def zscore_per_epoch(x_3d, eps=1e-8):
    # x_3d: (n_epochs, epoch_len, 1)
    mean_v = np.mean(x_3d, axis=1, keepdims=True)
    std_v = np.std(x_3d, axis=1, keepdims=True)
    std_v[std_v < eps] = 1.0
    x_3d = (x_3d - mean_v) / std_v
    x_3d = np.nan_to_num(x_3d, nan=0.0, posinf=0.0, neginf=0.0)
    return x_3d


def dreams_to_sleepnet_labels(y):
    # DREAMS(0..5) -> DeepSleepNet(0..4)
    # 0 Wake, 1 N1, 2 N2, 3 N3, 4 N4, 5 REM
    # 0 Wake, 1 N1, 2 N2, 3 N3, 4 REM
    y = y.copy()
    y[y == 4] = 3  # N4 -> N3
    y[y == 5] = 4  # REM -> 4
    return y


def calculate_band_power(freqs, psd, fmin, fmax):
    idx = (freqs >= fmin) & (freqs <= fmax)
    if np.sum(idx) < 2:
        return 0.0
    return np.trapz(psd[idx], freqs[idx])


def v4_denoise_to_100hz_epochs(raw, ch_name, v4_model):
    # 对齐 Run_V4_Inference.py：
    # EDF(256Hz) -> 512块逐块标准化 -> V4 -> pred*std+m -> 再降采样 100Hz -> 切 30s -> epoch z-score
    if raw.info["sfreq"] != 256:
        raw_256 = raw.copy()
        raw_256.resample(256)
    else:
        raw_256 = raw

    raw_data_256 = raw_256.get_data(picks=ch_name)[0]

    slen = 512
    pad_len = int(np.ceil(len(raw_data_256) / slen) * slen) - len(raw_data_256)
    raw_padded = np.pad(raw_data_256, (0, pad_len), mode="constant")
    n_seg = len(raw_padded) // slen

    chunks = raw_padded.reshape(n_seg, slen).astype(np.float32)
    m = chunks.mean(axis=1, keepdims=True)
    std = chunks.std(axis=1, keepdims=True)
    std_fixed = np.where(std == 0, 1.0, std).astype(np.float32)

    X_v4_in = ((chunks - m) / std_fixed)[..., None]  # (n_seg, 512, 1)
    X_denoised = v4_model.predict(X_v4_in, batch_size=64, verbose=0)

    denoised_arr = np.asarray(X_denoised)
    if denoised_arr.ndim == 3:
        denoised_flat = denoised_arr[:, :, 0]
    else:
        denoised_flat = denoised_arr.reshape(n_seg, slen)

    denoised_chunks = denoised_flat * std_fixed + m
    denoised_256hz = denoised_chunks.reshape(-1)[: len(raw_data_256)]

    # 256Hz -> 100Hz
    num_samples_100hz = int(len(denoised_256hz) * 100 / 256)
    data_100hz = resample(denoised_256hz, num_samples_100hz)

    # 100Hz -> 30s epochs (3000点) + epoch z-score
    epoch_len = 30 * 100
    n_epochs = len(data_100hz) // epoch_len
    X_v4 = data_100hz[: n_epochs * epoch_len].reshape(n_epochs, epoch_len, 1)
    X_v4 = zscore_per_epoch(X_v4)
    return X_v4


def raw_to_100hz_epochs(raw, ch_name):
    # 对齐 Evaluate_N3_Fidelity.py 输入准备：
    # EDF(100Hz) -> 切 30s -> epoch z-score
    if raw.info["sfreq"] != 100:
        raw_100 = raw.copy()
        raw_100.resample(100)
    else:
        raw_100 = raw

    raw_data_100 = raw_100.get_data(picks=ch_name)[0]
    epoch_len = 30 * 100
    n_epochs = len(raw_data_100) // epoch_len

    X_raw = raw_data_100[: n_epochs * epoch_len].reshape(n_epochs, epoch_len, 1)
    X_raw = zscore_per_epoch(X_raw)
    return X_raw


def load_labels_30s(label_path, n_epochs):
    # 对齐 Evaluate_N3_Fidelity.py：
    # 读取 HypnogramR&K_subjectX.txt（第1行是 [HypnogramR&K] 头）
    # 标签是 5s 粒度：每个 30s epoch 对应 6 个标签 -> mode(少数服从多数)
    df = pd.read_csv(label_path, header=None, sep=r"\s+", skiprows=1)
    labels_raw = df.values.flatten()
    labels_raw = labels_raw[: n_epochs * 6]
    labels_reshaped = labels_raw.reshape(n_epochs, 6)

    mode_res = stats.mode(labels_reshaped, axis=1, keepdims=False)
    labels_30s = getattr(mode_res, "mode", None)
    if labels_30s is None:
        labels_30s = mode_res[0]
    y_true = np.asarray(labels_30s).flatten().astype(int)
    return y_true


def stage_precision_from_confusion(y_true, y_pred, stage_idx):
    # confusion_matrix rows=true, cols=pred
    # precision for class i = TP / (TP + FP) = cm[i,i] / sum_over_true cm[:,i]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4])
    denom = np.sum(cm[:, stage_idx])
    return (cm[stage_idx, stage_idx] / denom) * 100.0 if denom > 0 else np.nan


def energy_loss_by_stage(X_raw, X_v4, y_true, bands, fs=100, nperseg=1000):
    # 每个分期（以真实标签 y_true 为准）计算：loss = (bp_raw-bp_v4)/bp_raw*100
    out = {}
    for stage_name, stage_idx in [("N1", 1), ("N2", 2), ("N3", 3)]:
        mask = y_true == stage_idx
        if np.sum(mask) == 0:
            for band_name in bands.keys():
                out[f"loss_{band_name}_{stage_name}"] = np.nan
            continue

        raw_stage = X_raw[mask]
        v4_stage = X_v4[mask]
        flat_raw = raw_stage.flatten()
        flat_v4 = v4_stage.flatten()

        freqs, psd_raw = welch(flat_raw, fs=fs, nperseg=nperseg)
        _, psd_v4 = welch(flat_v4, fs=fs, nperseg=nperseg)

        for band_name, (fmin, fmax) in bands.items():
            bp_raw = calculate_band_power(freqs, psd_raw, fmin, fmax)
            bp_v4 = calculate_band_power(freqs, psd_v4, fmin, fmax)
            out[f"loss_{band_name}_{stage_name}"] = ((bp_raw - bp_v4) / (bp_raw + 1e-8)) * 100.0
    return out


def process_subject(subject_id, v4_model, referee_model):
    edf_path = os.path.join(RAW_EDF_DIR, f"{subject_id}.edf")
    label_path = os.path.join(RAW_EDF_DIR, f"HypnogramR&K_{subject_id}.txt")

    if not os.path.exists(edf_path):
        raise FileNotFoundError(f"Missing EDF: {edf_path}")
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Missing label: {label_path}")

    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    ch_name = next((c for c in raw.ch_names if ("C3" in c or "CZ" in c)), raw.ch_names[0])

    t0 = time.time()
    X_raw = raw_to_100hz_epochs(raw, ch_name)
    X_v4 = v4_denoise_to_100hz_epochs(raw, ch_name, v4_model)
    n_epochs = min(X_raw.shape[0], X_v4.shape[0])
    X_raw = X_raw[:n_epochs]
    X_v4 = X_v4[:n_epochs]

    y_true = load_labels_30s(label_path, n_epochs)
    y_true = dreams_to_sleepnet_labels(y_true)

    valid_idx = (y_true >= 0) & (y_true <= 4)
    X_raw_valid = X_raw[valid_idx]
    X_v4_valid = X_v4[valid_idx]
    y_valid = y_true[valid_idx]

    pred_probs_raw = referee_model.predict(X_raw_valid, batch_size=64, verbose=0)
    y_pred_raw = np.argmax(pred_probs_raw, axis=1)

    pred_probs_v4 = referee_model.predict(X_v4_valid, batch_size=64, verbose=0)
    y_pred_v4 = np.argmax(pred_probs_v4, axis=1)

    t_total = time.time() - t0

    # 1) 准确率（整体）
    acc_overall_raw = accuracy_score(y_valid, y_pred_raw) * 100.0
    acc_overall_v4 = accuracy_score(y_valid, y_pred_v4) * 100.0

    # 2) N1~N3 分期 precision（对照定义：TP/(TP+FP)）
    n1_raw = stage_precision_from_confusion(y_valid, y_pred_raw, 1)
    n2_raw = stage_precision_from_confusion(y_valid, y_pred_raw, 2)
    n3_raw = stage_precision_from_confusion(y_valid, y_pred_raw, 3)

    n1_v4 = stage_precision_from_confusion(y_valid, y_pred_v4, 1)
    n2_v4 = stage_precision_from_confusion(y_valid, y_pred_v4, 2)
    n3_v4 = stage_precision_from_confusion(y_valid, y_pred_v4, 3)

    # 3) 能量损失（按真实分期，delta/theta/sigma）
    bands = {"delta": (0.5, 4.0), "theta": (4.0, 7.0), "sigma": (12.0, 15.0)}
    loss_dict = energy_loss_by_stage(X_raw_valid, X_v4_valid, y_valid, bands=bands)

    return {
        "Subject": subject_id,
        "Acc_Overall_Raw(%)": acc_overall_raw,
        "Acc_Overall_V4(%)": acc_overall_v4,
        "Acc_Overall_Delta(pp)": acc_overall_v4 - acc_overall_raw,
        "N1_Precision_Raw(%)": n1_raw,
        "N1_Precision_V4(%)": n1_v4,
        "N1_Precision_Delta(pp)": n1_v4 - n1_raw if np.isfinite(n1_raw) and np.isfinite(n1_v4) else np.nan,
        "N2_Precision_Raw(%)": n2_raw,
        "N2_Precision_V4(%)": n2_v4,
        "N2_Precision_Delta(pp)": n2_v4 - n2_raw if np.isfinite(n2_raw) and np.isfinite(n2_v4) else np.nan,
        "N3_Precision_Raw(%)": n3_raw,
        "N3_Precision_V4(%)": n3_v4,
        "N3_Precision_Delta(pp)": n3_v4 - n3_raw if np.isfinite(n3_raw) and np.isfinite(n3_v4) else np.nan,
        "Time_Total(s)": t_total,
        **loss_dict,
    }


def main():
    v4_model, referee_model = load_models()

    def parse_subject_num(subject_id: str) -> int:
        # subject17 -> 17
        digits = "".join([ch for ch in subject_id if ch.isdigit()])
        return int(digits) if digits else -1

    edf_paths = sorted(glob.glob(os.path.join(RAW_EDF_DIR, "subject*.edf")))
    if len(edf_paths) == 0:
        raise RuntimeError(f"No subject*.edf found in {RAW_EDF_DIR}")

    results = []
    errors = []
    for edf_path in edf_paths:
        subject_id = os.path.splitext(os.path.basename(edf_path))[0]  # subject17

        subject_num = parse_subject_num(subject_id)
        if INCLUDE_SUBJECT_START:
            if subject_num < SUBJECT_START:
                continue
        else:
            if subject_num <= SUBJECT_START:
                continue

        print(f"Processing {subject_id} ...")
        try:
            row = process_subject(subject_id, v4_model, referee_model)
            results.append(row)
            print(f"  OK: Overall Raw={row['Acc_Overall_Raw(%)']:.2f}%, V4={row['Acc_Overall_V4(%)']:.2f}%")
        except Exception as e:
            print(f"  FAILED: {e}")
            errors.append({"Subject": subject_id, "Error": str(e)})

    if len(results) == 0:
        print("No results generated. Check errors.")
        if errors:
            print(pd.DataFrame(errors).to_string(index=False))
        return

    df = pd.DataFrame(results)
    numeric_cols = [c for c in df.columns if c != "Subject" and c != "Time_Total(s)" and df[c].dtype != object]
    avg_row = {"Subject": "AVERAGE"}
    for c in numeric_cols:
        avg_row[c] = float(np.nanmean(df[c].values))
    avg_row["Time_Total(s)"] = float(np.nanmean(df["Time_Total(s)"].values))

    df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)

    # 保存输出
    out_csv = os.path.join(BASE_DIR, "V4_StageTable_EnergyLoss.csv")
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print("\n==== Result Table (preview) ====")
    # 预览重点列
    preview_cols = [c for c in df.columns if c in [
        "Subject",
        "Acc_Overall_Raw(%)", "Acc_Overall_V4(%)", "Acc_Overall_Delta(pp)",
        "N1_Precision_Raw(%)", "N1_Precision_V4(%)", "N1_Precision_Delta(pp)",
        "N2_Precision_Raw(%)", "N2_Precision_V4(%)", "N2_Precision_Delta(pp)",
        "N3_Precision_Raw(%)", "N3_Precision_V4(%)", "N3_Precision_Delta(pp)",
        "loss_delta_N1", "loss_theta_N1", "loss_sigma_N1",
        "loss_delta_N2", "loss_theta_N2", "loss_sigma_N2",
        "loss_delta_N3", "loss_theta_N3", "loss_sigma_N3",
        "Time_Total(s)"
    ] if c in df.columns]
    print(df[preview_cols].to_string(index=False))

    print(f"\nSaved: {out_csv}")
    if errors:
        err_df = pd.DataFrame(errors)
        out_err = os.path.join(BASE_DIR, "V4_StageTable_EnergyLoss_errors.csv")
        err_df.to_csv(out_err, index=False, encoding="utf-8-sig")
        print(f"Errors saved: {out_err}")


if __name__ == "__main__":
    main()

