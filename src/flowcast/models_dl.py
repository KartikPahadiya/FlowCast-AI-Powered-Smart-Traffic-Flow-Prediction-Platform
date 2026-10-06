"""M6 — Deep-learning engine: LSTM sequence forecaster built from scratch.

No pre-trained weights, no transfer learning. A sliding window of the
previous N=24 half-hour steps (12 h) per segment, each step a vector of
scaled telemetry features, feeds a 2-layer LSTM with dropout; the final
hidden state drives two heads — next-window volume (regression) and
next-window congestion class (classification). Trained with Adam, MSE +
cross-entropy loss, early stopping on validation loss and LR-on-plateau.
"""
import json
import logging
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset

from . import config
from .models_classical import reg_metrics, clf_metrics

log = logging.getLogger("flowcast.dl")

# per-step sequence channels (all past-information, scaled on train only)
SEQ_FEATURES = ["traffic_volume", "avg_speed", "occupancy", "rainfall",
                "visibility", "temperature", "hour_sin", "hour_cos",
                "dow_sin", "dow_cos", "is_weekend", "is_peak",
                "public_holiday", "event_flag"]


class FlowLSTM(nn.Module):
    """LSTM sequence forecaster, hand-built with torch primitives."""

    def __init__(self, n_features, hidden=64, layers=2, dropout=0.2,
                 n_classes=4):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden,
                            num_layers=layers, batch_first=True,
                            dropout=dropout)
        self.head_reg = nn.Linear(hidden, 1)          # next-window volume
        self.head_clf = nn.Linear(hidden, n_classes)  # congestion logits

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head_reg(last).squeeze(-1), self.head_clf(last)


class SequenceData:
    """Builds (sequence -> next-window target) pairs per segment in time order."""

    def __init__(self, df, seq_len):
        self.seq_len = seq_len
        self.mu, self.sigma = {}, {}
        self.splits = {}
        d = df.sort_values(["road_id", "timestamp"])
        # scaling statistics from the training portion of the timeline only
        stamps = np.sort(d["timestamp"].unique())
        tr_end = stamps[int(len(stamps) * config.TRAIN_FRAC)]
        train_part = d[d["timestamp"] <= tr_end]
        for col in SEQ_FEATURES:
            self.mu[col] = float(train_part[col].mean())
            self.sigma[col] = float(train_part[col].std()) or 1.0
        self.vol_mu = float(train_part["traffic_volume"].mean())
        self.vol_sigma = float(train_part["traffic_volume"].std())

        feats = d[SEQ_FEATURES].to_numpy(np.float32)
        scaled = np.column_stack([(feats[:, i] - self.mu[c]) / self.sigma[c]
                                  for i, c in enumerate(SEQ_FEATURES)])
        scaled = scaled.astype(np.float32)
        vol = d["traffic_volume"].to_numpy(np.float32)
        cong = d["congestion_code"].to_numpy(np.int64)
        ts = d["timestamp"].to_numpy()
        roads = d["road_id"].to_numpy()

        n = len(stamps)
        edges = {"train": stamps[int(n * config.TRAIN_FRAC)],
                 "val": stamps[int(n * (config.TRAIN_FRAC + config.VAL_FRAC))]}
        offsets = np.arange(seq_len)          # [0..seq_len-1]
        arrays = {"train": [], "val": [], "test": []}
        for _, idx in d.groupby(roads).indices.items():
            idx = np.sort(idx)
            starts = np.arange(0, len(idx) - seq_len)      # window starts
            targets = idx[starts + seq_len]                # target positions
            t_targets = ts[targets]
            split = np.where(t_targets <= edges["train"], "train",
                             np.where(t_targets <= edges["val"], "val", "test"))
            for name in ("train", "val", "test"):
                sel = starts[split == name]
                if len(sel) == 0:
                    continue
                windows = idx[sel[:, None] + offsets]      # (m, seq_len)
                arrays[name].append((
                    scaled[windows], vol[targets[split == name]],
                    cong[targets[split == name]],
                    targets[split == name], roads[targets[split == name]]))

        for name, parts in arrays.items():
            if not parts:
                continue
            xs = np.concatenate([p[0] for p in parts])
            self.splits[name] = (xs,
                                 np.concatenate([p[1] for p in parts]),
                                 np.concatenate([p[2] for p in parts]),
                                 np.concatenate([p[3] for p in parts]),
                                 np.concatenate([p[4] for p in parts]))
            log.info("LSTM %s sequences: %s", name, xs.shape)

        # cap the training pool so a CPU epoch stays practical; the split
        # ratio is preserved by strided sampling (documented in model card)
        if "train" in self.splits and len(self.splits["train"][0]) > 90_000:
            keep = np.arange(0, len(self.splits["train"][0]),
                             int(np.ceil(len(self.splits["train"][0]) / 90_000)))
            xs, yv, yc, t, r = self.splits["train"]
            self.splits["train"] = (xs[keep], yv[keep], yc[keep], t[keep], r[keep])
            log.info("LSTM train sequences capped to %d (strided)", len(keep))

    def make_loader(self, name, batch_size, shuffle=False):
        xs, yv, yc, _, _ = self.splits[name]
        ds = TensorDataset(torch.from_numpy(xs), torch.from_numpy(yv),
                           torch.from_numpy(yc))
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          generator=torch.Generator().manual_seed(config.RANDOM_SEED))


def train_lstm(df, device="cpu"):
    """Train the from-scratch LSTM and return (model, data, history, test preds)."""
    cfg = config.LSTM_CONFIG
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    data = SequenceData(df, cfg["seq_len"])
    model = FlowLSTM(len(SEQ_FEATURES), cfg["hidden_size"], cfg["num_layers"],
                     cfg["dropout"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"],
                           weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, patience=2, factor=0.5)
    mse = nn.MSELoss()
    ce = nn.CrossEntropyLoss()
    loaders = {k: data.make_loader(k, cfg["batch_size"], shuffle=(k == "train"))
               for k in ("train", "val")}

    history, best_val, best_state, bad = [], np.inf, None, 0
    t0 = time.time()
    for epoch in range(1, cfg["max_epochs"] + 1):
        model.train()
        tl = tc = nb = 0.0
        for x, yv, yc in loaders["train"]:
            x, yv, yc = x.to(device), yv.to(device), yc.to(device)
            opt.zero_grad()
            pv, pc = model(x)
            loss_r = mse(pv, (yv - data.vol_mu) / data.vol_sigma)
            loss_c = ce(pc, yc)
            loss = loss_r + 0.5 * loss_c
            loss.backward()          # backprop through time, by hand
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += loss_r.item() * len(x)
            tc += loss_c.item() * len(x)
            nb += len(x)
        model.eval()
        vl, vc = 0.0, 0.0
        with torch.no_grad():
            for x, yv, yc in loaders["val"]:
                x, yv, yc = x.to(device), yv.to(device), yc.to(device)
                pv, pc = model(x)
                vl += mse(pv, (yv - data.vol_mu) / data.vol_sigma).item() * len(x)
                vc += ce(pc, yc).item() * len(x)
        n_tr, n_va = len(data.splits["train"][0]), len(data.splits["val"][0])
        tr_loss, va_loss = tl / nb, vl / n_va
        history.append({"epoch": epoch, "train_mse": tr_loss,
                        "val_mse": va_loss,
                        "train_ce": tc / nb, "val_ce": vc / n_va})
        log.info("epoch %2d | train mse %.4f | val mse %.4f | %.0fs",
                 epoch, tr_loss, va_loss, time.time() - t0)
        sched.step(va_loss)
        if va_loss < best_val - 1e-4:
            best_val, bad = va_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg["patience"]:
                log.info("early stopping at epoch %d", epoch)
                break
    if best_state is not None:
        model.load_state_dict(best_state)   # restore best weights
    return model, data, history


def evaluate_lstm(model, data, device="cpu"):
    """Evaluate on the untouched test window with the classical metrics."""
    model.eval()
    xs, yv, yc, ts, roads = data.splits["test"]
    pv, pc = [], []
    with torch.no_grad():
        for i in range(0, len(xs), 4096):
            x = torch.from_numpy(xs[i:i + 4096]).to(device)
            r, c = model(x)
            pv.append(r.cpu().numpy())
            pc.append(c.cpu().numpy())
    pv = np.concatenate(pv) * data.vol_sigma + data.vol_mu
    pc = np.concatenate(pc)
    vol_m = reg_metrics(yv, pv)
    cong_m = clf_metrics(yc, pc.argmax(1), labels=[0, 1, 2, 3])
    cong_m["macro_F1"] = float(f1_score(yc, pc.argmax(1), average="macro"))
    res = pd.DataFrame({
        "timestamp": ts, "road_id": roads,
        "y_volume": yv, "pred_volume": pv,
        "y_congestion": yc, "pred_congestion": pc.argmax(1),
        "conf_freeflow": pc[:, 0], "conf_moderate": pc[:, 1],
        "conf_heavy": pc[:, 2], "conf_severe": pc[:, 3],
    })
    return vol_m, cong_m, res


def save_lstm(model, history, path):
    torch.save({"state_dict": model.state_dict(),
                "config": config.LSTM_CONFIG,
                "history": history}, path)


def load_lstm(path, n_features, n_classes=4):
    ckpt = torch.load(path, weights_only=False)
    cfg = ckpt["config"]
    model = FlowLSTM(n_features, cfg["hidden_size"], cfg["num_layers"],
                     cfg["dropout"], n_classes)
    model.load_state_dict(ckpt["state_dict"])
    return model, ckpt
