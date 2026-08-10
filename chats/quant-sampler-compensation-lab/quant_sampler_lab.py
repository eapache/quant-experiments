#!/usr/bin/env python3
"""Low-sample post-hoc compensation for quantization-shifted LM logits.

This is deliberately dependency-light: NumPy/SciPy/scikit-learn are enough.
It contains two experiments:

1. Controlled logit shifts whose true structure is known.
2. A trained small next-token MLP, followed by groupwise weight quantization.

The calibrators only see paired reference and quantized logits on a small set of
contexts.  At inference they transform quantized logits before ordinary
temperature/top-p sampling.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Callable

# This runtime's home directory is read-only; make matplotlib's cache explicit.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/quant-sampler-matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPClassifier


EPS = 1e-30


def center(x: np.ndarray) -> np.ndarray:
    return x - x.mean(axis=-1, keepdims=True)


def log_softmax(x: np.ndarray) -> np.ndarray:
    y = x - x.max(axis=-1, keepdims=True)
    return y - np.log(np.exp(y).sum(axis=-1, keepdims=True))


def softmax(x: np.ndarray) -> np.ndarray:
    return np.exp(log_softmax(x))


def top_p_probs(logits: np.ndarray, temperature: float, top_p: float) -> np.ndarray:
    p = softmax(logits / temperature)
    if top_p >= 1.0:
        return p
    order = np.argsort(-p, axis=1)
    sorted_p = np.take_along_axis(p, order, axis=1)
    cumulative = np.cumsum(sorted_p, axis=1)
    # Keep the first token crossing the threshold, matching common samplers.
    keep = cumulative - sorted_p < top_p
    sorted_p = np.where(keep, sorted_p, 0.0)
    sorted_p /= sorted_p.sum(axis=1, keepdims=True)
    out = np.zeros_like(p)
    np.put_along_axis(out, order, sorted_p, axis=1)
    return out


def distribution_metrics(reference_logits: np.ndarray, candidate_logits: np.ndarray,
                         temperature: float = 1.0) -> dict[str, float]:
    lp = log_softmax(reference_logits / temperature)
    lq = log_softmax(candidate_logits / temperature)
    p, q = np.exp(lp), np.exp(lq)
    m = 0.5 * (p + q)
    kl = np.sum(p * (lp - lq), axis=1)
    js = 0.5 * np.sum(p * (lp - np.log(m + EPS)), axis=1)
    js += 0.5 * np.sum(q * (lq - np.log(m + EPS)), axis=1)
    tv = 0.5 * np.abs(p - q).sum(axis=1)
    ref_order = np.argsort(-reference_logits, axis=1)[:, :10]
    cand_order = np.argsort(-candidate_logits, axis=1)[:, :10]
    overlap = np.array([
        len(set(a.tolist()).intersection(b.tolist())) / 10.0
        for a, b in zip(ref_order, cand_order)
    ])
    return {
        "kl": float(kl.mean()),
        "js": float(js.mean()),
        "tv": float(tv.mean()),
        "top1_flip": float((reference_logits.argmax(1) != candidate_logits.argmax(1)).mean()),
        "top10_overlap": float(overlap.mean()),
    }


def probability_metrics(p: np.ndarray, q: np.ndarray) -> dict[str, float]:
    m = 0.5 * (p + q)
    js = 0.5 * np.sum(p * (np.log(p + EPS) - np.log(m + EPS)), axis=1)
    js += 0.5 * np.sum(q * (np.log(q + EPS) - np.log(m + EPS)), axis=1)
    tv = 0.5 * np.abs(p - q).sum(axis=1)
    return {"js": float(js.mean()), "tv": float(tv.mean())}


class Calibrator:
    name = "identity"

    def fit(self, z_ref: np.ndarray, z_quant: np.ndarray) -> "Calibrator":
        return self

    def transform(self, z_quant: np.ndarray) -> np.ndarray:
        return z_quant


class TemperatureCalibrator(Calibrator):
    """One scalar logit multiplier: exactly a temperature correction."""

    name = "temperature"

    def __init__(self) -> None:
        self.scale = 1.0

    def fit(self, z_ref: np.ndarray, z_quant: np.ndarray) -> "TemperatureCalibrator":
        p = softmax(z_ref)
        q = center(z_quant)

        def objective(log_scale: float) -> float:
            return float(-np.sum(p * log_softmax(math.exp(log_scale) * q), axis=1).mean())

        result = minimize_scalar(objective, bounds=(math.log(0.2), math.log(5.0)), method="bounded")
        self.scale = float(math.exp(result.x))
        return self

    def transform(self, z_quant: np.ndarray) -> np.ndarray:
        return self.scale * center(z_quant)


def log_rank_buckets(logits: np.ndarray, bucket_count: int) -> np.ndarray:
    order = np.argsort(-logits, axis=1)
    ranks = np.empty_like(order)
    rows = np.arange(logits.shape[0])[:, None]
    ranks[rows, order] = np.arange(logits.shape[1])[None, :]
    buckets = np.floor(np.log2(ranks + 1)).astype(np.int32)
    return np.minimum(buckets, bucket_count - 1)


class RankBiasCalibrator(Calibrator):
    """Temperature plus one additive correction per logarithmic rank bucket."""

    name = "temperature+rank"

    def __init__(self, bucket_count: int = 11, regularization: float = 2e-3) -> None:
        self.bucket_count = bucket_count
        self.regularization = regularization
        self.scale = 1.0
        self.bias = np.zeros(bucket_count)

    def fit(self, z_ref: np.ndarray, z_quant: np.ndarray) -> "RankBiasCalibrator":
        p = softmax(z_ref)
        q = center(z_quant)
        buckets = log_rank_buckets(q, self.bucket_count)
        temp = TemperatureCalibrator().fit(z_ref, q)
        x0 = np.r_[math.log(temp.scale), np.zeros(self.bucket_count)]

        def objective_and_grad(x: np.ndarray) -> tuple[float, np.ndarray]:
            scale = math.exp(float(x[0]))
            bias = x[1:] - x[1:].mean()
            corrected = scale * q + bias[buckets]
            lp = log_softmax(corrected)
            pred = np.exp(lp)
            loss = -np.sum(p * lp, axis=1).mean()
            smooth = np.diff(bias)
            loss += self.regularization * float(np.sum(smooth * smooth))

            dlogits = (pred - p) / q.shape[0]
            grad = np.zeros_like(x)
            grad[0] = np.sum(dlogits * (scale * q))
            gb = np.bincount(buckets.ravel(), weights=dlogits.ravel(), minlength=self.bucket_count)
            gb -= gb.mean()
            gs = np.zeros_like(bias)
            gs[:-1] += 2 * self.regularization * (bias[:-1] - bias[1:])
            gs[1:] += 2 * self.regularization * (bias[1:] - bias[:-1])
            grad[1:] = gb + gs
            return float(loss), grad

        result = minimize(objective_and_grad, x0, jac=True, method="L-BFGS-B",
                          options={"maxiter": 250, "ftol": 1e-11})
        self.scale = float(math.exp(result.x[0]))
        self.bias = result.x[1:] - result.x[1:].mean()
        return self

    def transform(self, z_quant: np.ndarray) -> np.ndarray:
        q = center(z_quant)
        return self.scale * q + self.bias[log_rank_buckets(q, self.bucket_count)]


class IsotonicLogitCalibrator(Calibrator):
    """A monotone nonlinear map from centered quant logits to reference logits."""

    name = "isotonic"

    def __init__(self) -> None:
        self.model = IsotonicRegression(out_of_bounds="clip", y_min=-30.0, y_max=30.0)

    def fit(self, z_ref: np.ndarray, z_quant: np.ndarray) -> "IsotonicLogitCalibrator":
        self.model.fit(center(z_quant).ravel(), center(z_ref).ravel())
        return self

    def transform(self, z_quant: np.ndarray) -> np.ndarray:
        shape = z_quant.shape
        return self.model.predict(center(z_quant).ravel()).reshape(shape)


class TokenBiasCalibrator(Calibrator):
    """Temperature plus a shrunken per-token mean paired-logit residual."""

    name = "temperature+token-bias"

    def __init__(self, prior_contexts: float = 8.0) -> None:
        self.prior_contexts = prior_contexts
        self.temp = TemperatureCalibrator()
        self.bias: np.ndarray | None = None

    def fit(self, z_ref: np.ndarray, z_quant: np.ndarray) -> "TokenBiasCalibrator":
        self.temp.fit(z_ref, z_quant)
        residual = center(z_ref) - center(self.temp.transform(z_quant))
        shrink = z_ref.shape[0] / (z_ref.shape[0] + self.prior_contexts)
        self.bias = shrink * residual.mean(axis=0)
        self.bias -= self.bias.mean()
        return self

    def transform(self, z_quant: np.ndarray) -> np.ndarray:
        assert self.bias is not None
        return self.temp.transform(z_quant) + self.bias


def context_features(q: np.ndarray, basis: np.ndarray | None = None) -> np.ndarray:
    q = center(q)
    sorted_q = np.sort(q, axis=1)
    p = softmax(q)
    entropy = -np.sum(p * np.log(p + EPS), axis=1)
    feats = [
        q.std(axis=1), entropy,
        sorted_q[:, -1], sorted_q[:, -1] - sorted_q[:, -2],
        sorted_q[:, -1] - sorted_q[:, -5],
        np.quantile(q, 0.9, axis=1) - np.quantile(q, 0.5, axis=1),
    ]
    out = np.stack(feats, axis=1)
    if basis is not None and len(basis):
        out = np.c_[out, q @ basis.T]
    return out


class ResidualPCACalibrator(Calibrator):
    """Heretic-style low-rank paired residual directions plus a coefficient predictor.

    The vocabulary-aligned residual matrix has one row per calibration context.
    Its dominant right singular vectors are output-space distortion directions.
    A tiny ridge model predicts their context-dependent amplitudes from the
    quantized logit vector and simple uncertainty features.
    """

    name = "temperature+token-bias+PCA"

    def __init__(self, max_components: int = 4, prior_contexts: float = 8.0,
                 ridge_alpha: float = 10.0) -> None:
        self.max_components = max_components
        self.prior_contexts = prior_contexts
        self.ridge_alpha = ridge_alpha
        self.temp = TemperatureCalibrator()
        self.mean_residual: np.ndarray | None = None
        self.basis: np.ndarray = np.empty((0, 0))
        self.predictor: Ridge | None = None

    def fit(self, z_ref: np.ndarray, z_quant: np.ndarray) -> "ResidualPCACalibrator":
        self.temp.fit(z_ref, z_quant)
        base = center(self.temp.transform(z_quant))
        residual = center(z_ref) - base
        shrink = z_ref.shape[0] / (z_ref.shape[0] + self.prior_contexts)
        self.mean_residual = shrink * residual.mean(axis=0)
        centered_residual = residual - residual.mean(axis=0, keepdims=True)
        k = min(self.max_components, max(0, z_ref.shape[0] - 3))
        if k == 0:
            self.basis = np.empty((0, z_ref.shape[1]))
            return self
        _, _, vt = np.linalg.svd(centered_residual, full_matrices=False)
        self.basis = vt[:k]
        targets = centered_residual @ self.basis.T
        features = context_features(z_quant, self.basis)
        self.predictor = Ridge(alpha=self.ridge_alpha).fit(features, targets)
        return self

    def transform(self, z_quant: np.ndarray) -> np.ndarray:
        assert self.mean_residual is not None
        corrected = self.temp.transform(z_quant) + self.mean_residual
        if self.predictor is not None and len(self.basis):
            coeff = self.predictor.predict(context_features(z_quant, self.basis))
            if coeff.ndim == 1:
                coeff = coeff[:, None]
            corrected = corrected + coeff @ self.basis
        return corrected


CALIBRATORS: dict[str, Callable[[], Calibrator]] = {
    "identity": Calibrator,
    "temperature": TemperatureCalibrator,
    "temperature+rank": RankBiasCalibrator,
    "isotonic": IsotonicLogitCalibrator,
    "temperature+token-bias": TokenBiasCalibrator,
    "temperature+token-bias+PCA": ResidualPCACalibrator,
}


def controlled_logits(n: int, vocab: int, seed: int,
                      scenario: str) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    latent_dim = 20
    token_latent = rng.normal(size=(latent_dim, vocab)) / math.sqrt(latent_dim)
    frequency = -0.82 * np.log1p(np.arange(vocab))
    frequency = frequency[rng.permutation(vocab)]
    h = rng.normal(size=(n, latent_dim))
    z = 1.25 * (h @ token_latent) + frequency
    z = center(z)
    if scenario == "scale":
        q = 0.82 * z + rng.normal(scale=0.035, size=z.shape)
        return z, q
    if scenario != "mixed":
        raise ValueError(f"unknown scenario: {scenario}")

    stable_bias = rng.normal(scale=0.11, size=vocab)
    residual_basis = rng.normal(size=(3, vocab))
    residual_basis /= np.linalg.norm(residual_basis, axis=1, keepdims=True)
    # Coefficients are partly visible in the quantized logit vector, making a
    # learned context-dependent residual correction possible but not oracle-like.
    coeff = 0.34 * h[:, :3] + rng.normal(scale=0.10, size=(n, 3))
    low_rank = coeff @ residual_basis * math.sqrt(vocab / 4)

    order = np.argsort(-z, axis=1)
    ranks = np.empty_like(order)
    rows = np.arange(n)[:, None]
    ranks[rows, order] = np.arange(vocab)[None, :]
    rank_shift = -0.18 * np.exp(-ranks / 10.0) + 0.055 * np.log1p(ranks) / math.log(vocab)
    noise_scale = 0.045 + 0.025 * (ranks / vocab)
    noise = rng.normal(size=z.shape) * noise_scale
    q = 0.87 * z + stable_bias + low_rank + rank_shift + noise
    return z, q


def make_language_data(n: int, vocab: int, context_len: int,
                       seed: int, context_distribution: str = "zipf") -> tuple[np.ndarray, np.ndarray]:
    """Sample a next-token task from a latent nonlinear teacher."""
    rng = np.random.default_rng(seed)
    if context_distribution == "zipf":
        contexts = rng.zipf(1.35, size=(n, context_len))
        contexts = np.minimum(contexts - 1, vocab - 1).astype(np.int32)
    elif context_distribution == "uniform":
        contexts = rng.integers(0, vocab, size=(n, context_len), dtype=np.int32)
    else:
        raise ValueError(f"unknown context distribution: {context_distribution}")
    d = 18
    embed = rng.normal(size=(vocab, d))
    out = rng.normal(size=(d, vocab)) / math.sqrt(d)
    base = -0.55 * np.log1p(np.arange(vocab))
    position_weights = np.linspace(0.25, 1.0, context_len)
    hidden = sum(position_weights[j] * embed[contexts[:, j]] for j in range(context_len))
    hidden += 0.35 * embed[(contexts[:, -1] + contexts[:, -2]) % vocab]
    logits = 1.6 * np.tanh(hidden) @ out + base
    probs = softmax(logits / 0.72)
    u = rng.random(n)
    targets = (u[:, None] > np.cumsum(probs, axis=1)).sum(axis=1).astype(np.int32)
    x = np.zeros((n, context_len * vocab), dtype=np.float32)
    rows = np.arange(n)
    for j in range(context_len):
        x[rows, j * vocab + contexts[:, j]] = 1.0
    return x, targets


def mlp_logits(model: MLPClassifier, x: np.ndarray,
               coefs: list[np.ndarray] | None = None) -> np.ndarray:
    weights = model.coefs_ if coefs is None else coefs
    activation = x
    for i, (w, b) in enumerate(zip(weights, model.intercepts_)):
        activation = activation @ w + b
        if i != len(weights) - 1:
            activation = np.maximum(activation, 0.0)
    return activation


def quantize_groupwise_symmetric(w: np.ndarray, bits: int, group_size: int = 32) -> np.ndarray:
    """Quantize each output column in groups along its input dimension."""
    levels = 2 ** (bits - 1) - 1
    out = np.empty_like(w)
    for start in range(0, w.shape[0], group_size):
        block = w[start:start + group_size]
        scale = np.max(np.abs(block), axis=0, keepdims=True) / levels
        scale = np.where(scale == 0, 1.0, scale)
        out[start:start + group_size] = np.clip(np.round(block / scale), -levels, levels) * scale
    return out


def fit_micro_lm(seed: int, quick: bool) -> tuple[
        np.ndarray, dict[int, np.ndarray], np.ndarray, dict[int, np.ndarray], dict[str, float]]:
    vocab, context_len = 96, 4
    n_train = 16000 if quick else 36000
    n_test = 2500 if quick else 6000
    x_train, y_train = make_language_data(n_train, vocab, context_len, seed)
    x_test, _ = make_language_data(n_test, vocab, context_len, seed + 1)
    x_ood, _ = make_language_data(n_test, vocab, context_len, seed + 2, "uniform")
    model = MLPClassifier(
        hidden_layer_sizes=(64,), activation="relu", solver="adam",
        batch_size=256, learning_rate_init=2e-3,
        max_iter=14 if quick else 24, early_stopping=False,
        random_state=seed, verbose=False,
    )
    start = time.time()
    model.fit(x_train, y_train)
    train_seconds = time.time() - start
    z_ref = mlp_logits(model, x_test)
    z_ood = mlp_logits(model, x_ood)
    quant_logits = {}
    quant_ood = {}
    weight_errors = {}
    for bits in (4, 3):
        qcoefs = [quantize_groupwise_symmetric(w, bits) for w in model.coefs_]
        quant_logits[bits] = mlp_logits(model, x_test, qcoefs)
        quant_ood[bits] = mlp_logits(model, x_ood, qcoefs)
        numerator = sum(float(np.sum((a - b) ** 2)) for a, b in zip(model.coefs_, qcoefs))
        denominator = sum(float(np.sum(a ** 2)) for a in model.coefs_)
        weight_errors[str(bits)] = math.sqrt(numerator / denominator)
    metadata = {
        "train_examples": n_train, "test_examples": n_test,
        "vocab": vocab, "context_len": context_len,
        "train_seconds": train_seconds,
        "training_iterations": int(model.n_iter_),
        "relative_weight_rmse_q4": weight_errors["4"],
        "relative_weight_rmse_q3": weight_errors["3"],
    }
    return z_ref, quant_logits, z_ood, quant_ood, metadata


def calibration_curve(z: np.ndarray, q: np.ndarray, sample_sizes: list[int],
                      trials: int, seed: int, source: str) -> list[dict]:
    rng = np.random.default_rng(seed)
    n = len(z)
    test_idx = np.arange(n // 3, n)
    pool = np.arange(0, n // 3)
    rows = []
    for trial in range(trials):
        order = rng.permutation(pool)
        for sample_size in sample_sizes:
            cal = order[:sample_size]
            for name, factory in CALIBRATORS.items():
                calibrator = factory().fit(z[cal], q[cal])
                corrected = calibrator.transform(q[test_idx])
                metrics = distribution_metrics(z[test_idx], corrected)
                row = {"source": source, "trial": trial, "calibration_contexts": sample_size,
                       "method": name, **metrics}
                if isinstance(calibrator, TemperatureCalibrator):
                    row["fitted_scale"] = calibrator.scale
                rows.append(row)
    return rows


def transfer_curve(z_cal_source: np.ndarray, q_cal_source: np.ndarray,
                   z_test: np.ndarray, q_test: np.ndarray,
                   sample_sizes: list[int], trials: int, seed: int,
                   source: str) -> list[dict]:
    """Fit on one context distribution and evaluate on a different one."""
    rng = np.random.default_rng(seed)
    pool = np.arange(len(z_cal_source) // 3)
    rows = []
    for trial in range(trials):
        order = rng.permutation(pool)
        for sample_size in sample_sizes:
            cal = order[:sample_size]
            for name, factory in CALIBRATORS.items():
                fitted = factory().fit(z_cal_source[cal], q_cal_source[cal])
                metrics = distribution_metrics(z_test, fitted.transform(q_test))
                rows.append({"source": source, "trial": trial,
                             "calibration_contexts": sample_size,
                             "method": name, **metrics})
    return rows


def temperature_transfer(z: np.ndarray, q: np.ndarray, calibration_contexts: int,
                         trials: int, seed: int, source: str) -> list[dict]:
    """Fit corrections at softmax T=1, then evaluate unchanged at other T."""
    rng = np.random.default_rng(seed)
    pool = np.arange(len(z) // 3)
    test = np.arange(len(z) // 3, len(z))
    methods = ["identity", "temperature", "temperature+token-bias",
               "temperature+token-bias+PCA"]
    rows = []
    for trial in range(trials):
        cal = rng.choice(pool, size=calibration_contexts, replace=False)
        for method in methods:
            fitted = CALIBRATORS[method]().fit(z[cal], q[cal])
            corrected = fitted.transform(q[test])
            for temp in (0.55, 0.8, 1.0, 1.3):
                metrics = distribution_metrics(z[test], corrected, temperature=temp)
                rows.append({"source": source, "trial": trial,
                             "calibration_contexts": calibration_contexts,
                             "method": method, "sampling_temperature": temp, **metrics})
    return rows


def fit_sampler_grid(z_ref: np.ndarray, z_quant: np.ndarray, target_temperature: float,
                     target_top_p: float, tune_top_p: bool) -> tuple[float, float]:
    p_ref = top_p_probs(z_ref, target_temperature, target_top_p)
    temperatures = np.linspace(0.50, 1.15, 27)
    top_ps = np.linspace(0.86, 1.0, 15) if tune_top_p else np.array([target_top_p])
    best = (float("inf"), target_temperature, target_top_p)
    for temp in temperatures:
        raw = softmax(z_quant / temp)
        order = np.argsort(-raw, axis=1)
        sorted_raw = np.take_along_axis(raw, order, axis=1)
        cumulative = np.cumsum(sorted_raw, axis=1)
        for top_p in top_ps:
            keep = cumulative - sorted_raw < top_p
            truncated = np.where(keep, sorted_raw, 0.0)
            truncated /= truncated.sum(axis=1, keepdims=True)
            q = np.zeros_like(raw)
            np.put_along_axis(q, order, truncated, axis=1)
            js = probability_metrics(p_ref, q)["js"]
            if js < best[0]:
                best = (js, float(temp), float(top_p))
    return best[1], best[2]


def sampler_comparison(z: np.ndarray, q: np.ndarray, sample_sizes: list[int],
                       trials: int, seed: int, source: str) -> list[dict]:
    rng = np.random.default_rng(seed)
    n = len(z)
    test_idx = np.arange(n // 3, n)
    pool = np.arange(n // 3)
    target_temp, target_p = 0.80, 0.95
    reference = top_p_probs(z[test_idx], target_temp, target_p)
    rows = []
    for trial in range(trials):
        order = rng.permutation(pool)
        for sample_size in sample_sizes:
            cal = order[:sample_size]
            candidates: list[tuple[str, np.ndarray, float, float]] = [
                ("same T,p", q[test_idx], target_temp, target_p),
            ]
            t_only, p_only = fit_sampler_grid(z[cal], q[cal], target_temp, target_p, False)
            t_both, p_both = fit_sampler_grid(z[cal], q[cal], target_temp, target_p, True)
            candidates.extend([
                ("tuned T", q[test_idx], t_only, p_only),
                ("tuned T,p", q[test_idx], t_both, p_both),
            ])
            for method_name in ("temperature+rank", "temperature+token-bias+PCA"):
                fitted = CALIBRATORS[method_name]().fit(z[cal], q[cal])
                candidates.append((method_name, fitted.transform(q[test_idx]), target_temp, target_p))
            for method, logits, temp, top_p in candidates:
                metrics = probability_metrics(reference, top_p_probs(logits, temp, top_p))
                rows.append({"source": source, "trial": trial,
                             "calibration_contexts": sample_size, "method": method,
                             "temperature": temp, "top_p": top_p, **metrics})
    return rows


def summarize_curves(frame: pd.DataFrame, metric: str = "kl") -> pd.DataFrame:
    return (frame.groupby(["source", "calibration_contexts", "method"])[metric]
            .agg(["mean", "std"]).reset_index())


def make_plots(curves: pd.DataFrame, sampler: pd.DataFrame, output_dir: Path) -> None:
    selected = ["identity", "temperature", "temperature+rank",
                "temperature+token-bias", "temperature+token-bias+PCA"]
    sources = [s for s in ["controlled:mixed", "micro-lm:Q3"] if s in set(curves.source)]
    fig, axes = plt.subplots(1, len(sources), figsize=(6.2 * len(sources), 4.4), squeeze=False)
    for ax, source in zip(axes[0], sources):
        subset = curves[(curves.source == source) & curves.method.isin(selected)]
        agg = subset.groupby(["calibration_contexts", "method"]).kl.mean().reset_index()
        for method, group in agg.groupby("method"):
            ax.plot(group.calibration_contexts, group.kl, marker="o", label=method)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_title(source)
        ax.set_xlabel("paired calibration contexts")
        ax.set_ylabel("KL(reference || corrected)")
        ax.grid(alpha=0.25)
    axes[0, -1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "low_sample_kl.png", dpi=170)
    plt.close(fig)

    focus = sampler[sampler.calibration_contexts == sampler.calibration_contexts.max()]
    agg = focus.groupby(["source", "method"]).js.mean().reset_index()
    fig, ax = plt.subplots(figsize=(9, 4.7))
    pivot = agg.pivot(index="source", columns="method", values="js")
    pivot.plot.bar(ax=ax)
    ax.set_ylabel("JS divergence vs reference sampler")
    ax.set_xlabel("")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "sampler_compensation_js.png", dpi=170)
    plt.close(fig)


def format_table(frame: pd.DataFrame, source: str, contexts: int, metric: str,
                 methods: list[str] | None = None) -> str:
    f = frame[(frame.source == source) & (frame.calibration_contexts == contexts)]
    if methods is not None:
        f = f[f.method.isin(methods)]
    agg = f.groupby("method")[metric].agg(["mean", "std"]).sort_values("mean")
    lines = ["| Method | Mean | Trial SD |", "|---|---:|---:|"]
    for method, row in agg.iterrows():
        lines.append(f"| {method} | {row['mean']:.6g} | {row['std']:.3g} |")
    return "\n".join(lines)


def transfer_temperature_table(frame: pd.DataFrame, source: str) -> str:
    f = frame[frame.source == source]
    pivot = f.groupby(["method", "sampling_temperature"]).kl.mean().unstack()
    pivot = pivot.sort_values(1.0)
    lines = ["| Method | T=0.55 | T=0.8 | T=1.0 | T=1.3 |",
             "|---|---:|---:|---:|---:|"]
    for method, row in pivot.iterrows():
        lines.append(f"| {method} | {row[0.55]:.5g} | {row[0.8]:.5g} | {row[1.0]:.5g} | {row[1.3]:.5g} |")
    return "\n".join(lines)


def write_report(output_dir: Path, curves: pd.DataFrame, sampler: pd.DataFrame,
                 temp_transfer: pd.DataFrame,
                 metadata: dict, elapsed: float) -> None:
    contexts = int(curves.calibration_contexts.max())
    methods = list(CALIBRATORS)
    scale_table = format_table(curves, "controlled:scale", contexts, "kl", methods)
    mixed_table = format_table(curves, "controlled:mixed", contexts, "kl", methods)
    q4_table = format_table(curves, "micro-lm:Q4", contexts, "kl", methods)
    q3_table = format_table(curves, "micro-lm:Q3", contexts, "kl", methods)
    q4_ood_table = format_table(curves, "micro-lm:Q4:OOD-uniform", contexts, "kl", methods)
    q3_ood_table = format_table(curves, "micro-lm:Q3:OOD-uniform", contexts, "kl", methods)
    sampler_table = format_table(sampler, "micro-lm:Q3", contexts, "js")
    temperature_table = transfer_temperature_table(temp_transfer, "micro-lm:Q3")

    report = f"""# Low-sample quantization sampling compensation: proof of concept

## Bottom line

Paired full-logit calibration is information-dense: one context supplies one aligned
error observation for every vocabulary item.  In these experiments, 16–64 paired
contexts were enough to distinguish three regimes:

1. **Nearly uniform logit scaling:** a single fitted temperature is the right answer.
2. **Rank-shaped error:** a small logarithmic rank-bias table improves on temperature.
3. **Stable token or low-rank residuals:** a shrunken vocabulary bias and a few residual
   PCA directions can improve further, but are more prone to workload overfit.

Ordinary temperature/top-p search can only repair the first regime.  Top-p is a support
cutoff, not a general distribution calibrator; tuning it sometimes lowers JS divergence
but cannot restore token-specific probability mass.

## Experiment A: controlled output shifts

Reference logits were generated by a latent softmax model.  The `scale` condition adds
only global flattening and small noise.  The `mixed` condition additionally contains a
stable vocabulary bias, a rank-shaped error, context-dependent low-rank residuals, and
heteroscedastic noise.

At {contexts} calibration contexts, KL(reference || corrected):

### Scale-only condition

{scale_table}

### Mixed condition

{mixed_table}

## Experiment B: trained and weight-quantized micro language model

A 384→64→96 next-token MLP was trained on {metadata['train_examples']:,} examples from a
latent nonlinear language generator, then every weight matrix was quantized groupwise
to signed 4-bit or 3-bit values (group size 32).  This is a genuine learned-weight
quantization experiment with exact paired logits, but it is **not evidence about a
specific transformer/GGUF format**.

- Training iterations: {metadata['training_iterations']}
- Relative weight RMSE: Q4={metadata['relative_weight_rmse_q4']:.4f}, Q3={metadata['relative_weight_rmse_q3']:.4f}
- Test contexts: {metadata['test_examples']:,}

### Q4

{q4_table}

### Q3

{q3_table}

### Context-distribution stress test

The same corrections were fit on Zipf-distributed language contexts and applied without
refitting to uniformly distributed token contexts, a deliberately severe covariate shift.

#### Q4, out of distribution

{q4_ood_table}

#### Q3, out of distribution

{q3_ood_table}

### Matching a reference sampler (T=0.8, top-p=0.95), Q3

JS divergence at {contexts} paired contexts:

{sampler_table}

### Transfer across downstream temperatures, Q3

Corrections were fit once at T=1 and then left unchanged upstream of the sampler:

{temperature_table}

## Proposed real-model calibration protocol

1. Select 32–128 short, diverse prompts from the actual workload.  Use one or several
   teacher-forced positions per prompt; do not roll out independently, because different
   sampled histories confound quantization error with context divergence.
2. Run the reference precision and target quant with identical tokens and save full logits
   at the selected positions.  Center each logit vector; softmax is invariant to the removed
   scalar offset.
3. Fit nested models in this order: temperature → rank buckets → shrunken token bias →
   residual PCA.  Choose the smallest model that improves held-out KL/JS across prompt
   strata.  Never select on the same contexts used to fit the higher-capacity correction.
4. Evaluate at several downstream temperatures and truncation settings.  Also report top-1
   flips, top-k overlap, and task/sequence evaluations; matching one-step distributions is
   necessary but not sufficient for matching rollout quality.
5. Export the correction upstream of the existing sampler.  Rank correction is tiny and
   quant-agnostic.  Token/PCA correction requires vocabulary identity and should be tied to
   the exact model, quant file, backend, and prompt template.

## Mathematical form

For reference logits `z` and quant logits `q`, fit `g(q; θ)` by minimizing

`mean_context KL(softmax(z / T*) || softmax(g(q; θ) / T*))`.

The nested correction used here is

`g(q) = a·center(q) + b_rank(q) + μ_token + Σ_j c_j(q) v_j`,

where `a` is temperature compensation, `b_rank` is a small log-rank table, `μ_token`
is a shrunken mean residual, and `v_j` are right singular vectors of the paired residual
matrix.  The last term is the closest analogue to finding a low-dimensional direction:
the directions are learned from reference-minus-quant output residuals, while a ridge
model predicts their amplitude using only the current quantized logits.

## Practical cautions

- A per-token bias looks statistically cheap because each context exposes the whole
  vocabulary, but rare-token logits can still shift by domain.  Shrink and validate it.
- PCA directions are only useful if their amplitudes are predictable from the quantized
  state.  An oracle projection using reference logits would be invalid at inference.
- Greedy top-1 flips often come from small local margin errors; a global temperature cannot
  change ranking at all.
- The best correction may differ by quantization method, backend/kernel, context length,
  prompt template, and workload.  It should be stored as metadata for one exact artifact,
  not advertised as a generic Q3/Q4 preset.

## Relation to existing work

- [LFQ](https://arxiv.org/abs/2605.29756) confirms that matching full-precision token
  probabilities at the final-block logit level improves low-bit generation, but changes
  the quantization procedure; this project asks how much can be recovered purely after
  the forward pass.
- [Quantized Reasoning Models Think They Need to Think Longer](https://arxiv.org/abs/2606.00206)
  uses teacher-forced token-level KL to isolate quantization drift and improves behavior
  with targeted token logit penalties.  The token-bias calibrator here is an automatic,
  vocabulary-wide version of that general intervention.
- [Bias Compensation](https://arxiv.org/abs/2404.01892) adds analytically fitted bias
  vectors to quantized layer outputs.  It is closely related in spirit, but acts inside
  the model rather than as a distribution-level sampler preprocessor.
- [Heretic](https://github.com/p-e-w/heretic) searches low-dimensional model edits while
  preserving the original distribution with a KL objective.  The residual-PCA method
  borrows the low-dimensional-direction idea, not Heretic's refusal-specific mechanism.
- [`llama-perplexity`](https://github.com/ggml-org/llama.cpp/tree/master/tools/perplexity)
  already records reference logits and computes teacher-forced KL, providing a natural
  extraction path for a real GGUF experiment.

## Reproduction

Run:

```bash
python3 quant_sampler_lab.py --output-dir results
```

Raw trial-level results are in `calibration_curves.csv` and `sampler_comparison.csv`.
Temperature-transfer trials are in `temperature_transfer.csv`.  The complete
machine-readable bundle is `results.json`.  Total runtime for this run was
{elapsed:.1f} seconds.
"""
    (output_dir / "REPORT.md").write_text(report)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    sample_sizes = [2, 4, 8, 16, 32] if args.quick else [2, 4, 8, 16, 32, 64]
    trials = 3 if args.quick else 5

    curve_rows: list[dict] = []
    sampler_rows: list[dict] = []
    temp_transfer_rows: list[dict] = []
    for i, scenario in enumerate(("scale", "mixed")):
        z, q = controlled_logits(1000 if args.quick else 1800, 512, args.seed + i, scenario)
        source = f"controlled:{scenario}"
        curve_rows += calibration_curve(z, q, sample_sizes, trials, args.seed + 100 + i, source)
        sampler_rows += sampler_comparison(z, q, [16, sample_sizes[-1]], trials,
                                           args.seed + 200 + i, source)
        if scenario == "mixed":
            temp_transfer_rows += temperature_transfer(z, q, sample_sizes[-1], trials,
                                                       args.seed + 250 + i, source)

    z_ref, quant_logits, z_ood, quant_ood, metadata = fit_micro_lm(args.seed + 300, args.quick)
    for bits, q in quant_logits.items():
        source = f"micro-lm:Q{bits}"
        curve_rows += calibration_curve(z_ref, q, sample_sizes, trials,
                                        args.seed + 400 + bits, source)
        sampler_rows += sampler_comparison(z_ref, q, [16, sample_sizes[-1]], trials,
                                           args.seed + 500 + bits, source)
        curve_rows += transfer_curve(z_ref, q, z_ood, quant_ood[bits],
                                     [16, sample_sizes[-1]], trials,
                                     args.seed + 550 + bits, source + ":OOD-uniform")
        if bits == 3:
            temp_transfer_rows += temperature_transfer(z_ref, q, sample_sizes[-1], trials,
                                                       args.seed + 600 + bits, source)

    curves = pd.DataFrame(curve_rows)
    sampler = pd.DataFrame(sampler_rows)
    temp_transfer_frame = pd.DataFrame(temp_transfer_rows)
    curves.to_csv(args.output_dir / "calibration_curves.csv", index=False)
    sampler.to_csv(args.output_dir / "sampler_comparison.csv", index=False)
    temp_transfer_frame.to_csv(args.output_dir / "temperature_transfer.csv", index=False)
    make_plots(curves, sampler, args.output_dir)
    elapsed = time.time() - start
    bundle = {
        "configuration": {"seed": args.seed, "quick": args.quick,
                          "sample_sizes": sample_sizes, "trials": trials},
        "micro_lm": metadata,
        "calibration_summary": summarize_curves(curves).to_dict(orient="records"),
        "sampler_summary": (sampler.groupby(["source", "calibration_contexts", "method"])
                            [["js", "tv"]].mean().reset_index().to_dict(orient="records")),
        "temperature_transfer_summary": (
            temp_transfer_frame.groupby(["source", "sampling_temperature", "method"])
            [["kl", "js", "tv"]].mean().reset_index().to_dict(orient="records")),
        "runtime_seconds": elapsed,
    }
    (args.output_dir / "results.json").write_text(json.dumps(bundle, indent=2))
    write_report(args.output_dir, curves, sampler, temp_transfer_frame, metadata, elapsed)
    print(json.dumps({"output_dir": str(args.output_dir), "runtime_seconds": elapsed,
                      "micro_lm": metadata}, indent=2))


if __name__ == "__main__":
    main()
