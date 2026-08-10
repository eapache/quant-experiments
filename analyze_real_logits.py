#!/usr/bin/env python3
"""Cross-validate post-logit corrections on llama-perplexity KLD files.

The input files contain quantized log-probabilities, not raw logits. This is enough:
softmax is invariant to a per-position additive constant, and every correction tested
here is applied before the user's ordinary temperature/nucleus sampler.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar


GAP_EDGES = np.array([0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0], dtype=np.float32)


@dataclass
class KLDFile:
    path: Path
    n_ctx: int
    n_vocab: int
    n_chunk: int
    tokens: np.ndarray
    codes: np.memmap
    scale: np.ndarray
    minimum: np.ndarray

    @property
    def rows_per_chunk(self) -> int:
        return self.n_ctx - 1 - self.n_ctx // 2

    @property
    def n_rows(self) -> int:
        return self.n_chunk * self.rows_per_chunk

    @property
    def chunk_ids(self) -> np.ndarray:
        return np.repeat(np.arange(self.n_chunk), self.rows_per_chunk)

    def decode(self) -> np.ndarray:
        values = self.codes[:, : self.n_vocab].astype(np.float32)
        values *= self.scale[:, None]
        values += self.minimum[:, None]
        return values

    def retained_mass(self) -> np.ndarray:
        result = np.empty(self.n_rows, dtype=np.float64)
        for row in range(self.n_rows):
            keep = self.codes[row, : self.n_vocab] > 0
            lp = self.minimum[row] + self.scale[row] * self.codes[row, : self.n_vocab][keep]
            result[row] = np.exp(lp.astype(np.float64)).sum()
        return result


def read_kld(path: Path) -> KLDFile:
    with path.open("rb") as stream:
        if stream.read(8) != b"_logits_":
            raise ValueError(f"{path} is not a llama.cpp KLD logit file")
        n_ctx, n_vocab, n_chunk = struct.unpack("<Iii", stream.read(12))
        tokens = np.fromfile(stream, dtype="<i4", count=n_ctx * n_chunk)
        offset = stream.tell()
    width = 2 * ((n_vocab + 1) // 2) + 4
    rows = n_chunk * (n_ctx - 1 - n_ctx // 2)
    raw = np.memmap(path, dtype="<u2", mode="r", offset=offset, shape=(rows, width))
    metadata = raw[:, :4].copy().view("<f4").reshape(rows, 2)
    expected_size = offset + rows * width * 2
    if path.stat().st_size != expected_size:
        raise ValueError(f"unexpected file size for {path}: {path.stat().st_size} != {expected_size}")
    return KLDFile(path, n_ctx, n_vocab, n_chunk, tokens, raw[:, 4:], metadata[:, 0], metadata[:, 1])


def check_alignment(reference: KLDFile, candidate: KLDFile) -> None:
    fields = ("n_ctx", "n_vocab", "n_chunk")
    for field in fields:
        if getattr(reference, field) != getattr(candidate, field):
            raise ValueError(f"unaligned {field}: {getattr(reference, field)} != {getattr(candidate, field)}")
    if not np.array_equal(reference.tokens, candidate.tokens):
        raise ValueError("token sequences are not aligned")


def distribution(logits: np.ndarray, mask: np.ndarray, temperature: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    x = logits / temperature
    x = np.where(mask, x, -np.inf)
    maximum = np.max(x, axis=1, keepdims=True)
    weights = np.exp(x - maximum, dtype=np.float32)
    weights = np.where(mask, weights, 0.0)
    totals = weights.sum(axis=1, keepdims=True, dtype=np.float64)
    probabilities = (weights / totals).astype(np.float32)
    log_probabilities = np.where(mask, x - maximum - np.log(totals), -np.inf)
    return probabilities, log_probabilities


def top_indices(logits: np.ndarray, mask: np.ndarray, k: int) -> np.ndarray:
    x = np.where(mask, logits, -np.inf)
    chosen = np.argpartition(x, -k, axis=1)[:, -k:]
    chosen_values = np.take_along_axis(x, chosen, axis=1)
    order = np.argsort(-chosen_values, axis=1)
    return np.take_along_axis(chosen, order, axis=1)


def distribution_metrics(reference: np.ndarray, candidate: np.ndarray, mask: np.ndarray,
                         temperature: float) -> dict[str, float]:
    p, lp = distribution(reference, mask, temperature)
    q, lq = distribution(candidate, mask, temperature)
    safe_lp = np.where(mask, lp, 0.0)
    safe_lq = np.where(mask, lq, 0.0)
    kl_rows = np.sum(p * (safe_lp - safe_lq), axis=1, dtype=np.float64)
    midpoint = 0.5 * (p + q)
    log_midpoint = np.zeros_like(midpoint)
    np.log(midpoint, out=log_midpoint, where=midpoint > 0)
    js_rows = 0.5 * np.sum(p * (safe_lp - log_midpoint), axis=1, dtype=np.float64)
    js_rows += 0.5 * np.sum(q * (safe_lq - log_midpoint), axis=1, dtype=np.float64)
    tv_rows = 0.5 * np.abs(p - q).sum(axis=1, dtype=np.float64)
    ref_top = top_indices(reference, mask, 10)
    candidate_top = top_indices(candidate, mask, 10)
    overlap = np.array([
        np.intersect1d(a, b, assume_unique=True).size / 10.0
        for a, b in zip(ref_top, candidate_top)
    ])
    return {
        "kl": float(kl_rows.mean()),
        "js": float(js_rows.mean()),
        "tv": float(tv_rows.mean()),
        "acceptance": float(1.0 - tv_rows.mean()),
        "top1_agreement": float(np.mean(ref_top[:, 0] == candidate_top[:, 0])),
        "top10_overlap": float(overlap.mean()),
        "ref_entropy": float(np.mean(-np.sum(p * safe_lp, axis=1))),
    }


def nucleus(probabilities: np.ndarray, threshold: float) -> np.ndarray:
    order = np.argsort(-probabilities)
    sorted_p = probabilities[order]
    keep = np.cumsum(sorted_p) - sorted_p < threshold
    sorted_p = np.where(keep, sorted_p, 0.0)
    sorted_p /= sorted_p.sum()
    result = np.zeros_like(probabilities)
    result[order] = sorted_p
    return result


def nucleus_rows(logits: np.ndarray, mask: np.ndarray, temperature: float,
                 top_p: float) -> list[np.ndarray]:
    probabilities, _ = distribution(logits, mask, temperature)
    return [nucleus(row[keep], top_p) for row, keep in zip(probabilities, mask)]


def sampler_metrics(reference_rows: list[np.ndarray], candidate: np.ndarray, mask: np.ndarray,
                    temperature: float, top_p: float) -> dict[str, float]:
    q, _ = distribution(candidate, mask, temperature)
    js_values = []
    tv_values = []
    support_jaccard = []
    for pp, q_row, keep in zip(reference_rows, q, mask):
        active = np.flatnonzero(keep)
        qq = nucleus(q_row[active], top_p)
        midpoint = 0.5 * (pp + qq)
        p_keep = pp > 0
        q_keep = qq > 0
        js = 0.5 * np.sum(pp[p_keep] * np.log(pp[p_keep] / midpoint[p_keep]))
        js += 0.5 * np.sum(qq[q_keep] * np.log(qq[q_keep] / midpoint[q_keep]))
        js_values.append(js)
        tv_values.append(0.5 * np.abs(pp - qq).sum())
        ps = pp > 0
        qs = qq > 0
        support_jaccard.append(np.count_nonzero(ps & qs) / max(1, np.count_nonzero(ps | qs)))
    return {
        "sampler_js": float(np.mean(js_values)),
        "sampler_tv": float(np.mean(tv_values)),
        "sampler_support_jaccard": float(np.mean(support_jaccard)),
    }


def fit_temperature(reference: np.ndarray, candidate: np.ndarray, mask: np.ndarray,
                    target_temperature: float = 1.0) -> float:
    p, _ = distribution(reference, mask, target_temperature)

    def objective(scale: float) -> float:
        _, lq = distribution(scale * candidate, mask, target_temperature)
        safe_lq = np.where(mask, lq, 0.0)
        return float(-np.sum(p * safe_lq, dtype=np.float64) / p.shape[0])

    result = minimize_scalar(objective, bounds=(0.4, 1.8), method="bounded", options={"xatol": 2e-5})
    return float(result.x)


def gap_buckets(candidate: np.ndarray, mask: np.ndarray) -> np.ndarray:
    maximum = np.max(np.where(mask, candidate, -np.inf), axis=1, keepdims=True)
    gaps = maximum - candidate
    return np.searchsorted(GAP_EDGES, gaps, side="right").astype(np.int8)


def fit_gap_bias(reference: np.ndarray, candidate: np.ndarray, mask: np.ndarray,
                 scale: float) -> np.ndarray:
    p, _ = distribution(reference, mask)
    q, _ = distribution(scale * candidate, mask)
    weights = 0.5 * (p + q)
    residual = reference - scale * candidate
    row_shift = np.sum(weights * residual, axis=1, keepdims=True, dtype=np.float64)
    residual = residual - row_shift
    buckets = gap_buckets(candidate, mask)
    count = len(GAP_EDGES) + 1
    numerator = np.bincount(buckets[mask], weights=(weights * residual)[mask], minlength=count)
    denominator = np.bincount(buckets[mask], weights=weights[mask], minlength=count)
    bias = numerator / np.maximum(denominator + 0.05, 1e-12)
    bias -= bias[0]
    return np.clip(bias, -1.0, 1.0).astype(np.float32)


def fit_token_bias(reference: np.ndarray, candidate: np.ndarray, exact: np.ndarray,
                   mask: np.ndarray, base: np.ndarray, prior_positions: float = 32.0) -> np.ndarray:
    p, _ = distribution(reference, mask)
    residual = reference - base
    row_shift = np.sum(p * residual, axis=1, keepdims=True, dtype=np.float64)
    residual -= row_shift
    numerator = np.sum(np.where(exact, residual, 0.0), axis=0, dtype=np.float64)
    counts = exact.sum(axis=0)
    bias = numerator / np.maximum(counts + prior_positions, 1.0)
    observed = counts > 0
    if np.any(observed):
        bias -= np.average(bias[observed], weights=counts[observed])
    return np.clip(bias, -1.0, 1.0).astype(np.float32)


def fit_methods(reference: np.ndarray, candidate: np.ndarray, mask: np.ndarray,
                exact: np.ndarray) -> tuple[dict[str, tuple[float, np.ndarray, np.ndarray]], float]:
    scale = fit_temperature(reference, candidate, mask)
    gap = fit_gap_bias(reference, candidate, mask, scale)
    buckets = gap_buckets(candidate, mask)
    scaled = scale * candidate
    gap_corrected = scaled + gap[buckets]
    token = fit_token_bias(reference, candidate, exact, mask, scaled)
    gap_token = fit_token_bias(reference, candidate, exact, mask, gap_corrected)
    zero_gap = np.zeros_like(gap)
    zero_token = np.zeros(candidate.shape[1], dtype=np.float32)
    return {
        "identity": (1.0, zero_gap, zero_token),
        "temperature": (scale, zero_gap, zero_token),
        "temperature+gap": (scale, gap, zero_token),
        "temperature+token": (scale, zero_gap, token),
        "temperature+gap+token": (scale, gap, gap_token),
    }, scale


def transform(candidate: np.ndarray, mask: np.ndarray,
              parameters: tuple[float, np.ndarray, np.ndarray]) -> np.ndarray:
    scale, gap, token = parameters
    return scale * candidate + gap[gap_buckets(candidate, mask)] + token


def mean_by(rows: list[dict], keys: tuple[str, ...], metric: str) -> dict[tuple, tuple[float, float]]:
    groups: dict[tuple, list[float]] = {}
    for row in rows:
        key = tuple(str(row[k]) for k in keys)
        groups.setdefault(key, []).append(float(row[metric]))
    return {key: (float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else 0.0)
            for key, values in groups.items()}


def analyze_pair(reference_file: KLDFile, candidate_file: KLDFile, name: str,
                 temperatures: list[float], top_p: float) -> tuple[list[dict], list[dict]]:
    check_alignment(reference_file, candidate_file)
    reference = reference_file.decode()
    candidate = candidate_file.decode()
    reference_exact = reference_file.codes[:, : reference_file.n_vocab] > 0
    candidate_exact = candidate_file.codes[:, : candidate_file.n_vocab] > 0
    active = reference_exact | candidate_exact
    exact = reference_exact & candidate_exact
    chunks = reference_file.chunk_ids
    rows: list[dict] = []
    scale_rows: list[dict] = []

    # Four outer folds, each holding out two complete contiguous chunks.
    for fold in range(4):
        held_out = np.array([2 * fold, 2 * fold + 1])
        test = np.isin(chunks, held_out)
        train = ~test
        methods, fitted_scale = fit_methods(reference[train], candidate[train], active[train], exact[train])
        sampler_temperature = temperatures[0]
        reference_nucleus = nucleus_rows(
            reference[test], active[test], sampler_temperature, top_p)
        scale_rows.append({
            "quant": name,
            "fold": fold,
            "train_positions": int(train.sum()),
            "test_positions": int(test.sum()),
            "scale": fitted_scale,
            "equivalent_temperature_at_0.8": 0.8 / fitted_scale,
        })
        for method, parameters in methods.items():
            corrected = transform(candidate[test], active[test], parameters)
            for temperature in temperatures:
                metrics = distribution_metrics(reference[test], corrected, active[test], temperature)
                if temperature == sampler_temperature:
                    sampler = sampler_metrics(reference_nucleus, corrected, active[test], temperature, top_p)
                else:
                    sampler = {
                        "sampler_js": math.nan,
                        "sampler_tv": math.nan,
                        "sampler_support_jaccard": math.nan,
                    }
                rows.append({
                    "quant": name,
                    "fold": fold,
                    "test_chunks": "+".join(map(str, held_out)),
                    "train_positions": int(train.sum()),
                    "test_positions": int(test.sum()),
                    "method": method,
                    "temperature": temperature,
                    "top_p": top_p,
                    **metrics,
                    **sampler,
                })
    del reference, candidate, reference_exact, candidate_exact, active, exact
    return rows, scale_rows


def analyze_path(reference_path: Path, candidate_path: Path, name: str,
                 temperatures: list[float], top_p: float) -> tuple[str, list[dict], list[dict], dict]:
    reference = read_kld(reference_path)
    candidate = read_kld(candidate_path)
    retained = candidate.retained_mass()
    rows, scales = analyze_pair(reference, candidate, name, temperatures, top_p)
    return name, rows, scales, {"mean": float(retained.mean()), "minimum": float(retained.min())}


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, rows: list[dict], scales: list[dict], metadata: dict) -> None:
    lines = [
        "# Real Qwen3.5-4B quantization compensation results",
        "",
        "## Experimental design",
        "",
        "Q8_K_XL is the available reference; Q4_K_XL and Q2_K_XL are candidates. "
        "All files were evaluated on identical teacher-forced tokens. Four outer folds each "
        "hold out two complete 128-token chunks, so adjacent held-out positions never leak "
        "into calibration. Each fold trains on 378 positions and tests on 126.",
        "",
        "Saved log-probabilities are clipped 16 log units below the row maximum. The analysis "
        "uses the union of non-clipped reference/candidate support and renormalizes it. Retained "
        "mass is recorded in `results.json`.",
        "",
        "Corrections are fit upstream of the ordinary sampler in nested order: a global logit "
        "scale, nine quant-logit-gap buckets, and a strongly shrunken per-token residual bias.",
        "",
        "## Cross-validated distribution results",
        "",
    ]
    summary = mean_by(rows, ("quant", "temperature", "method"), "kl")
    for quant in sorted({row["quant"] for row in rows}):
        for temperature in sorted({float(row["temperature"]) for row in rows}):
            lines += [f"### {quant} at downstream temperature {temperature:g}", "",
                      "| Method | KL mean | fold SD | Recovered vs identity | Top-1 agreement | Top-10 overlap |",
                      "|---|---:|---:|---:|---:|---:|"]
            identity = summary[(quant, str(temperature), "identity")][0]
            for method in ["identity", "temperature", "temperature+gap", "temperature+token", "temperature+gap+token"]:
                mean, sd = summary[(quant, str(temperature), method)]
                selected = [r for r in rows if r["quant"] == quant and float(r["temperature"]) == temperature and r["method"] == method]
                top1 = np.mean([r["top1_agreement"] for r in selected])
                top10 = np.mean([r["top10_overlap"] for r in selected])
                recovered = 1.0 - mean / identity
                lines.append(f"| {method} | {mean:.7f} | {sd:.7f} | {recovered:.1%} | {top1:.1%} | {top10:.1%} |")
            lines.append("")

    lines += ["## Nucleus sampler matching", "",
              "Jensen-Shannon divergence after applying the same temperature and top-p cutoff "
              "to the Q8 reference and corrected quant distributions:", "",
              "| Quant | Temperature | Method | sampler JS | Recovered vs identity | Support Jaccard |",
              "|---|---:|---|---:|---:|---:|"]
    sampler_rows = [row for row in rows if math.isfinite(float(row["sampler_js"]))]
    sampler_summary = mean_by(sampler_rows, ("quant", "temperature", "method"), "sampler_js")
    for quant in sorted({row["quant"] for row in sampler_rows}):
        for temperature in sorted({float(row["temperature"]) for row in sampler_rows}):
            identity = sampler_summary[(quant, str(temperature), "identity")][0]
            for method in ["identity", "temperature", "temperature+gap", "temperature+token", "temperature+gap+token"]:
                mean, _ = sampler_summary[(quant, str(temperature), method)]
                selected = [r for r in rows if r["quant"] == quant and float(r["temperature"]) == temperature and r["method"] == method]
                jaccard = np.mean([r["sampler_support_jaccard"] for r in selected])
                lines.append(f"| {quant} | {temperature:g} | {method} | {mean:.7f} | {1.0 - mean / identity:.1%} | {jaccard:.1%} |")
    lines += ["", "## Fitted temperature stability", "",
              "The fitted scale multiplies quant logits. To emulate a desired temperature of "
              "0.8 using only an ordinary temperature setting, use `0.8 / scale`.", "",
              "| Quant | Scale mean | fold SD | Equivalent T mean | fold SD |",
              "|---|---:|---:|---:|---:|"]
    for quant in sorted({row["quant"] for row in scales}):
        selected = [row for row in scales if row["quant"] == quant]
        values = np.array([row["scale"] for row in selected])
        temps = np.array([row["equivalent_temperature_at_0.8"] for row in selected])
        lines.append(f"| {quant} | {values.mean():.6f} | {values.std(ddof=1):.6f} | {temps.mean():.6f} | {temps.std(ddof=1):.6f} |")
    lines += ["", "## Interpretation", "",
              "See `EXPERIMENT_LOG.md` for the running interpretation, limitations, and exact extraction commands. "
              "Raw fold-level metrics are in `cross_validation.csv` and fitted scales in `temperature_fits.csv`.", ""]
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, default=Path("results/logits/q8.kld"))
    parser.add_argument("--q4", type=Path, default=Path("results/logits/q4.kld"))
    parser.add_argument("--q2", type=Path, default=Path("results/logits/q2.kld"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--temperatures", type=float, nargs="+", default=[0.8, 1.0])
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    reference = read_kld(args.reference)
    metadata = {
        "reference": str(args.reference),
        "n_ctx": reference.n_ctx,
        "n_chunks": reference.n_chunk,
        "rows_per_chunk": reference.rows_per_chunk,
        "positions": reference.n_rows,
        "vocabulary": reference.n_vocab,
        "top_p": args.top_p,
        "temperatures": args.temperatures,
        "retained_mass": {"q8": {
            "mean": float(reference.retained_mass().mean()),
            "minimum": float(reference.retained_mass().min()),
        }},
    }
    all_rows: list[dict] = []
    all_scales: list[dict] = []
    jobs = (("Q4_K_XL", args.q4), ("Q2_K_XL", args.q2))
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs) as executor:
        futures = [executor.submit(analyze_path, args.reference, path, name,
                                   args.temperatures, args.top_p) for name, path in jobs]
        for future in concurrent.futures.as_completed(futures):
            name, rows, scales, retained = future.result()
            print(f"finished {name}", flush=True)
            metadata["retained_mass"][name] = retained
            all_rows.extend(rows)
            all_scales.extend(scales)

    all_rows.sort(key=lambda row: (row["quant"], row["fold"], row["method"], row["temperature"]))
    all_scales.sort(key=lambda row: (row["quant"], row["fold"]))

    write_csv(args.output_dir / "cross_validation.csv", all_rows)
    write_csv(args.output_dir / "temperature_fits.csv", all_scales)
    (args.output_dir / "results.json").write_text(json.dumps({
        "metadata": metadata,
        "cross_validation": all_rows,
        "temperature_fits": all_scales,
    }, indent=2))
    write_report(args.output_dir / "REPORT.md", all_rows, all_scales, metadata)
    print(f"wrote {args.output_dir / 'REPORT.md'}")


if __name__ == "__main__":
    main()
