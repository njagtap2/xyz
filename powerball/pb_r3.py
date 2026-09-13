244967e13736b4496fb67a68c1d52657118f6d1a5ac96cb2354ec8d267423949  pb_shadow.py
"""PB-R3 reference math. Python standard library; no network, storage or orders."""
import hashlib
import json
import math
import random
from datetime import date, datetime, timezone
from pathlib import Path

VERSION = "PB-R3-1.0"
COMBINATIONS = math.comb(69, 5)
UNIFORM = 1 / (COMBINATIONS * 26)
P = 5 / 69
COMPONENTS = [(0, 1.0)] + [(i, r) for i in range(1, 70)
                         for r in (0.5, 0.8, 1.25, 2.0)]
KAPPAS = (1000, 10000, None)  # None means the exact uniform limit.
NULL_PRIORS = (0.90, 0.99, 1.0)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def valid_ticket(whites, red):
    return (isinstance(whites, (list, tuple)) and len(whites) == 5
            and all(type(x) is int and 1 <= x <= 69 for x in whites)
            and len(set(whites)) == 5 and type(red) is int and 1 <= red <= 26)


def history(rows, target):
    date.fromisoformat(target)
    clean = []
    seen = set()
    for row in rows:
        d = row["draw_date"]
        if date.fromisoformat(d).isoformat() != d or d >= target or d in seen:
            raise ValueError("Duplicate, noncanonical or future training draw")
        if not valid_ticket(row["whites"], row["red"]):
            raise ValueError("Invalid draw numbers")
        seen.add(d)
        clean.append(dict(draw_date=d, whites=sorted(row["whites"]), red=row["red"]))
    return sorted(clean, key=lambda x: x["draw_date"])


def posterior(counts, n, prior):
    if prior not in NULL_PRIORS:
        raise ValueError("Unregistered white prior")
    if prior == 1:
        return [1.0] + [0.0] * (len(COMPONENTS) - 1)
    logw = [math.log(prior)]
    lp = math.log((1 - prior) / (len(COMPONENTS) - 1))
    for i, r in COMPONENTS[1:]:
        logw.append(lp + counts[i-1] * math.log(r) - n * math.log1p(P * (r-1)))
    maximum = max(logw)
    weights = [math.exp(x-maximum) for x in logw]
    total = math.fsum(weights)
    return [x / total for x in weights]


def white_ratio(weights, whites):
    selected = set(whites)
    return math.fsum(w * (r if i in selected else 1.0) / (1-P+P*r)
                     for w, (i, r) in zip(weights, COMPONENTS))


def red_probs(counts, n, kappa):
    if kappa not in KAPPAS:
        raise ValueError("Unregistered red shrinkage")
    if kappa is None:
        return [1/26] * 26
    return [(c + kappa/26) / (n+kappa) for c in counts]


def calibrate(rows, target):
    rows = history(rows, target)
    if len(rows) < 300:
        raise ValueError("Need 300 verified same-rule draws for calibration")
    start = max(200, len(rows)-500)
    cw, cr = [0]*69, [0]*26
    wl, rl = {p: 0.0 for p in NULL_PRIORS}, {k: 0.0 for k in KAPPAS}
    for n, row in enumerate(rows):
        if n >= start:
            for p in NULL_PRIORS:
                wl[p] -= math.log(white_ratio(posterior(cw, n, p), row["whites"]))
            for k in KAPPAS:
                rl[k] -= math.log(26 * red_probs(cr, n, k)[row["red"]-1])
        for i in row["whites"]:
            cw[i-1] += 1
        cr[row["red"]-1] += 1
    # Deterministic tie rule favors stronger shrinkage, including uniform.
    bestp = max(p for p in NULL_PRIORS if wl[p] <= min(wl.values()) + 1e-10)
    bestk = max((k for k in KAPPAS if rl[k] <= min(rl.values()) + 1e-10),
                key=lambda k: math.inf if k is None else k)
    return {"kappa": bestk, "white_null_prior": bestp,
            "calibration": {"first": rows[start]["draw_date"],
                "last": rows[-1]["draw_date"], "draws": len(rows)-start,
                "training_sha256": digest(rows),
                "white_excess_log_loss": {str(k): v for k,v in wl.items()},
                "red_excess_log_loss": {str(k): v for k,v in rl.items()}}}


def uniform_forecast(target):
    date.fromisoformat(target)
    record = {"version": VERSION, "model": "M0", "target_draw": target,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "config": {"kappa": None, "white_null_prior": 1.0}, "n_training": 0,
        "training_cutoff": None, "training_sha256": digest([]),
        "white_weights": [1.0]+[0.0]*(len(COMPONENTS)-1),
        "red_probabilities": [1/26]*26}
    record["forecast_sha256"] = digest(record)
    return record


def fit(rows, target, config):
    rows = history(rows, target)
    if not rows:
        raise ValueError("Training history unavailable")
    cw, cr = [0]*69, [0]*26
    for row in rows:
        for i in row["whites"]:
            cw[i-1] += 1
        cr[row["red"]-1] += 1
    n = len(rows)
    record = {"version": VERSION, "model": "M1", "target_draw": target,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "config": config, "n_training": n, "training_cutoff": rows[-1]["draw_date"],
        "training_sha256": digest(rows),
        "white_weights": posterior(cw, n, config["white_null_prior"]),
        "red_probabilities": red_probs(cr, n, config["kappa"])}
    record["forecast_sha256"] = digest(record)
    return record


def validate_forecast(record):
    body = {k:v for k,v in record.items() if k != "forecast_sha256"}
    if digest(body) != record.get("forecast_sha256") or record["version"] != VERSION:
        raise ValueError("Forecast integrity/version failure")
    for key, size in (("white_weights", len(COMPONENTS)), ("red_probabilities", 26)):
        v = record[key]
        if (len(v) != size or any(not math.isfinite(x) or x < 0 for x in v)
                or not math.isclose(math.fsum(v), 1.0, rel_tol=0, abs_tol=1e-12)):
            raise ValueError("Invalid probability distribution")
    if min(record["red_probabilities"]) <= 0:
        raise ValueError("Red probabilities must have full support")


def draw_one(record, rng, baseline=False):
    if baseline:
        return sorted(rng.sample(range(1,70), 5)), rng.randrange(1,27)
    j = rng.choices(range(len(COMPONENTS)), weights=record["white_weights"])[0]
    i, r = COMPONENTS[j]
    if i == 0:
        whites = rng.sample(range(1,70), 5)
    else:
        others = [x for x in range(1,70) if x != i]
        include = rng.random() < P*r / (1-P+P*r)
        whites = rng.sample(others, 4 if include else 5) + ([i] if include else [])
    red = rng.choices(range(1,27), weights=record["red_probabilities"])[0]
    return sorted(whites), red


def picks(record, seed, baseline=False):
    validate_forecast(record)
    if type(seed) is not int:
        raise ValueError("Seed must be an independently generated integer")
    rng = random.Random(seed)
    tickets = []
    for _ in range(10000):
        w, b = draw_one(record, rng, baseline)
        ticket = {"whites": w, "red": b}
        if ticket not in tickets:
            tickets.append(ticket)
        if len(tickets) == 2:
            return {"model": "M0" if baseline else "M1", "seed": seed,
                    "rng": "Python random.Random", "tickets": tickets}
    raise RuntimeError("Could not generate two distinct tickets")


def score(record, actual):
    validate_forecast(record)
    if actual["draw_date"] != record["target_draw"] or not valid_ticket(actual["whites"], actual["red"]):
        raise ValueError("Invalid or mismatched result")
    wr = white_ratio(record["white_weights"], actual["whites"])
    rp = record["red_probabilities"][actual["red"]-1]
    lw, lr = math.log(wr), math.log(26*rp)
    return {"draw_date": actual["draw_date"], "forecast_sha256": record["forecast_sha256"],
            "log_loss_white": math.log(COMBINATIONS)-lw,
            "log_loss_red": math.log(26)-lr,
            "log_loss_joint": -math.log(UNIFORM)-lw-lr,
            "log_lr_white": lw, "log_lr_red": lr, "log_lr_joint": lw+lr}


def evidence(scored, candidate_index):
    if type(candidate_index) is not int or candidate_index < 1:
        raise ValueError("Candidate index must be registered before forecasts")
    dates = [x["draw_date"] for x in scored]
    if dates != sorted(set(dates)):
        raise ValueError("Duplicate or unordered scored draws")
    if any(not math.isfinite(x["log_lr_joint"]) for x in scored):
        raise ValueError("Invalid log likelihood ratio")
    alpha = 0.025 / (candidate_index*(candidate_index+1))
    limit, total, maximum, crossing = -math.log(alpha), 0.0, 0.0, None
    for row in scored:
        total += row["log_lr_joint"]
        maximum = max(maximum, total)
        if crossing is None and total >= limit:
            crossing = row["draw_date"]
    return {"draws": len(scored), "alpha": alpha, "threshold_log_e": limit,
            "log_e": total, "max_log_e": maximum, "first_crossing": crossing,
            "mean_log_gain": total/len(scored) if scored else None}
