"""M1-S1.0: a fixed, nondegenerate Bayesian shadow forecast; no tickets."""
import hashlib
from pathlib import Path
import pb_r3 as base


def forecast(rows, target):
    rows = base.history(rows, target)
    if len(rows) < 300:
        raise ValueError("Shadow requires 300 verified same-rule draws")
    config = {"kappa": 10000, "white_null_prior": 0.99,
              "candidate_name": "M1-S1.0", "selection": "fixed; no calibration"}
    record = base.fit(rows, target, config)
    record["model"] = "M1-S"
    record["shadow_version"] = "M1-S1.0"
    record["adapter_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    record["forecast_sha256"] = base.digest(
        {k: v for k, v in record.items() if k != "forecast_sha256"})
    base.validate_forecast(record)
    return record
