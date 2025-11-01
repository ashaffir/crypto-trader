from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple
import collections

import numpy as np


def _safe(val: Optional[float], default: float = 0.0) -> float:
    try:
        if val is None:
            return float(default)
        v = float(val)
        if math.isnan(v) or math.isinf(v):
            return float(default)
        return v
    except Exception:
        return float(default)


@dataclass
class _SymbolState:
    # Online EWMA state
    mean_z: np.ndarray
    cov_z: np.ndarray
    c_vec: np.ndarray
    sigma_r2: float
    updates: int = 0
    # Queues to align z at t with realized r at t+H
    pending: Deque[Tuple[int, np.ndarray, float]] = field(
        default_factory=collections.deque
    )  # (ts_ms, centered_z, p_t)
    px_history: Deque[Tuple[int, float]] = field(default_factory=collections.deque)


class DeterministicSignalEngine:
    """Deterministic linear-Gaussian signal engine.

    Implements the formulation in the provided slides using online EWMA updates.

    Notes:
    - Input is a per-symbol summary dict from FeatureEngine.summarize_window
    - The engine maintains its own price history to compute r_{t+H}
    - Direction, confidence and leverage are returned when a trading edge exists
    """

    def __init__(
        self,
        symbols: List[str],
        *,
        alpha: float = 0.02,
        lam: float = 1e-3,
        horizon_s: int = 45,
        risk_aversion_gamma: float = 1.0,  # gamma=1 => Kelly; >1 more conservative
        s_cap_bps: float = 25.0,  # cap to map to liquidity penalty
        k_cost_mult: float = 1.5,  # theta = k * cost_bps
    ) -> None:
        self.alpha = float(alpha)
        self.lam = float(lam)
        self.horizon_ms = int(horizon_s) * 1000
        self.gamma = float(risk_aversion_gamma)
        self.s_cap_bps = float(s_cap_bps)
        self.k_cost_mult = float(k_cost_mult)
        # Initialize per-symbol state lazily when first seen (dimension can vary)
        self._state: Dict[str, _SymbolState] = {}

    # ---------------------- Feature mapping ----------------------
    @staticmethod
    def _z_from_summary(summary: Dict[str, object]) -> Tuple[np.ndarray, List[str]]:
        """Map summary dict to deterministic feature vector z_t.

        All features are dimensionless or scaled to reasonable ranges.
        """
        mid = summary.get("mid") if isinstance(summary, dict) else None
        spread = summary.get("spread_bps") if isinstance(summary, dict) else None
        ob = summary.get("ob_imbalance") if isinstance(summary, dict) else None

        # Mid slope normalized by mean mid (approx. pct per sec)
        mid_mean = _safe(mid.get("mean")) if isinstance(mid, dict) else None
        mid_slope = _safe(mid.get("slope")) if isinstance(mid, dict) else None
        mid_slope_rel = (
            (mid_slope / max(1e-12, mid_mean)) if (mid_mean and mid_slope) else 0.0
        )

        spread_mean_bps = _safe(spread.get("mean")) if isinstance(spread, dict) else 0.0
        imb_mean = _safe(ob.get("mean")) if isinstance(ob, dict) else 0.0

        ofp = _safe(summary.get("orderflow_pressure"))
        dskew = _safe(summary.get("depth_skew"))
        cancel_int = _safe(summary.get("cancel_intensity"))  # [0,1]
        trade_aggr = _safe(summary.get("trade_aggression"))  # ~[0,1]
        oi_delta = _safe(summary.get("oi_delta"))  # [-1,1]
        liq_burst = _safe(summary.get("liquidation_burst"))  # >=0

        # Compact vector (order is fixed and exported for tests)
        names = [
            "mid_slope_rel",
            "imbalance_mean",
            "orderflow_pressure",
            "depth_skew",
            "cancel_intensity",
            "trade_aggression",
            "oi_delta",
            "liquidation_burst",
            "spread_bps_mean_scaled",
        ]
        # scale spread to percent units
        z = np.array(
            [
                mid_slope_rel,
                imb_mean,
                ofp,
                dskew,
                cancel_int,
                trade_aggr,
                oi_delta,
                liq_burst,
                spread_mean_bps / 1e2,  # 100 bps => 1.0
            ],
            dtype=float,
        )
        return z, names

    # ---------------------- Public API ----------------------
    def update_and_score(
        self,
        *,
        symbol: str,
        summary: Dict[str, object],
        ts_ms: int,
        last_mid: Optional[float],
        fee_rate_bps_roundtrip: float,
    ) -> Dict[str, object]:
        """Update EWMA state for `symbol` and return decision dict.

        The returned dict always contains fields: direction ("buy"|"sell"|"none"),
        leverage (int), confidence (0..1), score, prob, actionable (bool).
        """
        z, _ = self._z_from_summary(summary)

        # Lazily initialize state with correct dimension
        st = self._state.get(symbol)
        if st is None:
            m = int(z.shape[0])
            st = _SymbolState(
                mean_z=np.zeros((m,), dtype=float),
                cov_z=np.zeros((m, m), dtype=float),
                c_vec=np.zeros((m,), dtype=float),
                sigma_r2=1e-8,  # small to avoid div/0
            )
            self._state[symbol] = st

        a = self.alpha

        # --- Update mean and covariance of z (EWMA) ---
        st.mean_z = (1.0 - a) * st.mean_z + a * z
        centered = z - st.mean_z
        # rank-1 update for covariance
        st.cov_z = (1.0 - a) * st.cov_z + a * np.outer(centered, centered)

        # Maintain price history (ms) for realized returns
        if last_mid is not None and last_mid > 0:
            st.px_history.append((int(ts_ms), float(last_mid)))
            # Keep modest history (e.g., 10 horizons)
            cutoff = ts_ms - (self.horizon_ms * 10)
            while st.px_history and st.px_history[0][0] < cutoff:
                st.px_history.popleft()

        # Schedule centered z for c_vec update when r_{t+H} becomes available
        if last_mid is not None and last_mid > 0:
            st.pending.append((int(ts_ms), centered.astype(float), float(last_mid)))
            while st.pending and (ts_ms - st.pending[0][0]) >= self.horizon_ms:
                t0, centered_old, p0 = st.pending.popleft()
                # Find price at t0 + H
                tH = t0 + self.horizon_ms
                pH = None
                # Walk from left until >= tH
                for tpx, p in st.px_history:
                    if tpx >= tH:
                        pH = p
                        break
                if pH is None or p0 is None or p0 <= 0:
                    continue
                r = (pH - p0) / p0  # fraction, not bps
                st.c_vec = (1.0 - a) * st.c_vec + a * (centered_old * r)
                st.sigma_r2 = float((1.0 - a) * st.sigma_r2 + a * (r * r))
                st.updates += 1

        # --- Compute weights (ridge) ---
        m = st.mean_z.shape[0]
        regI = self.lam * np.eye(m)
        try:
            # Solve (Σ + λI) w = c for stability
            w = np.linalg.solve(st.cov_z + regI, st.c_vec)
        except np.linalg.LinAlgError:
            w = np.zeros((m,), dtype=float)

        # Standardize current features using diag of Σ
        diag = np.clip(np.diag(st.cov_z), 1e-10, None)
        D_inv_sqrt = 1.0 / np.sqrt(diag)
        z_tilde = centered * D_inv_sqrt

        s = float(np.dot(w, z_tilde))

        # Predictive variance of score
        sigma_s2 = float(np.dot(w, st.cov_z @ w) + self.lam * float(np.dot(w, w)))
        sigma_s2 = max(sigma_s2, 1e-12)

        # Probit approx for class probability
        zscore = s / math.sqrt(sigma_s2)
        p = 0.5 * (1.0 + math.erf(zscore / math.sqrt(2.0)))  # Φ(z)
        confidence = max(0.0, min(1.0, 2.0 * abs(p - 0.5)))

        # Direction with threshold theta. Use cost-aware threshold.
        # cost_bps is round-trip fees + expected slippage (use spread mean from summary)
        spread_stats = summary.get("spread_bps") if isinstance(summary, dict) else None
        spread_mean_bps = (
            _safe(spread_stats.get("mean")) if isinstance(spread_stats, dict) else 0.0
        )
        cost_bps = float(fee_rate_bps_roundtrip + spread_mean_bps)
        theta = self.k_cost_mult * (cost_bps / 1e4)  # convert to fraction

        direction: Optional[str]
        if s > theta:
            direction = "buy"
        elif s < -theta:
            direction = "sell"
        else:
            direction = None

        # Leverage using Kelly-style fraction
        # Forecast return rhat = s (already a standardized score; treat as fraction)
        rhat = s
        # Realized variance proxy over horizon
        var_r = max(st.sigma_r2, 1e-10)
        rhat_net = rhat - (cost_bps / 1e4)
        f = max(0.0, rhat_net / (self.gamma * var_r))

        # Liquidity penalty based on spread and cancel intensity
        cancel_int = _safe(summary.get("cancel_intensity"))
        Pt = min(1.0, (self.s_cap_bps / max(1e-6, spread_mean_bps))) * (
            1.0 - 0.5 * cancel_int
        )
        Pt = max(0.0, min(1.0, Pt))

        lev = int(np.clip(round(1 + 9 * min(1.0, f) * Pt), 1, 10))

        warmup = st.updates == 0
        if direction is None:
            return {
                "direction": "none",
                "leverage": 1,
                "confidence": float(confidence),
                "score": float(s),
                "prob": float(p),
                "actionable": False,
                "theta": float(theta),
                "cost_bps": float(cost_bps),
                "warmup": bool(warmup),
            }

        return {
            "direction": direction,
            "leverage": max(1, int(lev)),
            "confidence": float(confidence),
            "score": float(s),
            "prob": float(p),
            "actionable": True,
            "theta": float(theta),
            "cost_bps": float(cost_bps),
            "warmup": bool(warmup),
        }


__all__ = ["DeterministicSignalEngine"]
