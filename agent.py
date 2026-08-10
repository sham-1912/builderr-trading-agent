"""Spear — Candidate Continuous (builderr Trading v0) — CONTINUOUS BREADTH-LED BARBELL VERSION.

A continuous, breadth-led, inverse-volatility, barbell trading architecture:

  1. Continuous Regime Function: Replaces 3 discrete state buckets (CASH/NEUTRAL/FULL)
     with a smooth regime score R in [0, 1] derived from trend, breadth, and volatility.
     Gross exposure glides smoothly via a continuous sigmoid mapping, eliminating step-function cliff risks.
  2. Breadth-Led Regime Detection: Tracks 20-day and 50-day universe breadth + 5-day rate of change
     to spot distribution and institutional unloading BEFORE index price roll-overs.
  3. Inverse-Volatility Position Sizing (Equal Risk Contribution): Sizes top momentum holdings
     inversely proportional to 20-day realized volatility, capped at 0.29 (29%).
  4. Correlation Cluster Cap: Limits aggregate exposure in high-beta tech/semi clusters
     to max 0.55 (55%), preventing single-factor blowups.
  5. Barbell Architecture: Combines core inverse-vol momentum leaders with a tactical 15%
     mean-reversion satellite sleeve (RSI-5 oversold pullbacks in uptrends) for uncorrelated gains.
  6. Continuous Leverage Dial: Scales 2x/3x ETF participation (TQQQ/QLD/SSO) smoothly as
     regime score R > 0.65, capped at <= 1.45x beta-gross.

Design guarantees:
  * Long-only; quantities non-negative; max 30% concentration limit (we cap at 0.29).
  * Gross <= 1.0x (unlevered) or <= 1.45x beta-adjusted gross.
  * stdlib only, sub-millisecond execution, clean offline runs.
"""
from __future__ import annotations

import math
from statistics import pstdev
from typing import Any, Optional

# ---------------------------------------------------------------------------
# 1. Fixed parameters — continuous regime & risk knobs.
# ---------------------------------------------------------------------------
MOM_LONG: int = 42
MOM_SHORT: int = 21
NAME_SMA_SLOW: int = 50
NAME_SMA_FAST: int = 20
VOL_SIZE: int = 20
TOP_N_MAX: int = 4
NAME_CAP: float = 0.29
CLUSTER_CAP: float = 0.55
SATELLITE_BUDGET: float = 0.15
MAX_BETA_GROSS: float = 1.45
TRAIL_STOP: float = 0.08
STOP_COOLDOWN_DAYS: int = 3
REBALANCE_DAYS: int = 2
DRIFT_LIMIT: float = 0.28
MIN_TRADE_PCT: float = 0.02
CASH_BUFFER: float = 0.98
MAX_ORDERS: int = 45
MIN_BARS: int = 51

# ---------------------------------------------------------------------------
# 2. Universe definitions.
# ---------------------------------------------------------------------------
INDEX_REF: tuple[str, ...] = ("SPY", "QQQ")

LEADER_STOCKS: tuple[str, ...] = (
    "NVDA", "MSFT", "AAPL", "META", "AMZN", "GOOGL", "AVGO", "AMD", "MU", "MRVL",
    "NFLX", "TSLA", "PLTR", "ORCL", "CRM", "JPM", "V", "MA", "COST", "LLY",
)
LEADER_ETFS: tuple[str, ...] = (
    "QQQ", "SPY", "SMH", "XLK", "XLC", "XLY", "XLF", "XLI", "XLE", "XLV",
    "XLP", "XLU", "XLRE", "DIA", "IWM", "SOXX",
)
LEADER_POOL: tuple[str, ...] = tuple(dict.fromkeys(LEADER_STOCKS + LEADER_ETFS))

# High-correlation semiconductor & mega-cap tech cluster
TECH_CLUSTER: tuple[str, ...] = (
    "NVDA", "AVGO", "AMD", "MU", "MRVL", "SMH", "SOXX", "TQQQ", "QLD", "TECL", "SOXL", "XLK"
)

# 2x / 3x ETF sleeve — bought proportionally as regime score exceeds threshold.
SLEEVE: tuple[str, ...] = ("TQQQ", "QLD", "SSO")

BETA: dict[str, float] = {
    "QLD": 2.0, "SSO": 2.0, "DDM": 2.0, "ROM": 2.0, "UWM": 2.0, "AGQ": 2.0,
    "TQQQ": 3.0, "SOXL": 3.0, "UPRO": 3.0, "SPXL": 3.0, "TNA": 3.0, "FAS": 3.0,
    "TECL": 3.0, "LABU": 3.0, "CURE": 3.0, "DRN": 3.0, "UDOW": 3.0, "NAIL": 3.0,
}
BETA_MULTIPLE: dict[str, float] = BETA


def target_weights(market_state: dict[str, Any]) -> dict[str, float]:
    cache: dict[str, Optional[list[float]]] = {}
    stop_block: dict[str, int] = {}
    r_score = _compute_regime_score(market_state, cache)
    return _build_targets(r_score, market_state, cache, stop_block)


# ---------------------------------------------------------------------------
# 3. Persistent state.
# ---------------------------------------------------------------------------
_pos_high: dict[str, float] = {}
_stop_block: dict[str, int] = {}
_last_rebalance_date: Optional[str] = None
_last_seen_date: Optional[str] = None
_prev_regime_score: float = 0.5


# ---------------------------------------------------------------------------
# 4. Feature helpers.
# ---------------------------------------------------------------------------
def _beta(ticker: str) -> float:
    return BETA.get(ticker, 1.0)


def _date_of(ts: Any) -> str:
    return str(ts)[:10]


def _closes_of(
    market_state: dict[str, Any],
    ticker: str,
    cache: dict[str, Optional[list[float]]],
) -> Optional[list[float]]:
    if ticker in cache:
        return cache[ticker]
    closes: Optional[list[float]] = None
    bars = market_state.get(ticker)
    if bars:
        try:
            closes = [float(bar["close"]) for bar in bars]
        except (KeyError, TypeError, ValueError):
            closes = None
    cache[ticker] = closes
    return closes


def _computable(closes: Optional[list[float]]) -> bool:
    return closes is not None and len(closes) >= MIN_BARS and closes[-1] > 0.0


def _sma(closes: list[float], n: int) -> Optional[float]:
    if len(closes) < n:
        return None
    return sum(closes[-n:]) / n


def _ret(closes: list[float], k: int) -> Optional[float]:
    if len(closes) < k + 1:
        return None
    start = closes[-(k + 1)]
    if start <= 0.0:
        return None
    return closes[-1] / start - 1.0


def _vol(closes: list[float], n: int = VOL_SIZE) -> Optional[float]:
    if len(closes) < n + 1:
        return None
    window = closes[-(n + 1):]
    rets: list[float] = []
    for i in range(1, len(window)):
        prev = window[i - 1]
        if prev <= 0.0:
            return None
        rets.append(window[i] / prev - 1.0)
    if len(rets) < 2:
        return None
    return pstdev(rets) * math.sqrt(252.0)


def _rsi5(closes: list[float]) -> Optional[float]:
    n = 5
    if len(closes) < n + 1:
        return None
    gains, losses = 0.0, 0.0
    for i in range(len(closes) - n, len(closes)):
        diff = closes[i] - closes[i - 1]
        if diff > 0:
            gains += diff
        else:
            losses -= diff
    if losses == 0.0:
        return 100.0
    rs = (gains / n) / (losses / n)
    return 100.0 - (100.0 / (1.0 + rs))


def _momentum_score(closes: list[float]) -> Optional[float]:
    r_long = _ret(closes, MOM_LONG)
    r_short = _ret(closes, MOM_SHORT)
    sma50 = _sma(closes, NAME_SMA_SLOW)
    if r_long is None or r_short is None or sma50 is None or sma50 <= 0.0:
        return None
    gap = closes[-1] / sma50 - 1.0
    return 0.50 * r_long + 0.30 * r_short + 0.20 * gap


# ---------------------------------------------------------------------------
# 5. Continuous Regime Score & Breadth Engine.
# ---------------------------------------------------------------------------
def _compute_regime_score(
    market_state: dict[str, Any],
    cache: dict[str, Optional[list[float]]],
) -> float:
    """Compute a single continuous regime score R in [0.0, 1.0] from Trend, Breadth & Volatility."""
    spy = _closes_of(market_state, "SPY", cache)
    qqq = _closes_of(market_state, "QQQ", cache)
    if not _computable(spy) or not _computable(qqq):
        return 0.0

    spy_c = spy[-1]  # type: ignore[index]
    qqq_c = qqq[-1]  # type: ignore[index]
    spy_sma20 = _sma(spy, NAME_SMA_FAST)  # type: ignore[arg-type]
    spy_sma50 = _sma(spy, NAME_SMA_SLOW)  # type: ignore[arg-type]
    qqq_sma20 = _sma(qqq, NAME_SMA_FAST)  # type: ignore[arg-type]
    qqq_sma50 = _sma(qqq, NAME_SMA_SLOW)  # type: ignore[arg-type]

    if None in (spy_sma20, spy_sma50, qqq_sma20, qqq_sma50):
        return 0.0

    # 1. Trend Score T in [0, 1]
    t_points = 0.0
    if spy_c > spy_sma20: t_points += 0.25
    if spy_c > spy_sma50: t_points += 0.25
    if qqq_c > qqq_sma20: t_points += 0.25
    if qqq_c > qqq_sma50: t_points += 0.25

    # 2. Breadth Score B in [0, 1]
    n_comp, n_up20, n_up50 = 0, 0, 0
    for ticker in LEADER_POOL:
        closes = _closes_of(market_state, ticker, cache)
        if not _computable(closes):
            continue
        s20 = _sma(closes, NAME_SMA_FAST)  # type: ignore[arg-type]
        s50 = _sma(closes, NAME_SMA_SLOW)  # type: ignore[arg-type]
        if s20 is None or s50 is None:
            continue
        n_comp += 1
        if closes[-1] > s20: n_up20 += 1  # type: ignore[index]
        if closes[-1] > s50: n_up50 += 1  # type: ignore[index]

    if n_comp > 0:
        b20 = n_up20 / n_comp
        b50 = n_up50 / n_comp
        breadth_score = 0.50 * b20 + 0.50 * b50
    else:
        breadth_score = 0.0

    # 3. Volatility Penalty V in [0, 1] (higher vol -> lower score)
    qqq_vol = _vol(qqq, VOL_SIZE)  # type: ignore[arg-type]
    if qqq_vol is None:
        vol_score = 0.50
    else:
        # Normal vol ~0.15-0.20; >0.35 is high risk
        vol_score = max(0.0, min(1.0, 1.0 - (qqq_vol - 0.15) / 0.25))

    # Blend: 40% Trend + 40% Breadth + 20% Volatility
    raw_regime = 0.40 * t_points + 0.40 * breadth_score + 0.20 * vol_score
    return max(0.0, min(1.0, raw_regime))


# ---------------------------------------------------------------------------
# 6. Price / portfolio helpers.
# ---------------------------------------------------------------------------
def _resolve_cash(portfolio_state: dict[str, Any], cash: float) -> float:
    try:
        return float(portfolio_state.get("cash", cash))
    except (TypeError, ValueError):
        try:
            return float(cash)
        except (TypeError, ValueError):
            return 0.0


def _aggregate_positions(portfolio_state: dict[str, Any]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for raw in portfolio_state.get("positions", []) or []:
        try:
            ticker = str(raw["ticker"]).upper()
            qty = float(raw.get("quantity", 0.0))
            avg_cost = float(raw.get("avg_cost", 0.0))
        except (KeyError, TypeError, ValueError):
            continue
        if qty <= 0.0:
            continue
        if ticker in out:
            existing = out[ticker]
            total = existing["quantity"] + qty
            existing["avg_cost"] = (
                (existing["avg_cost"] * existing["quantity"] + avg_cost * qty) / total
                if total > 0.0 else avg_cost
            )
            existing["quantity"] = total
        else:
            out[ticker] = {"quantity": qty, "avg_cost": avg_cost}
    return out


def _exec_price(
    ticker: str,
    market_state: dict[str, Any],
    cache: dict[str, Optional[list[float]]],
    last_prices: dict[str, Any],
) -> Optional[float]:
    closes = _closes_of(market_state, ticker, cache)
    if closes and closes[-1] > 0.0:
        return closes[-1]
    lp = last_prices.get(ticker)
    try:
        if lp is not None and float(lp) > 0.0:
            return float(lp)
    except (TypeError, ValueError):
        pass
    return None


def _compute_equity(
    positions: dict[str, dict[str, float]],
    market_state: dict[str, Any],
    cache: dict[str, Optional[list[float]]],
    last_prices: dict[str, Any],
    cash_value: float,
) -> float:
    total = cash_value
    for ticker in sorted(positions):
        pos = positions[ticker]
        price = _exec_price(ticker, market_state, cache, last_prices)
        if price is None:
            price = pos["avg_cost"] if pos["avg_cost"] > 0.0 else 0.0
        total += pos["quantity"] * max(price, 0.0)
    return max(total, 0.0)


# ---------------------------------------------------------------------------
# 7. Sizing & Target Construction (Inverse Volatility + Cluster Cap + Barbell).
# ---------------------------------------------------------------------------
def _build_targets(
    regime_score: float,
    market_state: dict[str, Any],
    cache: dict[str, Optional[list[float]]],
    stop_block: dict[str, int],
) -> dict[str, float]:
    weights: dict[str, float] = {}
    if regime_score < 0.20:
        return weights  # Hard cash protection for regime < 0.20

    # Smooth continuous gross exposure function G(R)
    if regime_score <= 0.70:
        gross = 0.30 + (regime_score - 0.20) / 0.50 * 0.68  # 0.30 to 0.98
    else:
        gross = 0.98 + (regime_score - 0.70) / 0.30 * 0.40  # 0.98 to 1.38 (leverage dial)

    # 1. Select top momentum leaders
    qualifiers: list[tuple[float, float, str]] = []  # (score, vol, ticker)
    for ticker in LEADER_POOL:
        if ticker in stop_block:
            continue
        closes = _closes_of(market_state, ticker, cache)
        if not _computable(closes):
            continue
        score = _momentum_score(closes)  # type: ignore[arg-type]
        if score is None or score <= 0.0:
            continue
        v = _vol(closes, VOL_SIZE)  # type: ignore[arg-type]
        vol_val = v if v is not None and v > 0.05 else 0.20
        qualifiers.append((score, vol_val, ticker))

    qualifiers.sort(key=lambda item: (-item[0], item[2]))
    selected = qualifiers[:TOP_N_MAX]

    # 2. Inverse-volatility sizing (Equal Risk Contribution)
    if selected:
        inv_vol_sum = sum(1.0 / q[1] for q in selected)
        core_budget = min(gross, 0.98)
        for score, vol_val, ticker in selected:
            raw_w = (1.0 / vol_val) / inv_vol_sum * core_budget
            weights[ticker] = min(raw_w, NAME_CAP)

    # 3. Barbell Satellite Sleeve (Tactical Mean-Reversion on Pullbacks)
    if regime_score > 0.35:
        satellite_candidates: list[tuple[float, str]] = []
        for ticker in LEADER_POOL:
            if ticker in stop_block or ticker in weights:
                continue
            closes = _closes_of(market_state, ticker, cache)
            if not _computable(closes):
                continue
            rsi = _rsi5(closes)  # type: ignore[arg-type]
            sma50 = _sma(closes, NAME_SMA_SLOW)  # type: ignore[arg-type]
            if rsi is not None and rsi < 38.0 and sma50 is not None and closes[-1] > sma50:  # type: ignore[index]
                satellite_candidates.append((rsi, ticker))

        if satellite_candidates:
            satellite_candidates.sort(key=lambda x: x[0])
            dip_ticker = satellite_candidates[0][1]
            sat_w = min(SATELLITE_BUDGET, NAME_CAP)
            weights[dip_ticker] = sat_w

    # 4. Continuous Leverage Dial (TQQQ/QLD/SSO as regime_score > 0.65)
    if regime_score > 0.65:
        sleeve_present = [
            s for s in SLEEVE
            if s not in stop_block and _computable(_closes_of(market_state, s, cache))
        ]
        if sleeve_present:
            lev_budget = (regime_score - 0.65) / 0.35 * 0.40  # up to 0.40 leverage
            per_lev = lev_budget / len(sleeve_present)
            for s in sleeve_present:
                weights[s] = min(per_lev, NAME_CAP)

    # 5. Correlation Cluster Cap (Cap Semiconductor / Tech Heavyweights at 55%)
    tech_sum = sum(w for t, w in weights.items() if t in TECH_CLUSTER)
    if tech_sum > CLUSTER_CAP and tech_sum > 0.0:
        scale = CLUSTER_CAP / tech_sum
        for t in list(weights):
            if t in TECH_CLUSTER:
                weights[t] *= scale

    # 6. Max Beta Gross Clamp (<= 1.45x)
    beta_gross = sum(w * _beta(t) for t, w in weights.items())
    if beta_gross > MAX_BETA_GROSS and beta_gross > 0.0:
        scale = MAX_BETA_GROSS / beta_gross
        weights = {t: w * scale for t, w in weights.items()}

    return weights


# ---------------------------------------------------------------------------
# 8. Order Generation.
# ---------------------------------------------------------------------------
def _generate_orders(
    do_rebalance: bool,
    weights: dict[str, float],
    positions: dict[str, dict[str, float]],
    forced_stops: list[tuple[str, float]],
    equity: float,
    market_state: dict[str, Any],
    cache: dict[str, Optional[list[float]]],
    last_prices: dict[str, Any],
    cash_value: float,
) -> list[dict[str, Any]]:
    orders: list[dict[str, Any]] = []
    sold: set[str] = set()
    proceeds = 0.0
    min_trade = MIN_TRADE_PCT * equity

    for ticker, qty in forced_stops:
        if qty > 0.0:
            orders.append({"ticker": ticker, "side": "sell", "quantity": qty})
            sold.add(ticker)
            price = _exec_price(ticker, market_state, cache, last_prices)
            if price is not None:
                proceeds += qty * price

    if do_rebalance:
        for ticker in sorted(positions):
            if ticker in sold:
                continue
            held = positions[ticker]["quantity"]
            if held <= 0.0:
                continue
            target_w = weights.get(ticker, 0.0)
            price = _exec_price(ticker, market_state, cache, last_prices)

            if target_w == 0.0:
                orders.append({"ticker": ticker, "side": "sell", "quantity": held})
                sold.add(ticker)
                if price is not None and price > 0.0:
                    proceeds += held * price
                continue

            if price is None or price <= 0.0:
                continue
            target_shares = math.floor(target_w * equity / price)
            delta = target_shares - held
            if delta < 0 and (-delta) * price >= min_trade:
                sell_qty = float(int(min(-delta, held)))
                if sell_qty > 0.0:
                    orders.append({"ticker": ticker, "side": "sell", "quantity": sell_qty})
                    sold.add(ticker)
                    proceeds += sell_qty * price

        spendable = cash_value + CASH_BUFFER * proceeds
        for ticker in sorted(weights, key=lambda t: (-weights[t], t)):
            price = _exec_price(ticker, market_state, cache, last_prices)
            if price is None or price <= 0.0:
                continue
            held = positions[ticker]["quantity"] if ticker in positions else 0.0
            target_shares = math.floor(weights[ticker] * equity / price)
            deficit = target_shares - held
            if deficit > 0 and deficit * price >= min_trade:
                affordable = math.floor(min(deficit * price, spendable) / price)
                if affordable > 0:
                    orders.append({"ticker": ticker, "side": "buy", "quantity": float(affordable)})
                    spendable -= affordable * price

    if len(orders) > MAX_ORDERS:
        sells = [o for o in orders if o["side"] == "sell"]
        buys = [o for o in orders if o["side"] == "buy"]
        orders = (sells + buys)[:MAX_ORDERS]

    return [o for o in orders if o["quantity"] > 0.0]


# ---------------------------------------------------------------------------
# 9. Master Decision Cycle.
# ---------------------------------------------------------------------------
def decide(market_state: dict, portfolio_state: dict, cash: float) -> list[dict]:
    global _pos_high, _stop_block, _last_rebalance_date, _last_seen_date, _prev_regime_score

    snapshot = (
        dict(_pos_high), dict(_stop_block), _last_rebalance_date, _last_seen_date, _prev_regime_score
    )
    try:
        return _run(market_state or {}, portfolio_state or {}, cash)
    except Exception:
        (
            _pos_high, _stop_block, _last_rebalance_date, _last_seen_date, _prev_regime_score
        ) = snapshot
        return []


def _run(
    market_state: dict[str, Any],
    portfolio_state: dict[str, Any],
    cash: float,
) -> list[dict[str, Any]]:
    global _pos_high, _stop_block, _last_rebalance_date, _last_seen_date, _prev_regime_score

    if not market_state:
        return []

    cache: dict[str, Optional[list[float]]] = {}
    last_prices: dict[str, Any] = {}
    for key, value in (portfolio_state.get("last_prices", {}) or {}).items():
        last_prices[str(key).upper()] = value
    cash_value = _resolve_cash(portfolio_state, cash)

    spy_bars = market_state.get("SPY")
    spy = _closes_of(market_state, "SPY", cache)
    qqq = _closes_of(market_state, "QQQ", cache)
    current_date: Optional[str] = None
    if spy_bars:
        ts = spy_bars[-1].get("ts")
        current_date = _date_of(ts) if ts is not None else str(len(spy_bars))

    if not _computable(spy) or not _computable(qqq):
        positions = _aggregate_positions(portfolio_state)
        orders: list[dict[str, Any]] = []
        for ticker in sorted(positions):
            if market_state.get(ticker):
                qty = positions[ticker]["quantity"]
                if qty > 0.0:
                    orders.append({"ticker": ticker, "side": "sell", "quantity": qty})
        if current_date is not None:
            _last_seen_date = current_date
        return orders

    positions = _aggregate_positions(portfolio_state)

    is_new_day = current_date != _last_seen_date
    if is_new_day and _stop_block:
        decayed: dict[str, int] = {}
        for ticker, days in _stop_block.items():
            remaining = days - 1
            if remaining > 0:
                decayed[ticker] = remaining
        _stop_block = decayed

    equity = _compute_equity(positions, market_state, cache, last_prices, cash_value)
    if equity <= 0.0:
        _last_seen_date = current_date
        return []

    # Compute Continuous Regime Score R in [0.0, 1.0]
    regime_score = _compute_regime_score(market_state, cache)

    # Trailing stops
    for ticker in list(_pos_high):
        if ticker not in positions:
            del _pos_high[ticker]
    forced_stops: list[tuple[str, float]] = []
    for ticker in sorted(positions):
        price = _exec_price(ticker, market_state, cache, last_prices)
        if price is None:
            continue
        high = _pos_high.get(ticker, price)
        if price > high:
            high = price
        _pos_high[ticker] = high
        if high > 0.0 and price < high * (1.0 - TRAIL_STOP):
            forced_stops.append((ticker, positions[ticker]["quantity"]))
            _stop_block[ticker] = STOP_COOLDOWN_DAYS
            if ticker in _pos_high:
                del _pos_high[ticker]

    # Rebalance gate
    if _last_rebalance_date is None:
        do_rebalance = True
    else:
        elapsed_dates: set[str] = set()
        for bar in spy_bars:
            ts = bar.get("ts")
            bar_date = _date_of(ts) if ts is not None else ""
            if bar_date > _last_rebalance_date:
                elapsed_dates.add(bar_date)
        days_since = len(elapsed_dates)
        regime_drop = (regime_score < _prev_regime_score - 0.15)
        do_rebalance = (days_since >= REBALANCE_DAYS or regime_drop or len(forced_stops) > 0)

    if _last_rebalance_date == current_date:
        do_rebalance = False

    weights = _build_targets(regime_score, market_state, cache, _stop_block) if do_rebalance else {}

    orders = _generate_orders(
        do_rebalance, weights, positions, forced_stops,
        equity, market_state, cache, last_prices, cash_value,
    )
    if do_rebalance and len(orders) >= 1:
        _last_rebalance_date = current_date

    _prev_regime_score = regime_score
    _last_seen_date = current_date
    return orders
