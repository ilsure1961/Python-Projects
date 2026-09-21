"""
Option Strategy Valuation - Main Entry Point
==============================================
Interactive command-line tool for pricing options and evaluating strategies.
Outputs a dated CSV file and triggers the HTML dashboard generator.

Usage:
    python option_valuation.py

The script prompts for inputs step-by-step, then:
 1. Prices the single option (BSM or CRR)
 2. Runs put-call parity check
 3. Evaluates a chosen multi-leg strategy (or a custom leg list)
 4. Exports results to a CSV file
 5. Generates a self-contained HTML dashboard

Output files (in same directory as this script):
    option_results_YYYY-MM-DD.csv
    option_dashboard_YYYY-MM-DD.html

Dependencies:
    pip install numpy scipy pandas
"""

import os
import sys
import csv
import datetime

# ------------------------------------------------------------------
# Path setup - allow running from any working directory
# ------------------------------------------------------------------
_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)

from pricing.black_scholes import price_european, put_call_parity_check, sensitivity_grid
from pricing.binomial_tree import price_american
from strategies.leg import Leg
from strategies.strategy_builder import Strategy
from strategies.library import STRATEGY_MENU, build_strategy
from strategies.payoff import payoff_curves, build_S_range

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------
TODAY = datetime.date.today().isoformat()
_OUT_DIR = _DIR
_CSV_PATH = os.path.join(_OUT_DIR, f"option_results_{TODAY}.csv")
_DASH_PATH = os.path.join(_OUT_DIR, f"option_dashboard_{TODAY}.html")

ASSET_CLASSES = {
    "E": "equity",
    "B": "bond",
    "I": "index",
    "FX": "fx",
    "COM": "commodity",
}

# ------------------------------------------------------------------
# Input helpers
# ------------------------------------------------------------------

def _prompt(msg: str, default=None, cast=str):
    """Prompt user for input with an optional default."""
    suffix = f" [{default}]" if default is not None else ""
    while True:
        raw = input(f"{msg}{suffix}: ").strip()
        if raw == "" and default is not None:
            return cast(default)
        try:
            return cast(raw)
        except (ValueError, TypeError):
            print(f"    Invalid input. Expected {cast.__name__}.")


def _prompt_choice(msg: str, choices: list, default=None):
    """Prompt for a value that must be in choices (case-insensitive)."""
    choices_upper = [str(c).upper() for c in choices]
    suffix = f" [{default}]" if default is not None else ""
    while True:
        raw = input(f"{msg} ({'/'.join(str(c) for c in choices)}){suffix}: ").strip().upper()
        if raw == "" and default is not None:
            return default
        if raw in choices_upper:
            return raw
        print(f"    Please enter one of: {', '.join(str(c) for c in choices)}")


def _section(title: str):
    width = 60
    print()
    print("=" * width)
    print(f" {title}")
    print("=" * width)


def _banner():
    print()
    print("┌─────────────────────────────────────────────────────────┐")
    print("│    OPTION STRATEGY VALUATION TOOL  (Stages 1-2)         │")
    print("│    Black-Scholes-Merton + CRR Binomial Tree             │")
    print("└─────────────────────────────────────────────────────────┘")
    print()


# ------------------------------------------------------------------
# Stage 1: single-option inputs
# ------------------------------------------------------------------

def collect_inputs() -> dict:
    _section("STAGE 1 - Single Option Inputs")

    print("\n Asset class codes: E=Equity B=Bond I=Index FX=FX COM=Commodity")
    asset_raw = _prompt_choice("Asset class", list(ASSET_CLASSES.keys()), default="E")
    asset_class = ASSET_CLASSES[asset_raw]

    S = _prompt("Underlying spot price (S)", default=50.0, cast=float)
    K = _prompt("Strike price (K)", default=50.0, cast=float)

    print("\n Time to expiry: enter as years (e.g. 1.0) or as days (e.g. 252d)")
    T_raw = _prompt("Time to expiry (T)", default="1.0")
    if T_raw.lower().endswith("d"):
        T = float(T_raw[:-1]) / 365.0
    else:
        T = float(T_raw)

    r = _prompt("Risk-free rate (e.g. 0.05 for 5%)", default=0.02, cast=float)
    sigma = _prompt("Volatility (e.g. 0.30 for 30%)", default=0.30, cast=float)
    q = _prompt("Dividend yield q (0 if none)", default=0.0, cast=float)

    opt_type = _prompt_choice("Option type", ["call", "put"], default="call")
    style = _prompt_choice("Exercise style", ["european", "american"], default="european")

    return dict(
        S=S, K=K, T=T, r=r, sigma=sigma, q=q,
        option_type=opt_type.lower(), exercise_style=style.lower(),
        asset_class=asset_class
    )


# ------------------------------------------------------------------
# Stage 1: pricing + Greeks
# ------------------------------------------------------------------

def price_single_option(inputs: dict) -> dict:
    _section("STAGE 1 - Theoretical Price + Greeks")

    S, K, T, r, sigma, q = (inputs[k] for k in ("S", "K", "T", "r", "sigma", "q"))
    opt = inputs["option_type"]
    style = inputs["exercise_style"]

    if style == "european":
        result = price_european(S, K, T, r, sigma, q, opt)
        # Also compute the opposite type for put-call parity
        opp_type = "put" if opt == "call" else "call"
        opp_result = price_european(S, K, T, r, sigma, q, opp_type)
    else:
        result = price_american(S, K, T, r, sigma, q, opt)
        opp_result = None

    print(f"\n Model           : {result['model']}")
    print(f" Option type     : {opt.upper()} | Style: {style.upper()}")
    print(f" S={S}  K={K}  T={T:.4f}yr  r={r:.4f}  σ={sigma:.4f}  q={q:.4f}")
    print()
    print(f" {'Theoretical Price':20s}: {result['price']:>10.4f}")
    print()
    print(f" Greeks:")
    print(f"   {'Delta':12s}: {result['delta']:>10.6f}  (dV/dS per $1 move)")
    print(f"   {'Gamma':12s}: {result['gamma']:>10.6f}  (dΔ/dS per $1 move)")
    print(f"   {'Vega':12s}: {result['vega']:>10.6f}  (dV per 1% move)")
    print(f"   {'Theta':12s}: {result['theta']:>10.6f}  (dV per calendar day)")
    print(f"   {'Rho':12s}: {result['rho']:>10.6f}  (dV per 1% rate move)")

    # Put-call parity check
    if opp_result is not None:
        call_p = result["price"] if opt == "call" else opp_result["price"]
        put_p = opp_result["price"] if opt == "call" else result["price"]
        pcp = put_call_parity_check(call_p, put_p, S, K, T, r, q)
        status = "PASS" if pcp["passed"] else "FAIL"
        
print()
        print(f" Put-Call Parity Check: {status}")
        print(f"   C - P = {pcp['lhs']:.6f} | S·e^(-qT) - K·e^(-rT) = {pcp['rhs']:.6f} | diff = {pcp['diff']:.2e}")
    else:
        pcp = None

    result["parity_check"] = pcp
    result.update(inputs)
    return result


# ------------------------------------------------------------------
# Stage 2: strategy selection + evaluation
# ------------------------------------------------------------------

def collect_strategy_inputs(base_inputs: dict) -> dict:
    _section("STAGE 2 - Strategy Evaluator")

    print("\n Available strategies:")
    for k, v in STRATEGY_MENU.items():
        print(f"   {k:2d}. {v}")

    strat_id = _prompt("Select strategy number (or 0 to skip)", default=9, cast=int)
    if strat_id == 0:
        return {}

    S = base_inputs["S"]
    r = base_inputs["r"]
    sigma = base_inputs["sigma"]
    T = base_inputs["T"]
    q = base_inputs["q"]
    K = base_inputs["K"]
    style = base_inputs["exercise_style"]

    extra = {}

    # Collect extra strike/expiry inputs depending on strategy
    if strat_id in (5, 6, 7, 8, 10):
        print(f"\n This strategy requires two strikes.")
        extra["K_lo"] = _prompt("Lower strike K_lo", default=round(K * 0.95, 2), cast=float)
        extra["K_hi"] = _prompt("Upper strike K_hi", default=round(K * 1.05, 2), cast=float)
    elif strat_id == 11:
        print(f"\n Iron Condor requires 4 strikes (K1 < K2 < K3 < K4).")
        extra["K1"] = _prompt("K1 (lower long put)", default=round(K * 0.85, 2), cast=float)
        extra["K2"] = _prompt("K2 (short put)", default=round(K * 0.95, 2), cast=float)
        extra["K3"] = _prompt("K3 (short call)", default=round(K * 1.05, 2), cast=float)
        extra["K4"] = _prompt("K4 (upper long call)", default=round(K * 1.15, 2), cast=float)
    elif strat_id == 12:
        print(f"\n Butterfly requires 3 strikes (K_lo < K_mid < K_hi).")
        extra["K_lo"] = _prompt("K_lo", default=round(K * 0.95, 2), cast=float)
        extra["K_mid"] = _prompt("K_mid (peak)", default=K, cast=float)
        extra["K_hi"] = _prompt("K_hi", default=round(K * 1.05, 2), cast=float)
    elif strat_id == 13:
        print(f"\n Calendar spread requires two expiries.")
        extra["T_near"] = _prompt("Near-term expiry T_near (yrs)", default=round(T * 0.5, 4), cast=float)
        extra["T_far"] = _prompt("Far-term expiry T_far (yrs)", default=T, cast=float)

    return dict(strategy_id=strat_id, K=K, style=style, **extra)


def run_strategy(base_inputs: dict, strategy_kwargs: dict) -> dict:
    """Build, price, and summarise the selected strategy."""
    if not strategy_kwargs:
        return {}

    strat = build_strategy(
        S=base_inputs["S"],
        r=base_inputs["r"],
        sigma=base_inputs["sigma"],
        T=base_inputs["T"],
        q=base_inputs["q"],
        **strategy_kwargs,
    )
    strat.price()
    summary = strat.summary()

    print(f"\n Strategy     : {summary['strategy']}")
    print(f" Description  : {summary['description'].split(chr(10))[0]}")
    print()
    print(f" Net Premium  : {summary['net_premium']:>10.4f}  ({summary['debit_or_credit']})")
    print(f" Max Profit   : {str(summary['max_profit']):>10}")
    print(f" Max Loss     : {str(summary['max_loss']):>10}")
    be1 = summary.get("breakeven_1")
    be2 = summary.get("breakeven_2")
    if be1:
        print(f" Breakeven 1 : {be1:.4f}", end="")
        if be2:
            print(f" | Breakeven 2: {be2:.4f}", end="")
        print()

    print()
    print(" Aggregate Greeks (position-weighted):")
    for g in ("net_delta", "net_gamma", "net_vega", "net_theta", "net_rho"):
        print(f"   {g.replace('net_', '').capitalize():8s}: {summary[g]:>10.6f}")

    return {"strategy_obj": strat, "summary": summary}


# ------------------------------------------------------------------
# CSV export
# ------------------------------------------------------------------

def export_csv(single_result: dict, strategy_data: dict, path: str):
    """Write all results to a CSV file."""
    rows = []

    # Single option row
    row = {
        "run_date": TODAY,
        "S": single_result.get("S"),
        "K": single_result.get("K"),
        "T": single_result.get("T"),
        "r": single_result.get("r"),
        "sigma": single_result.get("sigma"),
        "q": single_result.get("q"),
        "option_type": single_result.get("option_type"),
        "exercise_style": single_result.get("exercise_style"),
        "asset_class": single_result.get("asset_class"),
        "model": single_result.get("model"),
        "price": single_result.get("price"),
        "delta": single_result.get("delta"),
        "gamma": single_result.get("gamma"),
        "vega": single_result.get("vega"),
        "theta": single_result.get("theta"),
        "rho": single_result.get("rho"),
        "strategy": "Single Option",
        "quantity": 1,
        "net_premium": single_result.get("price"),
        "max_profit": "",
        "max_loss": "",
        "breakeven_1": "",
        "breakeven_2": "",
    }
    rows.append(row)

    # Strategy legs
    if strategy_data:
        strat = strategy_data.get("strategy_obj")
        summary = strategy_data.get("summary", {})
        if strat:
            for leg_dict in strat.legs_to_dicts():
                r2 = {
                    "run_date": TODAY,
                    "S": leg_dict.get("S"),
                    "K": leg_dict.get("K"),
                    "T": leg_dict.get("T"),
                    "r": leg_dict.get("r"),
                    "sigma": leg_dict.get("sigma"),
                    "q": leg_dict.get("q"),
                    "option_type": leg_dict.get("option_type"),
                    "exercise_style": leg_dict.get("exercise_style"),
                    "asset_class": leg_dict.get("asset_class"),
                    "model": leg_dict.get("model"),
                    "price": leg_dict.get("premium"),
                    "delta": leg_dict.get("position_delta"),
                    "gamma": leg_dict.get("position_gamma"),
                    "vega": leg_dict.get("position_vega"),
                    "theta": leg_dict.get("position_theta"),
                    "rho": leg_dict.get("position_rho"),
                    "strategy": leg_dict.get("strategy"),
                    "quantity": leg_dict.get("quantity"),
                    "net_premium": leg_dict.get("position_cost"),
                    "max_profit": summary.get("max_profit", ""),
"max_loss": summary.get("max_loss", ""),
                    "breakeven_1": summary.get("breakeven_1", ""),
                    "breakeven_2": summary.get("breakeven_2", ""),
                }
                rows.append(r2)

    # BSM sensitivity grid (mirror Excel template)
    grid = sensitivity_grid(
        K=single_result.get("K"),
        T=single_result.get("T"),
        r=single_result.get("r"),
        sigma=single_result.get("sigma"),
        q=single_result.get("q", 0.0),
    )
    for g in grid:
        rows.append({
            "run_date": TODAY,
            "S": g["S"],
            "K": single_result.get("K"),
            "T": single_result.get("T"),
            "r": single_result.get("r"),
            "sigma": single_result.get("sigma"),
            "q": single_result.get("q", 0.0),
            "option_type": "call",
            "exercise_style": "european",
            "asset_class": single_result.get("asset_class", "equity"),
            "model": "BSM",
            "price": g["call_price"],
            "delta": g["nd1"],
            "gamma": "",
            "vega": "",
            "theta": "",
            "rho": "",
            "strategy": "BSM Sensitivity Grid (Call)",
            "quantity": "",
            "net_premium": "",
            "max_profit": "",
            "max_loss": "",
            "breakeven_1": "",
            "breakeven_2": "",
        })
        rows.append({
            "run_date": TODAY,
            "S": g["S"],
            "K": single_result.get("K"),
            "T": single_result.get("T"),
            "r": single_result.get("r"),
            "sigma": single_result.get("sigma"),
            "q": single_result.get("q", 0.0),
            "option_type": "put",
            "exercise_style": "european",
            "asset_class": single_result.get("asset_class", "equity"),
            "model": "BSM",
            "price": g["put_price"],
            "delta": "",
            "gamma": "",
            "vega": "",
            "theta": "",
            "rho": "",
            "strategy": "BSM Sensitivity Grid (Put)",
            "quantity": "",
            "net_premium": "",
            "max_profit": "",
            "max_loss": "",
            "breakeven_1": "",
            "breakeven_2": "",
        })

    fieldnames = [
        "run_date", "S", "K", "T", "r", "sigma", "q",
        "option_type", "exercise_style", "asset_class", "model",
        "price", "delta", "gamma", "vega", "theta", "rho",
        "strategy", "quantity", "net_premium",
        "max_profit", "max_loss", "breakeven_1", "breakeven_2",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n CSV exported -> {path}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    _banner()

    # Stage 1: collect inputs and price single option
    inputs = collect_inputs()
    result = price_single_option(inputs)

    # Stage 2: strategy evaluation
    strat_kwargs = collect_strategy_inputs(inputs)
    strat_data = run_strategy(inputs, strat_kwargs) if strat_kwargs else {}

    # Export CSV
    _section("OUTPUTS")
    export_csv(result, strat_data, _CSV_PATH)

    # Generate HTML dashboard
    try:
        import option_dashboard as od
        od.generate_dashboard(result, strat_data, _DASH_PATH)
        print(f"  HTML dashboard -> {_DASH_PATH}")
    except Exception as e:
        print(f"  Dashboard generation failed: {e}")

    print()
    print(" Done.")
    print()


if __name__ == "__main__":
    main()