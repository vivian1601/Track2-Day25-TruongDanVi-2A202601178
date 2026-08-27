"""M2 — Inference Cost Levers: $/1M-token, batch x cache x cascade (deck §7).

Run: python missions/m2_inference_levers.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from missions._common import load_csv, num
from finops import pricing, sustainability

# $/1M tokens (input, output) — illustrative 2026.
MODEL_PRICES = {"small": (0.20, 0.40), "large": (3.00, 15.00)}
REASONING_TRAFFIC_CAP = 0.05
REASONING_OUTPUT_MULTIPLIER = 6


def _traffic_metrics(requests: list[dict]) -> dict:
    """Aggregate cost and energy with token-normalized comparison metrics."""
    tokens = sum(r["input_tokens"] + r["output_tokens"] for r in requests)
    cost = sum(r["cost"] for r in requests)
    energy = sum(r["energy_wh"] for r in requests)
    return {
        "requests": len(requests),
        "tokens": tokens,
        "cost": round(cost, 4),
        "energy_wh": round(energy, 1),
        "cost_per_m_tokens": round(pricing.dollars_per_million(cost, tokens), 3),
        "energy_wh_per_m_tokens": round(energy / tokens * 1e6, 1) if tokens else 0.0,
    }


def run(verbose: bool = True) -> dict:
    rows = load_csv("token_usage.csv")
    base_cost = opt_cost = 0.0
    total_tokens = 0
    request_details = []
    for r in rows:
        inp, out = int(num(r["input_tokens"])), int(num(r["output_tokens"]))
        cached = int(num(r["cached_input_tokens"]))
        is_batch = bool(int(num(r["is_batch"])))
        total_tokens += inp + out
        # BASELINE: naive deployment — everything on the large model, no cache, no batch
        lin, lout = MODEL_PRICES["large"]
        base_cost += pricing.request_cost(inp, out, lin, lout)
        # OPTIMIZED: cascade (route_tier), prompt caching, batch API
        pin, pout = MODEL_PRICES[r["route_tier"]]
        optimized_request_cost = pricing.request_cost(
            inp, out, pin, pout, cached_in=cached, batch=is_batch
        )
        opt_cost += optimized_request_cost

        is_reasoning = bool(int(num(r["is_reasoning"])))
        request_details.append({
            "input_tokens": inp,
            "output_tokens": out,
            "cached_input_tokens": cached,
            "is_batch": is_batch,
            "is_reasoning": is_reasoning,
            "cost": optimized_request_cost,
            "energy_wh": sustainability.wh_per_query(
                inp + out, is_reasoning=is_reasoning
            ),
        })

    base_pm = pricing.dollars_per_million(base_cost, total_tokens)
    opt_pm = pricing.dollars_per_million(opt_cost, total_tokens)
    savings_pct = (1 - opt_cost / base_cost) * 100 if base_cost else 0.0

    # Extension: quantify the reasoning tax and model a governed traffic cap.
    reasoning_rows = [d for d in request_details if d["is_reasoning"]]
    standard_rows = [d for d in request_details if not d["is_reasoning"]]
    reasoning_cost = sum(d["cost"] for d in reasoning_rows)
    total_energy = sum(d["energy_wh"] for d in request_details)
    reasoning_energy = sum(d["energy_wh"] for d in reasoning_rows)
    allowed_reasoning = int(len(rows) * REASONING_TRAFFIC_CAP)
    reroute_count = max(0, len(reasoning_rows) - allowed_reasoning)

    # Reroute the costliest excess reasoning requests to the small model. Synthetic
    # reasoning responses are 6x longer, so the fallback estimate removes that tax.
    cap_cost_savings = cap_energy_savings = 0.0
    for d in sorted(reasoning_rows, key=lambda x: x["cost"], reverse=True)[:reroute_count]:
        fallback_out = max(1, round(d["output_tokens"] / REASONING_OUTPUT_MULTIPLIER))
        small_in, small_out = MODEL_PRICES["small"]
        fallback_cost = pricing.request_cost(
            d["input_tokens"], fallback_out, small_in, small_out,
            cached_in=d["cached_input_tokens"], batch=d["is_batch"],
        )
        fallback_energy = sustainability.wh_per_query(
            d["input_tokens"] + fallback_out, is_reasoning=False
        )
        cap_cost_savings += max(0.0, d["cost"] - fallback_cost)
        cap_energy_savings += max(0.0, d["energy_wh"] - fallback_energy)

    reasoning = {
        "requests": len(reasoning_rows),
        "traffic_pct": round(len(reasoning_rows) / len(rows) * 100, 1) if rows else 0.0,
        "cost_pct": round(reasoning_cost / opt_cost * 100, 1) if opt_cost else 0.0,
        "energy_pct": round(reasoning_energy / total_energy * 100, 1) if total_energy else 0.0,
        "cap_pct": REASONING_TRAFFIC_CAP * 100,
        "rerouted_requests": reroute_count,
        "projected_cost_savings_daily": round(cap_cost_savings, 2),
        "projected_cost_savings_monthly": round(cap_cost_savings * 30, 2),
        "projected_energy_savings_wh_daily": round(cap_energy_savings, 1),
        "projected_energy_savings_kwh_monthly": round(cap_energy_savings * 30 / 1000, 1),
        "reasoning_metrics": _traffic_metrics(reasoning_rows),
        "standard_metrics": _traffic_metrics(standard_rows),
    }

    if verbose:
        print("== M2 Inference Cost Levers ==")
        print(f"requests={len(rows)}  tokens={total_tokens:,}")
        print(f"baseline  : ${base_cost:,.2f}/day   ${base_pm:.3f}/1M-token")
        print(f"optimized : ${opt_cost:,.2f}/day   ${opt_pm:.3f}/1M-token")
        print(f"savings   : {savings_pct:.1f}%  (cascade + caching + batch)")
        print(f"discount stack (batch + 100% cache): {pricing.discount_stack(batch=True, cache_hit_frac=1.0):.3f} of naive")
        print(
            f"reasoning : {reasoning['traffic_pct']:.1f}% traffic -> "
            f"{reasoning['cost_pct']:.1f}% cost, {reasoning['energy_pct']:.1f}% energy"
        )
        print(
            "normalized: reasoning "
            f"${reasoning['reasoning_metrics']['cost_per_m_tokens']:.3f}/1M-token, "
            f"{reasoning['reasoning_metrics']['energy_wh_per_m_tokens']:,.1f} Wh/1M-token; "
            "standard "
            f"${reasoning['standard_metrics']['cost_per_m_tokens']:.3f}/1M-token, "
            f"{reasoning['standard_metrics']['energy_wh_per_m_tokens']:,.1f} Wh/1M-token"
        )
        print(
            f"5% reasoning cap: reroute {reasoning['rerouted_requests']} requests -> "
            f"save ${reasoning['projected_cost_savings_daily']:.2f}/day and "
            f"{reasoning['projected_energy_savings_wh_daily']:.1f} Wh/day"
        )

    return {
        "baseline_daily": round(base_cost, 2), "optimized_daily": round(opt_cost, 2),
        "baseline_per_m": round(base_pm, 3), "optimized_per_m": round(opt_pm, 3),
        "savings_pct": round(savings_pct, 1), "total_tokens": total_tokens,
        "reasoning": reasoning,
    }


if __name__ == "__main__":
    run()
