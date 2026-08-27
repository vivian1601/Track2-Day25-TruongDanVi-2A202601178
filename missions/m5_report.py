"""M5 — Optimization Report: combine M1-M4 into baseline-vs-optimized (deck §1/§11).

Run: python missions/m5_report.py   ->  outputs/report.md + outputs/savings.png
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import os
from missions._common import num, catalog_by_type, ROOT
from finops import report, sustainability
from missions import m1_efficiency_audit, m2_inference_levers, m3_purchasing

DAYS = 30
# one tier down for over-provisioned ("util-lie") GPUs
RIGHTSIZE_MAP = {"H100": "A100", "H200": "H100", "A100": "A10G", "A10G": "L4", "L4": "L4"}


def run(verbose: bool = True) -> dict:
    r1 = m1_efficiency_audit.run(verbose=False)
    r2 = m2_inference_levers.run(verbose=False)
    r3 = m3_purchasing.run(verbose=False)
    cat = catalog_by_type()

    # --- buckets ---
    infer_savings = (r2["baseline_daily"] - r2["optimized_daily"]) * DAYS
    purchasing_savings = r3["on_demand_monthly"] - r3["optimized_monthly"]

    idle_savings = r1["idle_waste_daily"] * DAYS
    rightsize_savings = 0.0
    for lie in r1["lies"]:
        cur = lie["gpu_type"]
        tgt = RIGHTSIZE_MAP.get(cur, cur)
        delta = num(cat[cur]["on_demand_hr"]) - num(cat[tgt]["on_demand_hr"])
        rightsize_savings += max(0.0, delta) * 24 * DAYS

    levers = {
        "Inference (cascade/cache/batch)": round(infer_savings),
        "Purchasing (spot/reserved)": round(purchasing_savings),
        "Right-size util-lies": round(rightsize_savings),
        "Kill idle GPUs": round(idle_savings),
    }
    baseline = r2["baseline_daily"] * DAYS + r3["on_demand_monthly"]
    optimized = baseline - sum(levers.values())
    total_pct = sum(levers.values()) / baseline * 100 if baseline else 0.0

    # --- sustainability snapshot ---
    median_tokens = 800
    wh = sustainability.wh_per_query(median_tokens)
    sust = {
        "wh_per_query": wh,
        "carbon_g": sustainability.carbon_g(wh, "us-east-1"),
        "best_region": min(sustainability.REGION_CARBON, key=sustainability.REGION_CARBON.get),
    }

    reasoning = r2["reasoning"]
    carbon = r3["carbon_aware"]
    analysis_sections = [
        "## Findings and prioritized actions",
        "",
        "1. **Move stable demand to reserved capacity and checkpoint interruptible jobs on spot.** "
        f"This is the largest measured lever at ${levers['Purchasing (spot/reserved)']:,.0f}/month; "
        "validate utilization before making a long commitment.",
        "2. **Apply inference routing, prompt caching, and batching.** "
        f"This saves ${levers['Inference (cascade/cache/batch)']:,.0f}/month and is a fast, low-risk "
        "software change that does not require a capacity commitment.",
        "3. **Profile and right-size the GPU-Util lies.** A high GPU-Util value only means the GPU clock "
        "was active; memory stalls, weak tensor-core occupancy, and kernel-launch overhead can keep useful "
        "FLOPs low. `gpu-h100-4` therefore bills as a full H100 while delivering about 20% MFU.",
        "4. **Auto-stop idle instances.** This is a low-risk guardrail and removes "
        f"${levers['Kill idle GPUs']:,.0f}/month from the current sample.",
        "",
        "## Extension: reasoning budget",
        "",
        f"Reasoning is {reasoning['traffic_pct']:.1f}% of requests but consumes "
        f"{reasoning['cost_pct']:.1f}% of optimized inference cost and "
        f"{reasoning['energy_pct']:.1f}% of estimated energy. Enforce a {reasoning['cap_pct']:.0f}% "
        "traffic budget: reserve reasoning for evaluation or high-complexity tasks; route excess requests "
        "to the small model after a cheap complexity check.",
        f"On this dataset that reroutes {reasoning['rerouted_requests']} requests/day and is projected to "
        f"save ${reasoning['projected_cost_savings_daily']:.2f}/day plus "
        f"{reasoning['projected_energy_savings_wh_daily']:,.1f} Wh/day.",
        "",
        "| Traffic | Requests | Tokens | $/1M tokens | Wh/1M tokens |",
        "|---|---:|---:|---:|---:|",
        f"| Reasoning | {reasoning['reasoning_metrics']['requests']:,} | "
        f"{reasoning['reasoning_metrics']['tokens']:,} | "
        f"${reasoning['reasoning_metrics']['cost_per_m_tokens']:.3f} | "
        f"{reasoning['reasoning_metrics']['energy_wh_per_m_tokens']:,.1f} |",
        f"| Standard | {reasoning['standard_metrics']['requests']:,} | "
        f"{reasoning['standard_metrics']['tokens']:,} | "
        f"${reasoning['standard_metrics']['cost_per_m_tokens']:.3f} | "
        f"{reasoning['standard_metrics']['energy_wh_per_m_tokens']:,.1f} |",
        "",
        f"At a 5% cap, the 30-day projection is "
        f"${reasoning['projected_cost_savings_monthly']:,.2f} and "
        f"{reasoning['projected_energy_savings_kwh_monthly']:,.1f} kWh saved.",
        "",
        "## Extension: carbon-aware scheduling",
        "",
        f"The interruptible pool uses {carbon['interruptible_energy_kwh']:,.1f} kWh per workload cycle. "
        f"Moving it from {carbon['source_region']} to {carbon['cleanest_region']} saves "
        f"{carbon['carbon_saved_kg']:,.2f} kgCO2e ({carbon['carbon_reduction_pct']:.1f}%) "
        "and reduces estimated electricity cost by "
        f"${abs(carbon['electricity_cost_change']):,.2f}.",
        "",
        "| Region | $/kWh | gCO2/kWh | Electricity cost | Carbon (kg) |",
        "|---|---:|---:|---:|---:|",
    ]
    for region, values in carbon["regions"].items():
        analysis_sections.append(
            f"| {region} | {values['price_per_kwh']:.3f} | "
            f"{values['carbon_g_per_kwh']:.0f} | ${values['electricity_cost']:,.2f} | "
            f"{values['carbon_kg']:,.2f} |"
        )
    analysis_sections += [
        "",
        "| Interruptible job | GPU | Energy (kWh) | Source carbon (kg) | Cleanest carbon (kg) | Saved (kg) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for job in carbon["per_job"]:
        analysis_sections.append(
            f"| {job['job_id']} | {job['gpu_type']} | {job['energy_kwh']:,.1f} | "
            f"{job['source_carbon_kg']:,.2f} | {job['cleanest_carbon_kg']:,.2f} | "
            f"{job['carbon_saved_kg']:,.2f} |"
        )
    analysis_sections += [
        "",
        f"Use **{carbon['cleanest_region']}** when carbon is the priority, "
        f"**{carbon['cheapest_region']}** for minimum electricity cost, or "
        f"**{carbon['balanced_region']}** for the simple cost-carbon balance. Validate data residency, "
        "capacity, and user latency before moving online inference; the recommendation applies first to "
        "checkpointable batch/training jobs.",
    ]

    md = report.build_report(
        baseline, optimized, levers, sustainability=sust,
        analysis_sections=analysis_sections,
    )
    out_md = os.path.join(ROOT, "outputs", "report.md")
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    with open(out_md, "w") as f:
        f.write(md)
    png = report.savings_waterfall(levers, os.path.join(ROOT, "outputs", "savings.png"))

    if verbose:
        print("== M5 Optimization Report ==")
        print(md)
        print(f"\nWritten: outputs/report.md" + (f" + outputs/savings.png" if png else " (matplotlib absent: PNG skipped)"))

    return {"baseline_monthly": round(baseline), "optimized_monthly": round(optimized),
            "levers": levers, "total_savings_pct": round(total_pct, 1)}


if __name__ == "__main__":
    run()
