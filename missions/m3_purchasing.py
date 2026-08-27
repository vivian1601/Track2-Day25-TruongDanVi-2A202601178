"""M3 — Purchasing Strategy: break-even, tier choice, spot-checkpoint sim (deck §4).

Run: python missions/m3_purchasing.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from missions._common import load_csv, num, catalog_by_type
from finops import pricing, sustainability

DAYS = 30


def carbon_aware_analysis(jobs: list[dict], catalog: dict,
                          source_region: str = "us-east-1") -> dict:
    """Compare regions for interruptible jobs using their measured GPU energy.

    Interruptible workloads are the safe scheduling pool: they can move in time or
    region without changing the always-on inference serving path.
    """
    energy_kwh = 0.0
    job_energy = []
    for job in jobs:
        if not bool(int(num(job["interruptible"]))):
            continue
        gpu = catalog[job["gpu_type"]]
        gpu_hours = num(job["hours_per_day"]) * num(job["days"]) * int(num(job["num_gpus"]))
        kwh = gpu_hours * num(gpu["watts"]) / 1000.0
        energy_kwh += kwh
        job_energy.append((job, kwh))

    comparison = {}
    for region, intensity in sustainability.REGION_CARBON.items():
        comparison[region] = {
            "price_per_kwh": sustainability.REGION_PRICE_KWH[region],
            "carbon_g_per_kwh": intensity,
            "electricity_cost": round(energy_kwh * sustainability.REGION_PRICE_KWH[region], 2),
            "carbon_kg": round(energy_kwh * intensity / 1000.0, 2),
        }

    cleanest = min(comparison, key=lambda r: comparison[r]["carbon_g_per_kwh"])
    cheapest = min(comparison, key=lambda r: comparison[r]["price_per_kwh"])
    max_price = max(v["price_per_kwh"] for v in comparison.values())
    max_carbon = max(v["carbon_g_per_kwh"] for v in comparison.values())
    balanced = min(
        comparison,
        key=lambda r: (
            comparison[r]["price_per_kwh"] / max_price
            + comparison[r]["carbon_g_per_kwh"] / max_carbon
        ),
    )
    source = comparison[source_region]
    target = comparison[cleanest]
    per_job = []
    for job, kwh in job_energy:
        source_carbon = kwh * sustainability.REGION_CARBON[source_region] / 1000.0
        target_carbon = kwh * sustainability.REGION_CARBON[cleanest] / 1000.0
        source_cost = kwh * sustainability.REGION_PRICE_KWH[source_region]
        target_cost = kwh * sustainability.REGION_PRICE_KWH[cleanest]
        per_job.append({
            "job_id": job["job_id"],
            "gpu_type": job["gpu_type"],
            "energy_kwh": round(kwh, 1),
            "source_carbon_kg": round(source_carbon, 2),
            "cleanest_carbon_kg": round(target_carbon, 2),
            "carbon_saved_kg": round(source_carbon - target_carbon, 2),
            "source_electricity_cost": round(source_cost, 2),
            "cleanest_electricity_cost": round(target_cost, 2),
        })
    return {
        "interruptible_energy_kwh": round(energy_kwh, 1),
        "source_region": source_region,
        "cleanest_region": cleanest,
        "cheapest_region": cheapest,
        "balanced_region": balanced,
        "carbon_saved_kg": round(source["carbon_kg"] - target["carbon_kg"], 2),
        "electricity_cost_change": round(target["electricity_cost"] - source["electricity_cost"], 2),
        "carbon_reduction_pct": round((1 - target["carbon_kg"] / source["carbon_kg"]) * 100, 1)
        if source["carbon_kg"] else 0.0,
        "per_job": per_job,
        "regions": comparison,
    }


def run(verbose: bool = True) -> dict:
    jobs = load_csv("workloads.csv")
    cat = catalog_by_type()
    on_demand_monthly = optimized_monthly = 0.0
    recs = []
    for j in jobs:
        gtype = j["gpu_type"]
        ngpu = int(num(j["num_gpus"]))
        hpd = num(j["hours_per_day"])
        interruptible = bool(int(num(j["interruptible"])))
        c = cat[gtype]
        gpu_hours = hpd * DAYS * ngpu
        od = num(c["on_demand_hr"])
        on_demand_cost = gpu_hours * od

        tier = pricing.recommend_tier(hpd, interruptible)
        if tier == "spot":
            sim = pricing.spot_checkpoint_cost(gpu_hours, num(c["spot_hr"]), od)
            opt_cost = sim["spot_cost"]
        elif tier == "reserved":
            opt_cost = gpu_hours * num(c["reserved_3yr_hr"])
        else:
            opt_cost = on_demand_cost

        on_demand_monthly += on_demand_cost
        optimized_monthly += opt_cost
        recs.append({"job_id": j["job_id"], "gpu_type": gtype, "tier": tier,
                     "on_demand": round(on_demand_cost), "optimized": round(opt_cost)})

    savings = on_demand_monthly - optimized_monthly
    savings_pct = savings / on_demand_monthly * 100 if on_demand_monthly else 0.0
    carbon = carbon_aware_analysis(jobs, cat)

    if verbose:
        print("== M3 Purchasing Strategy ==")
        print(f"break-even utilization @ 45% reserved discount = {pricing.break_even_utilization(0.45):.0%}")
        print(f"{'job':18}{'gpu':7}{'tier':11}{'on-demand':>12}{'optimized':>12}")
        for r in recs:
            print(f"{r['job_id']:18}{r['gpu_type']:7}{r['tier']:11}${r['on_demand']:>11,}${r['optimized']:>11,}")
        print(f"\nmonthly: on-demand ${on_demand_monthly:,.0f} -> optimized ${optimized_monthly:,.0f}  ({savings_pct:.1f}% saved)")
        print("\ncarbon-aware scheduling (interruptible jobs):")
        print(f"{'region':18}{'$/kWh':>8}{'gCO2/kWh':>11}{'energy $':>11}{'carbon kg':>12}")
        for region, values in carbon["regions"].items():
            print(
                f"{region:18}{values['price_per_kwh']:>8.3f}"
                f"{values['carbon_g_per_kwh']:>11.0f}"
                f"{values['electricity_cost']:>11.2f}"
                f"{values['carbon_kg']:>12.2f}"
            )
        print(
            f"cleanest={carbon['cleanest_region']}, cheapest={carbon['cheapest_region']}, "
            f"balanced={carbon['balanced_region']}; cleanest saves "
            f"{carbon['carbon_saved_kg']:.2f} kgCO2e "
            f"({carbon['carbon_reduction_pct']:.1f}%) vs {carbon['source_region']}"
        )
        print(f"\n{'interruptible job':18}{'GPU':7}{'kWh':>9}{'source kg':>12}{'clean kg':>11}{'saved kg':>11}")
        for job in carbon["per_job"]:
            print(
                f"{job['job_id']:18}{job['gpu_type']:7}{job['energy_kwh']:>9.1f}"
                f"{job['source_carbon_kg']:>12.2f}{job['cleanest_carbon_kg']:>11.2f}"
                f"{job['carbon_saved_kg']:>11.2f}"
            )

    return {"recommendations": recs, "on_demand_monthly": round(on_demand_monthly),
            "optimized_monthly": round(optimized_monthly), "savings_pct": round(savings_pct, 1),
            "carbon_aware": carbon}


if __name__ == "__main__":
    run()
