import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from missions import m2_inference_levers, m3_purchasing


def test_reasoning_budget_is_measured_and_saves_resources():
    result = m2_inference_levers.run(verbose=False)["reasoning"]
    assert result["traffic_pct"] > result["cap_pct"]
    assert result["cost_pct"] > result["traffic_pct"]
    assert result["energy_pct"] > result["traffic_pct"]
    assert result["projected_cost_savings_daily"] > 0
    assert result["projected_energy_savings_wh_daily"] > 0


def test_carbon_aware_schedule_compares_cost_and_carbon():
    result = m3_purchasing.run(verbose=False)["carbon_aware"]
    assert len(result["regions"]) == 5
    assert result["cleanest_region"] == "europe-north1"
    assert result["cheapest_region"] == "us-east-wa"
    assert result["carbon_saved_kg"] > 0
    assert result["regions"][result["cleanest_region"]]["carbon_kg"] < result["regions"][result["source_region"]]["carbon_kg"]
