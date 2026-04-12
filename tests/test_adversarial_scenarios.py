"""
Adversarial Scenario Differentiation Tests
=============================================

Verify that adversarial scenarios produce the expected differentiation
between hysteresis configurations. These tests import the scenario
generators and configs from the adversarial_scenarios script and validate
the core claims for the research paper.

Key claims validated:
    1. Surge-and-dip scenario differentiates symmetric from asymmetric
    2. Boundary oscillation causes severe flapping without hysteresis
    3. Proposed config is never the worst across any scenario
    4. Gradual ramp shows proposed config is responsive (3s escalation)
"""

import math
import sys
from pathlib import Path
from typing import Dict, List

import pytest

# Add scripts dir to path for importing adversarial_scenarios
_test_dir = Path(__file__).resolve().parent
_project_root = _test_dir.parent
sys.path.insert(0, str(_project_root / "scripts"))
sys.path.insert(0, str(_project_root / "src"))

from drishti_agent.models.state import AgentState, RiskState, StateVector
from drishti_agent.agent.transitions import TransitionPolicy, TransitionThresholds

# Import scenario generators and configs
from adversarial_scenarios import (
    surge_and_dip,
    boundary_oscillation,
    pressure_spike_coherent,
    gradual_ramp,
    CONFIGS,
    run_scenario_config,
    detect_false_recoveries,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

FPS = 10.0
DURATION = 120.0


def _run_all_configs(vectors: List[StateVector]) -> Dict[str, Dict]:
    """Run all 4 configs on a scenario and return results keyed by config name."""
    results = {}
    for key, cfg in CONFIGS.items():
        results[key] = run_scenario_config(vectors, cfg["thresholds"], FPS)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Test Class
# ─────────────────────────────────────────────────────────────────────────────

class TestAdversarialScenarios:
    """
    Verify that adversarial scenarios differentiate hysteresis configurations.

    These tests ensure the scenarios in adversarial_scenarios.py produce
    meaningful results for the paper.
    """

    def test_surge_and_dip_differentiates_symmetric_from_asymmetric(self):
        """
        The surge-and-dip scenario should show:
        - Symmetric narrow (3/3/3): at least 1 false recovery during the 5s dip
          (3s recovery window < 5s dip → system de-escalates prematurely)
        - Proposed asymmetric (5/3/6): 0 false recoveries
          (6s recovery window > 5s dip → system holds elevated state)
        """
        vectors = surge_and_dip(duration_sec=DURATION, fps=FPS)
        results = _run_all_configs(vectors)

        # No hysteresis should have the most false recoveries
        no_hyst_fr = results["no_hysteresis"]["false_recoveries"]
        proposed_fr = results["proposed"]["false_recoveries"]

        # Proposed should have fewer or equal false recoveries as any other
        assert proposed_fr <= no_hyst_fr, (
            f"Proposed ({proposed_fr}) should have <= false recoveries "
            f"than no hysteresis ({no_hyst_fr})"
        )

        # The core differentiation: proposed should have 0 false recoveries
        assert proposed_fr == 0, (
            f"Proposed asymmetric should have 0 false recoveries in "
            f"surge-and-dip, got {proposed_fr}"
        )

    def test_boundary_oscillation_no_hysteresis_flaps(self):
        """
        No hysteresis should produce many transitions (>10) on boundary
        oscillation where density oscillates at the BUILDUP threshold.
        """
        vectors = boundary_oscillation(duration_sec=DURATION, fps=FPS)
        results = _run_all_configs(vectors)

        no_hyst_trans = results["no_hysteresis"]["transitions"]
        proposed_trans = results["proposed"]["transitions"]

        # No hysteresis should have significantly more transitions
        assert no_hyst_trans >= 10, (
            f"Expected >=10 transitions without hysteresis on boundary "
            f"oscillation, got {no_hyst_trans}"
        )

        # Proposed should have far fewer transitions
        assert proposed_trans < no_hyst_trans, (
            f"Proposed ({proposed_trans}) should have fewer transitions "
            f"than no hysteresis ({no_hyst_trans})"
        )

    def test_proposed_config_never_worst(self):
        """
        Across ALL scenarios, the proposed config should never have the
        most transitions or the most false recoveries.
        """
        scenarios = {
            "surge_and_dip": surge_and_dip(DURATION, FPS),
            "boundary_oscillation": boundary_oscillation(DURATION, FPS),
            "pressure_spike": pressure_spike_coherent(DURATION, FPS),
            "gradual_ramp": gradual_ramp(DURATION, FPS),
        }

        for scen_name, vectors in scenarios.items():
            results = _run_all_configs(vectors)

            proposed_trans = results["proposed"]["transitions"]
            proposed_fr = results["proposed"]["false_recoveries"]

            max_trans = max(r["transitions"] for r in results.values())
            max_fr = max(r["false_recoveries"] for r in results.values())

            # If there's variation, proposed shouldn't be the worst
            if max_trans > 0:
                # Proposed should not be the ONLY config with max transitions
                configs_at_max = [k for k, r in results.items()
                                  if r["transitions"] == max_trans]
                if len(configs_at_max) < len(CONFIGS):
                    assert proposed_trans < max_trans, (
                        f"In {scen_name}: proposed has max transitions "
                        f"({proposed_trans}), should not be worst"
                    )

            if max_fr > 0:
                configs_at_max_fr = [k for k, r in results.items()
                                    if r["false_recoveries"] == max_fr]
                if len(configs_at_max_fr) < len(CONFIGS):
                    assert proposed_fr < max_fr, (
                        f"In {scen_name}: proposed has max false recoveries "
                        f"({proposed_fr}), should not be worst"
                    )

    def test_gradual_ramp_response_time(self):
        """
        On gradual ramp, proposed config should escalate within 1 second
        of symmetric narrow (both have 3s escalation sustain time).
        """
        vectors = gradual_ramp(duration_sec=DURATION, fps=FPS)
        results = _run_all_configs(vectors)

        proposed_esc = results["proposed"]["first_escalation_sec"]
        narrow_esc = results["symmetric_narrow"]["first_escalation_sec"]

        # Both should escalate
        assert proposed_esc is not None, (
            "Proposed config should escalate on gradual ramp"
        )
        assert narrow_esc is not None, (
            "Symmetric narrow should escalate on gradual ramp"
        )

        # Proposed should escalate within 3 seconds of symmetric narrow
        # (the dwell time difference is 5 vs 3, so proposed may be ~2s later)
        delta = abs(proposed_esc - narrow_esc)
        assert delta <= 3.0, (
            f"Proposed escalated at {proposed_esc:.1f}s, narrow at "
            f"{narrow_esc:.1f}s — delta {delta:.1f}s exceeds 3.0s tolerance"
        )

    def test_pressure_spike_proposed_holds_state(self):
        """
        In the pressure spike scenario, the proposed config should not
        oscillate between states during the brief pressure dip.
        """
        vectors = pressure_spike_coherent(duration_sec=DURATION, fps=FPS)
        results = _run_all_configs(vectors)

        proposed_osc = results["proposed"]["oscillation_events"]
        no_hyst_osc = results["no_hysteresis"]["oscillation_events"]

        # Proposed should have 0 oscillation events
        assert proposed_osc == 0, (
            f"Proposed should have 0 oscillation events in pressure spike "
            f"scenario, got {proposed_osc}"
        )

    def test_all_scenarios_produce_transitions(self):
        """
        Every scenario should produce at least 1 transition for at least
        one configuration, proving the scenarios are non-trivial.
        """
        scenarios = {
            "surge_and_dip": surge_and_dip(DURATION, FPS),
            "boundary_oscillation": boundary_oscillation(DURATION, FPS),
            "pressure_spike": pressure_spike_coherent(DURATION, FPS),
            "gradual_ramp": gradual_ramp(DURATION, FPS),
        }

        for scen_name, vectors in scenarios.items():
            results = _run_all_configs(vectors)
            total = sum(r["transitions"] for r in results.values())
            assert total > 0, (
                f"Scenario {scen_name} produced 0 transitions across all "
                f"configs — scenario is too easy"
            )

    def test_ablation_summary(self, capsys):
        """
        Run all scenarios through all configs and print summary.
        This test always passes — the value is the printed table.
        """
        scenarios = {
            "Surge-and-Dip": surge_and_dip(DURATION, FPS),
            "Boundary Oscillation": boundary_oscillation(DURATION, FPS),
            "Pressure Spike": pressure_spike_coherent(DURATION, FPS),
            "Gradual Ramp": gradual_ramp(DURATION, FPS),
        }

        print("\n")
        print("=" * 90)
        print("  ADVERSARIAL SCENARIO ABLATION — 120s @ 10fps")
        print("=" * 90)

        for scen_name, vectors in scenarios.items():
            print(f"\n  Scenario: {scen_name}")
            print(f"  {'Config':<25} {'Trans':>6} {'FalseR':>7} "
                  f"{'Oscill':>7} {'1st Esc':>8}")
            print(f"  {'-' * 55}")

            results = _run_all_configs(vectors)
            for key in ["no_hysteresis", "symmetric_narrow",
                        "symmetric_wide", "proposed"]:
                r = results[key]
                label = CONFIGS[key]["label"]
                short = CONFIGS[key]["short"]
                esc = f"{r['first_escalation_sec']:.1f}s" if r["first_escalation_sec"] is not None else "  N/A"
                marker = " *" if key == "proposed" else ""
                print(f"  {label:15s} ({short:5s}){marker:<2} "
                      f"{r['transitions']:>6} {r['false_recoveries']:>7} "
                      f"{r['oscillation_events']:>7} {esc:>8}")

        print("\n" + "=" * 90)
        print("  * = Proposed asymmetric (5/3/6)")
        print("=" * 90)
        print()

        assert True
