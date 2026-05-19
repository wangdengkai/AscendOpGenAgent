#!/usr/bin/env python3
"""
wm_ops.py — World Model CLI operations for LINGXI evolution.

CLI subcommands:
  select          --path <wm.json> --n <int>      Select top-N open nodes by utility score
  validate        --path <wm.json>                Validate invariants (non-zero exit = errors)
  summary         --path <wm.json> [--max-chars 1200]  Compact summary for sub-agent injection
  deep-profiling  --wm-path <wm.json> --node-id <id> --work-dir <dir> --op-name <name>
                  Run deep profiling analysis and write evidence to world model
  refine          --wm-path <wm.json> --round <int> --results-dir <dir> --parallel-map <json>
                  Deterministic world model update after a round (score, children, stagnation)
  diagnose        --wm-path <wm.json> --node-id <id> --failure-type <type> --failure-reason <str>
                  Write failure diagnosis for a node (impl_error → fix child, strategy_infeasible → seal)
"""

import argparse
import json
import os
import subprocess
import sys
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Optional state.json sync with state_ops.py
# ---------------------------------------------------------------------------
# wm_ops subcommands operate on world_model.json. The runtime state cursor
# lives in <evo_dir>/state.json (see state_ops.py / state_schema.md).
# These helpers infer evo_dir from the wm_path and re-derive state as a
# side effect. All failures are silently ignored — state.json is optional;
# wm_ops remains backward-compatible if it is absent.

def _maybe_update_state_stage(wm_path: str, stage: str, round_num: Optional[int] = None) -> None:
    """Best-effort: if <dirname(wm_path)>/state.json exists, update its stage.

    Silently noops if state.json missing or state_ops import fails. Never
    raises — wm_ops must not break when state.json infrastructure is absent.
    """
    try:
        evo_dir = os.path.dirname(os.path.abspath(wm_path))
        state_path = os.path.join(evo_dir, "state.json")
        if not os.path.isfile(state_path):
            return
        # Import lazily to avoid hard dependency at module load
        from state_ops import _read_state, _write_state, ALL_STAGES  # type: ignore
        if stage not in ALL_STAGES:
            return
        state = _read_state(evo_dir)
        prev = state.get("stage")
        state["stage"] = stage
        if round_num is not None:
            state["current_round"] = round_num
        _write_state(evo_dir, state)
        print(f"  [state] stage {prev} → {stage}" + (f" (round={round_num})" if round_num is not None else ""),
              file=sys.stderr)
    except Exception:
        # state.json sync is optional; never let it break wm_ops
        pass


# ---------------------------------------------------------------------------
# Drift circuit breaker
# ---------------------------------------------------------------------------
# When the global stagnation_count or branch stagnation_count_vs_base crosses
# DRIFT_THRESHOLD, write state.drift_status = "replan_required" so that the
# next round's GATE step diverts to the drift_replan flow (open_exploration
# saturation + forced fresh-source reading). When stagnation recovers, auto
# clear back to "normal".

DRIFT_THRESHOLD = 2  # consecutive stalled rounds to trigger drift


def _decide_drift_status(wm: dict) -> tuple[str, str]:
    """Return (new_drift_status, reason) based on world model stagnation counters.

    Args:
        wm: world model dict (post-refine state).

    Returns:
        (status, reason):
          status ∈ {"replan_required", "normal"}
          reason: human-readable one-liner for logging
    """
    sc = int(wm.get("stagnation_count", 0))
    scvb = int(wm.get("stagnation_count_vs_base", 0))

    if sc >= DRIFT_THRESHOLD:
        return "replan_required", (
            f"stagnation_count={sc} ≥ {DRIFT_THRESHOLD}: "
            f"no global-best progress for {sc} consecutive rounds"
        )
    if scvb >= DRIFT_THRESHOLD:
        return "replan_required", (
            f"stagnation_count_vs_base={scvb} ≥ {DRIFT_THRESHOLD}: "
            f"no variant beat its parent for {scvb} consecutive rounds"
        )
    return "normal", f"stagnation_count={sc}, vs_base={scvb} below threshold"


def _maybe_update_drift_status(wm_path: str, wm: dict) -> None:
    """Best-effort: detect stall and write state.drift_status.

    Called from cmd_refine after world_model.json is updated. Noops if
    state.json missing or import fails. Mirrors the noop discipline of
    _maybe_update_state_stage.
    """
    try:
        evo_dir = os.path.dirname(os.path.abspath(wm_path))
        state_path = os.path.join(evo_dir, "state.json")
        if not os.path.isfile(state_path):
            return
        from state_ops import _read_state, _write_state  # type: ignore

        state = _read_state(evo_dir)
        prev = state.get("drift_status", "normal")
        new_status, reason = _decide_drift_status(wm)

        if prev == new_status:
            return  # idempotent — no-op log spam

        state["drift_status"] = new_status
        _write_state(evo_dir, state)
        marker = "[DRIFT]" if new_status == "replan_required" else "[drift cleared]"
        print(f"  {marker} drift_status {prev} → {new_status} ({reason})", file=sys.stderr)
    except Exception:
        pass


def _read_state_field_safe(wm_path: str, field: str) -> Any:
    """Read a top-level field from <evo_dir>/state.json. Returns None on any error."""
    try:
        evo_dir = os.path.dirname(os.path.abspath(wm_path))
        state_path = os.path.join(evo_dir, "state.json")
        if not os.path.isfile(state_path):
            return None
        with open(state_path, "r", encoding="utf-8") as f:
            state = json.load(f)
        return state.get(field)
    except Exception:
        return None


def _maybe_infer_state(wm_path: str) -> None:
    """Re-derive state.stage/current_round/partial_status from filesystem
    evidence (replaces an older single-field _maybe_update_state_stage helper).

    Noop if state.json missing. Drift_status / stall_count / must_run are
    preserved (those are wm_ops/setup's responsibility, not LLM-trusted state).
    """
    try:
        evo_dir = os.path.dirname(os.path.abspath(wm_path))
        state_path = os.path.join(evo_dir, "state.json")
        if not os.path.isfile(state_path):
            return
        from state_ops import _read_state, _write_state, _infer_state_from_filesystem  # type: ignore
        inferred = _infer_state_from_filesystem(evo_dir)
        state = _read_state(evo_dir)
        prev_stage = state.get("stage")
        prev_round = state.get("current_round")
        state["stage"] = inferred["stage"]
        state["current_round"] = inferred["current_round"]
        state["partial_status"] = inferred["partial_status"]
        _write_state(evo_dir, state)
        if prev_stage != inferred["stage"] or prev_round != inferred["current_round"]:
            print(
                f"  [state] inferred: stage {prev_stage} → {inferred['stage']}, "
                f"round {prev_round} → {inferred['current_round']}",
                file=sys.stderr,
            )
    except Exception:
        pass


def _maybe_clear_drift(wm_path: str) -> None:
    """Drift_status auto-clear after SELECT consumed the drift signal.

    Called from cmd_select right after a drift-aware selection runs. The drift
    signal is a one-shot trigger; consuming it must reset state.drift_status to
    "normal" so subsequent rounds aren't repeatedly diverted.
    """
    try:
        evo_dir = os.path.dirname(os.path.abspath(wm_path))
        state_path = os.path.join(evo_dir, "state.json")
        if not os.path.isfile(state_path):
            return
        from state_ops import _read_state, _write_state  # type: ignore
        state = _read_state(evo_dir)
        if state.get("drift_status") == "replan_required":
            state["drift_status"] = "normal"
            _write_state(evo_dir, state)
            print("  [drift consumed] drift_status replan_required → normal", file=sys.stderr)
    except Exception:
        pass


def _profiling_complete(parallel_dir: str) -> bool:
    """Check whether msprof profiling produced its key artifact in a parallel dir.

    The canonical evidence is `parallel_K/profiling/.../op_summary_*.csv` —
    this CSV is what ascendc-profiling-analysis actually consumes. If absent,
    profiling either was skipped or failed silently.

    Returns True if at least one op_summary_*.csv exists under profiling/.
    """
    prof_dir = os.path.join(parallel_dir, "profiling")
    if not os.path.isdir(prof_dir):
        return False
    # Walk the directory tree; profiling output is nested deep
    for root, _dirs, files in os.walk(prof_dir):
        for f in files:
            if f.startswith("op_summary_") and f.endswith(".csv"):
                return True
    return False


def _maybe_mark_profiling_skipped(wm_path: str, results_dir: str, parallel_map: dict) -> None:
    """R9: detect if msprof was skipped for any passed partial.

    For each parallel slot whose evaluation_results.json reports
    precision_passed=True (i.e. the kernel ran), check whether
    profiling/.../op_summary_*.csv exists. If any passed partial lacks
    profiling, mark `msprof` into state.must_run_before_next_round so the
    next round's GATE / Stop hook R4 blocks until profiling is rerun.
    """
    try:
        evo_dir = os.path.dirname(os.path.abspath(wm_path))
        state_path = os.path.join(evo_dir, "state.json")
        if not os.path.isfile(state_path):
            return

        missing = []
        for p_idx_str, _node_id in parallel_map.items():
            parallel_dir = os.path.join(results_dir, f"parallel_{p_idx_str}")
            eval_path = os.path.join(parallel_dir, "evaluation_results.json")
            if not os.path.isfile(eval_path):
                continue
            try:
                with open(eval_path, "r", encoding="utf-8") as f:
                    eres = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
            # Only care about partials that actually ran successfully — failed
            # compiles legitimately have no profiling.
            if not eres.get("precision_passed"):
                continue
            if not _profiling_complete(parallel_dir):
                missing.append(p_idx_str)

        if not missing:
            return

        from state_ops import _read_state, _write_state  # type: ignore
        state = _read_state(evo_dir)
        pending = state.setdefault("must_run_before_next_round", [])
        if "msprof" not in pending:
            pending.append("msprof")
            _write_state(evo_dir, state)
            print(
                f"  [R9] msprof missing for passed partial(s) {missing}; "
                f"must_run_before_next_round += msprof",
                file=sys.stderr,
            )
    except Exception:
        pass


def _maybe_warn_precision_failures(parallel_map: dict, results_dir: str) -> None:
    """R10 (warn-only): if ≥50% of partials failed precision, alert.

    Pure stderr warning — does not block. Surfaces large-scale precision
    regressions early so the user notices before agent claims success.
    """
    try:
        total = 0
        failed = 0
        for p_idx_str in parallel_map:
            parallel_dir = os.path.join(results_dir, f"parallel_{p_idx_str}")
            eval_path = os.path.join(parallel_dir, "evaluation_results.json")
            if not os.path.isfile(eval_path):
                continue
            total += 1
            try:
                with open(eval_path, "r", encoding="utf-8") as f:
                    eres = json.load(f)
                if not eres.get("precision_passed", True):
                    failed += 1
            except (json.JSONDecodeError, OSError):
                continue
        if total >= 2 and failed * 2 >= total:
            print(
                f"  [R10 WARN] {failed}/{total} partials failed precision in this round. "
                f"Review evaluation_results.json before claiming success.",
                file=sys.stderr,
            )
    except Exception:
        pass


# Ensure state_ops is importable when wm_ops is run as a script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# Core algorithm: utility computation
# ---------------------------------------------------------------------------

def _baseline_mismatch_penalty(node: dict, wm: dict) -> float:
    """A5: soft penalty for nodes misaligned with root-level baseline_evidence.

    Returns:
      0.0 if wm has no baseline_evidence (fallback to pre-A5 behavior)
      -2.0 if node's strategy_combination intersects baseline anti_strategies
      -1.0 if node has a non-empty strategy_combination that has NO intersection
           with baseline suggested_strategies (primary + secondary)
      0.0 otherwise

    Cast so the penalty cannot flip cross-layer ordering (see design doc):
    abs(-2.0) < typical parent_score differential and w_root_explore, so
    a misaligned node with a very high-scoring parent can still outrank an
    aligned node with a weak parent — this is intentional.
    """
    be = wm.get("baseline_evidence") if isinstance(wm, dict) else None
    if not be or not isinstance(be, dict):
        return 0.0
    strats = node.get("strategy_combination") or []
    if not strats:
        return 0.0
    s_set = set(strats)
    anti = set(be.get("anti_strategies") or [])
    if s_set & anti:
        return -2.0
    suggested = set(be.get("suggested_strategies") or [])
    if suggested and not (s_set & suggested):
        return -1.0
    return 0.0


def compute_utility(node: dict, nodes: dict, wm: Optional[dict] = None) -> float:
    """Compute utility score for a given node using the unified world model formula.

    Unified formula (authoritative — all other docs reference this):
        utility = 3.0 × parent_score
                + 2.5 × (5 - difficulty)
                + 0.75 × depth
                + w_root_explore       (2.0 if parent is root, else 0.0)
                + w_evidence           (1.5 if parent has profiling_evidence, else 0.0)
                + w_baseline_mismatch  (A5: 0.0 / -1.0 / -2.0, requires wm)

    Rationale for each term:
        parent_score:          Exploit — children of high-performing nodes are more promising
        difficulty:            Prefer easier implementations first (low-hanging fruit)
        depth:                 Mild encouragement for depth-first exploitation
        w_root_explore:        Ensure first-layer breadth is explored before deep-diving
        w_evidence:            Prioritize nodes whose parent has instruction-level profiling
        w_baseline_mismatch:   Soft-penalize strategies misaligned with baseline bottleneck;
                               zero when wm.baseline_evidence is null (backwards-compatible)
    """
    parent_id = node.get("parent_id") or "root"
    parent = nodes.get(parent_id, {})
    parent_score = parent.get("score") or 1.0
    difficulty = node.get("difficulty", 3)
    depth = node.get("depth", 1)
    w_root_explore = 2.0 if parent_id == "root" else 0.0
    w_evidence = 1.5 if parent.get("profiling_evidence") else 0.0
    w_baseline_mismatch = _baseline_mismatch_penalty(node, wm) if wm else 0.0
    return (3.0 * parent_score
            + 2.5 * (5 - difficulty)
            + 0.75 * depth
            + w_root_explore
            + w_evidence
            + w_baseline_mismatch)


# ---------------------------------------------------------------------------
# Optimization type inference
# ---------------------------------------------------------------------------

_BANDWIDTH_STRATEGIES = frozenset({
    "P1", "P7", "P10", "P11", "R5",
    # P19-P26: buffer management + DataCopy params
    "P19", "P20", "P21", "P22",
    "P24", "P25", "P26",
    # P32-P33: special copy patterns
    "P32", "P33",
    # P34-P45: buffer resident / reuse
    "P34", "P35", "P37", "P38", "P40",
    "P41", "P42", "P43", "P44",
    "P45",
    # P49, P52: hardware dequant, L2 cache hint
    "P49", "P52",
    # P53-P88: bandwidth-related (data movement, format, buffer, cache)
    "P53", "P56", "P59", "P60", "P61", "P63",
    "P64", "P65", "P66", "P69", "P70", "P71",
    "P74", "P76", "P78", "P81", "P83", "P85",
})
_TILING_STRATEGIES = frozenset({
    "P2", "P4", "P5", "P8",
    # P28-P30: sync & pipeline control
    "P28", "P29", "P30",
    # P47, P51: diagonal scheduling, dynamic core ratio
    "P47", "P51",
    # P53-P88: tiling-related (sync, partition, pipeline, core ratio)
    "P54", "P55", "P57", "P58", "P67", "P68",
    "P72", "P73", "P75", "P77", "P80", "P82",
    "P84", "P86", "P87", "P88",
})

_VALID_STRATEGY_IDS = frozenset({
    # D-series
    "D1", "D2", "D3", "D4", "D5",
    # P-series L0
    "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10", "P11", "P12", "P13",
    # P-series L1 (P14-P52)
    "P14", "P16", "P17", "P18",
    "P19", "P20", "P21", "P22",
    "P24", "P25", "P26",
    "P28", "P29", "P30",
    "P32", "P33",
    "P34", "P35", "P37", "P38",
    "P40", "P41", "P42", "P43", "P44", "P45",
    "P46", "P47", "P48", "P49", "P50", "P51", "P52",
    # P-series L1 (P53-P88, fork-B incremental)
    "P53", "P54", "P55", "P56",
    "P57", "P58",
    "P59", "P60", "P61", "P62", "P63", "P64", "P65", "P66",
    "P67", "P68", "P69",
    "P70", "P71", "P72",
    "P73", "P74", "P75",
    "P76", "P77", "P78",
    "P79", "P80", "P81", "P82", "P83", "P84",
    "P85", "P86", "P87", "P88",
    # A-series
    "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8",
    # X-series (discovered)
    "X1",
})
_REGISTER_OPT_STRATEGIES = frozenset({"R2"})
_VF_FUSION_STRATEGIES = frozenset({"R1"})
_INSTRUCTION_SCHED_STRATEGIES = frozenset({"R3", "R4", "R8"})


def infer_optimization_type(strategy_combination: list, mode: str = "strategy_guided") -> str:
    """Infer the performance optimization type from strategy combination.

    Returns one of: "bandwidth", "tiling", "algorithm",
    "register_opt", "vf_fusion", "instruction_sched".
    D/A-series strategies are ignored (precision constraints, not perf directions).
    P19-P88 (data movement / CV fusion / advanced) are classified as bandwidth, tiling (sync),
    or algorithm (default).
    A5 R-series strategies map to register_opt/vf_fusion/instruction_sched/bandwidth/algorithm.
    """
    if mode in ("open_exploration", "profiling_driven"):
        return "algorithm"
    s = set(strategy_combination) if strategy_combination else set()
    bw = len(s & _BANDWIDTH_STRATEGIES)
    tl = len(s & _TILING_STRATEGIES)
    ro = len(s & _REGISTER_OPT_STRATEGIES)
    vf = len(s & _VF_FUSION_STRATEGIES)
    isc = len(s & _INSTRUCTION_SCHED_STRATEGIES)
    counts = {"bandwidth": bw, "tiling": tl, "register_opt": ro,
              "vf_fusion": vf, "instruction_sched": isc}
    best_type = max(counts, key=counts.get)
    if counts[best_type] > 0:
        return best_type
    return "algorithm"


# ---------------------------------------------------------------------------
# soft prune
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# soft prune
# ---------------------------------------------------------------------------

def soft_prune_dead_branches(nodes: dict) -> list:
    """Soft-prune open nodes under sealed ancestors.

    An ancestor is considered "sealed" if any of:
      - status=="failed" AND difficulty>=5 (impl-level seal, legacy behavior)
      - direction_sealed is True (direction-level seal set by diagnose on a
        passed node whose direction has been semantically disproven)

    Walks each open node's parent chain upward. If a sealed ancestor is found
    before a healthy (passed but non-sealed / completed) ancestor, the open
    node is soft-pruned by setting its difficulty to 5 (never deleted).

    Returns list of pruned node IDs.
    """
    pruned = []
    for nid, nd in list(nodes.items()):
        if nd.get("status") != "open":
            continue
        current_id = nd.get("parent_id")
        should_prune = False
        visited = set()
        while current_id and current_id in nodes and current_id not in visited:
            visited.add(current_id)
            ancestor = nodes[current_id]
            sealed_failed = (ancestor.get("status") == "failed"
                             and ancestor.get("difficulty", 0) >= 5)
            sealed_direction = bool(ancestor.get("direction_sealed"))
            if sealed_failed or sealed_direction:
                should_prune = True
                break
            # A passed/completed ancestor without direction_sealed is healthy;
            # stop the upward walk (don't cross a healthy branch).
            if ancestor.get("status") in ("passed", "completed"):
                break
            current_id = ancestor.get("parent_id")
        if should_prune:
            nd["difficulty"] = 5
            pruned.append(nid)
    return pruned


# ---------------------------------------------------------------------------
# soft demote (A4: bridging stale-direction signal → SELECT without hard-prune)
# ---------------------------------------------------------------------------


def soft_demote_stale_directions(
    wm: dict,
    quality: str = "good",
    round_num: Optional[int] = None,
) -> list:
    """Penalize open descendants of passed nodes whose speedup barely beat
    their parent AND whose bottleneck did not shift.

    Signal:
      A passed node n with n.score < parent.score × stag_threshold is
      "soft disproven" when bottleneck_shift is not set — the variant ran,
      but it neither improved meaningfully nor opened a new bottleneck
      front. The direction has not been hard-disproven (that is A6 via
      diagnose), but exploring further down this branch is lower value
      than exploring elsewhere.

    Effect:
      All open descendants of such nodes get difficulty += 1, capped at 4
      (never 5 — that's reserved for hard seal via diagnose). This lets
      compute_utility naturally deprioritize them vs. siblings with
      stronger parents, while keeping them selectable as a last resort.

    Idempotence:
      Marks each demoted node with demoted_in_round so repeated refine
      calls in the same round do not re-inflate difficulty.

    Returns list of demoted node IDs.
    """
    stag_threshold = _THRESHOLDS.get(quality, _THRESHOLDS["good"])["stagnation"]
    nodes = wm.get("decision_tree", {}).get("nodes", {}) or {}
    demoted = []

    for nid, nd in list(nodes.items()):
        if nd.get("status") != "passed":
            continue
        if nd.get("bottleneck_shift"):
            continue
        parent_id = nd.get("parent_id")
        if not parent_id:
            continue
        parent = nodes.get(parent_id, {})
        parent_score = parent.get("score") or 1.0
        self_score = nd.get("score") or 0.0
        # Weak improvement under the current measurement-quality threshold.
        # Absolute regression (self < parent) is subsumed since stag_threshold ≥ 1.0.
        if self_score >= parent_score * stag_threshold:
            continue

        # Walk descendants via BFS
        frontier = list(nd.get("children", []))
        visited = set()
        while frontier:
            cid = frontier.pop()
            if cid in visited or cid not in nodes:
                continue
            visited.add(cid)
            ch = nodes[cid]
            # Don't cross a passed descendant (its own sub-tree is judged
            # by its own metrics).
            if ch.get("status") == "passed":
                continue
            # Only demote open nodes; also require idempotence on round.
            if ch.get("status") == "open":
                marker = ch.get("demoted_in_round")
                if round_num is not None and marker == round_num:
                    pass  # already demoted this round, skip bump but still walk children
                else:
                    old = ch.get("difficulty", 3) or 3
                    ch["difficulty"] = min(4, old + 1)
                    if round_num is not None:
                        ch["demoted_in_round"] = round_num
                    demoted.append(cid)
            frontier.extend(ch.get("children", []) or [])

    return demoted


# ---------------------------------------------------------------------------
# select
# ---------------------------------------------------------------------------

def _strategy_signature(strategies) -> frozenset:
    """Canonicalize a strategy_combination for sibling collision detection."""
    if not strategies:
        return frozenset()
    return frozenset(str(s).strip() for s in strategies if s)


def _sig_jaccard(sig_a: frozenset, sig_b: frozenset) -> float:
    """Jaccard similarity between two strategy signatures."""
    if not sig_a and not sig_b:
        return 1.0
    if not sig_a or not sig_b:
        return 0.0
    inter = len(sig_a & sig_b)
    union = len(sig_a | sig_b)
    return inter / union if union else 0.0


# A3 hard-filter threshold: two sibling variants with jaccard > this AND same
# optimization_type are considered a collision and the later one is skipped.
# 0.6 catches "one of two strategies swapped" for length-2 combos and "two of
# three kept" for length-3 combos — both are semantically redundant work.
_COLLISION_JACCARD_THRESHOLD = 0.6


def _collides_with_selected(
    candidate: dict, selected: list
) -> bool:
    """A3: True if candidate's strategy signature jaccard-overlaps with an
    already-selected sibling AND they share optimization_type.

    Same-parent is NOT a requirement for collision — two variants on different
    parents doing literally the same strategy combination still waste a slot.
    """
    cand_sig = _strategy_signature(candidate.get("strategy_combination"))
    if not cand_sig:
        # Empty-strategy nodes (e.g. free exploration) are never collision-filtered
        return False
    cand_type = candidate.get("optimization_type")
    for _nid, sel_nd in selected:
        sel_sig = _strategy_signature(sel_nd.get("strategy_combination"))
        if not sel_sig:
            continue
        if cand_type and sel_nd.get("optimization_type") != cand_type:
            continue
        if _sig_jaccard(cand_sig, sel_sig) > _COLLISION_JACCARD_THRESHOLD:
            return True
    return False


def select_nodes(wm: dict, n: int, force_open_exploration_min: Optional[int] = None) -> list:
    """
    Select top-N open nodes by utility score with open_exploration slot reservation
    and branch diversity constraint.

    Slot allocation (matches lingxi-evo/SKILL.md §4.0):
      - oe_slots = max(1, ⌈n/4⌉) reserved for open_exploration / profiling_driven
      - sg_slots = n - oe_slots for strategy_guided, sorted by utility descending
      - oe slots fall back to next strategy_guided if not enough oe nodes exist

    Branch diversity constraint:
      - Each parent_id contributes at most ceil(n / num_active_branches) slots
      - Prevents all slots from clustering on a single high-scoring branch
      - Falls back to unconstrained fill if constraint leaves slots empty

    Drift mode:
      - When `force_open_exploration_min` is set (e.g. by cmd_select when
        state.drift_status == "replan_required"), oe_slots is raised to at
        least that many. Default oe_slots formula still applies as a floor.
      - Typical drift call: force_open_exploration_min = max(1, ⌈n/2⌉) so at
        least half the round goes to fresh exploration when search stalls.

    If n == 1, the single slot goes to the highest-utility node regardless of mode.

    Pads with free-exploration placeholders if open nodes are insufficient.

    Returns a list of dicts with parallel_index, node_id, utility, mode, and
    all node fields needed for prompt construction.
    """
    nodes: dict = wm.get("decision_tree", {}).get("nodes", {})
    open_nodes = [
        (nid, nd) for nid, nd in nodes.items()
        if nd.get("status") == "open"
    ]

    # Split by mode
    strategy_guided = []
    open_exploration = []  # includes both open_exploration and profiling_driven
    for nid, nd in open_nodes:
        mode = nd.get("mode", "strategy_guided")
        if mode in ("open_exploration", "profiling_driven"):
            open_exploration.append((nid, nd))
        else:
            strategy_guided.append((nid, nd))

    # Sort each group by utility descending (wm passed so A5 baseline
    # mismatch penalty can fire when wm.baseline_evidence is present)
    strategy_guided.sort(key=lambda x: -compute_utility(x[1], nodes, wm))
    open_exploration.sort(key=lambda x: -compute_utility(x[1], nodes, wm))

    # --- Branch diversity constraint ---
    # Compute max slots per parent_id to prevent single-branch dominance
    sg_parent_ids = set(
        (nd.get("parent_id") or "root") for _, nd in strategy_guided
    )
    num_active_branches = max(1, len(sg_parent_ids))
    max_per_branch = max(1, -(-n // num_active_branches))  # ceil division
    branch_count: dict[str, int] = {}

    def _branch_ok(nd: dict) -> bool:
        """Check if the node's parent branch still has capacity."""
        pid = nd.get("parent_id") or "root"
        return branch_count.get(pid, 0) < max_per_branch

    def _branch_add(nd: dict) -> None:
        """Record that a slot was allocated from this node's parent branch."""
        pid = nd.get("parent_id") or "root"
        branch_count[pid] = branch_count.get(pid, 0) + 1

    selected: list[tuple[str, dict]] = []

    if n == 1:
        # Single slot: pick the overall best regardless of mode
        all_sorted = sorted(open_nodes, key=lambda x: -compute_utility(x[1], nodes, wm))
        if all_sorted:
            selected.append(all_sorted[0])
    else:
        # P0-a: ⌈n/4⌉ slots reserved for open_exploration (min 1), rest for strategy_guided
        oe_slots = max(1, -(-n // 4))
        # Drift mode raises oe_slots to force fresh-direction exploration
        if force_open_exploration_min is not None:
            oe_slots = max(oe_slots, min(force_open_exploration_min, n))
        sg_slots = n - oe_slots

        # --- Guaranteed round: each optimization type with open nodes gets 1 slot ---
        # Also respects branch diversity constraint
        type_best: dict[str, tuple[str, dict]] = {}
        for nid, nd in strategy_guided:
            t = nd.get("optimization_type") or infer_optimization_type(
                nd.get("strategy_combination", []),
                nd.get("mode", "strategy_guided"),
            )
            if t not in type_best:
                type_best[t] = (nid, nd)  # strategy_guided already sorted desc

        selected_sg: list[tuple[str, dict]] = []
        used_ids: set[str] = set()
        # Guaranteed round: each optimization type with open nodes gets 1 slot
        # Covers A3 types (bandwidth, tiling, algorithm) and A5 types
        # (register_opt, vf_fusion, instruction_sched)
        for t in type_best:
            if len(selected_sg) < sg_slots:
                nid, nd = type_best[t]
                if _branch_ok(nd) and not _collides_with_selected(nd, selected_sg):
                    selected_sg.append((nid, nd))
                    used_ids.add(nid)
                    _branch_add(nd)

        # --- Remaining slots: utility-driven with branch diversity + A3 collision filter ---
        for nid, nd in strategy_guided:
            if len(selected_sg) >= sg_slots:
                break
            if (nid not in used_ids
                    and _branch_ok(nd)
                    and not _collides_with_selected(nd, selected_sg)):
                selected_sg.append((nid, nd))
                used_ids.add(nid)
                _branch_add(nd)

        # --- Fallback: if branch+collision constraints left slots unfilled, relax ---
        # Relax order: (a) keep collision filter, drop branch constraint;
        #              (b) drop both (last resort — avoid empty slots)
        if len(selected_sg) < sg_slots:
            for nid, nd in strategy_guided:
                if len(selected_sg) >= sg_slots:
                    break
                if nid not in used_ids and not _collides_with_selected(nd, selected_sg):
                    selected_sg.append((nid, nd))
                    used_ids.add(nid)
        if len(selected_sg) < sg_slots:
            for nid, nd in strategy_guided:
                if len(selected_sg) >= sg_slots:
                    break
                if nid not in used_ids:
                    selected_sg.append((nid, nd))
                    used_ids.add(nid)

        selected.extend(selected_sg)

        # --- open_exploration dedicated slots (oe_slots total) ---
        # Fill up to oe_slots with open_exploration nodes; if not enough,
        # fall back to remaining strategy_guided nodes (collision-filtered).
        available_oe = [(nid, nd) for nid, nd in open_exploration if nid not in used_ids]
        oe_taken = 0
        for nid, nd in available_oe[:oe_slots]:
            selected.append((nid, nd))
            used_ids.add(nid)
            oe_taken += 1

        while oe_taken < oe_slots:
            remaining = [
                (nid, nd) for nid, nd in strategy_guided
                if nid not in used_ids and not _collides_with_selected(nd, selected)
            ]
            if not remaining:
                remaining = [(nid, nd) for nid, nd in strategy_guided if nid not in used_ids]
            if not remaining:
                break
            nid, nd = remaining[0]
            selected.append((nid, nd))
            used_ids.add(nid)
            oe_taken += 1

    # Pad with free-exploration placeholders if still not enough
    while len(selected) < n:
        idx = len(selected)
        selected.append((
            f"free_{idx}",
            {
                "id": f"free_{idx}",
                "mode": "strategy_guided",
                "status": "open",
                "strategy_combination": [],
                "description": "自由探索方向，基于已有经验选择多样化策略",
                "parent_id": "root",
                "difficulty": 3,
                "depth": 1,
                "score": None,
                "solution_ref": None,
                "parent_code_ref": None,
                "children": [],
            }
        ))

    result = []
    for i, (nid, nd) in enumerate(selected):
        parent_id = nd.get("parent_id") or "root"
        parent_node = nodes.get(parent_id, {})
        parent_score = parent_node.get("score") or 1.0
        parent_solution_ref = parent_node.get("solution_ref")  # may be None
        # Resolve parent profiling one-liner
        pi = parent_node.get("profiling_insight")
        parent_profiling_one_liner = (
            pi.get("profiling_one_liner") if isinstance(pi, dict) else None
        )

        entry = {
            "parallel_index": i,
            "node_id": nid,
            "utility": round(compute_utility(nd, nodes, wm), 4),
            "mode": nd.get("mode", "strategy_guided"),
            "description": nd.get("description", ""),
            "strategy_combination": nd.get("strategy_combination", []),
            "parent_id": parent_id,
            "parent_score": parent_score,
            "parent_solution_ref": parent_solution_ref,
            "parent_profiling_one_liner": parent_profiling_one_liner,
            "difficulty": nd.get("difficulty", 3),
            "depth": nd.get("depth", 1),
        }
        result.append(entry)

    return result


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------

def validate(wm: dict) -> list[str]:
    """
    Validate world model invariants.

    Returns a list of error strings. Empty list means valid.

    Invariants checked:
      1. All parent_id values point to existing nodes
      2. best_score >= 1.0
      3. status values are legal
      4. All passed nodes have at least 1 open child (continuation invariant)
      5. stagnation_count and stagnation_count_vs_base are non-negative integers
      6. profiling_evidence field structure is valid
      7. optimization_type values are legal (if present)
      8. session identity anchor present and valid
      9. baseline_evidence (root-level) structure is valid (if present)
    """
    errors: list[str] = []
    nodes: dict = wm.get("decision_tree", {}).get("nodes", {})
    legal_statuses = {"open", "in_progress", "passed", "failed", "completed"}

    # Invariant 1: parent_id references
    for nid, nd in nodes.items():
        pid = nd.get("parent_id")
        if pid is not None and pid not in nodes:
            errors.append(
                f"Invariant 1 FAIL: node '{nid}' has parent_id='{pid}' "
                f"which does not exist in nodes"
            )

    # Invariant 2: best_score >= 1.0
    best_score = wm.get("best_score", 1.0)
    if best_score is None or best_score < 1.0:
        errors.append(
            f"Invariant 2 FAIL: best_score={best_score} is less than 1.0"
        )

    # Invariant 3: legal status values
    for nid, nd in nodes.items():
        status = nd.get("status")
        if status not in legal_statuses:
            errors.append(
                f"Invariant 3 FAIL: node '{nid}' has illegal status='{status}'"
            )

    # Invariant 4: passed nodes must have >= 1 open descendant (continuation)
    # (Only applies to non-root passed nodes that are not leaf of a termination)
    all_open_ids = {nid for nid, nd in nodes.items() if nd.get("status") == "open"}
    for nid, nd in nodes.items():
        if nd.get("status") == "passed":
            children_ids = nd.get("children", [])
            has_open_child = any(cid in all_open_ids for cid in children_ids)
            # Also check recursively for open descendants
            if children_ids and not has_open_child:
                # Check if any descendant is open (breadth-first)
                queue = list(children_ids)
                found = False
                visited: set[str] = set()
                while queue and not found:
                    cid = queue.pop(0)
                    if cid in visited:
                        continue
                    visited.add(cid)
                    cnode = nodes.get(cid, {})
                    if cnode.get("status") == "open":
                        found = True
                    queue.extend(cnode.get("children", []))
                if not found:
                    errors.append(
                        f"Invariant 4 FAIL: passed node '{nid}' has no open "
                        f"descendant (continuation invariant violated)"
                    )

    # Invariant 5: stagnation counters are non-negative integers
    for key in ("stagnation_count", "stagnation_count_vs_base"):
        val = wm.get(key)
        if not isinstance(val, int) or val < 0:
            errors.append(
                f"Invariant 5 FAIL: '{key}'={val!r} is not a non-negative integer"
            )

    # Invariant 6: profiling_evidence field structure validation
    for nid, nd in nodes.items():
        pe = nd.get("profiling_evidence")
        if pe is not None:
            if not isinstance(pe, dict):
                errors.append(
                    f"Invariant 6 FAIL: node '{nid}' has profiling_evidence "
                    f"that is not a dict: {type(pe).__name__}"
                )
            else:
                required_keys = {"bottleneck_type", "suggested_strategies"}
                missing = required_keys - set(pe.keys())
                if missing:
                    errors.append(
                        f"Invariant 6 FAIL: node '{nid}' profiling_evidence "
                        f"missing required keys: {missing}"
                    )
                legal_bottlenecks = {
                    "mte2_stall", "mte3_stall", "tiling_imbalance",
                    "scalar_loading", "scalar_compute",
                    "compute_bound", "near_optimal",
                    "no_overlap", "partial_overlap",
                    "undersize_transfer", "bus_contention", "icache_miss",
                }
                bt = pe.get("bottleneck_type")
                if bt and bt not in legal_bottlenecks:
                    errors.append(
                        f"Invariant 6 FAIL: node '{nid}' profiling_evidence "
                        f"has illegal bottleneck_type='{bt}'"
                    )

    # Invariant 7: optimization_type values are legal (if present)
    legal_opt_types = {"bandwidth", "tiling", "algorithm",
                        "register_opt", "vf_fusion", "instruction_sched"}
    for nid, nd in nodes.items():
        ot = nd.get("optimization_type")
        if ot is not None and ot not in legal_opt_types:
            errors.append(
                f"Invariant 7 FAIL: node '{nid}' has illegal "
                f"optimization_type='{ot}'"
            )

    # Invariant 8: session identity anchor present and valid
    sess = wm.get("session")
    if not isinstance(sess, dict):
        errors.append("Invariant 8 FAIL: session field missing or not a dict")
    else:
        for key in ("session_id", "start_time", "evo_dir", "op_name"):
            if not sess.get(key):
                errors.append(f"Invariant 8 FAIL: session.{key} is empty or missing")
        actual = sess.get("actual_rounds_completed")
        requested = sess.get("requested_rounds")
        if actual is not None and requested is not None and actual > requested:
            errors.append(
                f"Invariant 8 FAIL: actual_rounds_completed({actual}) > "
                f"requested_rounds({requested})"
            )

    # Invariant 9: baseline_evidence (root-level) structure validation
    be = wm.get("baseline_evidence")
    if be is not None:
        if not isinstance(be, dict):
            errors.append(
                f"Invariant 9 FAIL: baseline_evidence is not a dict: "
                f"{type(be).__name__}"
            )
        else:
            legal_bottlenecks = {
                "mte2_stall", "mte3_stall", "tiling_imbalance",
                "scalar_loading", "scalar_compute",
                "compute_bound", "near_optimal",
                "no_overlap", "partial_overlap",
                "undersize_transfer", "bus_contention", "icache_miss",
            }
            bt = be.get("bottleneck_type")
            if bt is not None and bt not in legal_bottlenecks:
                errors.append(
                    f"Invariant 9 FAIL: baseline_evidence has illegal "
                    f"bottleneck_type='{bt}'"
                )
            if "suggested_strategies" not in be:
                errors.append(
                    "Invariant 9 FAIL: baseline_evidence missing "
                    "'suggested_strategies' field"
                )

    return errors


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------

def _find_best_path(wm: dict) -> str:
    """Trace the best path from root to the highest-scoring passed node."""
    nodes: dict = wm.get("decision_tree", {}).get("nodes", {})

    # Find node with best score
    best_nid = None
    best_score = -1.0
    for nid, nd in nodes.items():
        sc = nd.get("score")
        if sc is not None and sc > best_score:
            best_score = sc
            best_nid = nid

    if best_nid is None:
        return "root(baseline)"

    # Trace path from best node up to root
    path = []
    current = best_nid
    while current is not None:
        nd = nodes.get(current, {})
        sc = nd.get("score")
        label = f"{current}({sc:.2f}x)" if sc is not None else current
        path.append(label)
        current = nd.get("parent_id")

    path.reverse()
    return " → ".join(path)


def _count_unexplored_root_branches(wm: dict) -> str:
    """Summarize root's direct children that are still open."""
    nodes: dict = wm.get("decision_tree", {}).get("nodes", {})
    root = nodes.get("root", {})
    open_children = []
    for cid in root.get("children", []):
        cnode = nodes.get(cid, {})
        if cnode.get("status") == "open":
            strats = "+".join(cnode.get("strategy_combination", []) or ["free"])
            open_children.append(f"{cid}({strats})")
    return ", ".join(open_children) if open_children else "(none)"


def summary(wm: dict, max_chars: int = 1200) -> str:
    """
    Generate a compact world model summary for sub-agent prompt injection.

    Output is plain text, ≤ max_chars characters.
    """
    nodes: dict = wm.get("decision_tree", {}).get("nodes", {})
    open_count = sum(1 for nd in nodes.values() if nd.get("status") == "open")
    passed_count = sum(1 for nd in nodes.values() if nd.get("status") == "passed")
    failed_count = sum(1 for nd in nodes.values() if nd.get("status") == "failed")

    stagnation_count = wm.get("stagnation_count", 0)
    stagnation_window = wm.get("stagnation_window", 2)
    best_score = wm.get("best_score", 1.0)
    kernel_summary = wm.get("kernel_summary", "N/A")
    open_questions = wm.get("open_questions", [])

    active_path = _find_best_path(wm)
    unexplored_roots = _count_unexplored_root_branches(wm)

    lines = [
        "[World Model Summary]",
        f"Best: {best_score}x | Stagnation: {stagnation_count}/{stagnation_window}",
        f"Open nodes: {open_count} | Passed: {passed_count} | Failed: {failed_count}",
        f"Active path: {active_path}",
        f"Unexplored root branches: {unexplored_roots}",
    ]

    if open_questions:
        lines.append("Key findings:")
        for q in open_questions[:5]:
            lines.append(f"  - {q}")

    text = "\n".join(lines)

    # Truncate to max_chars if needed
    if len(text) > max_chars:
        text = text[:max_chars - 3] + "..."

    return text


# ---------------------------------------------------------------------------
# refine — deterministic world model update after a round
# ---------------------------------------------------------------------------


def _safe_float(val, default: float = 0.0) -> float:
    """Coerce a value to float, returning default on failure."""
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


# Dynamic thresholds based on measurement quality
_THRESHOLDS = {
    "good":       {"improve": 1.05, "stagnation": 1.02},
    "acceptable": {"improve": 1.08, "stagnation": 1.03},
    "noisy":      {"improve": 1.15, "stagnation": 1.05},
}


def _generate_child_id(parent_id: str, nodes: dict, suffix: str = "") -> str:
    """Generate a unique child node ID."""
    base = f"{parent_id}_{suffix}" if suffix else f"{parent_id}_c"
    if base not in nodes:
        return base
    for i in range(1, 100):
        candidate = f"{base}{i}"
        if candidate not in nodes:
            return candidate
    return f"{parent_id}_{suffix}_{id(nodes) % 10000}"


def _make_child_node(
    child_id: str, parent: dict, nodes: dict,
    mode: str, description: str,
    strategy_combination: list = None,
    optimization_type: str = None,
    difficulty_delta: int = 0,
) -> dict:
    """Create a child node dict with inherited fields."""
    difficulty = min(4, max(1, parent.get("difficulty", 3) + difficulty_delta))
    if optimization_type is None:
        optimization_type = parent.get("optimization_type", "algorithm")
    return {
        "id": child_id,
        "mode": mode,
        "strategy_combination": strategy_combination or [],
        "description": description,
        "optimization_type": optimization_type,
        "difficulty": difficulty,
        "depth": parent.get("depth", 1) + 1,
        "parent_id": parent["id"],
        "status": "open",
        "score": None,
        "solution_ref": None,
        "children": [],
        "failure_type": None,
        "failure_reason": None,
        "retry_count": 0,
        "profiling_insight": None,
        "profiling_evidence": None,
    }


def refine(
    wm: dict,
    round_num: int,
    results_dir: str,
    parallel_map: dict,
    task_type: str = "vector",
    profiling_script: str = None,
) -> dict:
    """Deterministic world model update after a round of evolution.

    Handles: score update, profiling_insight extraction, bottleneck shift
    detection, child node generation, stagnation counting.

    Does NOT handle (requires LLM): failure diagnosis (impl_error vs
    strategy_infeasible), open_questions update (Analyze).

    Args:
        wm: world model dict (modified in-place and returned)
        round_num: current round number
        results_dir: directory containing parallel_0/, parallel_1/, etc.
        parallel_map: {"0": "n1", "1": "n2", "2": "x0"} parallel_index→node_id
        task_type: vector/cube/cv-mix/unknown
        profiling_script: path to analyze_profiling.py (optional, for CSV profiling)

    Returns:
        dict with keys: round_summary (str), pending_diagnosis (list),
        best_score_before (float), best_score_after (float)
    """
    import os
    import subprocess as sp

    nodes = wm.setdefault("decision_tree", {}).setdefault("nodes", {})

    # Stale in_progress cleanup: nodes marked in_progress by a previous SELECT
    # but not in the current parallel_map are stale (their sub-agent failed
    # silently). Reset to "open" so they can be re-selected in future rounds.
    current_node_ids = set(parallel_map.values())
    stale_reset = []
    for nid, nd in nodes.items():
        if nd.get("status") == "in_progress" and nid not in current_node_ids:
            nd["status"] = "open"
            stale_reset.append(nid)

    best_score_before = wm.get("best_score", 1.0) or 1.0
    round_best_speedup = 0.0
    round_passed = 0
    round_failed = 0
    round_total = len(parallel_map)
    pending_diagnosis = []
    summary_lines = []
    worst_quality = "good"
    quality_rank = {"good": 0, "acceptable": 1, "noisy": 2}

    for p_idx_str, node_id in parallel_map.items():
        p_idx = int(p_idx_str)
        node = nodes.get(node_id)
        if node is None:
            summary_lines.append(f"  p{p_idx} [{node_id}]: SKIP (node not found)")
            round_failed += 1
            continue

        # Read evaluation_results.json
        eval_path = os.path.join(results_dir, f"parallel_{p_idx}", "evaluation_results.json")
        if not os.path.isfile(eval_path):
            node["status"] = "failed"
            node["failure_reason"] = "evaluation_results.json not found"
            summary_lines.append(f"  p{p_idx} [{node_id}]: FAIL (no results)")
            round_failed += 1
            pending_diagnosis.append({
                "node_id": node_id, "parallel_index": p_idx,
                "reason": "evaluation_results.json not found",
            })
            continue

        with open(eval_path, "r", encoding="utf-8") as f:
            eval_result = json.load(f)

        comp = eval_result.get("comparison", {})
        compilation_ok = comp.get("compilation_success", False)
        precision_ok = comp.get("precision_passed", False)
        speedup = comp.get("speedup", 0.0) or 0.0
        mq = comp.get("measurement_quality", "good")
        if quality_rank.get(mq, 0) > quality_rank.get(worst_quality, 0):
            worst_quality = mq

        if compilation_ok and precision_ok and speedup > 0:
            # --- PASSED ---
            node["status"] = "passed"
            node["score"] = round(speedup, 4)
            node["solution_ref"] = f"round_{round_num}/parallel_{p_idx}"
            round_passed += 1
            if speedup > round_best_speedup:
                round_best_speedup = speedup

            # Extract profiling_insight from evaluation_results.json
            # ops-evo 写的是 {"evolved": {...}} 嵌套结构；lingxi-evo/lingxi-partial
            # 写的是扁平结构（直接把 pipeline / bottleneck 放顶层）。兼容两种。
            evolved = eval_result.get("evolved") or eval_result
            pipeline = evolved.get("pipeline", {})
            bottleneck = evolved.get("bottleneck", "unknown")

            # Try to infer bottleneck from pipeline if still "unknown"
            if bottleneck == "unknown" and pipeline:
                mte2 = _safe_float(pipeline.get("aiv_mte2_ratio")) or _safe_float(pipeline.get("mte2_pct"))
                vec = _safe_float(pipeline.get("aiv_vec_ratio")) or _safe_float(pipeline.get("vec_pct"))
                scalar = _safe_float(pipeline.get("aiv_scalar_ratio")) or _safe_float(pipeline.get("scalar_pct"))
                # Values may be ratios (0-1) or percentages (0-100); normalize
                if max(mte2, vec, scalar) <= 1.0 and max(mte2, vec, scalar) > 0:
                    mte2 *= 100
                    vec *= 100
                    scalar *= 100
                if mte2 > 50:
                    bottleneck = "memory_bound"
                elif vec > 60:
                    bottleneck = "compute_bound"
                elif scalar > 30:
                    bottleneck = "scalar_bound"
                else:
                    bottleneck = "balanced"

            # Build profiling_insight
            one_liner = f"speedup={speedup:.2f}x | bottleneck={bottleneck}"
            if pipeline:
                parts = [f"{k}={v}" for k, v in sorted(pipeline.items()) if v]
                if parts:
                    one_liner = " | ".join(parts[:4]) + f" | bottleneck={bottleneck}"

            node["profiling_insight"] = {
                "bottleneck": bottleneck,
                "pipeline": pipeline,
                "recommended_strategies": [],
                "profiling_one_liner": one_liner,
            }

            # --- A1: CSV-level profiling_evidence synthesis (main-loop hook) ---
            # Synthesize profiling_analysis from pipeline ratios and run it
            # through extract_profiling_evidence so the hard mapping table
            # (BOTTLENECK_STRATEGY_MAP) is consulted every round, not only
            # when deep-profiling is conditionally triggered.
            try:
                from evolution.world_model.profiling_evidence import (
                    synthesize_analysis_from_pipeline, extract_profiling_evidence,
                )
            except ImportError:
                _pr = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
                if _pr not in sys.path:
                    sys.path.insert(0, _pr)
                from evolution.world_model.profiling_evidence import (
                    synthesize_analysis_from_pipeline, extract_profiling_evidence,
                )

            pa = synthesize_analysis_from_pipeline(pipeline, bottleneck_hint=bottleneck)
            csv_evidence = extract_profiling_evidence({"profiling_analysis": pa}) if pa else None
            # Don't overwrite deep-profiling evidence (has pattern_type etc.)
            if csv_evidence and not node.get("profiling_evidence"):
                csv_evidence["_source"] = "csv_synth"
                node["profiling_evidence"] = csv_evidence
                # Also mirror into profiling_insight.recommended_strategies so
                # prompts that read that legacy field see the mapping too.
                sug = csv_evidence.get("suggested_strategies") or []
                node["profiling_insight"]["recommended_strategies"] = sug[:5]
            # --- end A1 ---

            # Bottleneck shift detection
            parent_id = node.get("parent_id", "root")
            parent_node = nodes.get(parent_id, {})
            parent_pi = parent_node.get("profiling_insight")
            if parent_pi and isinstance(parent_pi, dict):
                parent_bn = parent_pi.get("bottleneck")
                if parent_bn and parent_bn != bottleneck:
                    node["bottleneck_shift"] = f"{parent_bn} → {bottleneck}"

            # Dynamic threshold
            thresholds = _THRESHOLDS.get(mq, _THRESHOLDS["good"])
            improve_threshold = thresholds["improve"]
            parent_score = parent_node.get("score") or 1.0
            substantial = speedup > parent_score * improve_threshold

            # Generate child nodes
            mode = node.get("mode", "strategy_guided")
            if mode == "strategy_guided":
                if substantial:
                    for i in range(2):
                        cid = _generate_child_id(node_id, nodes, f"s{i+1}")
                        sc = list(node.get("strategy_combination", []))
                        desc = f"深度探索: 在 {node_id}({speedup:.2f}x) 基础上继续优化"
                        child = _make_child_node(cid, node, nodes, "strategy_guided", desc, sc)
                        nodes[cid] = child
                        node.setdefault("children", []).append(cid)
                else:
                    cid = _generate_child_id(node_id, nodes, "cont")
                    desc = f"延续探索: {node_id}({speedup:.2f}x) 无显著提升，尝试变化"
                    child = _make_child_node(cid, node, nodes, "strategy_guided", desc)
                    nodes[cid] = child
                    node.setdefault("children", []).append(cid)

            elif mode == "open_exploration":
                cid = _generate_child_id(node_id, nodes, "x1")
                if substantial:
                    desc = f"在开放探索({node_id}, {speedup:.2f}x)基础上继续自主推理"
                else:
                    desc = f"延续探索: {node_id}({speedup:.2f}x) 开放探索方向"
                child = _make_child_node(cid, node, nodes, "open_exploration", desc)
                nodes[cid] = child
                node.setdefault("children", []).append(cid)

            elif mode == "profiling_driven":
                if substantial:
                    cid = _generate_child_id(node_id, nodes, "x1")
                    desc = f"Profiling驱动({node_id}, {speedup:.2f}x)已解决瓶颈，继续探索"
                    child = _make_child_node(cid, node, nodes, "open_exploration", desc)
                    nodes[cid] = child
                    node.setdefault("children", []).append(cid)
                # No child if profiling_driven didn't improve

            # Profiling-driven child generation
            if bottleneck not in ("balanced", "unknown"):
                existing_pd = [
                    c for c in node.get("children", [])
                    if nodes.get(c, {}).get("mode") == "profiling_driven"
                ]
                if not existing_pd:
                    pd_id = _generate_child_id(node_id, nodes, "pd1")
                    pd_desc = f"[Profiling驱动] 针对 {bottleneck}: {one_liner}"
                    opt_type = "bandwidth" if "memory" in bottleneck else (
                        "tiling" if "compute" in bottleneck else "algorithm"
                    )
                    pd_child = _make_child_node(
                        pd_id, node, nodes, "profiling_driven", pd_desc,
                        optimization_type=opt_type,
                    )
                    nodes[pd_id] = pd_child
                    node.setdefault("children", []).append(pd_id)

            # --- A1 (cont'd): merge evidence into this round's new children ---
            # Any child just generated inherits parent's strategy_combination but
            # with anti-strategies filtered out and suggested strategies injected.
            # profiling_driven children start empty and get the suggested list.
            if node.get("profiling_evidence"):
                try:
                    from evolution.world_model.profiling_evidence import merge_strategies_with_evidence
                except ImportError:
                    _pr = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
                    if _pr not in sys.path:
                        sys.path.insert(0, _pr)
                    from evolution.world_model.profiling_evidence import merge_strategies_with_evidence

                ev = node["profiling_evidence"]
                for cid in node.get("children", []):
                    ch = nodes.get(cid)
                    if not ch or ch.get("status") != "open":
                        continue
                    # Skip children created before this round (have scores or were
                    # touched by prior rounds) — only touch nodes with empty
                    # solution_ref and no profiling data of their own.
                    if ch.get("score") is not None or ch.get("profiling_insight"):
                        continue
                    old = ch.get("strategy_combination", [])
                    ch["strategy_combination"] = merge_strategies_with_evidence(old, ev)
            # --- end A1 ---

            summary_lines.append(
                f"  p{p_idx} [{node_id}]: PASS {speedup:.2f}x "
                f"({bottleneck}, quality={mq})"
            )

        else:
            # --- FAILED ---
            node["status"] = "failed"
            error = comp.get("precision_message", eval_result.get("error", "unknown"))
            node["failure_reason"] = str(error)[:200]
            round_failed += 1

            pending_diagnosis.append({
                "node_id": node_id,
                "parallel_index": p_idx,
                "compilation_success": compilation_ok,
                "precision_passed": precision_ok,
                "error": str(error)[:200],
                "implementation_note_path": os.path.join(
                    results_dir, f"parallel_{p_idx}", "implementation_note.txt"
                ),
            })

            summary_lines.append(
                f"  p{p_idx} [{node_id}]: FAIL "
                f"(compile={'OK' if compilation_ok else 'FAIL'}, "
                f"precision={'OK' if precision_ok else 'FAIL'})"
            )

    # --- Top-level updates ---
    best_score_after = max(best_score_before, round_best_speedup)
    wm["best_score"] = round(best_score_after, 4)

    # Update session anchor: actual_rounds_completed
    sess = wm.setdefault("session", {})
    sess["actual_rounds_completed"] = max(
        sess.get("actual_rounds_completed", 0), round_num
    )

    # Stagnation counting
    stag_threshold = _THRESHOLDS.get(worst_quality, _THRESHOLDS["good"])["stagnation"]
    if round_best_speedup <= best_score_before * stag_threshold:
        wm["stagnation_count"] = wm.get("stagnation_count", 0) + 1
    else:
        wm["stagnation_count"] = 0

    # Stagnation vs base (did any variant beat its parent?)
    any_beat_parent = False
    for p_idx_str, node_id in parallel_map.items():
        node = nodes.get(node_id, {})
        if node.get("status") == "passed" and node.get("score"):
            parent_id = node.get("parent_id", "root")
            parent_score = nodes.get(parent_id, {}).get("score") or 1.0
            if node["score"] > parent_score:
                any_beat_parent = True
                break
    if any_beat_parent:
        wm["stagnation_count_vs_base"] = 0
    else:
        wm["stagnation_count_vs_base"] = wm.get("stagnation_count_vs_base", 0) + 1

    # Build round summary
    stale_line = f"  [cleanup] Reset stale in_progress to open: {stale_reset}\n" if stale_reset else ""
    round_summary = (
        stale_line
        + f"Round {round_num}: {round_passed}/{round_total} passed, "
        f"{round_failed}/{round_total} failed, "
        f"best={round_best_speedup:.2f}x, "
        f"global_best={best_score_after:.2f}x, "
        f"stagnation={wm.get('stagnation_count', 0)}/{wm.get('stagnation_count_vs_base', 0)}\n"
        + "\n".join(summary_lines)
    )

    return {
        "round_summary": round_summary,
        "pending_diagnosis": pending_diagnosis,
        "best_score_before": best_score_before,
        "best_score_after": best_score_after,
        "worst_quality": worst_quality,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def cmd_select(args: argparse.Namespace) -> None:
    with open(args.path, "r", encoding="utf-8") as f:
        wm = json.load(f)

    # Drift breaker — if state.drift_status == replan_required, force
    # at least ⌈n/2⌉ open_exploration slots so the search escapes local minima.
    force_oe = None
    drift = _read_state_field_safe(args.path, "drift_status")
    if drift == "replan_required":
        force_oe = max(1, -(-args.n // 2))  # ceil(n/2)
        print(
            f"  [DRIFT] state.drift_status=replan_required → force "
            f"open_exploration_min={force_oe} (of n={args.n})",
            file=sys.stderr,
        )

    result = select_nodes(wm, args.n, force_open_exploration_min=force_oe)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    # Re-infer state from filesystem (replaces older single-field write helper)
    _maybe_infer_state(args.path)
    # Drift signal is one-shot; auto-clear after SELECT consumed it
    if drift == "replan_required":
        _maybe_clear_drift(args.path)


def cmd_validate(args: argparse.Namespace) -> None:
    with open(args.path, "r", encoding="utf-8") as f:
        wm = json.load(f)
    errors = validate(wm)
    if errors:
        print("Validation FAILED:")
        for e in errors:
            print(f"  {e}")
        sys.exit(1)
    else:
        print("Validation PASSED: all invariants satisfied.")


def cmd_summary(args: argparse.Namespace) -> None:
    with open(args.path, "r", encoding="utf-8") as f:
        wm = json.load(f)
    print(summary(wm, max_chars=args.max_chars))


def cmd_deep_profiling(args: argparse.Namespace) -> None:
    """Run deep profiling on a node and write results to world_model.json."""
    import os

    # Step 1: Run run_deep_profiling.py via subprocess
    script_path = os.path.join(
        os.path.dirname(__file__), "..", "..",
        ".claude", "skills", "ascendc-profiling-analysis", "scripts",
        "run_deep_profiling.py",
    )
    script_path = os.path.abspath(script_path)
    result_path = os.path.join(args.work_dir, "deep_profiling_result.json")

    cmd = [
        sys.executable, script_path,
        "--work-dir", args.work_dir,
        "--op-name", args.op_name,
        "--output", result_path,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    if proc.returncode != 0:
        print(f"deep-profiling: run_deep_profiling.py failed:\n{proc.stderr}", file=sys.stderr)
        sys.exit(1)

    # Step 2: Read profiling result
    with open(result_path, "r", encoding="utf-8") as f:
        prof_result = json.load(f)

    evidence = prof_result.get("profiling_evidence")
    if not evidence:
        print("deep-profiling: no profiling_evidence produced", file=sys.stderr)
        sys.exit(1)

    # Step 3: Write evidence into world_model.json node
    with open(args.wm_path, "r", encoding="utf-8") as f:
        wm = json.load(f)

    nodes = wm.get("decision_tree", {}).get("nodes", {})
    node = nodes.get(args.node_id)
    if not node:
        print(f"deep-profiling: node '{args.node_id}' not found", file=sys.stderr)
        sys.exit(1)

    node["profiling_evidence"] = evidence

    # Step 4: Optionally merge strategies into open children
    if args.merge_children:
        try:
            from evolution.world_model.profiling_evidence import merge_strategies_with_evidence
        except ImportError:
            _proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            sys.path.insert(0, _proj_root)
            from evolution.world_model.profiling_evidence import merge_strategies_with_evidence

        updated_children = []
        for cid in node.get("children", []):
            child = nodes.get(cid)
            if child and child.get("status") == "open":
                old_strats = child.get("strategy_combination", [])
                child["strategy_combination"] = merge_strategies_with_evidence(old_strats, evidence)
                updated_children.append(cid)
        if updated_children:
            print(f"deep-profiling: updated strategies for children: {updated_children}")

    # Step 5: Write back
    with open(args.wm_path, "w", encoding="utf-8") as f:
        json.dump(wm, f, ensure_ascii=False, indent=2)

    print(f"deep-profiling: wrote profiling_evidence to node '{args.node_id}'")
    print(f"  bottleneck_type: {evidence.get('bottleneck_type')}")
    print(f"  suggested_strategies: {evidence.get('suggested_strategies')}")


def cmd_attach_baseline_evidence(args: argparse.Namespace) -> None:
    """Attach root-level baseline_evidence to world_model.json from
    baseline_evaluation.json's pipeline/bottleneck fields.

    Called once per evolution session, right after Phase 3.6 baseline profiling
    produces baseline_evaluation.json. Subsequent SELECT rounds then use
    wm["baseline_evidence"] to penalize nodes misaligned with the baseline
    bottleneck (compute_utility w_baseline_mismatch) and inject the Baseline
    row into partial-agent prompts.

    Gracefully no-ops (writes baseline_evidence=None) when the baseline lacks
    pipeline data — all downstream consumers handle None by falling back.
    """
    import os

    try:
        from evolution.world_model.profiling_evidence import (
            synthesize_analysis_from_pipeline, extract_profiling_evidence,
        )
    except ImportError:
        _proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        sys.path.insert(0, _proj_root)
        from evolution.world_model.profiling_evidence import (
            synthesize_analysis_from_pipeline, extract_profiling_evidence,
        )

    with open(args.baseline_eval, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    # baseline_evaluation.json written by evaluate_ops_direct.py contains a
    # top-level "baseline" dict with pipeline/bottleneck; some variants nest
    # under "evolved" when baseline is the only run. Try both.
    baseline = eval_data.get("baseline") or eval_data.get("evolved") or {}
    pipeline = baseline.get("pipeline") or {}
    bottleneck_hint = baseline.get("bottleneck")

    with open(args.wm_path, "r", encoding="utf-8") as f:
        wm = json.load(f)

    if not pipeline:
        wm["baseline_evidence"] = None
        with open(args.wm_path, "w", encoding="utf-8") as f:
            json.dump(wm, f, ensure_ascii=False, indent=2)
        print("attach-baseline-evidence: no pipeline data in baseline; "
              "wrote baseline_evidence=null", file=sys.stderr)
        return

    profiling_analysis = synthesize_analysis_from_pipeline(pipeline, bottleneck_hint)
    evidence = extract_profiling_evidence({"profiling_analysis": profiling_analysis})

    if not evidence:
        wm["baseline_evidence"] = None
        print("attach-baseline-evidence: evidence synthesis returned None; "
              "wrote baseline_evidence=null", file=sys.stderr)
    else:
        wm["baseline_evidence"] = evidence
        print(f"attach-baseline-evidence: baseline bottleneck_type="
              f"{evidence.get('bottleneck_type')}, "
              f"suggested={evidence.get('suggested_strategies')[:5]}, "
              f"anti={evidence.get('anti_strategies')}")

    with open(args.wm_path, "w", encoding="utf-8") as f:
        json.dump(wm, f, ensure_ascii=False, indent=2)


def cmd_refine(args: argparse.Namespace) -> None:
    """Deterministic world model update after a round."""
    import os

    with open(args.wm_path, "r", encoding="utf-8") as f:
        wm = json.load(f)

    parallel_map = json.loads(args.parallel_map)

    result = refine(
        wm=wm,
        round_num=args.round,
        results_dir=args.results_dir,
        parallel_map=parallel_map,
        task_type=args.task_type,
    )

    # Write updated world_model.json
    with open(args.wm_path, "w", encoding="utf-8") as f:
        json.dump(wm, f, ensure_ascii=False, indent=2)

    # A4: soft-demote open descendants of passed-but-stale-direction nodes.
    # Runs BEFORE soft_prune so that demoted nodes that happen to be under a
    # sealed ancestor still get hard-pruned correctly.
    nodes = wm.get("decision_tree", {}).get("nodes", {})
    demoted = soft_demote_stale_directions(
        wm, quality=result.get("worst_quality", "good"), round_num=args.round
    )
    if demoted:
        with open(args.wm_path, "w", encoding="utf-8") as f:
            json.dump(wm, f, ensure_ascii=False, indent=2)
        print(f"refine: soft-demoted {len(demoted)} stale-direction descendants: {demoted}")

    # Soft prune dead branches after refine
    pruned = soft_prune_dead_branches(nodes)
    if pruned:
        # Write again with pruned nodes
        with open(args.wm_path, "w", encoding="utf-8") as f:
            json.dump(wm, f, ensure_ascii=False, indent=2)
        print(f"refine: soft-pruned {len(pruned)} orphaned open nodes: {pruned}")

    # Write pending_diagnosis.json if there are failed nodes
    if result["pending_diagnosis"]:
        diag_path = os.path.join(args.results_dir, "pending_diagnosis.json")
        with open(diag_path, "w", encoding="utf-8") as f:
            json.dump(result["pending_diagnosis"], f, ensure_ascii=False, indent=2)
        print(f"Pending diagnosis written to: {diag_path}")

    # Print round summary
    print(result["round_summary"])
    # Re-infer state from filesystem after refine writes wm.session.actual_rounds_completed
    _maybe_infer_state(args.wm_path)
    # Drift circuit breaker — auto-set state.drift_status based on
    # post-refine stagnation counters. Noops if state.json absent.
    _maybe_update_drift_status(args.wm_path, wm)
    # R9: detect missing profiling artifacts and gate next round
    _maybe_mark_profiling_skipped(args.wm_path, args.results_dir, parallel_map)
    # R10 (warn-only): flag if ≥50% partials failed precision
    _maybe_warn_precision_failures(parallel_map, args.results_dir)


def cmd_session(args: argparse.Namespace) -> None:
    """Write or update session identity anchor in world_model.json.

    Called once at the start of evolution (step 3 init) to pin the session
    to a unique directory and prevent later steps from dynamically
    discovering unrelated historical directories.
    """
    import os
    import datetime as dt

    if os.path.isfile(args.wm_path):
        with open(args.wm_path, "r", encoding="utf-8") as f:
            wm = json.load(f)
    else:
        wm = {}

    now = dt.datetime.now(dt.timezone(dt.timedelta(hours=8)))
    evo_dir = os.path.abspath(args.evo_dir)

    wm["session"] = {
        "session_id": args.session_id,
        "start_time": now.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "requested_rounds": args.requested_rounds,
        "actual_rounds_completed": 0,
        "evo_dir": evo_dir,
        "op_name": args.op_name,
    }

    os.makedirs(os.path.dirname(args.wm_path), exist_ok=True)
    with open(args.wm_path, "w", encoding="utf-8") as f:
        json.dump(wm, f, ensure_ascii=False, indent=2)

    print(f"session: anchored to {evo_dir}")
    print(f"  session_id={args.session_id}, requested_rounds={args.requested_rounds}")
    # Re-infer state from filesystem; session step initializes wm.json
    # which lets the next infer call resolve stage from "shared_prep" to "wm_init"
    _maybe_infer_state(args.wm_path)


def cmd_session_verify(args: argparse.Namespace) -> None:
    """Verify that a directory belongs to the current session.

    Exits with non-zero code if the evo_dir does NOT match the session anchor.
    Used in step 5 (final report) to prevent attribution errors.
    """
    import os

    if not os.path.isfile(args.wm_path):
        print(f"session-verify: FATAL — world_model.json not found at {args.wm_path}",
              file=sys.stderr)
        sys.exit(1)

    with open(args.wm_path, "r", encoding="utf-8") as f:
        wm = json.load(f)

    sess = wm.get("session")
    if not isinstance(sess, dict):
        print("session-verify: FATAL — session field missing in world_model.json",
              file=sys.stderr)
        sys.exit(2)

    expected_dir = os.path.abspath(sess.get("evo_dir", ""))
    actual_dir = os.path.abspath(args.evo_dir)

    if expected_dir != actual_dir:
        print(
            f"session-verify: FATAL — directory mismatch\n"
            f"  expected (from session anchor): {expected_dir}\n"
            f"  actual (provided):              {actual_dir}\n"
            f"  Do NOT dynamically discover directories. Use the evo_dir "
            f"from session.anchor only.",
            file=sys.stderr,
        )
        sys.exit(3)

    actual_rounds = sess.get("actual_rounds_completed", 0)
    requested_rounds = sess.get("requested_rounds", 0)
    if actual_rounds < requested_rounds:
        print(
            f"[WARNING] session-verify: WARNING — only {actual_rounds}/{requested_rounds} "
            f"rounds completed. Report must clearly state this.",
            file=sys.stderr,
        )

    print(f"session-verify: OK — {actual_dir} matches session anchor")
    print(f"  rounds: {actual_rounds}/{requested_rounds} completed")


def cmd_diagnose(args: argparse.Namespace) -> None:
    """Write failure/direction diagnosis for a node (called by agent after LLM reasoning).

    Supports two semantic modes depending on the target node's status:

      1. status=="failed" (legacy):
           - impl_error → generate fix child node, difficulty++
           - strategy_infeasible / retry>=2 → seal node (difficulty=5)

      2. status=="passed" + failure_type=="strategy_infeasible" (new: A6):
           Direction-level seal. The node itself stays passed (it did run
           successfully), but the agent has determined via semantic review
           (e.g. comparing baseline bottleneck vs evolved bottleneck, or
           observing a sibling variant along a different direction produced
           a substantially better speedup) that continuing down this
           direction is unlikely to yield further gains. We set
           direction_sealed=True and difficulty=5; soft_prune_dead_branches
           then demotes all open descendants. No fix child is generated
           because the direction has been disproven, not broken.

    Other status + failure_type combinations write failure_type/reason without
    sealing or generating children (informational).
    """
    with open(args.wm_path, "r", encoding="utf-8") as f:
        wm = json.load(f)

    nodes = wm.get("decision_tree", {}).get("nodes", {})
    node = nodes.get(args.node_id)
    if not node:
        print(f"ERROR: node '{args.node_id}' not found", file=sys.stderr)
        sys.exit(1)

    node["failure_type"] = args.failure_type
    node["failure_reason"] = args.failure_reason
    status = node.get("status")

    if status == "passed" and args.failure_type == "strategy_infeasible":
        # A6: direction-level seal on a passed node.
        node["direction_sealed"] = True
        node["difficulty"] = 5
        print(f"diagnose: passed+strategy_infeasible → direction-sealed "
              f"node '{args.node_id}' (difficulty=5, status stays passed)")
    elif args.failure_type == "impl_error" and node.get("retry_count", 0) < 2:
        # Generate fix child node
        node["difficulty"] = min(4, node.get("difficulty", 3) + 1)
        retry = node.get("retry_count", 0) + 1
        fix_id = f"{args.node_id}_fix{retry}"
        mode = node.get("mode", "strategy_guided")
        fix_child = {
            "id": fix_id,
            "mode": mode,
            "strategy_combination": list(node.get("strategy_combination", [])),
            "description": f"[修复实现] {args.failure_reason}",
            "optimization_type": node.get("optimization_type", "algorithm"),
            "difficulty": node["difficulty"],
            "depth": node.get("depth", 1) + 1,
            "parent_id": args.node_id,
            "status": "open",
            "score": None,
            "solution_ref": None,
            "children": [],
            "failure_type": None,
            "failure_reason": None,
            "retry_count": retry,
            "profiling_insight": None,
            "profiling_evidence": None,
        }
        nodes[fix_id] = fix_child
        node.setdefault("children", []).append(fix_id)
        print(f"diagnose: impl_error → generated fix child '{fix_id}'")
    elif args.failure_type == "strategy_infeasible" or node.get("retry_count", 0) >= 2:
        node["difficulty"] = 5
        print(f"diagnose: {args.failure_type} → sealed node (difficulty=5)")
    else:
        print(f"diagnose: wrote failure_type={args.failure_type}")

    # Run soft prune after diagnosis — a newly sealed node may orphan open descendants
    pruned = soft_prune_dead_branches(nodes)
    if pruned:
        print(f"diagnose: soft-pruned {len(pruned)} orphaned open nodes: {pruned}")

    with open(args.wm_path, "w", encoding="utf-8") as f:
        json.dump(wm, f, ensure_ascii=False, indent=2)


def cmd_prune(args: argparse.Namespace) -> None:
    """Standalone soft-prune: set difficulty=5 on open nodes under sealed ancestors."""
    with open(args.path, "r", encoding="utf-8") as f:
        wm = json.load(f)

    nodes = wm.get("decision_tree", {}).get("nodes", {})
    pruned = soft_prune_dead_branches(nodes)

    if pruned:
        with open(args.path, "w", encoding="utf-8") as f:
            json.dump(wm, f, ensure_ascii=False, indent=2)
        print(f"prune: soft-pruned {len(pruned)} nodes: {pruned}")
    else:
        print("prune: no orphaned open nodes found")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="World Model CLI operations for LINGXI evolution."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # select
    p_select = subparsers.add_parser("select", help="Select top-N open nodes")
    p_select.add_argument("--path", required=True, help="Path to world_model.json")
    p_select.add_argument("--n", type=int, required=True, help="Number of nodes to select")
    p_select.set_defaults(func=cmd_select)

    # validate
    p_validate = subparsers.add_parser("validate", help="Validate invariants")
    p_validate.add_argument("--path", required=True, help="Path to world_model.json")
    p_validate.set_defaults(func=cmd_validate)

    # summary
    p_summary = subparsers.add_parser("summary", help="Compact summary for prompt injection")
    p_summary.add_argument("--path", required=True, help="Path to world_model.json")
    p_summary.add_argument(
        "--max-chars", type=int, default=1200,
        help="Maximum characters in output (default: 1200)"
    )
    p_summary.set_defaults(func=cmd_summary)

    # deep-profiling
    p_dp = subparsers.add_parser("deep-profiling", help="Run deep profiling and write to world model")
    p_dp.add_argument("--wm-path", required=True, help="Path to world_model.json")
    p_dp.add_argument("--node-id", required=True, help="Node ID to write profiling_evidence to")
    p_dp.add_argument("--work-dir", required=True, help="Operator work directory (with build artifacts)")
    p_dp.add_argument("--op-name", required=True, help="Operator name")
    p_dp.add_argument("--merge-children", action="store_true",
                       help="Merge evidence strategies into open child nodes")
    p_dp.set_defaults(func=cmd_deep_profiling)

    # attach-baseline-evidence
    p_abe = subparsers.add_parser(
        "attach-baseline-evidence",
        help="Write root-level baseline_evidence from baseline_evaluation.json",
    )
    p_abe.add_argument("--wm-path", required=True, help="Path to world_model.json")
    p_abe.add_argument(
        "--baseline-eval", required=True,
        help="Path to baseline_evaluation.json (from evaluate_ops_direct.py)",
    )
    p_abe.set_defaults(func=cmd_attach_baseline_evidence)

    # refine
    p_refine = subparsers.add_parser(
        "refine", help="Deterministic world model update after a round"
    )
    p_refine.add_argument("--wm-path", required=True, help="Path to world_model.json")
    p_refine.add_argument("--round", type=int, required=True, help="Round number")
    p_refine.add_argument("--results-dir", required=True,
                          help="Directory containing parallel_0/, parallel_1/, etc.")
    p_refine.add_argument("--parallel-map", required=True,
                          help='JSON: {"0":"n1","1":"n2","2":"x0"}')
    p_refine.add_argument("--task-type", default="vector",
                          choices=["vector", "cube", "cv-mix", "unknown"])
    p_refine.set_defaults(func=cmd_refine)

    # diagnose
    p_diag = subparsers.add_parser(
        "diagnose", help="Write failure diagnosis for a node"
    )
    p_diag.add_argument("--wm-path", required=True, help="Path to world_model.json")
    p_diag.add_argument("--node-id", required=True, help="Failed node ID")
    p_diag.add_argument("--failure-type", required=True,
                        choices=["impl_error", "strategy_infeasible"])
    p_diag.add_argument("--failure-reason", required=True, help="One-line reason")
    p_diag.set_defaults(func=cmd_diagnose)

    # prune
    p_prune = subparsers.add_parser(
        "prune", help="Soft-prune open nodes under sealed ancestors"
    )
    p_prune.add_argument("--path", required=True, help="Path to world_model.json")
    p_prune.set_defaults(func=cmd_prune)

    # session — write session identity anchor
    p_sess = subparsers.add_parser(
        "session", help="Write session identity anchor into world_model.json"
    )
    p_sess.add_argument("--wm-path", required=True, help="Path to world_model.json")
    p_sess.add_argument("--session-id", required=True, help="Unique session ID")
    p_sess.add_argument("--evo-dir", required=True, help="Absolute path to evolution output dir")
    p_sess.add_argument("--op-name", required=True, help="Operator name")
    p_sess.add_argument("--requested-rounds", type=int, default=5, help="Requested max rounds")
    p_sess.set_defaults(func=cmd_session)

    # session-verify — verify directory matches session anchor
    p_sv = subparsers.add_parser(
        "session-verify", help="Verify evo_dir matches session anchor"
    )
    p_sv.add_argument("--wm-path", required=True, help="Path to world_model.json")
    p_sv.add_argument("--evo-dir", required=True, help="Directory to verify")
    p_sv.set_defaults(func=cmd_session_verify)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
