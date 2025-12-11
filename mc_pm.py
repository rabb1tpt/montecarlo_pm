#!/usr/bin/env python3
"""
Monte Carlo project timeline simulator (JSON + dependencies + risk events).

Features:
- Tasks with triangular durations (optimistic / most_likely / pessimistic)
- Dependencies via integer task IDs (DAG)
- Arbitrary integer IDs allowed (internally reindexed to 0..n-1)
- Risk events that add days to selected tasks with some probability
- Critical-path frequency analysis
- CLI: --project, --sims, --no-plot, --save-plot, --seed
"""

import argparse
import json
import os
from collections import deque

import numpy as np
import matplotlib.pyplot as plt


# ===== 1. Utility: topological sort from predecessor index lists =====

def topo_sort_from_preds(preds_by_idx):
    """
    Topologically sort tasks given predecessors as index lists.

    preds_by_idx: list of lists
        preds_by_idx[i] = list of task indices that must finish before task i starts.

    Returns:
        order: list of task indices in topological order.

    Raises:
        ValueError if a cycle is detected.
    """
    n = len(preds_by_idx)
    indegree = [0] * n
    graph = [[] for _ in range(n)]

    # Build graph (pred -> task) and indegree
    for i, preds in enumerate(preds_by_idx):
        for p in preds:
            graph[p].append(i)
            indegree[i] += 1

    # Kahn's algorithm
    q = deque([i for i in range(n) if indegree[i] == 0])
    order = []

    while q:
        u = q.popleft()
        order.append(u)
        for v in graph[u]:
            indegree[v] -= 1
            if indegree[v] == 0:
                q.append(v)

    if len(order) != n:
        raise ValueError("Cycle detected in task dependencies; project is not a DAG.")

    return order


# ===== 2. Simulation with dependencies + risk events =====

def simulate_project(tasks, risk_events, n_sim=1000, seed=None):
    """
    Run Monte Carlo simulation.

    Args:
        tasks: list of task dicts as loaded from JSON.
        risk_events: list of risk event dicts as loaded from JSON.
        n_sim: number of simulations.
        seed: random seed (int or None).

    Returns:
        project_duration: np.ndarray shape (n_sim,)
        start_times: np.ndarray shape (n_sim, n_tasks)
        finish_times: np.ndarray shape (n_sim, n_tasks)
        tasks_sorted: list of tasks sorted by id (used for reporting)
        preds_by_idx: list of predecessor index lists (for critical path analysis)
    """
    if seed is not None:
        np.random.seed(seed)

    # Normalize tasks: sort by id, build id -> index mapping
    tasks_sorted = sorted(tasks, key=lambda t: t["id"])
    id_to_idx = {t["id"]: i for i, t in enumerate(tasks_sorted)}
    n_tasks = len(tasks_sorted)

    # Build predecessor indices per task
    preds_by_idx = []
    for t in tasks_sorted:
        pred_ids = t.get("predecessors", [])
        pred_indices = [id_to_idx[pid] for pid in pred_ids]
        preds_by_idx.append(pred_indices)

    # Topological order based on index-based predecessors
    order = topo_sort_from_preds(preds_by_idx)

    # Pre-sample base durations: triangular distributions
    durations = np.zeros((n_sim, n_tasks))
    for idx, t in enumerate(tasks_sorted):
        o = float(t["optimistic"])
        m = float(t["most_likely"])
        p = float(t["pessimistic"])

        if not (o <= m <= p):
            raise ValueError(
                f"Task id {t['id']} ('{t.get('name', '')}'): "
                f"optimistic <= most_likely <= pessimistic violated: {o}, {m}, {p}"
            )

        durations[:, idx] = np.random.triangular(o, m, p, size=n_sim)

    # Apply risk events (additive delays in days)
    for event in risk_events:
        prob = float(event["probability"])
        impact_days = float(event["impact_days"])
        task_ids = event["tasks"]

        if prob <= 0 or impact_days == 0 or not task_ids:
            continue

        # For each simulation, does this event trigger?
        triggers = (np.random.rand(n_sim) < prob).astype(float)  # shape (n_sim,)

        # Map event's task IDs to indices
        task_indices = [id_to_idx[tid] for tid in task_ids]

        # Add impact to those tasks when event triggers
        for tidx in task_indices:
            durations[:, tidx] += triggers * impact_days

    # Now propagate schedule through DAG
    start_times = np.zeros((n_sim, n_tasks))
    finish_times = np.zeros((n_sim, n_tasks))

    for idx in order:
        preds = preds_by_idx[idx]
        if preds:
            # Start when all predecessors are done
            start_times[:, idx] = finish_times[:, preds].max(axis=1)
        else:
            # No predecessors: start at time 0
            start_times[:, idx] = 0.0

        finish_times[:, idx] = start_times[:, idx] + durations[:, idx]

    project_duration = finish_times.max(axis=1)
    return project_duration, start_times, finish_times, tasks_sorted, preds_by_idx


# ===== 3. Critical-path frequency analysis =====

def critical_path_frequency(preds_by_idx, start_times, finish_times):
    """
    For each simulation, reconstruct one critical path (backwards from last-finishing task)
    and count how often each task is on the critical path.

    Args:
        preds_by_idx: list of predecessor index lists.
        start_times: np.ndarray (n_sim, n_tasks).
        finish_times: np.ndarray (n_sim, n_tasks).

    Returns:
        cp_counts: np.ndarray shape (n_tasks,) with counts per task index.
    """
    n_sim, n_tasks = finish_times.shape
    cp_counts = np.zeros(n_tasks, dtype=int)

    tol = 1e-6

    for s in range(n_sim):
        # Last finishing task for this simulation
        current = int(np.argmax(finish_times[s, :]))
        visited = set([current])

        # Backtrack until a task with no predecessors
        while preds_by_idx[current]:
            preds = preds_by_idx[current]
            start_current = start_times[s, current]
            pred_finish_times = finish_times[s, preds]

            # Prefer predecessors whose finish_time == start_current (within tolerance)
            close_mask = np.isclose(pred_finish_times, start_current, atol=tol)

            if close_mask.any():
                local_idx = np.where(close_mask)[0][0]
                next_task = preds[local_idx]
            else:
                # Fallback: choose predecessor with max finish time
                local_idx = int(np.argmax(pred_finish_times))
                next_task = preds[local_idx]

            if next_task in visited:
                # Shouldn't happen in a DAG, but guard anyway.
                break

            visited.add(next_task)
            current = next_task

        # Update counts for tasks on this critical path
        for tid in visited:
            cp_counts[tid] += 1

    return cp_counts


# ===== 4. Reporting & plotting =====

def print_summary(project_duration, target_duration=None):
    mean = project_duration.mean()
    median = np.percentile(project_duration, 50)
    p80, p90, p95 = np.percentile(project_duration, [80, 90, 95])

    print("\n=== Project Duration Summary (Monte Carlo) ===")
    print(f"Simulations run    : {len(project_duration)}")
    print(f"Mean duration      : {mean:6.2f}")
    print(f"Median (50th pct)  : {median:6.2f}")
    print(f"80th percentile    : {p80:6.2f}")
    print(f"90th percentile    : {p90:6.2f}")
    print(f"95th percentile    : {p95:6.2f}")

    if target_duration is not None:
        prob_meet = (project_duration <= target_duration).mean()
        print(f"\nTarget duration    : {target_duration:.2f}")
        print(f"P(finish ≤ target) : {prob_meet * 100:5.1f}%")


def print_critical_path_stats(tasks_sorted, cp_counts, n_sim):
    print("\n=== Critical Path Frequency ===")
    print("Task ID | %-on-critical-path | Name")
    print("---------------------------------------------")

    indices = np.argsort(-cp_counts)  # sort descending by frequency
    for idx in indices:
        freq = cp_counts[idx] / n_sim * 100.0
        name = tasks_sorted[idx].get("name", f"Task {tasks_sorted[idx]['id']}")
        tid = tasks_sorted[idx]["id"]
        print(f"{tid:6d} | {freq:18.1f}% | {name}")


def plot_histogram(project_duration, target_duration=None, save_path=None):
    """
    Plot histogram of project durations.
    - If save_path is provided, save to file (PNG, etc.) and do NOT show GUI.
    - If save_path is None, show interactive window (if backend supports it).
    """
    plt.figure()
    plt.hist(project_duration, bins=30, edgecolor="black")
    plt.xlabel("Total project duration")
    plt.ylabel("Frequency (simulations)")
    plt.title("Monte Carlo Simulation of Project Duration")

    if target_duration is not None:
        plt.axvline(target_duration, linestyle="--")
        ymax = plt.ylim()[1]
        plt.text(
            target_duration,
            ymax * 0.9,
            f"Target = {target_duration:.1f}",
            rotation=90,
            va="top"
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Histogram saved to {save_path}")
        plt.close()
    else:
        plt.show()


# ===== 5. JSON loading & validation =====

def load_project(path):
    """
    Load and validate a project JSON file.

    Returns:
        tasks, risk_events, target_duration
    """
    # --- 1. Check path validity ---
    if path is None or not isinstance(path, str) or path.strip() == "":
        raise ValueError("Error: No project file path was provided.")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Error: Project file not found at: {path}")

    if not os.path.isfile(path):
        raise ValueError(f"Error: Path exists but is not a file: {path}")

    # --- 2. Try loading and parsing JSON ---
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Error: Failed to parse JSON in {path}. Details: {e}")
    except Exception as e:
        raise RuntimeError(f"Error: Unable to read file {path}. Details: {e}")

    # --- 3. Validate top-level structure ---
    if "tasks" not in data:
        raise ValueError(f"Error: JSON file '{path}' is missing required field: 'tasks'")

    tasks = data["tasks"]
    if not isinstance(tasks, list):
        raise ValueError("Error: 'tasks' field must be a list.")

    risk_events = data.get("risk_events", [])
    if not isinstance(risk_events, list):
        raise ValueError("Error: 'risk_events' field must be a list if present.")

    target_duration = data.get("target_duration", None)
    if target_duration is not None and not isinstance(target_duration, (int, float)):
        raise ValueError("Error: 'target_duration' must be a number if present.")

    # --- 4. Basic ID validation: existence, type, uniqueness ---
    ids = []
    for t in tasks:
        if "id" not in t:
            raise ValueError("Error: One or more tasks are missing the 'id' field.")
        tid = t["id"]
        if not isinstance(tid, int):
            raise ValueError(f"Error: Task id must be an integer. Got: {tid!r}")
        ids.append(tid)

    if len(set(ids)) != len(ids):
        raise ValueError(f"Error: Task IDs must be unique. Found duplicates in: {ids}")

    id_set = set(ids)

    # Validate predecessors
    for t in tasks:
        preds = t.get("predecessors", [])
        if preds is None:
            preds = []
            t["predecessors"] = preds
        if not isinstance(preds, list):
            raise ValueError(
                f"Error: 'predecessors' for task id {t['id']} must be a list of IDs."
            )
        for pid in preds:
            if not isinstance(pid, int):
                raise ValueError(
                    f"Error: predecessor id for task id {t['id']} must be int. Got: {pid!r}"
                )
            if pid not in id_set:
                raise ValueError(
                    f"Error: task id {t['id']} has unknown predecessor id {pid}."
                )

    # Validate risk events
    for event in risk_events:
        if "name" not in event:
            raise ValueError("Error: each risk event must have a 'name' field.")
        if "probability" not in event or "impact_days" not in event or "tasks" not in event:
            raise ValueError(
                f"Error: risk event '{event.get('name')}' must have "
                "'probability', 'impact_days', and 'tasks' fields."
            )
        p = event["probability"]
        if not isinstance(p, (int, float)) or not (0.0 <= p <= 1.0):
            raise ValueError(
                f"Error: risk event '{event['name']}' has invalid probability: {p!r}"
            )
        impact = event["impact_days"]
        if not isinstance(impact, (int, float)):
            raise ValueError(
                f"Error: risk event '{event['name']}' has non-numeric impact_days: {impact!r}"
            )
        rtasks = event["tasks"]
        if not isinstance(rtasks, list) or not rtasks:
            raise ValueError(
                f"Error: risk event '{event['name']}' must have a non-empty 'tasks' list."
            )
        for tid in rtasks:
            if not isinstance(tid, int):
                raise ValueError(
                    f"Error: risk event '{event['name']}' has non-integer task id: {tid!r}"
                )
            if tid not in id_set:
                raise ValueError(
                    f"Error: risk event '{event['name']}' references unknown task id: {tid}"
                )

    return tasks, risk_events, target_duration


# ===== 6. CLI =====

def parse_args():
    parser = argparse.ArgumentParser(
        description="Monte Carlo project timeline simulator (JSON + dependencies + risk events)."
    )
    parser.add_argument(
        "--project",
        required=True,
        help="Path to project JSON file (e.g. project.json)",
    )
    parser.add_argument(
        "--sims",
        type=int,
        default=1000,
        help="Number of Monte Carlo simulations (default: 1000)",
    )
    parser.add_argument(
        "--save-plot",
        type=str,
        default=None,
        help="If set, save histogram image to this path (e.g. hist.png) instead of showing it.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (int). Use for reproducibility.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    tasks, risk_events, target_duration = load_project(args.project)

    print(f"Loaded project from {args.project}")
    print(f"Tasks        : {len(tasks)}")
    print(f"Risk events  : {len(risk_events)}")
    print(f"Simulations  : {args.sims}")
    if target_duration is not None:
        print(f"Target dur.  : {target_duration}")
    if args.seed is not None:
        print(f"Random seed  : {args.seed}")
    if args.save_plot:
        print(f"Plot will be saved to: {args.save_plot}")

    # Run simulation
    project_duration, start_times, finish_times, tasks_sorted, preds_by_idx = simulate_project(
        tasks=tasks,
        risk_events=risk_events,
        n_sim=args.sims,
        seed=args.seed,
    )

    # Summary
    print_summary(project_duration, target_duration=target_duration)

    # Critical-path stats
    cp_counts = critical_path_frequency(preds_by_idx, start_times, finish_times)
    print_critical_path_stats(tasks_sorted, cp_counts, n_sim=args.sims)

    # Plot behavior:
    # - If --save-plot is set: always save to file, no GUI.
    # - Else: no plotting at all.
    if args.save_plot:
        plot_histogram(project_duration, target_duration=target_duration, save_path=args.save_plot)

if __name__ == "__main__":
    main()

