# montecarlo_pm

Monte Carlo **project timeline simulator** for project managers, tech leads and anyone who's doing work under uncertainty.

Here's how you can use it:

1. Define your project plan with all the tasks in a `.json` file
2. Add your optimistic, most-likely and pessimistic task estimations
3. Set dependencies between tasks
4. Add optional risk events
5. Run `mc_pm.py`

`mc_pm.py` then simulates `n` runs (as many as you want) and shows:

- Distribution of total project duration  
- Probability of finishing within a target duration  
- Which tasks most often sit on the **critical path**  

**Important Note**: The time unit is **generic**, meaning that you can treat it as *days, working days, weeks, sprints*. Whatever unit you choose, remember it and make sure you stay consistent.

**Sample output**
```bash
Loaded project from ./demo/project.json
Tasks        : 6
Risk events  : 3
Simulations  : 5000
Target dur.  : 20
Plot will be saved to: ./demo/project.png

=== Project Duration Summary (Monte Carlo) ===
Simulations run    : 5000
Mean duration      :  21.96
Median (50th pct)  :  21.98
80th percentile    :  24.49
90th percentile    :  25.71
95th percentile    :  26.60

Target duration    : 20.00
P(finish ≤ target) :  24.1%

=== Critical Path Frequency ===
Task ID | %-on-critical-path | Name
---------------------------------------------
     1 |              100.0% | Design
     4 |              100.0% | Integration
     6 |              100.0% | Deploy
     5 |              100.0% | Testing
     2 |               69.5% | Backend API
     3 |               30.5% | Frontend
Histogram saved to ./demo/project.png
```

---

## Features

- **Triangular task durations**: `optimistic`, `most_likely`, `pessimistic` per task
- **Dependencies via task IDs**: `predecessors` field, internally topologically sorted (DAG check)
- **Risk events**: 
  - Each event has `probability`, `impact_days`, and a list of affected tasks  
  - When triggered, it adds time to the selected tasks
- **Monte Carlo engine**  
  - Configurable number of simulations (`--sims`)
  - Optional random seed for reproducible runs (`--seed`)
- **Critical path frequency analysis**: shows how often each task appears on the simulated critical path
- **Histogram plotting**: show a GUI (`matplotlib`) or save to file (`--save-plot`)
- **JSON input with validation**: clear errors for missing fields, invalid IDs, malformed risk events

---

## Installation

Installing is as easy as:

```bash
git clone https://github.com/rabb1tpt/montecarlo_pm.git
cd montecarlo_pm
./setup.sh
```
If you want to run `mc_pm` from any folder, create a symlink into ~/.local/bin (which is usually in your PATH).

From the repo root, run this:
```bash
ln -s "$PWD/bin/mc_pm" ~/.local/bin/mc_pm
```

Now, can simply call `mc_pm` instead of `python3 mc_pm.py ...`.

From this point on, I will assume that you did this.
Otherwise, just explictly call `python3 mc_pm.py` or `./bin/mc_pm`.

---

## Basic usage

```bash
mc_pm --help
mc_pm --project ./demo/project.json --sims 5000
mc_pm --project ./demo/project.json --sims 5000 --save-plot ./demo/hist.png
```

---

## 📚 JSON schema

### `tasks`

- Required fields: `id`, `optimistic`, `most_likely`, `pessimistic`
- Optional: `name`, `predecessors` (defaults to [])

### `risk_events` (optional)

- Required: `name`, `probability`, `impact_days`, `tasks`

### `target_duration` (optional)

Used to calculate the probability of finishing on time.
I assume that most of the time, you will want to use this.

### Example schema (`project.json`)

```json
{
  "tasks": [
    {
      "id": 1,
      "name": "Design",
      "optimistic": 2,
      "most_likely": 3,
      "pessimistic": 5,
      "predecessors": []
    },
    {
      "id": 2,
      "name": "Backend API",
      "optimistic": 3,
      "most_likely": 5,
      "pessimistic": 8,
      "predecessors": [1]
    },
    {
      "id": 3,
      "name": "Frontend",
      "optimistic": 3,
      "most_likely": 4,
      "pessimistic": 7,
      "predecessors": [1]
    },
    {
      "id": 4,
      "name": "Integration",
      "optimistic": 2,
      "most_likely": 3,
      "pessimistic": 6,
      "predecessors": [2, 3]
    },
    {
      "id": 5,
      "name": "Testing",
      "optimistic": 3,
      "most_likely": 4,
      "pessimistic": 5,
      "predecessors": [4]
    },
    {
      "id": 6,
      "name": "Deploy",
      "optimistic": 0.5,
      "most_likely": 1,
      "pessimistic": 2,
      "predecessors": [5]
    }
  ],
  "risk_events": [
    {
      "name": "API Scope Creep",
      "probability": 0.3,
      "impact_days": 3,
      "tasks": [2]
    },
    {
      "name": "Cross-team Coordination Delay",
      "probability": 0.2,
      "impact_days": 2,
      "tasks": [3, 4]
    },
    {
      "name": "Testing uncovers issues",
      "probability": 0.7,
      "impact_days": 4,
      "tasks": [5]
    }
  ],
  "target_duration": 20
}
```
---

## 🧠 Interpretation tips

- The **time unit is arbitrary** so pick one and stay consistent with it.
- Percentiles:
  - 50th (median) = half of the simulated runs ended in `x` time. The other half took longer.
  - 80th = 80% of the simulated runs ended in `x` time. 20% took longer.
- Critical path frequency highlights the risky paths on your project.

---

## 📄 License

CC BY 4.0
