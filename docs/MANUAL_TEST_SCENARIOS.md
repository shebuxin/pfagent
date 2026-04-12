# Manual Test Scenarios

Use these multi-turn scripts to exercise the agent after a change.
Each scenario has:

- a **goal** (what capability is being tested)
- **expected behavior** (what "pass" means at each turn)
- **watch for** (known pitfalls or regression risks)

Run them in order inside one chat session to exercise continuity.
Use **RAG** or **Fine-tuned + RAG** as the agent type.

---

## Scenario A: Trip-line by position (the Stage 5 regression case)

**Goal:** verify the new case-idx inventory fixes the trip-line
hardcoding bug from the 2026-04-05 user-feedback session.

**Case:** IEEE39 built-in

```
1.  Use ieee39 to run a power flow and report the bus voltage at bus 15
    and whether there is a PQ load connected to that bus.

2.  Trip line 18 and plot the change in active power flow through
    each branch.

3.  What is the idx of the 18th line?

4.  Now trip the line between bus 16 and bus 17 instead. Show me
    which branches see the biggest flow change.

5.  Undo the trip and plot the baseline active power flow through
    each branch.
```

**Expected behavior:**
- Turn 1: identifies PQ at bus 15 (check against `ieee39.xlsx`: bus 15 has PQ_6)
- Turn 2: **first attempt succeeds** — no error fix needed. The code
  should resolve `"line 18"` as `Line.idx.v[17]` (= `"Line_18"`) via
  the injected inventory, NOT compare `line_ids == "18"`.
- Turn 3: returns the string `"Line_18"` using a `print()` (not a plot)
- Turn 4: looks up the line by `bus1==16 and bus2==17` (symmetric pair)
  instead of guessing
- Turn 5: reuses the ieee39 active case via continuity context

**Watch for:**
- Any turn that invokes the Codex error-fixer ⇒ a regression
- Any turn that hardcodes `"18"` as a literal idx string
- Turn 3 emitting a `plt.*` call (query-vs-plot regression)

---

## Scenario B: PQ load modification (idx resolution by bus)

**Goal:** verify the agent resolves PQ devices by bus number, not
by guessing an idx string.

**Case:** IEEE14 built-in

```
1.  Run a power flow on the IEEE14 built-in case and report the
    slack bus voltage and the three highest bus voltages.

2.  Double the P and Q of the load at bus 6.

3.  Re-run power flow and report the new voltage at bus 6.

4.  Now increase by 20% the load at bus 9 as well.

5.  Which two buses have the lowest voltage after both changes?
```

**Expected behavior:**
- Turn 1: slack bus = 1, V≈1.06 p.u.
- Turn 2: code inspects `ssa.PQ.bus.v` + `ssa.PQ.idx.v` to find the
  PQ entry at bus 6, then calls `ssa.PQ.set(src="p0", idx=[...],
  attr="v", value=[...])` with the resolved idx
- Turn 3: voltage at bus 6 should drop slightly
- Turn 4: modifies PQ at bus 9 without touching bus-6 changes
- Turn 5: uses argsort/sorted (validator pins this)

**Watch for:**
- Turn 2 using a literal like `idx=["PQ_5"]` without resolving from bus
- Turn 4 losing the turn-2 state (should not reset the bus-6 change)
- Turn 5 emitting a plot instead of printing the two bus IDs

---

## Scenario C: Uploaded case + N-1 screening

**Goal:** exercise the uploaded-case path (kundur, but re-uploaded as
a user file) + the N-1 contingency workflow.

**Setup:** before starting, upload `kundur_full.xlsx` via the sidebar
(download from ANDES built-ins first, or use any `.xlsx` case you
have locally).

```
1.  Load the uploaded case and plot the voltage profile across all
    buses after running power flow.

2.  Screen all single-line outages and report which line, when
    tripped, causes the largest |ΔV| across buses.

3.  Re-run power flow after tripping that worst line. Print the
    five lowest voltages.

4.  Is there any islanding after this trip?

5.  Now explain in plain English why this particular line is the
    most critical.
```

**Expected behavior:**
- Turn 1: uses `andes.load("<filename>", ...)` directly (NOT
  `andes.get_case(...)`), matches the uploaded-case template from
  prompt_builder
- Turn 2: iterates every line in `ssa.Line.idx.v`, tracks convergence
  + islanding per the structured-codegen N-1 pattern
- Turn 3: uses the trip identified in turn 2 (continuity check)
- Turn 4: calls `ssa.Bus.island_sets` / `ssa.Bus.nosw_island` /
  `ssa.Bus.n_islanded_buses`
- Turn 5: **prose only**, no Python — the `is_explanatory_followup_request`
  detector should short-circuit to the prose path

**Watch for:**
- Turn 1 wrapping the uploaded filename with `andes.get_case(...)`
  (validator rule should reject this)
- Turn 2 skipping the islanding guard (validator requires 2+ status markers)
- Turn 5 emitting code (the prose-mode detector + retry nudge should
  catch it)

---

## Scenario D: Branch-flow analysis (active + reactive)

**Goal:** stress the branch-flow API mapping (`p1/p2 → a1.e/a2.e`,
`q1/q2 → v1.e/v2.e`) and the branch-flow validator rules.

**Case:** IEEE39 built-in

```
1.  Run ieee39 power flow and plot the active power flow of every
    branch as a bar chart, sorted by absolute magnitude.

2.  Now plot the reactive power flow of every branch on the same
    axes.

3.  Which three lines carry the highest active power flow?

4.  Is line 7 one of them?

5.  Show the sending-end and receiving-end active power for line 7
    side by side.
```

**Expected behavior:**
- Turn 1: uses `ssa.Line.a1.e` or `ssa.Line.a2.e`, not `p1`/`p2`
- Turn 2: uses `ssa.Line.v1.e` / `ssa.Line.v2.e`
- Turn 3: argsort-based ranking, prints three idx values + metrics
- Turn 4: answers "yes" or "no" using the ranking from turn 3 (the
  inventory + continuity should make this trivial)
- Turn 5: plots `a1.e` and `a2.e` for the specific line, resolving
  `"line 7"` → `Line.idx.v[6]` via the inventory

**Watch for:**
- Turn 1 using `Line.p1.v` (validator rejects)
- Turn 3 skipping argsort/sorted (validator rejects)
- Turn 4 re-running power flow instead of reusing turn-3 results
  (no hard validator for this; it's a UX smell)

---

## Scenario E: Voltage bound threshold + ranked reports

**Goal:** the structured-codegen threshold reports (Stage 1 structured/scripts.py).

**Case:** IEEE14 built-in

```
1.  Run ieee14 power flow. Report buses whose voltage is below
    1.00 p.u. in a JSON object.  The JSON object must contain
    these keys: threshold, selected_bus_ids, selected_count,
    lowest_bus_ids, lowest_voltages.

2.  Now scale every PQ load by 1.3 and rerun. Return the same JSON.

3.  List the top-3 lowest-voltage buses.

4.  What caused the voltage drop in turn 2?
```

**Expected behavior:**
- Turn 1: should route through the structured-codegen path
  (`baseline_threshold_low_rank_report`); you'll see a
  "Applied structured ANDES code generation" log line
- Turn 2: structured codegen again, this time with a scale_factor
- Turn 3: argsort-based ranking
- Turn 4: **prose only**, explains PQ load increase → voltage dip

**Watch for:**
- Turn 1/2 going through the RAG path instead of structured codegen
  (means `structured_codegen_is_applicable` missed the JSON contract)
- JSON output not matching the exact key set

---

## Scenario F: Continuity & reset

**Goal:** verify that the `active_andes_case` continuity block survives
a rerun and that users can explicitly switch cases.

```
1.  Run ieee14 and print the slack bus voltage.

2.  Plot the voltage profile.            # implicit: reuse ieee14

3.  Plot the voltage profile.            # explicit repeat, same case

4.  Now switch to ieee39 and plot the
    voltage profile on THAT case.

5.  What was the slack bus voltage on
    ieee14?                              # fallback: model memory only
```

**Expected behavior:**
- Turn 2-3: `ANDES continuity context` pins ieee14; no case switch
- Turn 4: explicit case switch, active_case updates to ieee39
- Turn 5: answers from chat-history memory (not from case load);
  should recall "≈1.06 p.u." from turn 1 without re-running PF

**Watch for:**
- Turn 2/3 regenerating a new `andes.load(...)` from scratch instead
  of naming `andes.get_case("ieee14/ieee14_full.xlsx")`
- Turn 4 not updating the active case (continuity stays on ieee14)
- Turn 5 emitting code instead of answering in prose

---

## Scenario G: Error recovery

**Goal:** verify the Codex fixer path when first-attempt code has
a genuine bug that the inventory cannot prevent.

**Setup:** no special setup; just type a prompt that's likely to
trip the model into generating bad code, like a partial request.

```
1.  Load ieee14. Sort lines by their active power flow from lowest
    to highest.

2.  [Click "Fix Error with AI" on any error that appears — or
     continue if turn 1 succeeded.]

3.  Can you also show me the top-5 lines by absolute magnitude,
    regardless of direction?

4.  Save that top-5 table to a file named 'top5_lines.csv' in the
    output directory.
```

**Expected behavior:**
- Turn 1: may require the Codex fixer (the "sorted by flow" phrasing
  is ambiguous — could mean ascending vs. descending, signed vs. abs)
- Turn 2: if triggered, Codex fixer produces a fix that executes
  cleanly in the local validation loop
- Turn 3: argsort-based, handles abs() per validator rule
- Turn 4: writes CSV to `code_executions/<session>/data/output/` —
  exercise the file-listing widgets in the sidebar

**Watch for:**
- Codex fixer loop exceeding retry budget (default 2 validation retries)
- Turn 4 using a relative path that doesn't land in the right dir
  (`files.py:extract_generated_image_paths` and friends handle
  only `output/` — text files may just end up in session root)

---

## Scenario H: Query vs plot disambiguation

**Goal:** verify the "give me X" style queries don't get accidentally
turned into plots (the second regression from 2026-04-05).

**Case:** IEEE39 built-in (or continuation)

```
1.  Give me the idx of each line.

2.  Give me the idx of each PQ load.

3.  List every bus that has both a PQ load and a PV device.

4.  How many lines are in this case?

5.  Which bus has the most lines connected to it?
```

**Expected behavior:**
- **Every turn uses print() or prose, NOT plt.***
- Turn 1: prints 46 line idx strings
- Turn 3: set-intersects `ssa.PQ.bus.v` with `ssa.PV.bus.v`
- Turn 4: prints `len(ssa.Line.idx.v)`
- Turn 5: counts occurrences across `bus1.v + bus2.v`

**Watch for:**
- Any `plt.plot(...)` / `plt.bar(...)` call — this was the
  Turn 10/11 regression pattern

---

## How to score a run

After each scenario, note:

1. **First-attempt success rate** — did the first generated code run?
2. **Codex-fixer invocations** — how many turns triggered "Fix Error"?
3. **Prose-mode accuracy** — did turn-5-style questions stay in prose?
4. **Continuity** — did the agent remember the active case across turns?
5. **User-visible errors** — any `st.error(...)` banner?

A good baseline (RAG + latest main) should produce:
- ≥ 80% first-attempt success across all 8 scenarios
- ≤ 2 Codex-fixer invocations in Scenario G (where error recovery
  is the explicit test target)
- Zero plots in Scenario H

Record anomalies in `docs/MANUAL_TEST_RESULTS_<date>.md` so we can
track regression velocity over time.
