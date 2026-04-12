# Manual Test Scenarios

This file records 10 randomly sampled scenarios from the 164 verified passing scenarios for manual regression testing.

- Sampling source: `verification.suite.build_verification_suite(164)`
- Sampling seed: `20260412`
- Sampling result: 5 built-in cases and 5 uploaded cases
- Execution order: Run Turn 1 -> Turn 2 -> Turn 3 for each scenario in the same conversation
- Output check: Each response should provide a runnable Python script and print one final line starting with `RESULT_JSON=`
- Uploaded case setup: Before testing, copy or upload the corresponding source case using the uploaded filename shown in the table

## Sampled Scenarios

| # | Scenario | Case source | Case family | Source case | Uploaded filename | Blueprint |
|---|---|---|---|---|---|---|
| 1 | `scenario_006` | built-in | IEEE 14 | `ieee14/ieee14_full.xlsx` | N/A | `threshold_slack_add_extremes` |
| 2 | `scenario_025` | uploaded | IEEE 14 | `ieee14/ieee14_full.xlsx` | `verify_ieee14_025.xlsx` | `extremes_pv_scale_barplot` |
| 3 | `scenario_028` | uploaded | IEEE 14 | `ieee14/ieee14_full.xlsx` | `verify_ieee14_028.xlsx` | `low_buses_add_slack_plot` |
| 4 | `scenario_032` | built-in | IEEE 39 | `ieee39/ieee39.xlsx` | N/A | `voltage_rank_add_scale_plot` |
| 5 | `scenario_035` | built-in | IEEE 39 | `ieee39/ieee39.xlsx` | N/A | `threshold_slack_add_extremes` |
| 6 | `scenario_070` | built-in | Kundur | `kundur/kundur_full.xlsx` | N/A | `low_buses_add_slack_plot` |
| 7 | `scenario_074` | uploaded | Kundur | `kundur/kundur_full.xlsx` | `verify_kundur_074.xlsx` | `voltage_rank_add_scale_plot` |
| 8 | `scenario_079` | uploaded | Kundur | `kundur/kundur_full.xlsx` | `verify_kundur_079.xlsx` | `extremes_pv_scale_barplot` |
| 9 | `scenario_101` | built-in | IEEE 14 | `ieee14/ieee14_full.xlsx` | N/A | `targeted_pq_edit_then_n1_screening` |
| 10 | `scenario_152` | uploaded | IEEE 14 | `ieee14/ieee14_full.xlsx` | `verify_ieee14_152.xlsx` | `generalized_targeted_pv_then_branch_trip` |

## 1. `scenario_006`

- Case source: built-in
- Case family: IEEE 14
- Source case: `ieee14/ieee14_full.xlsx`
- Blueprint: `threshold_slack_add_extremes`

Turn 1 prompt:

```text
Please answer with one runnable Python script only and nothing else.
Use the built-in IEEE 14 full case.
Run power flow, count all buses above 1.020 p.u., and also return the two lowest-voltage buses.
- The script must end by printing exactly one line that starts with RESULT_JSON=
- The JSON object must contain these keys: threshold, selected_bus_ids, selected_count, lowest_bus_ids, lowest_voltages.
- Use plain Python ints/floats/lists in RESULT_JSON, not NumPy scalar types.
- Round float values in RESULT_JSON to 6 decimals.
- `selected_bus_ids` should list every bus above the threshold.
- `lowest_bus_ids` should contain exactly two buses in ascending voltage order.
```

Turn 2 prompt:

```text
Follow-up request: update the previous study and return a fresh complete script.
Keep using the same built-in case from earlier in this conversation.
Keep the same study, set the slack-bus voltage target to 1.025, rerun power flow, and report the slack bus voltage plus how many buses fall below 1.005 p.u.
- The script must end by printing exactly one line that starts with RESULT_JSON=
- The JSON object must contain these keys: slack_bus, slack_setpoint, slack_voltage, selected_count.
- Use plain Python ints/floats/lists in RESULT_JSON, not NumPy scalar types.
- Round float values in RESULT_JSON to 6 decimals.
```

Turn 3 prompt:

```text
Please revise the prior script for this next step and return one full script only.
Keep the adjusted slack-bus setting from the last turn.
Also add a new PQ load before setup at bus 5 with idx 'PQ_VERIFY_006_B', p0=0.014, and q0=0.009.
Rerun power flow and report the maximum-voltage bus, minimum-voltage bus, and the total number of PQ loads now present.
- The script must end by printing exactly one line that starts with RESULT_JSON=
- The JSON object must contain these keys: added_load_idx, max_bus, max_voltage, min_bus, min_voltage, total_pq_count.
- Use plain Python ints/floats/lists in RESULT_JSON, not NumPy scalar types.
- Round float values in RESULT_JSON to 6 decimals.
```

## 2. `scenario_025`

- Case source: uploaded
- Case family: IEEE 14
- Source case: `ieee14/ieee14_full.xlsx`
- Uploaded filename: `verify_ieee14_025.xlsx`
- Blueprint: `extremes_pv_scale_barplot`

Turn 1 prompt:

```text
Please answer with one runnable Python script only and nothing else.
Use my uploaded file verify_ieee14_025.xlsx from the current working directory.
Run power flow and report the maximum-voltage bus and minimum-voltage bus.
- The script must end by printing exactly one line that starts with RESULT_JSON=
- The JSON object must contain these keys: max_bus, max_voltage, min_bus, min_voltage.
- Use plain Python ints/floats/lists in RESULT_JSON, not NumPy scalar types.
- Round float values in RESULT_JSON to 6 decimals.
```

Turn 2 prompt:

```text
Please revise the prior script for this next step and return one full script only.
Keep using the same uploaded study file from earlier in this conversation.
Keep the same case, set the first PV voltage target to 1.020, rerun power flow, and report the affected PV bus voltage together with how many buses are above 1.025 p.u.
- The script must end by printing exactly one line that starts with RESULT_JSON=
- The JSON object must contain these keys: pv_bus, pv_setpoint, pv_voltage, selected_count.
- Use plain Python ints/floats/lists in RESULT_JSON, not NumPy scalar types.
- Round float values in RESULT_JSON to 6 decimals.
```

Turn 3 prompt:

```text
Next follow-up: rebuild the script for the updated study, code only.
Keep the PV setpoint adjustment from the previous turn.
Also scale every PQ load by 1.050, rerun power flow, and save a bar chart of the bus voltages to 'scenario_025_turn3_bar.png'.
- The script must end by printing exactly one line that starts with RESULT_JSON=
- The JSON object must contain these keys: scale_factor, min_bus, min_voltage, max_bus, max_voltage, plot_file.
- Use plain Python ints/floats/lists in RESULT_JSON, not NumPy scalar types.
- Round float values in RESULT_JSON to 6 decimals.
- Use a bar chart, not a line chart.
```

## 3. `scenario_028`

- Case source: uploaded
- Case family: IEEE 14
- Source case: `ieee14/ieee14_full.xlsx`
- Uploaded filename: `verify_ieee14_028.xlsx`
- Blueprint: `low_buses_add_slack_plot`

Turn 1 prompt:

```text
Please answer with one runnable Python script only and nothing else.
Use my uploaded file verify_ieee14_028.xlsx from the current working directory.
Run power flow and report the 4 lowest-voltage buses.
- The script must end by printing exactly one line that starts with RESULT_JSON=
- The JSON object must contain these keys: selected_bus_ids, selected_voltages.
- Use plain Python ints/floats/lists in RESULT_JSON, not NumPy scalar types.
- Round float values in RESULT_JSON to 6 decimals.
- `selected_bus_ids` and `selected_voltages` should represent the lowest-voltage buses in ascending voltage order.
```

Turn 2 prompt:

```text
Please revise the prior script for this next step and return one full script only.
Keep using the same uploaded study file from earlier in this conversation.
Keep the same study and add a new PQ load before setup at bus 9 with idx 'PQ_VERIFY_028_D', p0=0.018, and q0=0.012.
After rerunning, report the slack-bus voltage and every bus below 1.010 p.u.
- The script must end by printing exactly one line that starts with RESULT_JSON=
- The JSON object must contain these keys: added_load_idx, slack_bus, slack_voltage, threshold, selected_bus_ids, selected_count.
- Use plain Python ints/floats/lists in RESULT_JSON, not NumPy scalar types.
- Round float values in RESULT_JSON to 6 decimals.
```

Turn 3 prompt:

```text
Next follow-up: rebuild the script for the updated study, code only.
Keep the added load from the previous turn.
Also set the slack-bus voltage target to 1.035, rerun power flow, and save a line plot of bus voltages to 'scenario_028_turn3_line.png'.
- The script must end by printing exactly one line that starts with RESULT_JSON=
- The JSON object must contain these keys: slack_setpoint, slack_voltage, selected_bus_ids, selected_voltages, plot_file.
- Use plain Python ints/floats/lists in RESULT_JSON, not NumPy scalar types.
- Round float values in RESULT_JSON to 6 decimals.
- `selected_bus_ids` and `selected_voltages` should again represent the lowest-voltage buses in ascending voltage order.
```

## 4. `scenario_032`

- Case source: built-in
- Case family: IEEE 39
- Source case: `ieee39/ieee39.xlsx`
- Blueprint: `voltage_rank_add_scale_plot`

Turn 1 prompt:

```text
Return exactly one runnable Python code block and no prose.
Use the built-in IEEE 39 case.
Run power flow and report the top-5 highest-voltage buses plus the slack-bus voltage.
- The script must end by printing exactly one line that starts with RESULT_JSON=
- The JSON object must contain these keys: slack_bus, slack_voltage, selected_bus_ids, selected_voltages.
- Use plain Python ints/floats/lists in RESULT_JSON, not NumPy scalar types.
- Round float values in RESULT_JSON to 6 decimals.
- `selected_bus_ids` and `selected_voltages` should represent the top highest-voltage buses in descending order.
```

Turn 2 prompt:

```text
Please revise the prior script for this next step and return one full script only.
Keep using the same built-in case from earlier in this conversation.
Keep the same case, add one new PQ load before setup at bus 20 with idx 'PQ_VERIFY_032_A', p0=0.019, and q0=0.012.
After rerunning power flow, report every bus below 0.980 p.u. together with the minimum-voltage bus.
- The script must end by printing exactly one line that starts with RESULT_JSON=
- The JSON object must contain these keys: added_load_idx, added_load_bus, threshold, selected_bus_ids, selected_count, min_bus, min_voltage.
- Use plain Python ints/floats/lists in RESULT_JSON, not NumPy scalar types.
- Round float values in RESULT_JSON to 6 decimals.
- `selected_bus_ids` should list all buses below the threshold in ascending bus order.
```

Turn 3 prompt:

```text
Next follow-up: rebuild the script for the updated study, code only.
Keep the added load from the previous step.
Also scale every PQ load by a factor of 1.050 after setup, rerun power flow, and save a line plot of bus voltage magnitude to 'scenario_032_turn3_line.png'.
- The script must end by printing exactly one line that starts with RESULT_JSON=
- The JSON object must contain these keys: scale_factor, max_bus, max_voltage, min_bus, min_voltage, plot_file.
- Use plain Python ints/floats/lists in RESULT_JSON, not NumPy scalar types.
- Round float values in RESULT_JSON to 6 decimals.
- `plot_file` must exactly match the saved filename.
```

## 5. `scenario_035`

- Case source: built-in
- Case family: IEEE 39
- Source case: `ieee39/ieee39.xlsx`
- Blueprint: `threshold_slack_add_extremes`

Turn 1 prompt:

```text
Return exactly one runnable Python code block and no prose.
Use the built-in IEEE 39 case.
Run power flow, count all buses above 1.040 p.u., and also return the two lowest-voltage buses.
- The script must end by printing exactly one line that starts with RESULT_JSON=
- The JSON object must contain these keys: threshold, selected_bus_ids, selected_count, lowest_bus_ids, lowest_voltages.
- Use plain Python ints/floats/lists in RESULT_JSON, not NumPy scalar types.
- Round float values in RESULT_JSON to 6 decimals.
- `selected_bus_ids` should list every bus above the threshold.
- `lowest_bus_ids` should contain exactly two buses in ascending voltage order.
```

Turn 2 prompt:

```text
Please revise the prior script for this next step and return one full script only.
Keep using the same built-in case from earlier in this conversation.
Keep the same study, set the slack-bus voltage target to 1.030, rerun power flow, and report the slack bus voltage plus how many buses fall below 0.970 p.u.
- The script must end by printing exactly one line that starts with RESULT_JSON=
- The JSON object must contain these keys: slack_bus, slack_setpoint, slack_voltage, selected_count.
- Use plain Python ints/floats/lists in RESULT_JSON, not NumPy scalar types.
- Round float values in RESULT_JSON to 6 decimals.
```

Turn 3 prompt:

```text
Next follow-up: rebuild the script for the updated study, code only.
Keep the adjusted slack-bus setting from the last turn.
Also add a new PQ load before setup at bus 15 with idx 'PQ_VERIFY_035_B', p0=0.016, and q0=0.011.
Rerun power flow and report the maximum-voltage bus, minimum-voltage bus, and the total number of PQ loads now present.
- The script must end by printing exactly one line that starts with RESULT_JSON=
- The JSON object must contain these keys: added_load_idx, max_bus, max_voltage, min_bus, min_voltage, total_pq_count.
- Use plain Python ints/floats/lists in RESULT_JSON, not NumPy scalar types.
- Round float values in RESULT_JSON to 6 decimals.
```

## 6. `scenario_070`

- Case source: built-in
- Case family: Kundur
- Source case: `kundur/kundur_full.xlsx`
- Blueprint: `low_buses_add_slack_plot`

Turn 1 prompt:

```text
Write one complete runnable Python script only, inside a single ```python block.
Use the built-in Kundur full case.
Run power flow and report the 4 lowest-voltage buses.
- The script must end by printing exactly one line that starts with RESULT_JSON=
- The JSON object must contain these keys: selected_bus_ids, selected_voltages.
- Use plain Python ints/floats/lists in RESULT_JSON, not NumPy scalar types.
- Round float values in RESULT_JSON to 6 decimals.
- `selected_bus_ids` and `selected_voltages` should represent the lowest-voltage buses in ascending voltage order.
```

Turn 2 prompt:

```text
Please keep the conversation context and send one new complete script only.
Keep using the same built-in case from earlier in this conversation.
Keep the same study and add a new PQ load before setup at bus 7 with idx 'PQ_VERIFY_070_D', p0=0.018, and q0=0.012.
After rerunning, report the slack-bus voltage and every bus below 0.960 p.u.
- The script must end by printing exactly one line that starts with RESULT_JSON=
- The JSON object must contain these keys: added_load_idx, slack_bus, slack_voltage, threshold, selected_bus_ids, selected_count.
- Use plain Python ints/floats/lists in RESULT_JSON, not NumPy scalar types.
- Round float values in RESULT_JSON to 6 decimals.
```

Turn 3 prompt:

```text
Follow-up request: update the previous study and return a fresh complete script.
Keep the added load from the previous turn.
Also set the slack-bus voltage target to 1.010, rerun power flow, and save a line plot of bus voltages to 'scenario_070_turn3_line.png'.
- The script must end by printing exactly one line that starts with RESULT_JSON=
- The JSON object must contain these keys: slack_setpoint, slack_voltage, selected_bus_ids, selected_voltages, plot_file.
- Use plain Python ints/floats/lists in RESULT_JSON, not NumPy scalar types.
- Round float values in RESULT_JSON to 6 decimals.
- `selected_bus_ids` and `selected_voltages` should again represent the lowest-voltage buses in ascending voltage order.
```

## 7. `scenario_074`

- Case source: uploaded
- Case family: Kundur
- Source case: `kundur/kundur_full.xlsx`
- Uploaded filename: `verify_kundur_074.xlsx`
- Blueprint: `voltage_rank_add_scale_plot`

Turn 1 prompt:

```text
Give me code only: one full Python script in one fenced block.
Use my uploaded file verify_kundur_074.xlsx from the current working directory.
Run power flow and report the top-5 highest-voltage buses plus the slack-bus voltage.
- The script must end by printing exactly one line that starts with RESULT_JSON=
- The JSON object must contain these keys: slack_bus, slack_voltage, selected_bus_ids, selected_voltages.
- Use plain Python ints/floats/lists in RESULT_JSON, not NumPy scalar types.
- Round float values in RESULT_JSON to 6 decimals.
- `selected_bus_ids` and `selected_voltages` should represent the top highest-voltage buses in descending order.
```

Turn 2 prompt:

```text
Please keep the conversation context and send one new complete script only.
Keep using the same uploaded study file from earlier in this conversation.
Keep the same case, add one new PQ load before setup at bus 9 with idx 'PQ_VERIFY_074_A', p0=0.019, and q0=0.012.
After rerunning power flow, report every bus below 0.970 p.u. together with the minimum-voltage bus.
- The script must end by printing exactly one line that starts with RESULT_JSON=
- The JSON object must contain these keys: added_load_idx, added_load_bus, threshold, selected_bus_ids, selected_count, min_bus, min_voltage.
- Use plain Python ints/floats/lists in RESULT_JSON, not NumPy scalar types.
- Round float values in RESULT_JSON to 6 decimals.
- `selected_bus_ids` should list all buses below the threshold in ascending bus order.
```

Turn 3 prompt:

```text
Follow-up request: update the previous study and return a fresh complete script.
Keep the added load from the previous step.
Also scale every PQ load by a factor of 1.060 after setup, rerun power flow, and save a line plot of bus voltage magnitude to 'scenario_074_turn3_line.png'.
- The script must end by printing exactly one line that starts with RESULT_JSON=
- The JSON object must contain these keys: scale_factor, max_bus, max_voltage, min_bus, min_voltage, plot_file.
- Use plain Python ints/floats/lists in RESULT_JSON, not NumPy scalar types.
- Round float values in RESULT_JSON to 6 decimals.
- `plot_file` must exactly match the saved filename.
```

## 8. `scenario_079`

- Case source: uploaded
- Case family: Kundur
- Source case: `kundur/kundur_full.xlsx`
- Uploaded filename: `verify_kundur_079.xlsx`
- Blueprint: `extremes_pv_scale_barplot`

Turn 1 prompt:

```text
Write one complete runnable Python script only, inside a single ```python block.
Use my uploaded file verify_kundur_079.xlsx from the current working directory.
Run power flow and report the maximum-voltage bus and minimum-voltage bus.
- The script must end by printing exactly one line that starts with RESULT_JSON=
- The JSON object must contain these keys: max_bus, max_voltage, min_bus, min_voltage.
- Use plain Python ints/floats/lists in RESULT_JSON, not NumPy scalar types.
- Round float values in RESULT_JSON to 6 decimals.
```

Turn 2 prompt:

```text
Please keep the conversation context and send one new complete script only.
Keep using the same uploaded study file from earlier in this conversation.
Keep the same case, set the first PV voltage target to 0.990, rerun power flow, and report the affected PV bus voltage together with how many buses are above 0.990 p.u.
- The script must end by printing exactly one line that starts with RESULT_JSON=
- The JSON object must contain these keys: pv_bus, pv_setpoint, pv_voltage, selected_count.
- Use plain Python ints/floats/lists in RESULT_JSON, not NumPy scalar types.
- Round float values in RESULT_JSON to 6 decimals.
```

Turn 3 prompt:

```text
Follow-up request: update the previous study and return a fresh complete script.
Keep the PV setpoint adjustment from the previous turn.
Also scale every PQ load by 1.030, rerun power flow, and save a bar chart of the bus voltages to 'scenario_079_turn3_bar.png'.
- The script must end by printing exactly one line that starts with RESULT_JSON=
- The JSON object must contain these keys: scale_factor, min_bus, min_voltage, max_bus, max_voltage, plot_file.
- Use plain Python ints/floats/lists in RESULT_JSON, not NumPy scalar types.
- Round float values in RESULT_JSON to 6 decimals.
- Use a bar chart, not a line chart.
```

## 9. `scenario_101`

- Case source: built-in
- Case family: IEEE 14
- Source case: `ieee14/ieee14_full.xlsx`
- Blueprint: `targeted_pq_edit_then_n1_screening`

Turn 1 prompt:

```text
Write one complete runnable Python script only, inside a single ```python block.
Use the built-in IEEE 14 full case.
Run power flow, locate the existing PQ load connected to bus 2, and report its device idx, its current p0 and q0, and the solved slack-bus voltage.
- The script must end by printing exactly one line that starts with RESULT_JSON=
- The JSON object must contain these keys: target_pq_bus, target_pq_idx, target_p0, target_q0, slack_bus, slack_voltage.
- Use plain Python ints/floats/lists in RESULT_JSON, not NumPy scalar types.
- Round float values in RESULT_JSON to 6 decimals.
```

Turn 2 prompt:

```text
Next follow-up: rebuild the script for the updated study, code only.
Keep using the same built-in case from earlier in this conversation.
Keep the same study, locate the existing PQ load at bus 2, scale both p0 and q0 of that load by 1.030, rerun power flow, and report the updated device idx, updated p0/q0, and the minimum-voltage bus.
- The script must end by printing exactly one line that starts with RESULT_JSON=
- The JSON object must contain these keys: target_pq_bus, target_pq_idx, scale_factor, target_p0, target_q0, min_bus, min_voltage.
- Use plain Python ints/floats/lists in RESULT_JSON, not NumPy scalar types.
- Round float values in RESULT_JSON to 6 decimals.
```

Turn 3 prompt:

```text
Please keep the conversation context and send one new complete script only.
Keep the targeted PQ-load scaling from the previous turn.
Now perform an N-1 screening over these candidate lines, one outage at a time, always starting from the same modified case: 1-2, 1-5, 2-3.
For each contingency, open only that one line, rerun power flow, and identify which outage gives the lowest minimum bus voltage.
- The script must end by printing exactly one line that starts with RESULT_JSON=
- The JSON object must contain these keys: scale_factor, candidate_line_ids, worst_line_id, worst_line_bus_pair, worst_min_bus, worst_min_voltage.
- Use plain Python ints/floats/lists in RESULT_JSON, not NumPy scalar types.
- Round float values in RESULT_JSON to 6 decimals.
- `candidate_line_ids` must list the screened line ids in the same order as the candidate bus-pair list.
```

## 10. `scenario_152`

- Case source: uploaded
- Case family: IEEE 14
- Source case: `ieee14/ieee14_full.xlsx`
- Uploaded filename: `verify_ieee14_152.xlsx`
- Blueprint: `generalized_targeted_pv_then_branch_trip`

Turn 1 prompt:

```text
Give me code only: one full Python script in one fenced block.
Use my uploaded file verify_ieee14_152.xlsx from the current working directory.
Inspect the generator voltage-control record tied to bus 3.
After solving the case, return that record's idx, its present v0 target, and the solved voltage at that bus.
- The script must end by printing exactly one line that starts with RESULT_JSON=
- The JSON object must contain these keys: pv_bus, pv_idx, pv_setpoint, pv_voltage.
- Use plain Python ints/floats/lists in RESULT_JSON, not NumPy scalar types.
- Round float values in RESULT_JSON to 6 decimals.
```

Turn 2 prompt:

```text
Please revise the prior script for this next step and return one full script only.
Keep working from the same study state.
Move that same voltage-control record on bus 3 to a v0 target of 1.015, solve again, and report the updated idx, the applied setpoint, the solved PV-bus voltage, and how many buses are higher than 1.020 p.u.
- The script must end by printing exactly one line that starts with RESULT_JSON=
- The JSON object must contain these keys: pv_bus, pv_idx, pv_setpoint, pv_voltage, threshold, selected_count.
- Use plain Python ints/floats/lists in RESULT_JSON, not NumPy scalar types.
- Round float values in RESULT_JSON to 6 decimals.
```

Turn 3 prompt:

```text
Please keep the conversation context and send one new complete script only.
Keep that generator voltage-target change in place.
Now trip the branch joining buses 2 and 3, solve the modified network again, and report the opened branch id, the opened bus pair, the slack-bus voltage, and the minimum-voltage bus.
- The script must end by printing exactly one line that starts with RESULT_JSON=
- The JSON object must contain these keys: pv_setpoint, opened_line_id, opened_line_bus_pair, slack_bus, slack_voltage, min_bus, min_voltage.
- Use plain Python ints/floats/lists in RESULT_JSON, not NumPy scalar types.
- Round float values in RESULT_JSON to 6 decimals.
```
