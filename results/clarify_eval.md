# Feature F5 - Clarify: no-over-ask + correct-ask

- Commit: `df16eff5ca8fddc1677a549e9dce31bf42ece74e`  |  As-of: 2026-07-20  |  0 tokens (pure function)

## Headline
- Decision accuracy: **1.000** (21 items)
- OVER-ASK (load-bearing; must be 0): **0** (rate 0.000)
- UNDER-ASK: 0  |  warm-lead failures: 0

## By bucket
- well_specified: 3/3
- profile_resolvable: 5/5
- history_resolvable: 3/3
- truly_ambiguous: 6/6
- interaction: 2/2
- not_slot: 2/2

| id | bucket | expect_ask | ask | reason | ok |
| --- | --- | --- | --- | --- | --- |
| c-ws-01 | well_specified | False | False | resolved_message | ok |
| c-ws-02 | well_specified | False | False | resolved_message | ok |
| c-ws-03 | well_specified | False | False | resolved_message | ok |
| c-pr-01 | profile_resolvable | False | False | resolved_message | ok |
| c-pr-02 | profile_resolvable | False | False | resolved_profile | ok |
| c-pr-03 | profile_resolvable | False | False | resolved_profile | ok |
| c-pr-04 | profile_resolvable | False | False | resolved_message | ok |
| c-pr-05 | profile_resolvable | False | False | named_not_owned | ok |
| c-hr-01 | history_resolvable | False | False | resolved_history | ok |
| c-hr-02 | history_resolvable | False | False | resolved_history | ok |
| c-hr-03 | history_resolvable | False | False | resolved_message | ok |
| c-ta-01 | truly_ambiguous | True | True | missing | ok |
| c-ta-02 | truly_ambiguous | True | True | missing | ok |
| c-ta-03 | truly_ambiguous | True | True | missing_multiple | ok |
| c-ta-04 | truly_ambiguous | True | True | missing_multiple | ok |
| c-ta-05 | truly_ambiguous | True | True | missing_no_products | ok |
| c-int-01 | interaction | False | False | defer_escalation | ok |
| c-int-02 | interaction | False | False | reask_guard | ok |
| c-int-03 | truly_ambiguous | True | True | missing_multiple | ok |
| c-ns-01 | not_slot | False | False |  | ok |
| c-ns-02 | not_slot | False | False |  | ok |
