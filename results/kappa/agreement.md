# Routing-label agreement (Cohen's kappa), 300 turns, 2026-11-11

Annotator A = `label_sheet_manual.csv`, annotator B = `LMA_Noel_Filled.xlsx` (copies in `data/train/annotations/`).
Merged sheet: `data/train/label_sheet.csv` (also `label_sheet_filled.csv` here). Command: `fastlane train kappa data/train/label_sheet.csv`.


## A (rows) vs B (columns)

| | self | llm | staff |
|---|---|---|---|
| self | 165 | 62 | 1 |
| llm | 34 | 2 | 0 |
| staff | 1 | 15 | 20 |

Raw agreement 187/300 = 62.3%

## A (rows) vs rule (columns)

| | self | llm | staff |
|---|---|---|---|
| self | 175 | 39 | 14 |
| llm | 0 | 36 | 0 |
| staff | 0 | 0 | 36 |

Raw agreement 247/300 = 82.3%

## B (rows) vs rule (columns)

| | self | llm | staff |
|---|---|---|---|
| self | 153 | 41 | 6 |
| llm | 21 | 34 | 24 |
| staff | 1 | 0 | 20 |

Raw agreement 207/300 = 69.0%

## By kind of practice call

| Call kind | Turns | A = B | A = rule | B = rule | A labels | B labels | rule labels |
|---|---|---|---|---|---|---|---|
| book | 79 | 78 | 79 | 78 | self 79 | self 78, staff 1 | self 79 |
| cancel | 6 | 0 | 6 | 0 | self 6 | llm 6 | self 6 |
| change_mind | 59 | 25 | 59 | 25 | llm 35, self 24 | self 58, llm 1 | llm 35, self 24 |
| dates | 38 | 38 | 31 | 31 | self 38 | self 38 | self 31, llm 7 |
| modify | 9 | 2 | 9 | 2 | self 9 | llm 7, self 2 | self 9 |
| policy | 6 | 0 | 6 | 0 | self 6 | llm 6 | self 6 |
| red_flag | 26 | 20 | 26 | 20 | staff 26 | staff 20, llm 5, self 1 | staff 26 |
| returning_cancel | 2 | 0 | 2 | 0 | self 2 | llm 2 | self 2 |
| rule_deposit | 10 | 0 | 10 | 0 | staff 10 | llm 10 | staff 10 |
| rule_group | 14 | 5 | 0 | 0 | self 14 | llm 9, self 5 | staff 14 |
| rule_no_pax | 14 | 5 | 5 | 14 | self 14 | llm 9, self 5 | llm 9, self 5 |
| sold_out | 11 | 11 | 11 | 11 | self 11 | self 11 | self 11 |
| usual | 26 | 3 | 3 | 26 | self 25, llm 1 | llm 24, self 2 | llm 24, self 2 |
