# Detailed Analysis Notes

These notes expand the one-page case study. The headline metric is the official A.S.E `overall` score.

## Four-Way Comparison

| Scheme | Runs | Overall Mean | Overall Std | Best Overall | Delta vs Static | Delta vs Random |
|---|---:|---:|---:|---:|---:|---:|
| AI General Search | 16 | 30.22 | 1.14 | 34.08 | +2.25 | +1.25 |
| AI Prompt-Iterative | 16 | 30.10 | 1.46 | 34.08 | +2.13 | +1.13 |
| Random Baseline | 16 | 28.97 | 2.43 | 30.10 | +1.00 | 0.00 |
| Static Baseline | 16 | 27.97 | 4.17 | 30.10 | 0.00 | -1.00 |

## Interpretation

The strongest claim supported by the official metric is modest but clear: both AI-driven schemes improve the mean score over random/static baselines under the same evaluation budget.

The most useful result is not a smooth increasing curve across rounds. Instead, AI search occasionally discovers better candidates, while random/static baselines do not reach the same best score. This is why the case study should emphasize candidate search and robustness rather than monotonic optimization.

Variance is also meaningful. Static baseline has the largest standard deviation (4.17), while AI General Search has the smallest (1.14). This suggests the AI-guided search space is producing more consistently viable candidates, even though the absolute mean improvement is small.

## Iteration-Level Detail

| Route | Round 1 Mean | Round 2 Mean | Round 3 Mean | Round 4 Mean | Best Round |
|---|---:|---:|---:|---:|---|
| AI General Search | 31.10 | 30.10 | 29.60 | 30.10 | Round 1 |
| AI Prompt-Iterative | 29.60 | 31.09 | 29.60 | 30.10 | Round 2 |

AI General Search finds its best candidate early. AI Prompt-Iterative improves in Round 2, which is more consistent with the feedback-loop story. Later rounds flatten because the official score is coarse and many candidate differences map to the same outcomes.

## Metric Policy for Report

Use official A.S.E metrics for the main result:

- `overall_mean`
- `overall_std`
- `best_overall`

Mention custom metrics only as auxiliary diagnostics:

- `combined`
- `contrast`
- `refined`

Do not mix official and custom metrics in the same headline table.

## Suggested Wording

"Our AI-driven search does not show a perfectly monotonic improvement curve, but it does improve the final candidate distribution: both AI schemes achieve higher mean official A.S.E scores than random/static baselines, reach a higher best score, and show lower variance than the static baseline. This suggests that the AI feedback loop is useful as a candidate generator, while the coarse official score and small task slice limit how much smooth optimization can be observed."
