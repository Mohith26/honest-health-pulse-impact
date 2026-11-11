# PULSE impact ranking for Honest Health (concept)

**Live demo:** https://mohith26.github.io/honest-health-pulse-impact/

An unofficial concept feature for [Honest Health](https://honesthealth.com)'s PULSE: rank post-discharge follow-up calls by **readmissions a call can prevent** (uplift), not by readmission risk. Care teams have capacity for only some of the day's discharges; the sickest patients often readmit whether or not anyone calls, while a call changes outcomes most for patients going home alone, on many medications, without a recent PCP visit.

Unofficial concept demo by Mohith Gajjela. Not affiliated with Honest Health. Synthetic data; no patient information.

## What the demo shows

* **Worklist.** 64 Medicare patients discharged in the last 48 hours across three partner health systems, capacity 13 calls (20%). Each row has 30-day risk, the estimated risk reduction from a call with a 90% interval, and the three features that move that patient's benefit most. Toggle between ranking by impact and by risk; tags show who each list adds.
* **Impact.** Cross-fitted, doubly robust evaluation on 30,000 synthetic discharges (Jan to Sep 2026) with a capacity slider and a capacity curve.
* **Calibration.** Predicted vs observed benefit by fifth of predicted impact.
* **Check against the truth.** Because the data is synthetic, the true effect of every call is known, so the page shows the true numbers next to the estimates.

Headline numbers (all computed by the page; `node scripts/generate.mjs` prints the same in `data/report.json`):

| At 20% capacity, per 1,000 discharges | Estimated (AIPW, 90% CI) | True (synthetic) |
|---|---|---|
| Readmissions avoided, rank by impact | 19.0 (16.1 to 21.7) | 18.3 |
| Readmissions avoided, rank by risk | 10.8 (6.0 to 14.7) | 13.0 |
| Extra from re-ranking the same calls | +8.2 (4.2 to 12.3) | +5.3 (+41%) |
| Current practice (care managers called 41% of discharges) | 16.6 | 18.4 |
| Best possible with perfect knowledge | | 21.4 |

So in this world, calling 20% of discharges by impact avoids about as many readmissions as the 41% that were actually called, and captures 86% of the best achievable at that capacity (risk ranking: 61%). The estimator over-states the re-ranking gain on this seed (+8.2 vs a true +5.3), but the truth is inside its interval; across seeds the AIPW error averages near zero (see tests).

## How it works

`core/impact.js` (pure ES module shared by the page and the tests):

1. **Nuisance models, cross-fitted.** L2 logistic regressions (Newton/IRLS) for the probability of being called (`e`), and readmission without (`m0`) and with (`m1`) a call. History is non-random: care managers favored sicker patients with a PCP on record.
2. **DR-learner for the benefit of a call** (Kennedy 2023). Pseudo-outcome `φ = m0 − m1 + (1−T)(Y−m0)/(1−e) − T(Y−m1)/e` (propensity clipped to [0.05, 0.95]); ridge regression of φ on patient features plus discharge-home interactions gives each patient's estimated risk reduction. Per-patient 90% intervals use the ridge sandwich (HC0) covariance.
3. **Capacity.** Call the top k by estimated benefit. "Why" chips are the largest coefficient x (feature − average) contributions.
4. **Evaluation.** 5 blocked folds over the 9 months: each block is scored by models trained on the other blocks. The readmission rate under any call list is estimated with AIPW (`π·(m1 + T(Y−m1)/e) + (1−π)·(m0 + (1−T)(Y−m0)/(1−e))`), with a 200-sample paired bootstrap for the intervals. Calibration groups patients into fifths by predicted benefit.

### The synthetic world

`buildWorld()` (seed `20261110`) draws 30,000 Medicare discharges with age, sex, diagnosis group, HCC score, prior admissions, ED visits, discharge medications, living alone, recent PCP visit, disposition, length of stay, dual eligibility and market. True readmission risk rises with HCC, prior admissions and diagnosis; the true effect of a 48-hour call is largest for home discharges, living alone, 12+ medications, no recent PCP visit and CHF/COPD, and fades for SNF discharges, elective joints and the very sickest. The historical calling rule favors high-HCC, high-utilization patients with a PCP, so naive comparisons of called vs not-called patients are confounded.

## Run it

```bash
python3 -m http.server     # open http://localhost:8000
npm test                   # node --test test/impact.test.mjs (11 tests)
node scripts/generate.mjs  # writes data/discharges_jan_sep_2026.csv and data/report.json
```

Tests: deterministic world; logistic regression recovers known coefficients; sandwich SE matches the empirical spread; estimated uplift correlates with true uplift (> 0.6) while risk does not (< 0.4); AIPW policy values are close to the oracle at 20% and the true re-ranking gain is inside the bootstrap interval; impact beats risk in truth at 10-30% capacity; impact at 20% matches current practice at 41%; AIPW error averages near zero across 6 seeds; capacity and fold coverage; calibration ordering; worklist intervals and reasons.

## Honest limitations

* **Synthetic world.** Effect sizes and who benefits are planted. The real gain depends on how much the benefit of a call varies across patients and how well features capture it.
* **Needs variation in who was called** and overlap (every kind of patient sometimes called, sometimes not). Unmeasured reasons for calling (a nurse's judgment, patient reachability) would bias the history. A production version should start with a small randomized slice of the worklist to confirm effects and keep it running as a check.
* **Linear benefit model.** The ridge CATE stage with hand-picked interactions ranks well at the top but is noisy in the middle (see calibration); a causal forest or boosted DR-learner would likely do better with real data.
* **Interval estimates are noisy** at the policy level with 30,000 discharges; the page shows intervals and the truth for that reason.
* **UI.** No in-app screenshots of PULSE are public. The UI follows Honest Health's web design system (Nunito Sans, teal #37a38f, purple #6a4c91, blue #6985c3, the Honest Estimator's vertical stepper, gray option tiles, dotted dividers and lavender cards) and uses the public logo.

## Novelty note

Checked honesthealth.com (PULSE launch, Mar 17 2026; announcements through Aug 2026; ACO REACH/MSSP pages; Honest Estimator; Our Team) and press coverage: PULSE "detects critical signals and highlights the next best actions," with initial focus on post-discharge follow-up and avoidable readmissions, but nothing public describes ranking by the expected effect of outreach or validating it causally. "Impactability" targeting exists elsewhere (Evolent, ClosedLoop, CCPN, a 2018 Medicaid study in Population Health Management), so this is new for PULSE rather than new to the field; the contribution is the capacity-constrained worklist with per-patient intervals and an honest, doubly robust validation loop.

## Files

`index.html`, `styles.css`, `app.js` (page), `core/impact.js` (world, models, evaluation), `assets/` (logo, favicon), `scripts/generate.mjs`, `test/impact.test.mjs`, `data/`.
