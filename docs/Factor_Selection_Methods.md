# Factor Selection Analysis – NYC Environment & Cancer

> Markdown copy of the Claude Doc "Factor Selection Analysis – NYC Environment & Cancer" (https://claude.ai/artifact/9gGejEKoigtX5LMCaYYhKT). Every number comes from `outputs/analysis/analysis_results.json`; the interactive charts are in the online doc and in tab 5 of `outputs/NYC_Environment_Cancer_Explorer.html`.

Nov 20, 2026 · Duc

## Summary

**Short answer: of the 16 candidate factors, one, smoking, is selected as an important factor for lung cancer incidence across NYC neighborhoods.** No air-pollution, greenness, food-environment or climate factor shows an independent association in these data. The machine-learning models agree: the environmental factors alone predict nothing across boroughs (R² 0.00), while smoking, poverty and insurance together predict about half of the variation between unseen neighborhoods (0.54). This is an association between neighborhood averages, not proof of cause, and for the environmental factors it is a lack of evidence here, not evidence that they do not matter.

Hao asked for four things, and each has its own step below:

- *Feature selection (Steps 4 and 5).* Eight methods, from simple correlations to Random Forest and Gradient Boosting, each score every factor. A factor is kept only if most methods agree, the direction matches theory, the result survives holding out a whole borough, and there is a scientific reason for it.
- *Multicollinearity (Step 2).* Factors that rise and fall together, such as the traffic and heating pollutants (PM₂.₅, NO₂, NO, black carbon), cannot be told apart by a regression, like two people always pushing the same car. With all 16 factors PM₂.₅ has a variance inflation factor of 60 (above 10 is serious) and adding its look-alikes makes its confidence interval 5.6 times wider. We cut the set to 7 factors by rules fixed in advance (highest VIF 4.8).
- *Theoretical justification (Step 3).* A first draft of the scientific case for each factor, for the group to review together.
- *Model results and interpretation (Step 5 and the section after it).* Regression, Poisson, panel and machine-learning results, benchmarks, sensitivity checks and limits.

The final factor list (lung cancer, age-adjusted):

- *Selected (1):* smoking.
- *Flagged (3), a signal that fails a check:* precipitation (no mechanism, a negative control), no insurance (most likely a detection effect) and SO₂ (a location fingerprint).
- *Hypothesis only (5), a good scientific case but no support in these data:* PM₂.₅, NO₂, black carbon, greenness and poverty.
- *Not supported (7):* ozone, NO, SNAP retailers, supermarkets, convenience stores, mean temperature and summer maximum temperature.

## Data and design

We explain age-adjusted lung and bronchus cancer incidence across 55 NYC neighborhoods using 16 candidate factors, in two designs: one between neighborhoods and one over time.

**Outcomes.** The main outcome is the age-adjusted incidence of lung and bronchus cancer per 100,000 residents (direct standardization to the 2000 US standard population, both sexes, 19 age groups), from the SEER\*Stat export of New York State Cancer Registry public-use data, 2010–2023. Age-adjusting makes neighborhoods with older or younger residents comparable. The 2010–2023 totals are 60,799 lung cases and 586,569 cases of all sites combined, over 120,185,192 person-years. All cancers combined is a secondary outcome, and the crude lung rate is kept only as a contrast.

**Units.** 55 neighborhoods (PUMA-based; the Midtown PUMA is split into two units). 55 is a small number, which drives several choices below.

**Two designs.**

- *Between neighborhoods (n = 55).* The outcome is pooled over 2010–2023 and each factor is its period average. This asks: do neighborhoods with more of a factor have more cancer?
- *Annual panel (55 neighborhoods × 11 years, 2012–2022 = 605 rows).* This asks: when a factor changes inside a neighborhood, does incidence change too? Sulfur dioxide is left out of the panel because its data end in 2016; the other 15 factors are complete.

**The 16 candidate factors.** Smoking, insurance and greenness come from 34 UHF health areas, and the six air pollutants from 59 community districts. Both are blended onto the 55 neighborhoods by area overlap, so they are smoother than the truth. Poverty uses the nearest 5-year ACS window. Food-environment and climate factors are measured directly for each neighborhood. Factors that are strongly right-skewed (skewness above 1) are log-transformed, then every factor is converted to a z-score, so each regression coefficient means "per one standard deviation of the factor".

| Factor | Group | Measured for | Years | Used as |
| --- | --- | --- | --- | --- |
| Smoking | Behaviour & socioeconomic | 34 UHF health areas | 2010–2022 | z-score |
| Poverty | Behaviour & socioeconomic | ACS 5-year windows | 2010–2023 | z-score |
| No insurance | Behaviour & socioeconomic | 34 UHF health areas | 2012–2022 | z-score |
| Greenness (NDVI) | Green space | 34 UHF health areas | 2010–2023 | z-score |
| SO₂ | Air quality | 59 community districts | 2010–2016 | log, then z-score |
| Ozone (O₃) | Air quality | 59 community districts | 2010–2023 | z-score |
| NO₂ | Air quality | 59 community districts | 2010–2023 | z-score |
| NO | Air quality | 59 community districts | 2010–2023 | log, then z-score |
| PM₂.₅ | Air quality | 59 community districts | 2010–2023 | log, then z-score |
| Black carbon | Air quality | 59 community districts | 2010–2023 | z-score |
| SNAP retailers | Food environment | 55 neighborhoods (direct) | 2010–2023 | z-score |
| Supermarkets | Food environment | 55 neighborhoods (direct) | 2010–2023 | z-score |
| Convenience stores | Food environment | 55 neighborhoods (direct) | 2010–2023 | z-score |
| Mean temperature | Climate | 55 neighborhoods (direct) | 2010–2023 | z-score |
| Summer max temp. | Climate | 55 neighborhoods (direct) | 2010–2023 | z-score |
| Precipitation | Climate | 55 neighborhoods (direct) | 2010–2023 | z-score |

## Step 1: exploring and screening the factors

Of the 16 factors, only smoking has a strong, clean relationship with age-adjusted lung incidence on its own (ρ = +0.61); the other 15 are weak or move with location or neighborhood income. This is the "explore, don't include everything" step, done before any model.

**What we looked at, per factor.**

- *Spatial share.* How much of the factor's variation is a real difference between neighborhoods, as opposed to change from year to year. A factor with a low spatial share (mean temperature 9%, precipitation 1%) hardly distinguishes one neighborhood from another.
- *Borough share.* How much of the variation between neighborhoods is explained by the borough alone. A high value means the factor mostly says *where* a neighborhood is, not what it is like (SO₂ 68%, NO 66%, black carbon 62%).
- *Spearman rank correlation (ρ)* with age-adjusted lung incidence, with a 95% bootstrap interval. +1 means neighborhoods that rank high on the factor also rank high on incidence; 0 means no relation. With 55 neighborhoods, a ρ smaller than about 0.27 in either direction cannot be told apart from zero.
- The same ρ for the *crude* lung rate and for *all sites combined*, to see whether the age adjustment changes the picture.

| Factor | Spatial share | Borough share | ρ, lung age-adjusted (95% CI) | ρ, lung crude | ρ, all sites |
| --- | --- | --- | --- | --- | --- |
| Smoking | 33% | 33% | **+0.61** (+0.40 to +0.78) | +0.19 | +0.34 |
| Precipitation | 1% | 36% | **+0.36** (+0.08 to +0.59) | +0.08 | +0.30 |
| No insurance | 49% | 29% | −0.24 (−0.48 to +0.03) | −0.53 | −0.47 |
| Summer max temp. | 16% | 42% | +0.18 (−0.14 to +0.48) | −0.04 | +0.13 |
| SO₂ | 32% | 68% | −0.17 (−0.45 to +0.13) | −0.26 | −0.11 |
| NO | 34% | 66% | −0.16 (−0.44 to +0.14) | −0.40 | −0.14 |
| Supermarkets | 96% | 48% | −0.13 (−0.41 to +0.18) | −0.34 | −0.06 |
| NO₂ | 60% | 57% | −0.11 (−0.40 to +0.20) | −0.40 | −0.10 |
| SNAP retailers | 96% | 34% | −0.11 (−0.40 to +0.19) | −0.51 | −0.19 |
| Convenience stores | 94% | 28% | −0.11 (−0.39 to +0.18) | −0.54 | −0.24 |
| PM₂.₅ | 24% | 50% | −0.05 (−0.32 to +0.26) | −0.33 | −0.03 |
| Ozone (O₃) | 54% | 58% | +0.02 (−0.26 to +0.30) | +0.23 | −0.12 |
| Poverty | 95% | 35% | +0.02 (−0.23 to +0.26) | −0.40 | −0.13 |
| Black carbon | 40% | 62% | −0.02 (−0.30 to +0.29) | −0.25 | −0.02 |
| Mean temperature | 9% | 18% | −0.01 (−0.29 to +0.27) | −0.30 | −0.18 |
| Greenness (NDVI) | 91% | 45% | −0.00 (−0.28 to +0.26) | +0.31 | +0.05 |

Bold = the 95% interval excludes zero. Sorted by the size of ρ for age-adjusted lung incidence.

**What this shows.**

- Smoking and precipitation are the only two factors whose interval excludes zero. Precipitation has no plausible mechanism and a 1% spatial share, so its ρ is a location fingerprint, not an effect.
- Most factors look *negatively* related to the crude lung rate (−0.25 to −0.54) but not to the age-adjusted one. The crude rate partly reflects how old a neighborhood's residents are, which is why the project uses age-adjusted rates.
- Poverty, greenness and every air pollutant are within noise of zero on their own. That does not show they have no effect: smoking, income and detection differences can hide a small effect in a one-factor-at-a-time look. Steps 4 and 5 test them with the other factors in the model.

**Does the factor relate to other cancer sites too?** We repeated the screen for 19 cancer sites (16 factors × 19 sites = 304 tests) and kept only the results that survive a Benjamini–Hochberg correction for false discoveries at 5%: 77 of 304. A real behavioral or environmental cause should leave a pattern that makes biological sense across sites, whereas a chance finding or a stand-in for something else should not.

- *Smoking* is significant for lung, all respiratory, larynx, oral cavity, esophagus, pancreas, kidney, liver and colorectal cancer, a set of sites with an established smoking link.
- *Poverty, no insurance, SNAP retailers and convenience stores* share one pattern: positive for larynx, stomach and liver, negative for melanoma, thyroid, bladder and leukemia. This looks like a neighborhood-wealth and screening gradient (more diagnoses of screening-detected cancers in richer areas), not a set of separate environmental effects.
- *The air pollutants* show no significant link to lung cancer in this screen. Their significant cells are for colorectal and kidney cancer, with a negative sign, which has no known pollution mechanism and likely reflects the urban core's population mix.
- *Greenness* has no significant cell at all.

The full 16 × 19 heat map is in section 1 of tab 5 of the explorer page.

## Step 2: multicollinearity

All 16 factors together are too tangled for a regression to separate (condition number 32); we cut them to a core set of 7 (condition number 5.0, highest VIF 4.8) by rules fixed in advance.

**What multicollinearity means.** Picture two people who always push a car together. You can measure how far the car moves, but you cannot tell how hard each person pushes, because one is never pushing without the other. Regression has the same problem with factors that rise and fall together: NO₂, NO, black carbon and PM₂.₅ all come from the same traffic and heating sources, so wherever one is high the others are high too. Put them all in one model and it cannot decide which one deserves the credit. The estimates become unstable, the confidence intervals balloon, and a factor can even flip sign depending on which others are included. The model may still predict well, but its coefficients can no longer be interpreted, and interpretation is the goal here.

**How we measure it.**

- *Correlation matrix.* Shows which pairs of factors overlap. We use Spearman rank correlation because several factors are skewed.
- *Variance inflation factor (VIF).* For each factor, ask how well all the other factors together can predict it: VIF = 1 / (1 − R²). A VIF of 5 means the uncertainty of that factor's coefficient is about √5 ≈ 2.2 times larger than if it stood alone. 5 to 10 is the usual warning range; above 10 is serious.
- *Condition number.* One number for the whole set of factors; above about 30 signals strong collinearity.

**What we found.**

- The strongest overlaps are PM₂.₅ with black carbon (+0.97), NO₂ with NO (+0.96), SNAP retailers with convenience stores (+0.96), and NO or NO₂ with PM₂.₅ (+0.92). Ozone is the mirror image of the combustion pollutants (about −0.85 with PM₂.₅ and NO₂), because nitric oxide from traffic destroys ozone.
- With all 16 factors, five have a VIF above 35 (PM₂.₅ 60.2, NO₂ 55.6, black carbon 52.0, ozone 42.1, SNAP retailers 35.4) and the condition number is 32.
- A concrete case: the PM₂.₅ coefficient with smoking in the model is −0.3 rate points per standard deviation (95% interval −2.6 to +1.9). Add NO₂, NO and black carbon and it becomes +3.0 (−9.8 to +15.8). The interval is 5.6 times wider and the estimate has moved to the other side of zero, although the data are exactly the same.

**What we did about it.** The regression models use a *core set*: one well-justified representative from each group of near-duplicates, and none of the factors with no scientific case. The core set was chosen by rule, not by trial and error:

1. Set aside factors excluded by design: no plausible mechanism, or too little data (SO₂, mean temperature, summer maximum temperature, precipitation).
2. Where two factors correlate at 0.85 or more, keep the one with the stronger scientific case (PM₂.₅ over NO₂, NO and black carbon; convenience stores over SNAP retailers).
3. Drop any factor whose VIF is still 5 or more. Smoking (the main confounder) and the main exposures of interest, PM₂.₅ and greenness, are protected from this step. Only ozone (VIF 8.7) was dropped here.

A purely automatic rule (repeatedly drop the highest VIF) would have removed PM₂.₅ first, the pollutant with the strongest scientific case, and kept temperature and precipitation, which have none. That is why theory guides the choice. Dropped factors still enter the machine-learning models, which tolerate correlated inputs for prediction; to deal with shared credit there, their importance is also reported for whole themes shuffled together.

| Factor | VIF, all 16 | Decision | Reason / VIF in the core set |
| --- | --- | --- | --- |
| PM₂.₅ | 60.2 | **kept** | core-set VIF 2.0 |
| NO₂ | 55.6 | dropped | near-duplicate of PM₂.₅ (correlation ≥ 0.85); PM₂.₅ has the stronger scientific case |
| Black carbon | 52.0 | dropped | near-duplicate of PM₂.₅ (correlation ≥ 0.85); PM₂.₅ has the stronger scientific case |
| Ozone (O₃) | 42.1 | dropped | variance inflation factor 8.7 (≥ 5) within the core set |
| SNAP retailers | 35.4 | dropped | near-duplicate of Convenience stores (correlation ≥ 0.85); Convenience stores has the stronger scientific case |
| NO | 23.1 | dropped | near-duplicate of PM₂.₅ (correlation ≥ 0.85); PM₂.₅ has the stronger scientific case |
| Convenience stores | 19.1 | **kept** | core-set VIF 4.8 |
| Supermarkets | 11.4 | **kept** | core-set VIF 4.0 |
| Summer max temp. | 10.7 | dropped | No established mechanism; only 16% of its variation is between neighborhoods |
| SO₂ | 9.7 | dropped | Data end in 2016, so it cannot be used in the annual panel; weak mechanism; collinear with the other air pollutants |
| Mean temperature | 9.0 | dropped | No established mechanism; only 9% of its variation is between neighborhoods |
| Greenness (NDVI) | 7.1 | **kept** | core-set VIF 1.8 |
| Precipitation | 5.6 | dropped | No plausible mechanism; only 1% of its variation is between neighborhoods; kept in the machine-learning models as a negative control |
| Poverty | 4.6 | **kept** | core-set VIF 3.3 |
| Smoking | 3.2 | **kept** | core-set VIF 1.6 |
| No insurance | 2.7 | **kept** | core-set VIF 1.5 |

The interactive correlation matrix and VIF bars are in section 2 of tab 5 of the explorer page.

## Step 3: theoretical justification (initial draft for group review)

Smoking and PM₂.₅ have the strongest published case for lung cancer; greenness and NO₂ are plausible; the food-environment factors are weak proxies and the climate factors have no mechanism at all. This is a first draft from published sources, to be checked together.

Statistical importance alone is not enough: a factor can look important because it stands in for something else (location, income, how often cancer is looked for). So each factor gets a *scientific case* rating (strong, moderate, weak, none) and an *expected direction* before it meets the models. A factor is only recommended when the data agree with the theory on both counts.

| Factor | Role | Case | Expected | Why it might matter, and the caveats |
| --- | --- | --- | --- | --- |
| Smoking | Behaviour (confounder / main exposure) | strong | more cancer | Cigarette smoking is the number-one risk factor for lung cancer and is linked to about 80% of US lung-cancer deaths ([CDC](https://www.cdc.gov/lung-cancer/risk-factors/index.html)). The survey measures current adult smoking, while lung cancer today reflects smoking decades ago, so it is a proxy for each area's smoking history. |
| Poverty | Socioeconomic context | moderate | more cancer | Lower-income areas have more smoking, more occupational and indoor exposures and worse air. The link is partly mediated by smoking, and for all cancers combined the gradient can reverse because screening-detected cancers are commoner in richer areas. |
| No insurance | Access to care (detection) | weak | unclear | Lack of insurance mainly changes whether a cancer is found (screening, diagnosis), not whether it starts, so the sign is ambiguous: poorer access lowers recorded incidence, while the socioeconomic disadvantage behind it raises risk. |
| Greenness (NDVI) | Environment (protective, hypothesis) | moderate | less cancer | Vegetation can filter and dilute air pollution, encourages physical activity and lowers heat and stress. Observational cohorts report lower lung-cancer risk with higher NDVI (hazard ratio 0.89 per 0.1 NDVI in a [2025 Korean cohort](https://www.frontiersin.org/journals/public-health/articles/10.3389/fpubh.2025.1661937/full)), but NDVI is an imperfect exposure measure and greenness also tracks lower-density suburban areas. |
| SO₂ | Air pollution (combustion) | weak | more cancer | Marker of sulfur-containing fuel combustion (heating oil). Limited evidence for lung cancer on its own, and the NYC series in this project ends in 2016. |
| Ozone (O₃) | Air pollution (secondary) | weak | unclear | Ozone is a respiratory irritant but is not an established lung carcinogen. In cities its summer mean is lower near traffic (nitric oxide destroys ozone), so it behaves as an inverse marker of the urban core. |
| NO₂ | Air pollution (traffic) | moderate | more cancer | Traffic-related pollutant used as a marker of the traffic mixture; several cohort studies link it to lung-cancer incidence, but it is hard to separate from PM₂.₅ and other traffic emissions. |
| NO | Air pollution (fresh combustion) | weak | more cancer | Fresh combustion emission that converts to NO₂ within minutes. No separate carcinogenic mechanism is established; it is almost a duplicate of NO₂. |
| PM₂.₅ | Air pollution (particles) | strong | more cancer | IARC classified outdoor air pollution, and particulate matter within it, as carcinogenic to humans (Group 1) in 2013, with sufficient evidence for lung cancer and a rising risk with rising exposure ([IARC press release 221](https://www.iarc.who.int/wp-content/uploads/2018/07/pr221_E.pdf)). |
| Black carbon | Air pollution (diesel / combustion) | moderate | more cancer | Black carbon tracks diesel and other combustion soot; IARC classified diesel engine exhaust as carcinogenic to humans (Group 1, lung cancer) in 2012 ([IARC press release 213](https://www.iarc.who.int/wp-content/uploads/2018/07/pr213_E.pdf)). Few studies use black carbon itself, and in NYC it is almost identical to PM₂.₅. |
| SNAP retailers | Food environment (proxy) | weak | unclear | Count of SNAP-authorised stores per km². Mostly reflects how dense and commercial an area is; diet effects on lung cancer are modest and not captured by store counts. |
| Supermarkets | Food environment (diet quality) | weak | less cancer | Better access to supermarkets could improve diet (fruit and vegetable intake is weakly protective for lung cancer). Per-km² counts mainly track urban density. |
| Convenience stores | Food environment / tobacco retail | weak | more cancer | Convenience stores are a main tobacco retail channel, so store density may act as a smoking-access proxy (a hypothesis; **not yet checked against the literature**, to be verified by the group). As a density measure it also tracks the urban core. |
| Mean temperature | Climate | none | unclear | No established mechanism for lung-cancer incidence. Annual mean temperature mostly separates the dense, hotter core from the outer areas (urban heat island). |
| Summer max temp. | Climate | none | unclear | Summer heat is a health hazard for other outcomes (heat illness, ozone formation) but has no established link to lung-cancer incidence. Skin cancer depends on UV, not temperature. |
| Precipitation | Climate (negative control) | none | unclear | No plausible mechanism. Almost all of its variation is year-to-year weather shared by the whole city, so any neighborhood-level pattern is a location fingerprint. |

**How the ratings are used.** "Strong" and "moderate" factors can be *selected* when several methods agree and the direction matches. "Weak" or "none" factors can at most be *flagged* as a statistical signal that needs another explanation. A factor whose direction is opposite to the expected one is marked as probable confounding or detection rather than an effect.

**For the group to check.**

- Is there literature on tobacco retail density (convenience stores) and smoking or lung cancer, and does it apply to NYC?
- Other well-known lung-cancer factors are missing from the data: radon, second-hand smoke, occupational exposures (the [CDC](https://www.cdc.gov/lung-cancer/risk-factors/index.html) lists all three), and immigrant and smoking histories.
- Should NO₂ or black carbon replace PM₂.₅ as the representative of the air-pollution block? Sensitivity checks in Step 5 do this swap.

## Step 4: methods, what we used and why

We ran eight methods, from simple correlations to tree-based machine learning, on the same 55 neighborhoods, and keep a factor only when several of them agree and the scientific case supports it.

No single method is trusted on its own. Each one answers a slightly different question and has different blind spots, so consistency across methods (Hao's point 4) is the evidence.

| # | Method | The question it answers | Why it is here | Main weakness |
| --- | --- | --- | --- | --- |
| 1 | Spearman screen | Do neighborhoods that rank high on this factor also rank high on cancer incidence? | Simple, no model assumptions, robust to outliers and skewed factors | One factor at a time: cannot separate it from smoking or from look-alike factors |
| 2 | OLS adjusted for smoking | Does the factor still matter once smoking is taken into account? | Smoking is the dominant known cause of lung cancer, so this is the minimum fair test | Smoking is measured as current adult smoking, not smoking history |
| 3 | OLS on the core set (7 factors) | What is the effect of each factor with the others held fixed? | Gives an effect size and a confidence interval that a reader can interpret | Needs low multicollinearity, hence the core set; assumes straight-line effects |
| 4 | Poisson regression with age groups | The same, but using the actual case counts and person-years in each age group | Handles the count nature of cancer data and age structure directly, instead of a pre-computed rate | Assumes the log of the rate is linear in the factors |
| 5 | Panel two-way fixed effects | Do neighborhoods where the factor *changed* also see incidence change? | Removes everything constant within a neighborhood (borough, history, demographics) | For most factors almost nothing changes within a neighborhood (2–3% of the variation in poverty, greenness, food stores and PM₂.₅), and incidence lags exposure by years or decades, so the test has little power |
| 6 | Elastic Net with stability selection | Which factors does a penalised regression keep when it is forced to be sparse, and is that choice stable? | Built for many correlated predictors; repeating on 400 random half-samples shows which choices are stable | Picks one of two look-alikes more or less at random |
| 7 | Random Forest | Which factors do many decision trees need to predict unseen neighborhoods? | Handles curved and interacting effects; no transformation of factors needed | Importance is shared out between look-alike factors; can latch on to location |
| 8 | Gradient Boosting | The same question with a different way of building the trees | A second tree method guards against one algorithm's quirks | Same as the Random Forest; easier to overfit with only 55 neighborhoods |

**How a method "supports" a factor.** Spearman, both OLS versions, Poisson and the fixed-effects model: p < 0.05. Elastic Net: the factor is kept in at least 60% of 400 random half-samples. Random Forest and Gradient Boosting: shuffling the factor costs at least 1 R² point on average on held-out neighborhoods, and the cost is positive in at least 80% of the cross-validation folds.

**Regression details.** OLS uses heteroscedasticity-robust (HC3) standard errors. The Poisson model uses 12 age groups for lung cancer and 19 for all sites, with log person-years as the exposure offset and standard errors clustered by neighborhood. Every factor enters as a z-score (log first when skewed), so a coefficient is the change per one standard deviation.

**Over time (annual panel, 2012–2022).** Four versions are fitted: pooled regression with year effects (weighted by population), the two-way fixed-effects regression (neighborhood and year effects, so only *changes within* a neighborhood count), and Poisson versions of both. A Random Forest is also fitted on year-centred data, so it cannot learn the citywide trend, with whole neighborhoods held out.

**Machine-learning settings.** Random Forest: 300 trees, one third of the factors tried at each split, at least 3 neighborhoods per leaf. Gradient Boosting: 200 shallow trees (depth 2), learning rate 0.05, each tree built on a random 70% of the data. Elastic Net: penalty strength and mix chosen by cross-validation.

**Two ways of testing "unseen" neighborhoods.** *Random 5-fold cross-validation* hides one fifth of the neighborhoods, trains on the rest, predicts the hidden ones, and repeats eight times with different splits. *Leave-one-borough-out* hides a whole borough and predicts it from the other four. The second is much harder, because nearby neighborhoods resemble each other: a factor that only works as a stand-in for "where in the city" helps in the first test and fails in the second. This is how we spot a location fingerprint. We report R² on the hidden neighborhoods: 1 is perfect, 0 is no better than guessing the city average, and a negative value is worse than that.

**Permutation importance.** After a model is trained, the values of one factor are shuffled across the held-out neighborhoods. The factor's importance is how many R² points the predictions lose (average of 6 shuffles per fold). It is computed on held-out data, not the training data, so that a factor cannot look useful merely because the model memorised it. Factors are also shuffled in themed groups (smoking, poverty and insurance, greenness, air quality, food, climate) because look-alike factors share credit when shuffled one at a time.

**Partial dependence.** For the most important factors, the model's average prediction is plotted as that factor varies and the others stay as observed. This gives the direction and shape of the relationship (a straight rise, a plateau, a threshold).

**Benchmarks.** To see what each group of factors adds, the same cross-validation is run for: the map location alone (x, y coordinates), smoking alone, poverty and insurance alone, smoking plus poverty plus insurance, the 13 environmental factors alone, the core set, and all 16 factors (OLS, Ridge, Elastic Net, Random Forest, Gradient Boosting), plus a Random Forest without smoking.

**Residual check.** Moran's I of the regression residuals (five nearest neighbors, 999 permutations) tests whether nearby neighborhoods have similar errors. If they do, ordinary p-values are too optimistic.

**Sensitivity.** The core-set regression is repeated eight ways: without Staten Island; swapping PM₂.₅ for NO₂ or for black carbon; without insurance (a detection variable); with the crude rate; with all cancers as the outcome; and without 2020 (the pandemic diagnosis dip).

**Combining the methods into a verdict.** For each factor we count how many of the methods that could test it support it. Two thresholds decide the label:

- *Selected:* at least 60% of the methods support it, they agree on the direction, the scientific case is strong or moderate, the direction matches the theory, and the Random Forest or Gradient Boosting support does not vanish when a whole borough is held out.
- *Flag:* a statistical signal (some methods support it) that fails one of those checks: the scientific case is weak or none, the direction is opposite to the theory, or the support looks like a location fingerprint.
- *Hypothesis only:* a strong or moderate scientific case, but fewer than 25% of the methods support it.
- *Not supported:* neither a statistical signal nor a scientific case.

**Tools.** Everything is written in Python with NumPy, pandas, SciPy and scikit-learn. The regression, Poisson and fixed-effects code was written directly in NumPy (the statsmodels package could not be installed in this environment) and checked against scikit-learn and SciPy; the test file is `scripts/analysis/test_stats_lib.py`.

## Step 5: results

### Regression: each factor with the others held fixed

In the regressions only smoking has a clear, consistent positive effect on lung cancer incidence; the neighborhood-level link of greenness and PM₂.₅ to incidence is indistinguishable from zero. Effects are per one standard deviation of the factor, in new cases per 100,000 (rate points), or as a rate ratio for the Poisson model; \* marks p < 0.05.

| Factor | OLS, core set: rate points per SD (p) | OLS adjusted for smoking only (p) | Poisson, core set: rate ratio per SD (p) | Panel fixed effects (p) |
| --- | --- | --- | --- | --- |
| Smoking | +6.8\* (<0.001) | +5.3\* (<0.001) | 1.172\* (<0.001) | +0.0 (0.97) |
| Poverty | −1.9 (0.26) | −3.5\* (<0.001) | 0.963 (0.22) | +2.0 (0.30) |
| No insurance | −2.2\* (0.008) | −3.3\* (<0.001) | 0.948\* (0.006) | −0.5 (0.24) |
| Greenness (NDVI) | +0.6 (0.56) | +1.2 (0.26) | 1.006 (0.78) | −1.4 (0.26) |
| SO₂ | – | −1.6 (0.11) | – | – |
| Ozone (O₃) | – | −0.6 (0.47) | – | – |
| NO₂ | – | −0.6 (0.66) | – | – |
| NO | – | −0.8 (0.52) | – | – |
| PM₂.₅ | −0.2 (0.84) | −0.3 (0.76) | 1.007 (0.73) | +1.2 (0.39) |
| Black carbon | – | −0.4 (0.69) | – | – |
| SNAP retailers | – | −2.4\* (0.016) | – | – |
| Supermarkets | +1.0 (0.51) | −1.0 (0.36) | 1.022 (0.47) | +2.2 (0.21) |
| Convenience stores | −1.0 (0.60) | −2.6\* (0.014) | 0.982 (0.63) | −2.1 (0.28) |
| Mean temperature | – | −0.9 (0.34) | – | – |
| Summer max temp. | – | +0.2 (0.80) | – | – |
| Precipitation | – | +1.1 (0.23) | – | – |

A dash means the factor is not in that model: the core-set OLS, Poisson and panel models all use the same 7 core factors. The core-set OLS explains 56% of the variation (adjusted R²) in lung incidence.

- **Smoking** is the one robust factor: +6.8 rate points per SD (rate ratio 1.17, so about 17% higher incidence per SD) and stable across the OLS and Poisson versions. It disappears in the over-time model (+0.0, p = 0.97), discussed below.
- **No insurance and poverty point the "wrong" way.** Neighborhoods with more uninsured residents (−2.2 per SD) or more poverty (−3.5 per SD once smoking is held fixed) have *lower* recorded lung incidence, as do areas with more SNAP and convenience stores. This is the opposite of what poverty does to lung cancer risk for individuals, so we read it as detection and population mix (richer, better-insured neighborhoods diagnose and record more cancers), not as an environmental effect. It is flagged, not selected.
- **PM₂.₅ and greenness** show no independent association (−0.2, p = 0.84 and +0.6, p = 0.56). Their intervals are wide, so the data cannot rule out effects of a few rate points either way.
- **Residual clustering.** Moran's I of the core-set residuals is 0.05 (p = 0.16): after the core set, nearby neighborhoods no longer have similar errors. With smoking alone it is 0.26 (p = 0.001), so most of the remaining geographic pattern is explained by the other core factors. For all cancers combined the clustering stays (I = 0.21, p = 0.002), so p-values there are too optimistic.

### How well do the models predict unseen neighborhoods?

*Figure (interactive in the online doc and in tab 5 of the explorer): analysis_results.json, cross_bench for lung_aa · 55 neighborhoods · random 5-fold repeated 8 times; leave-one-borough-out*

A three-variable regression (smoking, poverty, no insurance) is the best predictor, and no machine-learning model beats it on whole boroughs.

- *Smoking alone* explains 30% of the variation across boroughs; poverty and insurance without smoking explain none (−0.01). They only help together with smoking, which lifts R² from 0.30 to 0.54.
- *Environmental factors alone do not generalize.* The 13 greenness, air-quality, food and climate factors together reach 0.15 on random hold-outs but 0.00 across boroughs, and the Random Forest without smoking drops to 0.01.
- *More factors do not help.* The 7-factor core set (which adds PM₂.₅, greenness, supermarkets and convenience stores) predicts worse (0.33) than the 3-variable model (0.54). Gradient Boosting on all 16 factors reaches 0.38.
- *Why we prune before regressing.* OLS on all 16 factors scores 0.07 on random hold-outs and −1.06 across boroughs, far worse than guessing the average: the overlapping factors make its coefficients unstable.
- The map location alone scores 0.06 across boroughs, so the factors carry more than a location fingerprint.

### Which factors do the machine-learning models need?

*Figure (interactive in the online doc and in tab 5 of the explorer): analysis_results.json, cross.lung_aa.rf / gbm · permutation importance, mean over folds (random 5-fold × 8 repeats; leave one borough out × 3 repeats)*

Both tree models put smoking far ahead: shuffling it costs the Random Forest 23 R² points (15 with a borough held out) and Gradient Boosting 47 (37).

- *Precipitation is second in both models* (6 and 18 points; 3 and 6 across boroughs), although it has no mechanism and only 1% of its variation lies between neighborhoods. Read this as a warning: importance rewards any variable that follows the city's geographic gradients, so machine-learning importance alone cannot show that a factor matters. It is why precipitation is flagged, not selected.
- *The environmental factors of interest are at or below the 1-point threshold.* In the Random Forest PM₂.₅ scores −0.7, NO₂ 0.0, black carbon −0.5 and greenness 0.1; in Gradient Boosting −0.6, −0.2, −0.3 and 0.4. A negative value means shuffling the factor did not hurt at all.
- *Some factors look important only on random hold-outs.* SO₂ (2.7 points), summer maximum temperature (2.4) and NO (2.0) in the Random Forest fall to −0.6, −0.2 and 0.0 when a whole borough is held out: location fingerprints.
- *Shared credit.* Shuffled one at a time, each air pollutant looks unimportant because its look-alikes stand in for it. Shuffled together, the air-quality theme costs the Random Forest 9.3 points, but only 1.6 across boroughs, so most of that is the urban-versus-outer gradient rather than a pollution effect.
- *One result to watch.* In Gradient Boosting, convenience stores gain importance when a borough is held out (5.2 points) while the Random Forest does not see it (0.8). It fails the random hold-out bar (2.8 points, positive in only 70% of folds), so it is not counted as support.

### Over time: do changes within a neighborhood show up?

The comparisons above are between places. The annual panel asks a different question: when the *same* neighborhood's factor rises or falls from one year to the next, does its lung cancer rate move too? Comparing a neighborhood only with itself removes everything fixed about it (borough, history, who lives there, how cases are found), so it is the cleanest test the data allows. The panel has 55 neighborhoods × 11 years (2012–2022) = 605 neighborhood-years; SO₂ is left out because its series ends in 2016. Effects are per one standard deviation of the factor, in rate points per 100,000 or, for Poisson, as a rate ratio (1.00 = no change). Parentheses are 95% confidence intervals; \* marks p < 0.05.

| Factor (per 1 SD) | Pooled, year effects | Two-way fixed effects | Poisson, year effects | Poisson, fixed effects |
| --- | --- | --- | --- | --- |
| Smoking | +4.5 (+2.6 to +6.3)\* | +0.0 (−0.7 to +0.8) | 1.11 (1.07 to 1.16)\* | 1.00 (0.98 to 1.01) |
| Poverty | +1.2 (−1.2 to +3.6) | +2.0 (−1.8 to +5.8) | 1.04 (0.98 to 1.10) | 1.08 (0.98 to 1.18) |
| No insurance | −3.1 (−4.6 to −1.6)\* | −0.5 (−1.4 to +0.4) | 0.92 (0.89 to 0.96)\* | 0.99 (0.97 to 1.01) |
| Greenness (NDVI) | +0.6 (−1.9 to +3.0) | −1.4 (−3.9 to +1.1) | 1.00 (0.95 to 1.06) | 0.96 (0.92 to 1.01) |
| PM₂.₅ | +0.1 (−3.7 to +3.8) | +1.2 (−1.6 to +4.0) | 1.00 (0.92 to 1.09) | 1.03 (0.96 to 1.10) |
| Supermarkets | −0.2 (−2.4 to +2.1) | +2.2 (−1.3 to +5.8) | 0.99 (0.94 to 1.05) | 1.05 (0.96 to 1.15) |
| Convenience stores | −1.3 (−4.0 to +1.3) | −2.1 (−6.0 to +1.8) | 0.96 (0.90 to 1.03) | 0.96 (0.88 to 1.05) |

- *Smoking is strong across places and absent within places.* Pooled over neighborhoods and years, one SD more smoking goes with 4.5 more lung cancer cases per 100,000 (11% more in the Poisson model). Compare each neighborhood only with itself and the effect is +0.0 (−0.7 to +0.8; p = 0.97). It is not for lack of movement: after removing the citywide trend, 39% of the variation in smoking is neighborhood-specific change, and the citywide average fell from 15.7% of adults in 2012 to about 9% in 2022. A likely reason is timing: lung cancer appears years to decades after smoking, so a change during 2012–2022 should not show up in the same years' incidence, while the difference between neighborhoods reflects long-standing smoking habits. This data cannot test that explanation.
- *No insurance behaves the same way:* −3.1 pooled (p < 0.001) becomes −0.5 within neighborhoods (p = 0.24), so the pattern belongs to what kind of place it is, not to year-to-year change.
- *For the environmental and food factors the panel is nearly blind.* Only 2–3% of the variation in poverty, greenness, supermarkets and convenience stores is neighborhood-specific change, and PM₂.₅ moves mostly citywide (66%; 2% neighborhood-specific). Their within-neighborhood estimates (PM₂.₅ +1.2, greenness −1.4) are consistent with no effect and with modest effects in either direction. So for these factors a null here means not detected, not ruled out.
- *Machine learning on the panel* (year-centred data, whole neighborhoods held out) predicts 0.23 of the variation on a random hold-out but only 0.02 with a whole borough held out. Smoking is again its top factor (7.2 R² points; about 4 across boroughs).

### Putting it together: which factors survive?

Each row is a factor and each column a method (lung cancer, age-adjusted). ● = the method supports the factor, ○ = tested but not supported, – = the method did not include that factor (the regressions and the panel use only the 7-factor core set, and the smoking-adjusted regression cannot test smoking itself). Support counts the methods that agree out of those that tested the factor.

| Factor | Spearman screen | OLS + smoking | OLS (core set) | Poisson (core set) | Elastic Net | Random Forest | Gradient Boosting | Panel two-way FE | Support | Direction | Theory | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Smoking | ● | – | ● | ● | ● | ● | ● | ○ | 6/7 | higher | strong | **Selected** |
| Precipitation | ● | ○ | – | – | ● | ● | ● | – | 4/5 | higher | none | Flag |
| No insurance | ○ | ● | ● | ● | ● | ○ | ○ | ○ | 4/8 | lower | weak | Flag |
| SO₂ | ○ | ○ | – | – | ● | ○ | ● | – | 2/5 | lower | weak | Flag |
| Poverty | ○ | ● | ○ | ○ | ○ | ○ | ○ | ○ | 1/8 | lower | moderate | Hypothesis only |
| Greenness (NDVI) | ○ | ○ | ○ | ○ | ● | ○ | ○ | ○ | 1/8 | higher | moderate | Hypothesis only |
| NO₂ | ○ | ○ | – | – | ○ | ○ | ○ | – | 0/5 | none | moderate | Hypothesis only |
| PM₂.₅ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | 0/8 | none | strong | Hypothesis only |
| Black carbon | ○ | ○ | – | – | ○ | ○ | ○ | – | 0/5 | none | moderate | Hypothesis only |
| NO | ○ | ○ | – | – | ● | ○ | ○ | – | 1/5 | lower | weak | Not supported |
| SNAP retailers | ○ | ● | – | – | ○ | ○ | ○ | – | 1/5 | lower | weak | Not supported |
| Summer max temp. | ○ | ○ | – | – | ● | ○ | ○ | – | 1/5 | none | none | Not supported |
| Convenience stores | ○ | ● | ○ | ○ | ○ | ○ | ○ | ○ | 1/8 | lower | weak | Not supported |
| Ozone (O₃) | ○ | ○ | – | – | ○ | ○ | ○ | – | 0/5 | none | weak | Not supported |
| Supermarkets | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | 0/8 | none | weak | Not supported |
| Mean temperature | ○ | ○ | – | – | ○ | ○ | ○ | – | 0/5 | none | none | Not supported |

- *Selected, one factor: smoking (6 of 7 methods).* It is supported by the screen, both regression types, Elastic Net, and both tree models (including with a borough held out); the direction is the expected one and the scientific case is strong. The one miss is the within-neighborhood panel, discussed above.
- *Flagged, a signal that fails a check (3).* **Precipitation** (4 of 5) is the negative control: it has no known mechanism for cancer and only 1% of its variation lies between neighborhoods, yet four methods pick it, which is what a place-level gradient looks like. This is why a statistical signal alone never selects a factor. **No insurance** (4 of 8) goes with *lower* incidence, the opposite of a harmful effect; the most plausible reading is detection (fewer diagnoses where fewer people see a doctor), so treat it as a control, not a cause. **SO₂** (2 of 5) is picked up by Elastic Net and Gradient Boosting, but its support disappears when a whole borough is held out (68% of its variation is between boroughs), its sign is opposite to theory, and its series ends in 2016: a location fingerprint.
- *Hypothesis only, a good scientific case but no support in these data (5).* **PM₂.₅** (0 of 8), **NO₂** (0 of 5) and **black carbon** (0 of 5) are the air-pollution block with the strongest or moderate case (IARC Group 1 for outdoor air pollution and diesel exhaust), and they are too correlated to be told apart (Step 2), so PM₂.₅ stands for all three. **Greenness** (1 of 8, Elastic Net only) has the wrong sign: more green space, slightly higher incidence. **Poverty** (1 of 8) is negative once smoking is accounted for, again opposite to expectation. The PM₂.₅ and greenness estimates (−0.2 and +0.6 rate points per SD, intervals −2.3 to +1.9 and −1.4 to +2.7) rule out an effect as large as smoking's (+6.8) but not a small one.
- *Not supported (7).* Ozone, NO, SNAP retailers, supermarkets, convenience stores, mean and summer maximum temperature. NO, SNAP retailers and convenience stores each have one supporting method, in the *lower* direction, opposite to what theory predicts; the tobacco-retail argument for convenience stores (Step 3) is unverified, so it stays a question for the group.
- *All-sites outcome.* The same single factor is selected (smoking, 6 of 7), but nine factors are flagged instead of three: the all-sites rate mixes many cancers and is more tied to screening and place (no insurance is supported by 7 of 8 methods, again in the lower direction). That is why lung cancer is the primary outcome.

### Does the answer change if we change the choices?

The core-set regression is repeated eight ways (effects in rate points per SD; \* marks p < 0.05; the PM₂.₅ column shows the replacement factor where one was swapped in).

| Variation | n | adj. R² | Smoking | Poverty | No insurance | Greenness (NDVI) | PM₂.₅ | Supermarkets | Convenience stores |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline: core set, lung age-adjusted, all 55 neighborhoods | 55 | 0.56 | +6.8\* | −1.9 | −2.2\* | +0.6 | −0.2 | +1.0 | −1.0 |
| Without Staten Island (3 neighborhoods) | 52 | 0.38 | +5.8\* | −1.3 | −2.0\* | +0.2 | −0.1 | +0.9 | −1.1 |
| PM2.5 replaced by NO2 | 55 | 0.56 | +6.8\* | −1.9 | −2.2\* | +0.7 | (no2) +0.0 | +0.9 | −0.9 |
| PM2.5 replaced by black carbon | 55 | 0.56 | +6.8\* | −1.8 | −2.2\* | +0.6 | (bc) −0.2 | +1.0 | −1.0 |
| Without no-insurance (a detection variable) | 55 | 0.51 | +7.4\* | −2.6 | – | +1.2 | +0.1 | +2.1 | −2.1 |
| Crude (not age-adjusted) lung rate as outcome | 55 | 0.55 | +8.5\* | −3.7 | −5.1\* | +3.0 | −4.9 | +7.2\* | −7.2\* |
| All sites age-adjusted as outcome | 55 | 0.54 | +19.1\* | −2.9 | −16.4\* | +4.4 | −4.1 | +9.0 | −8.6 |
| Outcome without 2020 (pandemic diagnosis dip) | 55 | 0.55 | +6.8\* | −1.9 | −2.2\* | +0.6 | −0.3 | +1.0 | −0.8 |

- *Smoking holds up everywhere:* significantly positive in all eight versions (+5.8 to +8.5 rate points per SD for lung cancer; +19.1 for all sites, where rates are larger). Dropping Staten Island lowers the overall fit (adjusted R² 0.56 to 0.38) but not the smoking effect, and dropping insurance raises it slightly (+7.4), so smoking is not borrowing strength from the detection variable.
- *Air pollution and greenness stay at noise level whatever we do.* Swapping PM₂.₅ for NO₂ (+0.0) or black carbon (−0.2) changes nothing, and no variation makes PM₂.₅ or greenness significant.
- *The crude rate manufactures results.* Without age adjustment, supermarkets (+7.2\*) and convenience stores (−7.2\*) become significant, because neighborhoods differ in age mix. This is the reason the main analysis uses age-adjusted rates.

## Interpretation, cautions and limits

### What the results mean

- *Smoking is the one factor the evidence supports.* Holding the other six core factors fixed, neighborhoods with one standard deviation more smokers (about 2.4 percentage points of adults) have 6.8 more lung cancer cases per 100,000 residents (95% interval +5.2 to +8.5). It holds across the regressions, the Poisson model and both tree models.
- *Air pollution, greenness and the food environment show no independent association with lung cancer incidence between NYC neighborhoods.* That is absence of evidence here, not evidence of absence: the intervals for PM₂.₅ and greenness exclude an effect as large as smoking's but not a small one, and the panel could barely test them (Step 5, over time).
- *Machine learning agrees with the regressions.* The 13 environmental factors together explain 15% of the variation between unseen neighborhoods on a random hold-out and 0% when a whole borough is held out; the map location alone explains 23% and 6%. Smoking, poverty and insurance together explain 54% on both tests, and nothing more complicated beats that.
- *Importance is not evidence of an effect.* A factor with no mechanism (precipitation) ranks second in both tree models. We therefore never select a factor on machine-learning importance alone: it must also pass the regressions, hold up across boroughs, and have a scientific case.

### Cautions

- *Association, not causation.* The design is ecological (neighborhood averages), so it says nothing about any individual and cannot show that a factor causes cancer. "Selected" means "the strongest statistical candidate with a scientific case", nothing more.
- *Few independent data points.* There are 55 neighborhoods, but smoking, insurance and greenness are measured for only 34 health areas and the air pollutants for 59 community districts, then blended onto neighborhoods. Intervals are wide and blending makes the factors smoother than reality.
- *Smoking is measured as current smoking.* Lung cancer reflects smoking over past decades, so the smoking coefficient also stands in for anything that goes with long-term smoking (such as second-hand smoke and occupation).
- *Several known risks are not in the data:* radon, second-hand smoke and occupational exposures (the CDC lists all three alongside smoking and air pollution). Nor do we know how long people have lived in a neighborhood, and screening and diagnosis differ between neighborhoods.
- *Detection and population mix.* Poverty (−3.5 per SD) and no insurance (−3.3) are negative once smoking is accounted for: more poverty or uninsurance goes with fewer diagnosed cases. This most likely reflects who gets diagnosed and who lives where, not a protective effect, but we have not tested that.
- *Multiple testing.* With 16 factors and 8 methods some "supports" appear by chance; precipitation is the example. This is why the rule asks for agreement across methods plus a scientific case.
- *Spatial clustering of errors.* For the lung model Moran's I of the residuals is 0.05 (p = 0.16), so little pattern is left over. Smoking alone leaves a clear one (0.26, p = 0.001), which poverty and insurance absorb. For all cancers combined, clustering remains (0.21, p = 0.002), so p-values for that outcome are optimistic.
- *Tools.* statsmodels and SHAP could not be installed here, so regression, Poisson and fixed-effects models are our own NumPy code (checked against scikit-learn and SciPy; 12 tests in `test_stats_lib.py`) and importance is permutation importance, not SHAP.
- *Convenience stores.* The tobacco-retail argument (Step 3) is unverified; the data show a negative sign, opposite to it.

## For Friday's meeting

### Talking points

- Multicollinearity (Hao's point 2): with all 16 factors the model cannot separate the traffic and heating pollutants (PM₂.₅ VIF 60, condition number 32), so we reduced to a 7-factor core set by rules fixed in advance (VIF below 5, condition number 5.0). The result does not change when PM₂.₅ is swapped for NO₂ or black carbon.
- Feature selection (point 1) and model results (point 4): eight methods, one clear result. Smoking is selected; no air-pollution, greenness or food-environment factor is, and the three flagged factors (precipitation, no insurance, SO₂) fail the checks for reasons we can name.
- Theory (point 3): the draft is in Step 3. The air-pollution case is strong on paper (IARC Group 1) but not visible in neighborhood data, which is the most interesting tension for the group.

### Questions for the group

- Do we keep lung cancer as the primary outcome, or add a cancer site with a stronger environmental case?
- Poverty and no insurance behave like detection variables here. Do we treat them as controls rather than factors of interest?
- Is PM₂.₅ the right representative of the pollution block, or do we want NO₂ or black carbon (results are the same)?
- Does anyone have a source for the convenience-store and tobacco-retail link in Step 3?
- Is a neighborhood-level, 2010–2023 design enough, or do we need earlier exposure years, since lung cancer follows exposure by decades?

### Possible next steps

- Finer geography (census tracts) for more independent data points, and longer exposure histories if earlier air-quality years exist.
- Add lung-specific confounders (radon, occupation, second-hand smoke, screening uptake) before asking again whether air pollution matters.
- Review the Step 3 theory text together and agree on the final factor list.
- Use the same pipeline on other cancer sites, where the explorer's one-factor screen already shows correlations (for example colon and kidney cancer with the air pollutants). Those are unadjusted and may be place gradients, so they need the same checks.

## Appendix: glossary, files, how to reproduce, sources

**Glossary (plain language).**

| Term | What it means here |
| --- | --- |
| Age-adjusted rate | New cases per 100,000 people after removing the effect of different age mixes (here: the 2000 US standard population), so neighborhoods can be compared |
| Crude rate | Cases divided by people, with no age correction |
| PUMA | Census area of about 100,000+ residents; our 55 neighborhoods |
| UHF area / community district | Larger health-reporting areas (34 / 59 in NYC) where smoking, insurance, greenness and air quality were measured; blended onto neighborhoods by area overlap |
| Spearman ρ | Rank correlation: do neighborhoods that rank high on one measure also rank high on the other? +1 yes, 0 no relation, −1 reversed |
| 95% confidence interval | The range of values that fits the data reasonably well; if it includes zero, the data do not show a clear effect |
| p-value | How surprising the result would be if there were no real effect; below 0.05 is the usual cut-off |
| False discovery rate (FDR) | When many tests are run, some look significant by chance; the Benjamini–Hochberg correction limits the share of false alarms to 5% |
| OLS (linear regression) | Fits a straight-line relationship between the factors and the outcome |
| Poisson regression | Regression for counts of cases, using the people at risk in each age group |
| Robust / clustered standard errors | Uncertainty estimates that stay valid when the spread of the errors varies or when rows from the same neighborhood are related |
| Multicollinearity | Factors that rise and fall together, so a regression cannot tell their effects apart |
| VIF | Variance inflation factor: how much a factor's uncertainty is inflated by overlap with the other factors (5+ is a warning) |
| Condition number | One summary of overlap across the whole set of factors (about 30+ is a warning) |
| Core set | The 7 factors used in the regressions after removing near-duplicates and factors with no scientific case |
| Fixed effects | Regression that gives every neighborhood (and every year) its own baseline, so only changes within a neighborhood count |
| Elastic Net | Regression that shrinks unimportant coefficients to zero; good when factors overlap |
| Stability selection | Repeat a selection on many random half-samples and keep what is chosen most of the time |
| Random Forest | Hundreds of decision trees on random subsets of the data and factors, averaged |
| Gradient Boosting | Small trees built one after another, each correcting the previous ones |
| Cross-validation | Hide some neighborhoods, train on the others, and score the predictions on the hidden ones |
| R² | Share of the variation that a model explains (here always on hidden neighborhoods); 0 is no better than the average, negative is worse |
| Leave-one-borough-out | Cross-validation that hides a whole borough, a harder test of whether a pattern is general |
| Permutation importance | How many R² points predictions lose when one factor's values are shuffled |
| Partial dependence | The model's average prediction as one factor varies and the others stay as observed |
| Moran's I | Measures whether nearby neighborhoods have similar values (or similar model errors) |
| Location fingerprint | A factor that predicts incidence only because it differs by area (borough, urban core), not because of a cause |
| Negative control | A factor with no plausible mechanism (precipitation); if it looks important, the method is picking up something else |
| Ecological design | The data describe areas, not individuals, so a finding about neighborhoods may not hold for the people living in them |

**Files in the repository** ([dmb488-ops/environment-and-cancer-research-nyc](https://github.com/dmb488-ops/environment-and-cancer-research-nyc), branch `factor-importance-analysis`).

| File | Contents |
| --- | --- |
| `outputs/NYC_Environment_Cancer_Explorer.html` | Offline explorer; tab 5 "Important factors" shows all results interactively |
| `outputs/analysis/analysis_results.json` | Every number in this document and in tab 5 |
| `outputs/analysis/puma_year_panel_2010_2023.csv` | 55 neighborhoods × 14 years: outcomes and the 16 factors (for teammates' own analyses) |
| `outputs/analysis/puma_period_means_2010_2023.csv` | 55 rows: pooled outcomes and factor period means |
| `outputs/analysis/age_strata_lung_all.csv` | Cases and person-years by neighborhood, year and age group |
| `outputs/analysis/factor_years_available.csv` | Which years each factor covers |
| `scripts/analysis/` | `build_tables.py`, `run_analysis.py`, `stats_lib.py`, `ml_lib.py`, `theory.py`, `common.py`, `test_stats_lib.py` |
| `scripts/web/` | `template.html` and `assemble.py`, which rebuild the explorer page |
| `outputs/analysis/*.csv` (summaries) | Spreadsheet-friendly results written by export_summaries.py: factor_verdicts_lung_aa / all_aa, collinearity_vif, model_performance, regression_coefficients_lung_aa, panel_variation_shares |
| `docs/Factor_Selection_Methods.md` | A markdown copy of this document |

**How to reproduce.** Needs Python with NumPy, pandas, SciPy and scikit-learn.

```
python3 scripts/analysis/build_tables.py [path to SEER*Stat export]   # tidy tables
python3 scripts/analysis/test_stats_lib.py                             # checks the hand-written statistics
python3 scripts/analysis/run_analysis.py                               # about 25 minutes
python3 scripts/analysis/export_summaries.py                           # flat CSV summaries
python3 scripts/web/assemble.py                                        # re-embed results in the explorer page
```

Random seeds are fixed, so a re-run gives the same numbers. `QUICK=1 python3 run_analysis.py` is a 1-minute dry run with fewer repeats (not for reporting).

**Sources.** Everything else (data) was provided by the research group.

- CDC, [Lung Cancer Risk Factors](https://www.cdc.gov/lung-cancer/risk-factors/index.html): cigarette smoking is linked to about 80% of lung cancer deaths; secondhand smoke, radon, air pollution and occupational exposures are further risk factors.
- IARC press release 221, 17 October 2013, [IARC: Outdoor air pollution a leading environmental cause of cancer deaths](https://www.iarc.who.int/wp-content/uploads/2018/07/pr221_E.pdf): outdoor air pollution and particulate matter are Group 1, with sufficient evidence for lung cancer.
- IARC press release 213, 12 June 2012, IARC: [Diesel engine exhaust carcinogenic](https://www.iarc.who.int/wp-content/uploads/2018/07/pr213_E.pdf): diesel engine exhaust is Group 1, with sufficient evidence for lung cancer.
- Frontiers in Public Health, 2025, [residential greenness (NDVI) and lung cancer in a South Korean national cohort](https://www.frontiersin.org/journals/public-health/articles/10.3389/fpubh.2025.1661937/full): adjusted hazard ratio 0.885 (95% CI 0.858–0.912) per 0.1 NDVI; the authors note that NDVI ignores access, quality and use of green space.
- The tobacco-retail (convenience store) argument in Step 3 is **not sourced yet** and is listed as a hypothesis for the group to check.
