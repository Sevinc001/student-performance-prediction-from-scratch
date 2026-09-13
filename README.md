# Student Performance Prediction — Logistic Regression from Scratch vs scikit-learn

A math teacher's take on predicting student performance — logistic regression built from scratch (using only NumPy), compared against scikit-learn, plus statistical hypothesis testing to check which factors actually matter, instead of just assuming.

## Motivation

I teach math at a secondary school, so I've always been curious which factors *really* affect whether a student passes or struggles - is it just study time, or do things like family support, going out with friends, or past failures matter more? This project is my attempt to answer that with data instead of just intuition, and to properly understand logistic regression by building it myself instead of only calling `sklearn.fit()`.

## Dataset

[UCI Student Performance Dataset](https://archive.ics.uci.edu/dataset/320/student+performance) (Cortez & Silva, 2008), Math course - 395 students from two Portuguese secondary schools. Target variable: final grade `G3` converted to pass/fail (pass = G3 >= 10).

`G1` and `G2` (earlier period grades) were dropped from the features on purpose - they are extremely close to the final grade and would basically let the model "cheat" by predicting a grade from an earlier grade, instead of predicting from the actual social/school/family factors I'm interested in.

## Project structure

```
data/raw/          -> original dataset (student-mat.csv)
notebooks/         -> 01 to 05, in order (exploration -> cleaning -> hypothesis tests -> model)
report/            -> final PDF write-up
```

## Method

1. Exploratory data analysis (distributions, pass/fail split - 67.1% pass rate, 95% CI: 0.625-0.717)
2. Data cleaning: binary encoding for yes/no columns, one-hot encoding for multi-category columns (Mjob, Fjob, reason, guardian)
3. Hypothesis testing (two-sample t-tests / chi-square, alpha = 0.05) on a few features I expected to matter
4. Logistic regression implemented from scratch in NumPy (sigmoid, binary cross-entropy cost, batch gradient descent)
5. Logistic regression with scikit-learn on the same train/test split, for comparison

## Results

### Hypothesis tests

| Feature | Test | p-value | Significant? |
|---|---|---|---|
| failures | t-test | < 0.0001 | **Yes** |
| studytime | t-test | 0.128 | No |
| absences | t-test | 0.118 | No |
| famsup | chi-square | 0.286 | No |

Honestly this surprised me a bit - I expected absences and study time to come out significant too, and they didn't (at least not on their own, not accounting for other variables). Past failures turned out to be by far the strongest single factor, which in hindsight makes sense - it's a much more direct signal of a student already struggling.

### Model performance (test set, n=79)

| Metric | From scratch (NumPy) | scikit-learn |
|---|---|---|
| Accuracy | 0.709 | 0.709 |
| Precision | 0.738 | 0.738 |
| Recall | 0.865 | 0.865 |
| F1 | 0.796 | 0.796 |
| ROC-AUC | 0.740 | 0.738 |

The two implementations landed almost exactly on the same numbers, which was a good sanity check that my from-scratch gradient descent is actually correct - sklearn uses a different (more advanced) optimizer under the hood, so getting near-identical results isn't guaranteed, it just means the underlying math is right.

### Which features mattered most

By coefficient size, both models agreed the same top features mattered: **failures** (negative - more past failures, lower chance of passing), **goout** (negative - more going out with friends, lower chance), and a few others like family educational support, age, and mother's job.

One interesting (and a bit counterintuitive) result: `famsup` (family educational support) had a *negative* coefficient - students receiving family support were slightly *less* likely to pass in this model. My best guess is reverse causality: families are probably more likely to step in and offer support *because* a student is already struggling, not the other way around. This dataset can't confirm that, but it's worth flagging rather than ignoring.

## Limitations

- Sample size is fairly small (395 students, single region in Portugal) - results might not generalize to other school systems.
- Correlation isn't causation - especially for something like famsup above.
- Only used the Math course dataset, not the Portuguese course one (`student-por.csv`) that's also in the original UCI release - could be a good extension.

## Author

Sevinc Qasimova
