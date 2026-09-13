# Student Performance Prediction — Logistic Regression from Scratch vs scikit-learn

A math teacher's take on predicting student performance — logistic regression built from scratch (using only NumPy), compared against scikit-learn, plus some statistical hypothesis testing to see which factors actually matter.

## Motivation

I teach math at a secondary school, so I've always been curious which factors *really* affect whether a student passes or struggles — is it just study time, or do things like family support, going out with friends, or past failures matter more? This project is my attempt to answer that with data instead of just intuition.

## Dataset

[UCI Student Performance Dataset](https://archive.ics.uci.edu/dataset/320/student+performance) (Cortez & Silva, 2008) — collected from two Portuguese secondary schools. Includes grades, demographic info, and social/school-related features for ~650 students (Portuguese language course) and ~395 students (Math course).

- `student-mat.csv` — Math course
- `student-por.csv` — Portuguese course

## Project structure

```
data/raw/        -> original dataset files (not modified)
notebooks/        -> exploration, modeling, evaluation notebooks
src/              -> from-scratch logistic regression implementation
report/           -> final PDF write-up
```

## What this project covers

1. Data cleaning & preprocessing
2. Exploratory data analysis (univariate + multivariate)
3. Statistical hypothesis testing on key factors
4. Logistic regression implemented from scratch (NumPy)
5. Logistic regression using scikit-learn
6. Model comparison & evaluation

## Status

Work in progress — day 1 of a 1-week build.

## Author

Sevinc Qasimova
