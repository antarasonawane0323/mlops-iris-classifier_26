# Hyperparameter Tuning Analysis

## 1. Baseline Model

**Model:** DecisionTreeClassifier

**CV F1 Macro:** 0.9328

**Test Accuracy:** 0.9333

## 2. Grid Search

**Model:** RandomForestClassifier

**Total Combinations:** 72

**Cross-validation:** 5-fold

**Total Fits:** 360

**Best CV F1 Macro:** 0.9583

**Test Accuracy:** 0.9667

## 3. Random Search

**Model:** RandomForestClassifier

**Number of Iterations:** 30

**Cross-validation:** 5-fold

**Total Fits:** 150

**Best CV F1 Macro:** 0.9581

**Test Accuracy:** 0.9667

## 4. Comparison

Grid Search evaluates all combinations in the specified
hyperparameter grid. Random Search evaluates a selected number
of combinations from the search space.

In this experiment, Grid Search required 360 model fits,
while Random Search required only 150 model fits.

Both tuning approaches produced higher CV F1 Macro than the
baseline model.

Grid Search achieved a CV F1 Macro of 0.9583, while Random Search
achieved 0.9581. Both achieved the same test accuracy of 0.9667.

Random Search reached comparable performance using less than half
the number of model fits required by Grid Search.

## 5. Conclusion

Hyperparameter tuning improved the model performance compared with
the baseline Decision Tree.

Grid Search achieved the slightly higher CV F1 Macro, but Random
Search achieved almost the same performance with only 150 fits
compared with 360 fits for Grid Search.

Therefore, Random Search was more computationally efficient in
this experiment while maintaining comparable model performance.