# Temporal Data Leakage Analysis

## Problem Statement

**CRITICAL ISSUE**: Current feature configurations use events from year `t` to predict outcomes in year `t`, which can cause temporal data leakage.

## What is Temporal Data Leakage?

When predicting `y_moved` in 2023, using events that also happened in 2023 is problematic **if we don't know which occurred first**:

```
Scenario A (No Leakage):
├─ Jan 2023: Birth of child (feature)
├─ Jun 2023: Family moves (outcome)
└─ Prediction is valid ✅

Scenario B (DATA LEAKAGE):
├─ Jan 2023: Family moves (outcome)
├─ Jun 2023: Birth of child (feature)
└─ Using future information to predict the past! ❌
```

Without exact timestamps (month/day), we **cannot distinguish** between these scenarios.

## Current Problematic Features

### All Feature Configs Affected

Found in:
- `configs/data/socioec_features_1.yaml`
- `configs/data/socioec_features_med.yaml`
- `configs/data/socioec_features_ext.yaml`

### Direct Event Leakage

These features use events from year `t`:

```yaml
# ❌ PROBLEMATIC - Events in same year as prediction
- birth1_event          # Birth in year t
- birth2_event          # Second birth in year t
- divorce_event         # Divorce in year t
- getalifeother_event   # Other life event in year t
```

### Interaction Feature Leakage

These derived features may also contain leakage:

```yaml
# ❌ POTENTIALLY PROBLEMATIC (if based on year t events)
- divorce_x_age            # divorce_event × age
- birth_x_age              # birth_event × age
- partnership_income       # Based on coupled status
- income_drop              # Income change within year t
- income_rise              # Income change within year t
- low_income_birth         # birth_event × low_income
- high_income_divorce      # divorce_event × high_income
- low_income_divorce       # divorce_event × low_income
- family_break             # divorce_event × household_size
- constrained_young_family # birth_event × age × income
- recent_birth             # Ambiguous: lag1 or current year?
```

### Safe Features (No Leakage)

These are safe to use:

```yaml
# ✅ SAFE - Lagged events (year t-1)
- birth1_event_lag1
- birth2_event_lag1
- divorce_event_lag1
- getalifeother_event_lag1

# ✅ SAFE - Lagged events (year t-2)
- birth1_event_lag2
- birth2_event_lag2
- divorce_event_lag2
- getalifeother_event_lag2

# ✅ SAFE - Cumulative history (censored)
- birth1_event_censored    # Ever had event before year t
- birth2_event_censored
- divorce_event_censored
- getalifeother_event_censored

# ✅ SAFE - Lagged outcomes
- y_moved_lag1             # Moved in year t-1
- y_moved_lag2             # Moved in year t-2

# ✅ SAFE - State variables (not events)
- age                      # Age at start of year
- MS_ADI_PP                # Socioeconomic index
- MS_ADI_HH                # Household socioeconomic index
- coupled                  # Relationship status (if measured at year start)
- income_norm              # Income level (if from previous year)
- eerste_nationaliteit     # Nationality (stable)
- hh_pos                   # Household position (if at year start)
```

## Impact on Model Performance

### Optimistic Bias

Models trained with leakage will show:
- **Artificially high AUC** (e.g., 0.95 instead of 0.80)
- **Overestimated feature importance** for leaked features
- **Poor generalization** to real-world deployment

### Production Failure

At prediction time, you won't have access to:
- Events that haven't happened yet
- Features computed from future events
- Interaction terms based on future events

Result: **Model performs much worse in production than in testing**

## Solutions

### Solution 1: Remove Non-Lagged Events (Recommended)

**Most Conservative - Zero Risk of Leakage**

Remove all features from year `t`:

```yaml
# REMOVE from feature configs:
- birth1_event
- birth2_event
- divorce_event
- getalifeother_event
- divorce_x_age
- birth_x_age
- income_drop (if based on year t)
- income_rise (if based on year t)
- low_income_birth
- high_income_divorce
- low_income_divorce
- family_break
- constrained_young_family
```

**Benefits**:
- ✅ Guaranteed no temporal leakage
- ✅ Realistic model performance estimates
- ✅ Direct production deployment

**Costs**:
- ❌ Lower predictive power (lose information)
- ❌ Can't capture immediate precursors to moving

**Use case**: When you need guaranteed valid predictions for research/policy

### Solution 2: Predict Year t+1 Using Year t Features

**Alternative Temporal Strategy**

Instead of predicting `y_moved` in year `t` using features from year `t`, predict `y_moved` in year `t+1` using features from year `t`:

```python
# Create lagged labels
df = df.withColumn("y_moved_future", F.lead("y_moved", 1).over(window))

# Now use features from year t to predict y_moved_future (year t+1)
label_col = "y_moved_future"
```

**Benefits**:
- ✅ Can use all features from year `t`
- ✅ No temporal leakage (features precede outcome)
- ✅ More predictive power

**Costs**:
- ❌ Loses last year of data (no future label)
- ❌ Changes interpretation (predicting 1 year ahead, not same year)

**Use case**: When you want maximum predictive power with temporal validity

### Solution 3: Monthly/Daily Timestamps (Ideal but Data-Intensive)

**Best Solution if Data Available**

If you have month/day information:

```python
# Create explicit temporal ordering
df = df.withColumn("event_date", F.make_date("year", "month", "day"))
df = df.withColumn("moved_date", F.make_date("year", "moved_month", "moved_day"))

# Only use features where event_date < moved_date
features_before_move = df.filter(F.col("event_date") < F.col("moved_date"))
```

**Benefits**:
- ✅ Maximum predictive power
- ✅ No temporal leakage
- ✅ Can use events from same year if they precede move

**Costs**:
- ❌ Requires detailed timestamps (month/day)
- ❌ More complex data processing
- ❌ May not be available in your data

**Use case**: When you have precise timing information

## Recommended Actions

### Immediate Actions (Research Validity)

1. **Audit current results**: Re-run experiments without non-lagged events
2. **Compare AUC**: Likely to drop by 0.05-0.15
3. **Document difference**: Report both "with leakage" (optimistic) and "without leakage" (realistic)

### For Future Work

1. **Use leakage-free configs**: See `configs/data/socioec_features_no_leakage.yaml`
2. **Add temporal validation**: Test model on t+1 prediction
3. **Document assumptions**: Clearly state which features are used from which time period

### For Production Deployment

1. **Must use leakage-free features**: Otherwise model will fail
2. **Test temporal validity**: Simulate production by hiding future information
3. **Monitor feature availability**: Ensure all features are available at prediction time

## How to Fix Existing Configs

### Quick Fix

For each config file, remove these lines:

```yaml
# DELETE THESE:
- birth1_event
- birth2_event
- divorce_event
- getalifeother_event
# And all interaction features based on them
```

### Testing Impact

Compare model performance:

```bash
# Run with leakage (current)
python run_experiments.py --config socioec_features_1.yaml

# Run without leakage (fixed)
python run_experiments.py --config socioec_features_no_leakage.yaml

# Compare AUC, F1, feature importance
```

Expected result:
- AUC drops by ~0.05-0.15
- Feature importance shifts to lagged variables
- More realistic production estimates

## Discussion Questions

1. **Do you have month/day timestamps?**
   - If YES → Solution 3 (use precise timing)
   - If NO → Solution 1 or 2

2. **What's the use case?**
   - Research/publication → Solution 1 (conservative, valid)
   - Production system → Solution 1 or 3 (must avoid leakage)
   - Exploratory analysis → Document leakage, use carefully

3. **What's the prediction horizon?**
   - Same-year prediction → Solution 1 (remove events)
   - Next-year prediction → Solution 2 (predict t+1)
   - Multi-year → Use increasingly lagged features

## References

- [Common ML Mistakes: Temporal Leakage](https://machinelearningmastery.com/data-leakage-machine-learning/)
- [Kaggle: Time Series Data Leakage](https://www.kaggle.com/getting-started/140314)
- [Best Practices for Temporal Validation](https://developers.google.com/machine-learning/data-prep/construct/sampling-splitting/time-series)

## Summary

**TL;DR**:
- ❌ Current configs use events from year `t` to predict outcomes in year `t` → **DATA LEAKAGE**
- ✅ Use only lagged events (`_lag1`, `_lag2`) or predict year `t+1` instead
- 📊 Expect AUC to drop by 0.05-0.15 after fixing (this is realistic, not worse!)
- 🎯 Use `configs/data/socioec_features_no_leakage.yaml` for valid experiments
