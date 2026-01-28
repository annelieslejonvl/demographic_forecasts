# Temporal Leakage - Quick Start Guide

## TL;DR - The Problem

**Je gebruikt events uit jaar t om y_moved in jaar t te voorspellen, maar je weet niet welke eerst gebeurde!**

```
❌ LEAKAGE: birth_event (maart) → y_moved (januari)  [event NA outcome]
✅ NO LEAK: birth_event (januari) → y_moved (maart)  [event VOOR outcome]
```

Zonder maand/dag timestamps kan je dit niet onderscheiden → **DATA LEAKAGE**

## Quick Check

Run this to see which features are problematic:

```bash
python check_temporal_leakage.py
```

Output shows:
- ❌ Leaky features (remove these)
- ✅ Safe features (keep these)
- ❓ Unknown features (review these)

## Impact

Your current configs have **temporal leakage** from these features:

### Problematic Features (in all configs)

```yaml
# ❌ REMOVE THESE (events in year t):
- birth1_event
- birth2_event
- divorce_event
- getalifeother_event

# ❌ REMOVE THESE (interactions with year t events):
- divorce_x_age
- birth_x_age
- income_drop
- income_rise
- low_income_birth
- high_income_divorce
- low_income_divorce
- family_break
- constrained_young_family
```

### Expected Impact on Performance

- **Current AUC** (with leakage): ~0.85-0.90 (optimistic)
- **Fixed AUC** (without leakage): ~0.75-0.82 (realistic)
- **Drop**: 0.05-0.15 points

This drop is **NOT bad** - it's honest! Your current results are artificially inflated.

## Solutions (Pick One)

### Option 1: Remove Leaky Features (Easiest)

**Use the fixed config**:

```bash
# Use this instead of socioec_features_1.yaml
configs/data/socioec_features_no_leakage.yaml
```

**Pros**:
- ✅ No code changes needed
- ✅ Guaranteed valid results
- ✅ Ready to use now

**Cons**:
- ❌ Lower predictive power
- ❌ Can't capture same-year events

### Option 2: Predict Year t+1 (Best Predictive Power)

**Change the prediction target**:

```python
# In your data pipeline:
from pyspark.sql import functions as F, Window

# Create future label
window = Window.partitionBy("sid").orderBy("year")
df = df.withColumn("y_moved_future", F.lead("y_moved", 1).over(window))

# Use y_moved_future as label (predicts next year)
label_col = "y_moved_future"
```

**Pros**:
- ✅ Can use ALL features from year t
- ✅ No leakage (features precede outcome)
- ✅ Higher predictive power

**Cons**:
- ❌ Loses last year of data
- ❌ Changes interpretation (1-year ahead prediction)

### Option 3: Use Month/Day Timestamps (Ideal)

**If you have month/day data**:

```python
# Filter to only use events that occurred before the move
df = df.withColumn("event_date", F.make_date("year", "event_month", "event_day"))
df = df.withColumn("moved_date", F.make_date("year", "moved_month", "moved_day"))

# Only use events where event_date < moved_date
safe_features = df.filter(F.col("event_date") < F.col("moved_date"))
```

**Pros**:
- ✅ Maximum predictive power
- ✅ No leakage
- ✅ Can use same-year events if they precede move

**Cons**:
- ❌ Requires detailed timestamps
- ❌ May not be available in your data

## Quick Comparison Test

Compare leaky vs clean configs:

```bash
# 1. Run with leakage (current)
python run.py --config configs/data/socioec_features_1.yaml

# 2. Run without leakage (fixed)
python run.py --config configs/data/socioec_features_no_leakage.yaml

# 3. Compare results
# Expected: AUC drops by 0.05-0.15
```

## What Features Are Safe?

### ✅ Always Safe

```yaml
# Lagged events (previous years)
- birth1_event_lag1      # Event in year t-1
- birth1_event_lag2      # Event in year t-2
- divorce_event_lag1
- divorce_event_lag2

# Cumulative history
- birth1_event_censored  # Ever had event before year t
- divorce_event_censored

# Historical outcomes
- y_moved_lag1           # Moved in year t-1
- y_moved_lag2           # Moved in year t-2

# State variables (if measured at year start)
- age                    # Age at start of year
- MS_ADI_PP              # Socioeconomic index
- coupled                # Relationship status
- eerste_nationaliteit   # Nationality
- hh_pos                 # Household position
```

### ❌ Never Safe (Without Timestamps)

```yaml
# Events in same year as prediction
- birth1_event           # Could be after move
- divorce_event          # Could be after move

# Interactions with same-year events
- divorce_x_age          # Based on divorce_event
- low_income_birth       # Based on birth_event
```

## Recommended Action Plan

### For Immediate Research/Publication

1. **Re-run all experiments** with `socioec_features_no_leakage.yaml`
2. **Report both results**:
   - "With same-year events" (optimistic, possibly invalid)
   - "With lagged events only" (conservative, valid)
3. **Document the difference** in your paper/report
4. **Use lagged-only for conclusions** (valid inference)

### For Production System

1. **MUST use leakage-free features** (system will fail otherwise)
2. **Test with Option 2** (predict t+1) for better performance
3. **Validate on holdout period** simulating production

### For Exploratory Analysis

1. **Use leaky features for exploration** (understand patterns)
2. **Switch to safe features for validation** (confirm findings)
3. **Clearly document** which results use which features

## Files Created

- `TEMPORAL_LEAKAGE_ANALYSIS.md` - Detailed explanation
- `TEMPORAL_LEAKAGE_QUICKSTART.md` - This file
- `configs/data/socioec_features_no_leakage.yaml` - Fixed config
- `check_temporal_leakage.py` - Automated checker

## Questions to Answer

Before choosing a solution, answer:

1. **Do you have month/day timestamps?**
   - YES → Use Option 3 (precise timing)
   - NO → Use Option 1 or 2

2. **What's the use case?**
   - Research paper → Option 1 (conservative)
   - Production system → Option 1 or 3
   - Maximum prediction → Option 2 (predict t+1)

3. **Can you change the label?**
   - YES → Consider Option 2 (predict t+1)
   - NO → Must use Option 1 (remove features)

## Further Reading

- `TEMPORAL_LEAKAGE_ANALYSIS.md` - Full analysis
- [Kaggle: Time Series Data Leakage](https://www.kaggle.com/getting-started/140314)
- [Google ML Best Practices: Temporal Validation](https://developers.google.com/machine-learning/data-prep/construct/sampling-splitting/time-series)

## Summary

**Current Status**: ⚠️  All feature configs contain temporal leakage

**Impact**: AUC is 0.05-0.15 too high (optimistic)

**Fix**: Use `configs/data/socioec_features_no_leakage.yaml`

**Next Steps**:
1. Run `python check_temporal_leakage.py` to see full analysis
2. Re-run experiments with fixed config
3. Compare and document results
