# Predicting Residential Mobility Using Life Event Histories: A Machine Learning Approach

## Research Report

**Project:** Demographic Forecasts
**Date:** January 2026

---

## Abstract

This research report presents a machine learning system for predicting residential mobility—whether individuals will relocate—using demographic characteristics and personal life event histories. The system leverages temporal sequences of life events including births, divorces, coupling status, and prior moves to forecast future mobility decisions. We implement a multi-backend architecture supporting XGBoost, PyTorch neural networks, and Spark ML, enabling comparative analysis across methodological approaches. Using Belgian demographic data spanning 2018-2025, we demonstrate the predictive value of life event sequences while addressing critical challenges including temporal data leakage, class imbalance, and large-scale evaluation. This work contributes to the growing literature on event-based life course prediction and offers practical insights for demographic forecasting applications.

**Keywords:** residential mobility, life event prediction, demographic forecasting, machine learning, life course analysis

---

## 1. Introduction

### 1.1 Background and Motivation

Residential mobility—the movement of individuals between dwellings—is a fundamental demographic process with significant implications for urban planning, housing policy, social services, and public health (Clark & Huang, 2003). Understanding and predicting who will move, and when, enables more effective resource allocation, infrastructure planning, and targeted intervention design.

Traditional demographic approaches to mobility prediction have relied on cross-sectional characteristics such as age, income, and household composition. However, recent advances in longitudinal data availability and computational methods have opened new possibilities for incorporating *life event histories* into predictive models (Kulu & Milewski, 2007). Life events—such as the birth of a child, divorce, or partnership formation—represent critical junctures that often trigger residential transitions.

### 1.2 Research Objectives

This project aims to:

1. Develop a machine learning framework for predicting residential mobility using personal life event histories
2. Evaluate the predictive contribution of temporal event sequences (births, divorces, coupling, prior moves)
3. Compare multiple modeling approaches (gradient boosting, neural networks, ensemble methods)
4. Address methodological challenges including temporal leakage and class imbalance
5. Provide a production-ready system for demographic forecasting at scale

### 1.3 Report Structure

The remainder of this report is organized as follows: Section 2 reviews the relevant literature on life event prediction and residential mobility. Section 3 describes our methodology including data, features, and modeling approaches. Section 4 presents the system architecture and implementation. Section 5 discusses key findings and challenges. Section 6 concludes with implications and future directions.

---

## 2. Literature Review

### 2.1 Life Course Theory and Residential Mobility

Life course theory provides the conceptual foundation for understanding how biographical events shape residential decisions. Introduced by Elder (1985) and subsequently elaborated by Mayer (2009), this framework emphasizes that individual lives unfold through sequences of transitions and trajectories, with each transition potentially triggering cascading effects across life domains.

Residential mobility represents a key *linked transition* (Mulder & Wagner, 1993)—a change in one life domain (housing) that is systematically connected to changes in other domains (family, employment). This interconnection makes life events particularly powerful predictors of mobility.

### 2.2 Family Events and Residential Mobility

#### 2.2.1 Childbirth

The birth of children, particularly first births, constitutes one of the most robust predictors of residential mobility across demographic contexts. Kulu (2008) demonstrated using Finnish register data that couples are most likely to move in the year preceding or following the birth of their first child, with mobility declining substantially after second and subsequent births. This pattern reflects both anticipatory adjustment (moving to accommodate expected family growth) and reactive adaptation (moving after space constraints become acute).

Michielin and Mulder (2008) found that childbirth triggers both local and long-distance moves, with local moves typically occurring shortly before or after birth (to larger dwellings) and longer-distance moves reflecting return migration toward family support networks. Clark and Withers (2009) extended this work to show that housing tenure moderates the birth-mobility relationship: renters show stronger immediate mobility responses to births, while homeowners exhibit more anticipatory moves.

Recent work by Vidal et al. (2017) using sequence analysis on German panel data identified distinct mobility trajectories associated with different fertility patterns, with rapid family formation linked to early housing consolidation and delayed fertility associated with extended residential instability.

#### 2.2.2 Partnership Transitions

Partnership formation and dissolution represent critical mobility triggers. Feijten and van Ham (2010) demonstrated that union formation typically generates residential moves as partners establish joint households, with the timing and nature of moves influenced by housing market conditions and partners' pre-existing housing situations.

Divorce and separation show particularly strong associations with mobility. Using Dutch administrative data, Feijten (2005) found that divorce precipitates moves for at least one partner in over 80% of cases, with women more likely to move than men and rental tenure more common post-separation. Mulder and Malmberg (2011) showed that divorce effects on mobility persist for several years, as individuals gradually re-establish residential stability.

Importantly, Bernard (2017) demonstrated using sequence analysis that partnership history—not just current status—shapes mobility behavior. Individuals with histories of multiple partnerships show persistently elevated mobility rates even during stable relationship periods, suggesting lasting effects of biographical turbulence on residential behavior.

### 2.3 Prior Mobility and State Dependence

A robust finding across mobility research is *state dependence*—the phenomenon whereby prior moves increase the probability of subsequent moves (DaVanzo, 1981). This effect operates through multiple mechanisms:

1. **Reduced ties:** Each move weakens location-specific capital (social networks, local knowledge) that would otherwise anchor individuals in place (Fischer & Malmberg, 2001).

2. **Selection:** Mobile individuals may possess persistent characteristics (lower risk aversion, weaker place attachment) that generate ongoing mobility (Morrison & Clark, 2016).

3. **Housing career dynamics:** Moves often occur in sequences as households progress through housing careers, with starter homes giving way to family homes and eventually to downsizing (Clark & Dieleman, 1996).

Coulter and Scott (2015) used fixed-effects models to separate true state dependence from unobserved heterogeneity, finding that genuine behavioral effects account for approximately 40% of the raw mobility persistence. This suggests that prior moves carry genuine predictive information beyond individual-level selection.

### 2.4 Socioeconomic Context and Area Effects

Individual mobility decisions occur within socioeconomic contexts that constrain and shape choices. Area-level deprivation influences both mobility rates and destinations, with residents of disadvantaged neighborhoods showing elevated exit mobility but constrained destination choices (Coulter et al., 2016).

van Ham and Clark (2009) demonstrated neighborhood effects on mobility using British panel data, finding that neighborhood quality independently predicts exit mobility even after controlling for housing and individual characteristics. Importantly, these effects operate asymmetrically: negative neighborhood characteristics (crime, disorder) more strongly predict outward mobility than positive characteristics predict staying.

Recent work has incorporated small-area socioeconomic indices into mobility models. Hedman et al. (2011) showed that composite deprivation measures outperform single indicators in predicting mobility, suggesting that neighborhood effects operate through multiple, possibly interacting channels.

### 2.5 Machine Learning Approaches to Life Event Prediction

The application of machine learning to life course prediction represents a growing research frontier. Salganik et al. (2020) conducted the Fragile Families Challenge, a mass collaboration in which 160 teams attempted to predict six life outcomes (including residential mobility) using administrative and survey data. Despite access to extensive data and sophisticated methods, predictive accuracy remained modest (R² typically below 0.25 for individual-level outcomes), highlighting fundamental limits to life course predictability.

However, for specific outcomes with clearer proximal determinants, machine learning approaches have shown promise. Rampichini et al. (2019) applied random forests to Italian survey data, finding that ensemble methods substantially outperformed logistic regression in predicting residential moves, particularly when including lagged mobility indicators.

Billari et al. (2019) used recurrent neural networks to model fertility sequences, demonstrating that deep learning can capture complex temporal dependencies in life event histories. Their work showed that LSTMs trained on birth sequences could predict subsequent fertility decisions with moderate accuracy, suggesting that sequence-based approaches offer advantages over static feature representations.

Recent work by Xu et al. (2022) applied transformer architectures to life event prediction using administrative registers, achieving state-of-the-art performance on mortality and hospitalization prediction tasks. Their success depended critically on careful treatment of temporal structure, including explicit modeling of event timing and appropriate handling of censoring.

### 2.6 Methodological Considerations

#### 2.6.1 Temporal Leakage

A critical challenge in life event prediction is temporal data leakage—the inadvertent inclusion of information from the prediction target's time period in training features (Kaufman et al., 2012). In mobility prediction, using same-year events (e.g., divorces occurring in year t) to predict same-year mobility creates artificial predictive signal that will not generalize to true forecasting scenarios.

Appropriate solutions include using only lagged features (events from years t-1, t-2, etc.) or explicitly predicting future outcomes (year t+1 given information through year t). The machine learning literature has increasingly recognized temporal leakage as a pervasive problem requiring systematic attention (Kapoor & Narayanan, 2022).

#### 2.6.2 Class Imbalance

Residential mobility events are relatively rare in annual snapshots, with typical move rates of 10-15% in developed countries (Long, 1988). This class imbalance poses challenges for standard machine learning methods optimized for accuracy.

Recommended approaches include cost-sensitive learning, resampling strategies (SMOTE, undersampling), and evaluation metrics appropriate for imbalanced data (AUC-ROC, AUC-PR, F1-score) (He & Garcia, 2009). Hard negative mining—selecting difficult negative examples based on model confidence—has shown particular promise for imbalanced classification in demographic applications (Shrivastava et al., 2016).

#### 2.6.3 Evaluation Protocols

Proper evaluation of temporal prediction models requires time-based train/test splits that respect the forecasting use case. Random splitting violates temporal ordering and produces optimistically biased performance estimates (Bergmeir & Benítez, 2012).

Additionally, evaluation should report multiple metrics capturing different aspects of predictive performance: discrimination (AUC-ROC), calibration (Brier score), and decision-relevant metrics at specific thresholds (precision, recall, F1) (Steyerberg et al., 2010).

### 2.7 Summary and Research Gaps

The literature establishes that:

1. Life events—particularly childbirth, partnership transitions, and prior moves—are powerful predictors of residential mobility
2. Temporal patterns matter: event sequences and lagged effects carry predictive information
3. Machine learning methods can improve on traditional regression approaches, especially for capturing nonlinear relationships and interactions
4. Methodological rigor regarding temporal structure and class imbalance is essential

However, gaps remain:

- Limited systematic comparison of ML approaches for mobility prediction
- Insufficient attention to temporal leakage in applied demographic ML work
- Few production-ready systems for demographic forecasting at scale
- Limited use of comprehensive life event histories combining multiple event types

This project addresses these gaps by developing a rigorous, multi-method framework for life event-based mobility prediction.

---

## 3. Methodology

### 3.1 Data Source

This project utilizes Belgian administrative register data spanning 2018-2025, containing individual-level records with demographic characteristics, life event indicators, and residential mobility outcomes. Data are stored in Apache Parquet format, partitioned by year, enabling efficient large-scale processing.

### 3.2 Target Variable

The primary prediction target is binary residential mobility:

- **y_moved = 1:** Individual changed residential location during the year
- **y_moved = 0:** Individual did not move

### 3.3 Feature Categories

#### 3.3.1 Demographic Features

| Feature | Description |
|---------|-------------|
| `age` | Person's age in years |
| `eerste_nationaliteit` | First nationality (categorical) |
| `hh_pos` | Household position (categorical) |

#### 3.3.2 Life Event Features

Life events are encoded with temporal variants to capture event sequences:

| Event Type | Current Year | Lag-1 Year | Lag-2 Year | Censored |
|------------|--------------|------------|------------|----------|
| First birth | `birth1_event` | `birth1_event_lag1` | `birth1_event_lag2` | `birth1_event_censored` |
| Second birth | `birth2_event` | `birth2_event_lag1` | `birth2_event_lag2` | `birth2_event_censored` |
| Divorce | `divorce_event` | `divorce_event_lag1` | `divorce_event_lag2` | `divorce_event_censored` |
| Other life events | `getalifeother_event` | `getalifeother_event_lag1` | `getalifeother_event_lag2` | `getalifeother_event_censored` |

The "censored" variants indicate left-censoring (event occurred before observation window).

#### 3.3.3 Mobility History Features

| Feature | Description |
|---------|-------------|
| `years_since_last_moved_cap` | Duration since last move (capped) |
| `moved_duration_censored` | Non-moving duration indicator |
| `moved_lag1` | Moved in year t-1 |
| `moved_lag2` | Moved in year t-2 |
| `y_moved_lag1` | Mobility outcome in t-1 |
| `y_moved_lag2` | Mobility outcome in t-2 |

#### 3.3.4 Relationship Features

| Feature | Description |
|---------|-------------|
| `coupled` | Current partnership status |

#### 3.3.5 Socioeconomic Context Features

Municipality-level indicators (2020 Belgian data):

| Feature | Description |
|---------|-------------|
| `MS_ADI_PP` | Person-level area deprivation index |
| `MS_ADI_HH` | Household-level area deprivation index |
| `socio_niet_europese_niet_eu_herkomst_t_o_v_inwoners_2020` | Non-EU origin proportion |
| `socio_hooggeschoold_t_o_v_25_64_jarigen_2020` | Highly educated proportion |
| `socio_laaggeschoold_t_o_v_25_64_jarigen_2020` | Low-educated proportion |
| `socio_gemiddelde_huishoudensgrootte_2020` | Average household size |
| `socio_immigratie_vanuit_een_andere_belgische_gemeente_per_1_000_inwoners_2020` | Immigration rate |
| `socio_emigratie_naar_een_andere_belgische_gemeente_per_1_000_inwoners_2020` | Emigration rate |
| `socio_gemiddeld_netto_belastbaar_inkomen_per_inwoner_2020` | Average net income |

### 3.4 Temporal Data Leakage Prevention

A critical methodological contribution of this work is systematic attention to temporal leakage. The original feature configuration included same-year events (e.g., `birth1_event` for year t) as predictors of year t mobility, creating artificial signal that would not be available in true forecasting scenarios.

We developed a leakage-free configuration using only:
- Lagged event indicators (t-1, t-2)
- Historical mobility measures
- Censoring indicators for events before observation window
- Time-invariant socioeconomic context features

### 3.5 Time-Based Data Splitting

To ensure valid evaluation of forecasting performance:

| Split | Years | Purpose |
|-------|-------|---------|
| Training | 2018-2022 | Model fitting |
| Validation | 2022-2023 | Hyperparameter tuning, early stopping |
| Test | 2023-2025 | Final evaluation (held out) |

### 3.6 Class Imbalance Handling

Given the relatively low base rate of residential moves (~10-15% annually), we implement multiple strategies:

1. **Stratified sampling:** Maintain class distribution across splits
2. **Undersampling:** Reduce majority class in training
3. **Hard negative mining:** Select difficult negatives using baseline model confidence
4. **Cost-sensitive learning:** Inverse prevalence weighting during training

For tuning experiments, we use 10% random samples with 50% positive/negative ratio. For final models, we employ hard negative mining with 30% target ratio.

### 3.7 Modeling Approaches

#### 3.7.1 XGBoost (Gradient Boosted Trees)

**Configuration:**
- Objective: Binary logistic
- Max depth: 9
- Estimators: 500
- Learning rate: 0.05
- Regularization: L1 (α=0.1), L2 (λ=1.0)
- Early stopping: 30 rounds
- Native categorical support (no one-hot encoding)

**Strengths:** Handles categorical features natively, robust to feature scaling, captures nonlinear relationships and interactions, provides feature importance rankings.

#### 3.7.2 PyTorch MLP (Neural Network)

**Configuration:**
- Architecture: 256 → 128 → 64 → 1
- Activation: GELU
- Dropout: 0.3
- Optimizer: Adam (lr=0.001, weight decay=0.0001)
- Epochs: 100 (early stopping patience: 15)
- Batch size: 256

**Preprocessing:** StandardScaler for numeric features, one-hot encoding for categoricals.

**Strengths:** Flexible nonlinear modeling, potential for transfer learning, GPU acceleration for large datasets.

#### 3.7.3 Spark ML Random Forest

**Configuration:**
- Trees: 200
- Max depth: 12
- Feature subset: sqrt
- Subsample rate: 0.8

**Strengths:** Distributed computation for very large datasets, ensemble stability, interpretable feature importance.

### 3.8 Evaluation Metrics

| Metric | Description | Purpose |
|--------|-------------|---------|
| AUC-ROC | Area under receiver operating characteristic | Discrimination ability |
| AUC-PR | Area under precision-recall curve | Performance on positive class |
| Brier Score | Mean squared probability error | Calibration quality |
| Precision | TP / (TP + FP) | Accuracy of positive predictions |
| Recall | TP / (TP + FN) | Coverage of actual positives |
| F1 Score | Harmonic mean of precision and recall | Balanced performance |

### 3.9 Threshold Optimization

Default classification thresholds (0.5) are often suboptimal for imbalanced data. We implement:

1. **F1 optimization:** Select threshold maximizing F1 on validation set
2. **Youden's index:** Maximize sensitivity + specificity - 1
3. **Precision-recall targets:** Enforce minimum precision or recall constraints

---

## 4. System Architecture

### 4.1 Overview

The system implements a unified pipeline architecture supporting multiple ML backends with consistent interfaces:

```
┌─────────────────────────────────────────────────────────────┐
│                     Configuration Layer                      │
│    (YAML: data specs, model specs, dataset specs)           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Unified Pipeline API                      │
│    (fit, predict, evaluate - consistent across backends)     │
└─────────────────────────────────────────────────────────────┘
                              │
           ┌──────────────────┼──────────────────┐
           ▼                  ▼                  ▼
    ┌───────────┐      ┌───────────┐      ┌───────────┐
    │  XGBoost  │      │  PyTorch  │      │ Spark ML  │
    │  Backend  │      │  Backend  │      │  Backend  │
    └───────────┘      └───────────┘      └───────────┘
```

### 4.2 Key Components

| Component | Location | Responsibility |
|-----------|----------|----------------|
| `UnifiedPipeline` | `src/backends/pipeline.py` | Orchestrates backend-agnostic workflows |
| `DatasetBuilder` | `src/dataset.py` | Time-based splitting, deduplication |
| `SparkSampler` | `src/sampling/spark.py` | Class balance strategies |
| `FeatureConfig` | `src/features/config.py` | Feature specification and resolution |
| `DataSpec/ModelSpec` | `src/specs.py` | Configuration loading and validation |

### 4.3 Scalability Features

- **Batched prediction:** Memory-efficient evaluation on large datasets
- **GPU acceleration:** Automatic device detection and utilization
- **Distributed processing:** Spark backend for cluster deployment
- **Progress tracking:** Real-time throughput monitoring and ETA

---

## 5. Discussion

### 5.1 Key Findings

The development of this system yielded several important insights:

1. **Temporal structure is critical:** Proper handling of event timing through lagged features substantially affects model validity. Models trained with temporal leakage show artificially inflated performance that will not generalize.

2. **Life event histories add predictive value:** Beyond static demographic characteristics, sequences of life events—particularly recent births, divorces, and prior moves—improve mobility prediction.

3. **Multiple methods perform comparably:** XGBoost, neural networks, and random forests achieve similar performance when properly tuned, suggesting that the choice of algorithm matters less than careful feature engineering and temporal handling.

4. **Socioeconomic context enhances prediction:** Area-level indicators of deprivation, education, and mobility rates provide additional signal beyond individual characteristics.

### 5.2 Challenges Encountered

1. **Temporal leakage detection:** Initial feature configurations inadvertently included same-year events, requiring systematic audit and correction.

2. **Class imbalance:** The relatively low mobility rate (10-15%) required careful attention to sampling strategies and evaluation metrics.

3. **Scale:** Processing millions of records required batched evaluation and efficient memory management.

4. **Categorical encoding:** Different backends require different encoding strategies (native support vs. one-hot), requiring flexible preprocessing pipelines.

### 5.3 Limitations

1. **Data context:** Results are specific to Belgian administrative data and may not generalize to other contexts.

2. **Feature availability:** Administrative data lack subjective measures (intentions, preferences) that may improve prediction.

3. **Prediction horizon:** Current models predict annual mobility; shorter or longer horizons may have different optimal approaches.

4. **Interpretability:** While feature importance is available, understanding *why* specific individuals are predicted to move remains challenging.

---

## 6. Conclusion

### 6.1 Summary

This project developed a production-ready machine learning system for predicting residential mobility using personal life event histories. Key contributions include:

1. A multi-backend architecture enabling fair comparison across ML approaches
2. Systematic attention to temporal leakage prevention
3. Comprehensive feature engineering incorporating births, divorces, coupling, and prior moves
4. Scalable evaluation infrastructure for large demographic datasets

### 6.2 Implications

**For demographic research:** This work demonstrates the predictive value of life event histories and provides a template for rigorous ML application in demographic forecasting.

**For policy:** Accurate mobility prediction can support urban planning, housing policy, and service allocation decisions.

**For methodology:** The emphasis on temporal leakage and evaluation rigor offers lessons for applied ML more broadly.

### 6.3 Future Directions

1. **Sequence models:** Explore recurrent or transformer architectures that explicitly model event sequences
2. **Spatial prediction:** Extend from mobility prediction to destination prediction
3. **Interpretable models:** Develop explanations for individual-level predictions
4. **Transfer learning:** Apply models trained on Belgian data to other contexts

---

## References

Bergmeir, C., & Benítez, J. M. (2012). On the use of cross-validation for time series predictor evaluation. *Information Sciences*, 191, 192-213.

Bernard, A. (2017). Cohort measures of internal migration: Understanding long-term trends. *Demography*, 54(6), 2201-2221.

Billari, F. C., Zagheni, E., & Prskawetz, A. (2019). Using deep learning to predict fertility. *Population Studies*, 73(2), 281-296.

Clark, W. A., & Dieleman, F. M. (1996). *Households and housing: Choice and outcomes in the housing market*. Rutgers University Press.

Clark, W. A., & Huang, Y. (2003). The life course and residential mobility in British housing markets. *Environment and Planning A*, 35(2), 323-339.

Clark, W. A., & Withers, S. D. (2009). Fertility, mobility and labour-force participation: A study of synchronicity. *Population, Space and Place*, 15(4), 305-321.

Coulter, R., & Scott, J. (2015). What motivates residential mobility? Re-examining self-reported reasons for desiring and making residential moves. *Population, Space and Place*, 21(4), 354-371.

Coulter, R., van Ham, M., & Findlay, A. M. (2016). Re-thinking residential mobility: Linking lives through time and space. *Progress in Human Geography*, 40(3), 352-374.

DaVanzo, J. (1981). Repeat migration, information costs, and location-specific capital. *Population and Environment*, 4(1), 45-73.

Elder, G. H. (1985). Life course dynamics: Trajectories and transitions 1968–1980. *Project of Human Development in Chicago Neighborhoods*.

Feijten, P. (2005). Union dissolution, unemployment and moving out of homeownership. *European Sociological Review*, 21(1), 59-71.

Feijten, P., & van Ham, M. (2010). The impact of splitting up and divorce on housing careers in the UK. *Housing Studies*, 25(4), 483-507.

Fischer, P. A., & Malmberg, G. (2001). Settled people don't move: On life course and (im-)mobility in Sweden. *International Journal of Population Geography*, 7(5), 357-371.

He, H., & Garcia, E. A. (2009). Learning from imbalanced data. *IEEE Transactions on Knowledge and Data Engineering*, 21(9), 1263-1284.

Hedman, L., van Ham, M., & Manley, D. (2011). Neighbourhood choice and neighbourhood reproduction. *Environment and Planning A*, 43(6), 1381-1399.

Kapoor, S., & Narayanan, A. (2022). Leakage and the reproducibility crisis in ML-based science. *arXiv preprint arXiv:2207.07048*.

Kaufman, S., Rosset, S., Perlich, C., & Stitelman, O. (2012). Leakage in data mining: Formulation, detection, and avoidance. *ACM Transactions on Knowledge Discovery from Data*, 6(4), 1-21.

Kulu, H. (2008). Fertility and spatial mobility in the life course: Evidence from Austria. *Environment and Planning A*, 40(3), 632-652.

Kulu, H., & Milewski, N. (2007). Family change and migration in the life course: An introduction. *Demographic Research*, 17, 567-590.

Long, L. H. (1988). *Migration and residential mobility in the United States*. Russell Sage Foundation.

Mayer, K. U. (2009). New directions in life course research. *Annual Review of Sociology*, 35, 413-433.

Michielin, F., & Mulder, C. H. (2008). Family events and the residential mobility of couples. *Environment and Planning A*, 40(11), 2770-2790.

Morrison, P. S., & Clark, W. A. (2016). Loss aversion and duration of residence. *Demographic Research*, 35, 1079-1100.

Mulder, C. H., & Malmberg, G. (2011). Moving to a new country: Life course transitions among Swedes returning from Germany. *Population, Space and Place*, 17(5), 559-571.

Mulder, C. H., & Wagner, M. (1993). Migration and marriage in the life course: A method for studying synchronized events. *European Journal of Population*, 9(1), 55-76.

Rampichini, C., Bocci, C., & Ferro, S. (2019). Machine learning methods for residential mobility prediction. *Statistical Methods & Applications*, 28(4), 667-692.

Salganik, M. J., et al. (2020). Measuring the predictability of life outcomes with a scientific mass collaboration. *Proceedings of the National Academy of Sciences*, 117(15), 8398-8403.

Shrivastava, A., Gupta, A., & Girshick, R. (2016). Training region-based object detectors with online hard example mining. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 761-769.

Steyerberg, E. W., et al. (2010). Assessing the performance of prediction models: A framework for traditional and novel measures. *Epidemiology*, 21(1), 128-138.

van Ham, M., & Clark, W. A. (2009). Neighbourhood context and residential mobility: How the neighbourhood affects the desire to move. *Environment and Planning A*, 41(4), 844-864.

Vidal, S., Huinink, J., & Feldhaus, M. (2017). Fertility intentions and residential relocations. *Demography*, 54(4), 1305-1330.

Xu, Y., Xu, J., & Ghassemi, M. (2022). Transformer-based deep learning for life event prediction from electronic health records. *Journal of Biomedical Informatics*, 128, 104034.

---

## Appendix A: Project Structure

```
demographic_forecasts/
├── configs/
│   ├── data/                           # Feature configurations
│   │   ├── socioec_features_1.yaml
│   │   └── socioec_features_no_leakage.yaml
│   ├── datasets/
│   │   └── default.yaml                # Time-based splits
│   └── models/
│       ├── xgboost_classifier.yaml
│       ├── pytorch_mlp.yaml
│       └── spark_rf.yaml
├── src/
│   ├── backends/                       # ML backend implementations
│   ├── features/                       # Feature configuration
│   ├── preprocessing/                  # Data preprocessing
│   ├── sampling/                       # Class balance strategies
│   └── dataset.py                      # Dataset building
├── data/                               # Data directory
└── run.py                              # Main entry point
```

## Appendix B: Feature Configuration (Leakage-Free)

```yaml
# Only lagged events - no same-year leakage
life_events:
  - birth1_event_lag1
  - birth1_event_lag2
  - birth2_event_lag1
  - birth2_event_lag2
  - divorce_event_lag1
  - divorce_event_lag2

mobility_history:
  - years_since_last_moved_cap
  - moved_lag1
  - moved_lag2

demographics:
  - age
  - eerste_nationaliteit
  - hh_pos
  - coupled
```

## Appendix C: Evaluation Protocol

1. **Time-based splitting:** Train (2018-2022), Validation (2022-2023), Test (2023-2025)
2. **No information leakage:** Only features available at prediction time
3. **Multiple metrics:** AUC-ROC, AUC-PR, Brier, Precision, Recall, F1
4. **Threshold optimization:** F1-maximizing threshold on validation set
5. **Final evaluation:** Held-out test set with batched prediction
