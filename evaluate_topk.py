#!/usr/bin/env python3
"""
Evaluate model performance focusing on top-k predictions.
For rare event prediction, ranking matters more than binary classification.
"""
import json
import sys
from pathlib import Path

def load_checkpoint_data(checkpoint_path):
    """Load model metadata."""
    meta_path = Path(checkpoint_path) / "metadata.json"
    if not meta_path.exists():
        print(f"Error: {meta_path} not found")
        return None

    with open(meta_path) as f:
        return json.load(f)

def analyze_performance(meta):
    """Analyze model performance with focus on ranking."""
    print("="*80)
    print("TOP-K PERFORMANCE ANALYSIS")
    print("="*80)

    metrics = meta['metrics']
    data = meta['data']

    print(f"\nData:")
    train_rows = data.get('train_rows') or data.get('total_trained_rows', 0)
    test_rows = data.get('test_rows', 0)
    print(f"  Train: {train_rows:,} rows ({data.get('train_years', 'N/A')})")
    print(f"  Test:  {test_rows:,} rows ({data.get('test_years', 'N/A')})")

    print(f"\nMetrics:")
    print(f"  AUC-ROC:   {metrics['auc_roc']:.4f}")
    print(f"  AUC-PR:    {metrics['auc_pr']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1:        {metrics['f1_score']:.4f}")

    # Estimate base rate from precision/recall
    # precision = TP / (TP + FP)
    # recall = TP / (TP + FN)
    # If we know precision and recall, we can estimate base rate

    p = metrics['precision']
    r = metrics['recall']

    if r > 0 and p > 0:
        # From recall: TP = r * (TP + FN) = r * positives_true
        # From precision: TP = p * (TP + FP) = p * positives_pred
        # positives_pred = TP / p = (r * positives_true) / p
        # predicted_positive_rate = positives_pred / total = (r / p) * base_rate

        print("\n" + "="*80)
        print("INTERPRETATION")
        print("="*80)

        if p < 0.1 and r > 0.5:
            print("🔴 Model is OVER-PREDICTING positive class")
            print(f"   - Catches {r*100:.1f}% of movers (recall)")
            print(f"   - But {(1-p)*100:.1f}% of predictions are false positives")
            print(f"   - Model lacks discriminative power")

            # Estimate how many it predicts positive
            estimated_base_rate = 0.06  # assume ~6% from AUPRC
            predicted_rate = (r / p) * estimated_base_rate if p > 0 else 0
            print(f"\n   Estimated prediction behavior:")
            print(f"   - True move rate: ~{estimated_base_rate*100:.1f}%")
            print(f"   - Model predicts ~{predicted_rate*100:.1f}% will move")
            print(f"   - Predicting {predicted_rate/estimated_base_rate:.1f}x more positives than baseline")

        elif p > 0.2 and r < 0.3:
            print("⚠️  Model is UNDER-PREDICTING (too conservative)")
            print(f"   - Only catches {r*100:.1f}% of movers")
            print(f"   - But {p*100:.1f}% of predictions are correct")

        elif p < 0.1 and r < 0.1:
            print("🔴 Model has NO predictive power")
            print("   - Predicts very few positives")
            print("   - Those it predicts are mostly wrong")

        else:
            print(f"✓ Balanced performance")
            print(f"  Recall {r*100:.1f}% / Precision {p*100:.1f}%")

    # Analysis based on AUC
    auc_roc = metrics['auc_roc']
    auc_pr = metrics['auc_pr']

    print("\n" + "="*80)
    print("RANKING QUALITY")
    print("="*80)

    if auc_roc < 0.55:
        print("🔴 AUC-ROC < 0.55: Random or worse")
        print("   → Model cannot rank individuals by risk")
    elif auc_roc < 0.65:
        print("⚠️  AUC-ROC 0.55-0.65: Weak ranking")
        print("   → Model has minimal discriminative ability")
    elif auc_roc < 0.75:
        print("✓ AUC-ROC 0.65-0.75: Moderate ranking")
        print("   → Model provides useful risk scores")
    else:
        print("✅ AUC-ROC > 0.75: Good ranking")
        print("   → Model reliably ranks high-risk individuals")

    # Estimate top-k performance
    if auc_roc > 0.5:
        # Rough approximation: lift at top 10%
        # For AUC=0.6, top 10% contains ~1.3-1.5x baseline
        # For AUC=0.7, top 10% contains ~2-3x baseline
        estimated_lift = (auc_roc - 0.5) * 6  # rough formula

        print(f"\nEstimated Top 10% Performance:")
        print(f"  Expected lift: ~{estimated_lift:.1f}x baseline")
        print(f"  If baseline is 6%, top 10% should have ~{estimated_lift*6:.1f}% movers")

        if estimated_lift < 1.3:
            print("  → Not useful for targeting")
        elif estimated_lift < 2.0:
            print("  → Marginally useful for targeting")
        else:
            print("  → Useful for targeting top individuals")

    # Recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)

    if auc_pr < 0.10:
        print("🔴 AUPRC < 0.10 suggests:")
        print("   1. Features lack predictive power")
        print("   2. Individual prediction may not be feasible")
        print("   3. Consider:")
        print("      → Aggregate/group-level forecasting")
        print("      → Add contextual features (municipality, region)")
        print("      → Use microsimulation instead of direct prediction")

    if auc_roc < 0.65:
        print("\n🔴 AUC-ROC < 0.65 suggests:")
        print("   → Model cannot reliably rank individuals")
        print("   → Need stronger features or different approach")

    if p < 0.1 and r > 0.5:
        print("\n⚠️  High recall, low precision:")
        print("   → Adjust threshold for better precision/recall trade-off")
        print("   → Or focus on relative risk scores, not binary predictions")

    print("\n💡 ALTERNATIVE APPROACHES:")
    print("   1. Group-level forecasting:")
    print("      - Predict transition rates by age group × region")
    print("      - Aggregate individual scores to population level")
    print()
    print("   2. Microsimulation:")
    print("      - Use P(move) as input to stochastic simulation")
    print("      - Evaluate on aggregate outcomes, not individual prediction")
    print()
    print("   3. Feature enhancement:")
    print("      - Add municipality-level features (housing prices, unemployment)")
    print("      - Add regional migration flows")
    print("      - Add temporal context (year fixed effects)")
    print()

if __name__ == "__main__":
    # Find latest checkpoint
    checkpoint_dir = Path("checkpoints")

    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
    else:
        # Find most recent
        checkpoints = sorted(checkpoint_dir.glob("model_*"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not checkpoints:
            print("No checkpoints found in checkpoints/")
            sys.exit(1)
        checkpoint_path = checkpoints[0]
        print(f"Using most recent checkpoint: {checkpoint_path.name}\n")

    meta = load_checkpoint_data(checkpoint_path)
    if meta:
        analyze_performance(meta)
