"""
Test script for ground truth infrastructure
Run this to verify the pairmeta parser and evaluation modules work correctly
"""
import sys
import os

# Add app to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.utils.ground_truth import (
    parse_pairmeta,
    get_ground_truth,
    get_ground_truth_summary,
    get_all_datasets_with_ground_truth,
    map_prediction_to_numeric,
    map_numeric_to_string,
    check_regime_switch
)

from app.utils.evaluation import (
    calculate_regime_accuracy,
    detect_regime_switch,
    generate_comparison_table
)

def test_ground_truth_parser():
    """Test ground truth parsing"""
    print("=" * 60)
    print("TEST 1: Ground Truth Parser")
    print("=" * 60)
    
    # Parse the file
    gt_db = parse_pairmeta()
    print(f"✓ Loaded {len(gt_db)} datasets with ground truth\n")
    
    # List all datasets
    print("Datasets with ground truth:")
    for dataset in gt_db.keys():
        print(f"  - {dataset}")
    print()

def test_specific_dataset():
    """Test getting ground truth for auto_mpg"""
    print("=" * 60)
    print("TEST 2: Specific Dataset (auto_mpg_horsepower)")
    print("=" * 60)
    
    gt = get_ground_truth('auto_mpg_horsepower')
    if gt:
        print("✓ Ground truth found:")
        print(f"  Threshold variable: {gt['threshold_var']}")
        print(f"  Threshold value: {gt['threshold_val']}")
        print(f"  Regime low direction: {map_numeric_to_string(gt['regime_low_dir'])}")
        print(f"  Regime high direction: {map_numeric_to_string(gt['regime_high_dir'])}")
        print(f"  Has regime switch: {check_regime_switch(gt)}")
        print()
        print("Summary:")
        print(get_ground_truth_summary('auto_mpg_horsepower'))
    else:
        print("✗ Ground truth not found")
    print()

def test_evaluation():
    """Test evaluation metrics"""
    print("=" * 60)
    print("TEST 3: Evaluation Metrics")
    print("=" * 60)
    
    # Get ground truth for auto_mpg
    gt = get_ground_truth('auto_mpg_horsepower')
    
    # Simulate predictions from different methods
    # Ground truth: low=-1 (Y->X), high=1 (X->Y)
    predictions = {
        'ROCHE': {'regime_low': -1, 'regime_high': 1},  # Correct
        'LOCI': {'regime_low': -1, 'regime_high': -1},  # Low correct, high wrong
        'LCUBE': {'regime_low': 1, 'regime_high': 1},   # Low wrong, high correct
        'LLM': {'regime_low': -1, 'regime_high': 1},    # Correct
    }
    
    print("Predictions:")
    for method, pred in predictions.items():
        print(f"  {method}: Low={map_numeric_to_string(pred['regime_low'])}, "
              f"High={map_numeric_to_string(pred['regime_high'])}")
    print()
    
    # Generate comparison table
    comparison_df = generate_comparison_table(predictions, gt)
    print("Comparison Table:")
    print(comparison_df.to_string(index=False))
    print()
    
    # Test individual method accuracy
    print("Detailed Metrics for ROCHE:")
    roche_metrics = calculate_regime_accuracy(predictions['ROCHE'], gt)
    print(f"  Regime low correct: {roche_metrics['regime_low_correct']}")
    print(f"  Regime high correct: {roche_metrics['regime_high_correct']}")
    print(f"  Overall accuracy: {roche_metrics['overall_accuracy']*100:.0f}%")
    
    roche_switch = detect_regime_switch(predictions['ROCHE'], gt)
    print(f"  Switch detected: {roche_switch['switch_detected']}")
    print(f"  Switch correct: {roche_switch['switch_correct']}")
    print()

def test_direction_mapping():
    """Test direction string mapping"""
    print("=" * 60)
    print("TEST 4: Direction Mapping")
    print("=" * 60)
    
    test_strings = ["X->Y", "Y->X", "x→y", "y→x", "uncertain", "1", "-1", "bidirectional"]
    
    print("String to Numeric:")
    for s in test_strings:
        numeric = map_prediction_to_numeric(s)
        back_to_string = map_numeric_to_string(numeric)
        print(f"  '{s}' -> {numeric} -> '{back_to_string}'")
    print()

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Ground Truth Infrastructure Test Suite")
    print("=" * 60 + "\n")
    
    try:
        test_ground_truth_parser()
        test_specific_dataset()
        test_evaluation()
        test_direction_mapping()
        
        print("=" * 60)
        print("✅ ALL TESTS PASSED")
        print("=" * 60)
        
    except Exception as e:
        print("\n" + "=" * 60)
        print("❌ TEST FAILED")
        print("=" * 60)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
