import numpy as np
from sleep_next.evaluate.metrics import equalize_three_class_thresholds

def test_equalize_three_class_thresholds():
    # Generate mock data: 3 classes (0=wake, 1=nrem, 2=rem)
    np.random.seed(42)
    N = 1000
    
    # True labels
    true_labels = np.random.choice([0, 1, 2], size=N, p=[0.4, 0.4, 0.2])
    
    # Generate probabilities roughly aligned with the true labels with continuous noise
    probs = np.zeros((N, 3))
    for i in range(N):
        if true_labels[i] == 0:
            probs[i] = [0.6, 0.2, 0.2] + np.random.uniform(-0.15, 0.15, 3)
        elif true_labels[i] == 1:
            probs[i] = [0.2, 0.6, 0.2] + np.random.uniform(-0.15, 0.15, 3)
        else:
            probs[i] = [0.2, 0.2, 0.6] + np.random.uniform(-0.15, 0.15, 3)
            
    # Clip and normalize probabilities
    probs = np.clip(probs, 1e-5, 1.0)
            
    # Normalize probabilities
    probs = probs / probs.sum(axis=1, keepdims=True)
    
    # Run the equal-accuracy binary search algorithm
    perf = equalize_three_class_thresholds(
        true_labels=true_labels,
        class_probabilities=probs,
        wake_scored_as_sleep_interpolation_point=0.4
    )
    
    # Verify results
    assert perf.wake_correct == 0.6
    # Check that NREM and REM correct are balanced (within reasonable threshold search limits)
    assert abs(perf.nrem_correct - perf.rem_correct) < 0.15
    assert perf.accuracy > 0.5
    assert perf.kappa > 0.1
