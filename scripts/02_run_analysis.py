import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

# Add src to python path so we can import sleep_next
src_path = str(Path(__file__).parent.parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import argparse
import numpy as np
from sklearn.metrics import roc_curve, auc
from sleep_next.config import settings
from sleep_next.data import loader
from sleep_next.evaluate import cross_val, metrics
from sleep_next.models import legacy, modern
from sleep_next.visualize import plotters

# List of classifier mappings
CLASSIFIERS = {
    "random_forest": (legacy.RandomForestStrategy, {}, "Random Forest"),
    "logistic_regression": (legacy.LogisticRegressionStrategy, {}, "Logistic Regression"),
    "knn": (legacy.KNNStrategy, {}, "k-Nearest Neighbors"),
    "neural_net": (legacy.NeuralNetStrategy, {}, "Neural Net"),
    "xgboost": (modern.XGBoostStrategy, {}, "XGBoost"),
    "lightgbm": (modern.LightGBMStrategy, {}, "LightGBM")
}

FEATURE_SETS = [
    ["feature_count"],
    ["feature_hr"],
    ["feature_count", "feature_hr"],
    ["feature_count", "feature_hr", "feature_cosine"]
]

FEATURE_TO_KEY = {
    ("feature_count",): "motion",
    ("feature_hr",): "hr",
    ("feature_count", "feature_hr"): "motion_hr",
    ("feature_count", "feature_hr", "feature_cosine"): "all"
}

def summarize_fold_at_tpr(raw_performances, target_tpr: float):
    accuracies = []
    wake_corrects = []
    sleep_corrects = []
    kappas = []
    aucs = []
    
    for rp in raw_performances:
        true_labels = (rp.true_labels > 0).astype(np.int32)
        probs = rp.class_probabilities[:, 1] if rp.class_probabilities.shape[1] > 1 else rp.class_probabilities[:, 0]
        
        fpr, tpr, thresholds = roc_curve(true_labels, probs, pos_label=1, drop_intermediate=False)
        # Interpolate threshold at target TPR
        # Since np.interp expects increasing xp, and tpr is increasing:
        thresh = float(np.interp(target_tpr, tpr, thresholds))
        
        # Evaluate at this threshold
        perf = metrics.compute_sleep_wake_performance(true_labels, rp.class_probabilities, thresh)
        
        accuracies.append(perf.accuracy)
        wake_corrects.append(perf.wake_correct)
        sleep_corrects.append(perf.sleep_correct)
        kappas.append(perf.kappa)
        aucs.append(perf.auc)
        
    class Summary:
        def __init__(self, acc, wc, sc, kap, a_val):
            self.accuracy = acc
            self.wake_correct = wc
            self.sleep_correct = sc
            self.kappa = kap
            self.auc = a_val
            
    return Summary(np.mean(accuracies), np.mean(wake_corrects), np.mean(sleep_corrects), np.mean(kappas), np.mean(aucs))

def print_table_sw(classifier_name: str, performance_dict: dict):
    print(f"\n\\begin{{table}}  \\caption{{Sleep/wake differentiation performance by {classifier_name} across different feature inputs}}")
    print("\\begin{tabular}{l*{5}{c}} & Accuracy & Wake correct (specificity) & Sleep correct (sensitivity) & $\\kappa$ & AUC \\\\")
    
    target_tprs = [0.8, 0.9, 0.93, 0.95]
    
    for feature_set, raw_performances in performance_dict.items():
        print('\\hline ' + plotters.get_label(feature_set) + ' &')
        
        for tpr in target_tprs:
            perf = summarize_fold_at_tpr(raw_performances, tpr)
            if tpr == 0.8:
                print(
                    f"{perf.accuracy:.3f} & {perf.wake_correct:.3f} & {perf.sleep_correct:.3f} & {perf.kappa:.3f} & {perf.auc:.3f} \\\\"
                )
            else:
                print(
                    f"& {perf.accuracy:.3f} & {perf.wake_correct:.3f} & {perf.sleep_correct:.3f} & {perf.kappa:.3f} & \\\\"
                )
                
    print(f"\\hline \\end{{tabular}}  \\end{{table}}\n")

def print_table_three_class(classifier_summaries_list: list):
    # Each item: (classifier_name, three_class_results_dict)
    print("\n\\begin{tabular}{l | l | c | c | c | c | c } & & Wake Correct & NREM Correct & REM Correct & Best accuracy & $\\kappa$ \\")
    
    for classifier_name, results_dict in classifier_summaries_list:
        is_first = True
        for feature_set, perf_list in results_dict.items():
            # perf_list is a list of ThreeClassPerformance objects (one per fold)
            wake_c = np.mean([p.wake_correct for p in perf_list])
            nrem_c = np.mean([p.nrem_correct for p in perf_list])
            rem_c = np.mean([p.rem_correct for p in perf_list])
            acc = np.mean([p.accuracy for p in perf_list])
            kap = np.mean([p.kappa for p in perf_list])
            
            lbl = plotters.get_label(feature_set)
            if is_first:
                print(f"\\hline {classifier_name} & {lbl} & {wake_c:.3f} & {nrem_c:.3f} & {rem_c:.3f} & {acc:.3f} & {kap:.3f} \\\\")
                is_first = False
            else:
                print(f" & {lbl} & {wake_c:.3f} & {nrem_c:.3f} & {rem_c:.3f} & {acc:.3f} & {kap:.3f} \\\\")
                
    print("\\end{tabular}\n")

def main():
    parser = argparse.ArgumentParser(description="Run sleep classification model analysis and generate paper plots.")
    parser.add_argument("--classifier", type=str, choices=list(CLASSIFIERS.keys()) + ["all"], default="all", help="Classifier to evaluate.")
    parser.add_argument("--cv", type=str, choices=["mc", "loo"], default="loo", help="Cross-validation mode: 'loo' (Leave-One-Out) or 'mc' (Monte Carlo).")
    parser.add_argument("--trials", type=int, default=20, help="Number of Monte Carlo splits (only applicable if cv=mc).")
    args = parser.parse_args()
    
    subject_ids = loader.get_all_subject_ids()
    
    eval_classifiers = list(CLASSIFIERS.keys()) if args.classifier == "all" else [args.classifier]
    
    three_class_summaries = []
    
    for cls_key in eval_classifiers:
        strategy_cls, strategy_kwargs, cls_name = CLASSIFIERS[cls_key]
        print(f"\n==================================================")
        print(f"Running evaluation for {cls_name} ({args.cv.upper()})...")
        print(f"==================================================")
        
        # --- Run Sleep-Wake (Binary) Analysis ---
        sw_perf_dict = {}
        for feature_set in FEATURE_SETS:
            print(f"Training on features: {feature_set}")
            raw_performances = cross_val.run_cross_validation(
                subject_ids=subject_ids,
                strategy_cls=strategy_cls,
                strategy_kwargs=strategy_kwargs,
                feature_cols=feature_set,
                cv_type=args.cv,
                classification_type="sleep_wake",
                number_of_splits=args.trials,
                scoring="roc_auc"
            )
            sw_perf_dict[FEATURE_TO_KEY[tuple(feature_set)]] = raw_performances
            
        # Draw Sleep-Wake curves & histograms
        plotters.make_roc_sw(cls_name, sw_perf_dict)
        plotters.make_pr_sw(cls_name, sw_perf_dict)
        plotters.make_single_threshold_histograms(cls_name, sw_perf_dict)
        print_table_sw(cls_name, sw_perf_dict)
        
        # --- Run Three-Class Analysis ---
        tc_perf_dict = {}
        for feature_set in FEATURE_SETS:
            print(f"Training three-class model on features: {feature_set}")
            raw_performances = cross_val.run_cross_validation(
                subject_ids=subject_ids,
                strategy_cls=strategy_cls,
                strategy_kwargs=strategy_kwargs,
                feature_cols=feature_set,
                cv_type=args.cv,
                classification_type="three_class",
                number_of_splits=args.trials,
                scoring="neg_log_loss"
            )
            tc_perf_dict[FEATURE_TO_KEY[tuple(feature_set)]] = raw_performances
            
        # Draw Three-Class curves & Bland-Altman plots
        plotters.make_roc_one_vs_rest(cls_name, tc_perf_dict)
        three_class_results = plotters.make_three_class_roc(cls_name, tc_perf_dict)
        plotters.make_bland_altman(cls_name, tc_perf_dict)
        
        three_class_summaries.append((cls_name, three_class_results))
        
    if len(three_class_summaries) > 0:
        print_table_three_class(three_class_summaries)
        
    # Combine plots if multiple classifiers were run
    if len(eval_classifiers) > 1:
        plotters.combine_plots_as_grid(
            [CLASSIFIERS[k][2] for k in eval_classifiers],
            args.trials if args.cv == "mc" else len(subject_ids),
            "sw_roc"
        )
        plotters.combine_plots_as_grid(
            [CLASSIFIERS[k][2] for k in eval_classifiers],
            args.trials if args.cv == "mc" else len(subject_ids),
            "sw_pr"
        )
        print("Combined figures grid created successfully.")

if __name__ == "__main__":
    main()
