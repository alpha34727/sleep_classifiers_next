import sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use('Agg')

# Add src to python path so we can import sleep_next
src_path = str(Path(__file__).parent.parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import os
import argparse
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_curve
from sleep_next.config import settings
from sleep_next.data import loader
from sleep_next.evaluate import cross_val, metrics
from sleep_next.models import legacy, modern
from sleep_next.visualize import plotters

# List of classifier mappings
CLASSIFIERS = {
    "logistic_regression": (legacy.LogisticRegressionStrategy, {}, "Logistic Regression"),
    "knn": (legacy.KNNStrategy, {}, "k-Nearest Neighbors"),
    "random_forest": (legacy.RandomForestStrategy, {}, "Random Forest"),
    "neural_net": (legacy.NeuralNetStrategy, {}, "Neural Net"),
    "lightgbm": (modern.LightGBMStrategy, {}, "LightGBM"),
    "xgboost": (modern.XGBoostStrategy, {}, "XGBoost")
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

FEATURE_LABELS = {
    "motion": "Motion only",
    "hr": "HR only",
    "motion_hr": "Motion, HR",
    "all": "Motion, HR, and Clock"
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
        thresh = float(np.interp(target_tpr, tpr, thresholds))
        
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

def main():
    parser = argparse.ArgumentParser(description="Run the full Phase 2 sleep classification Monte Carlo benchmark.")
    
    # Split settings
    parser.add_argument("--binary-splits", type=int, default=50, help="Number of Monte Carlo splits for Sleep/Wake.")
    parser.add_argument("--three-class-splits", type=int, default=20, help="Number of Monte Carlo splits for 3-class staging.")
    
    # Execution targets
    parser.add_argument("--run-all", action="store_true", help="Run all tasks and generate all figures (default if nothing else specified).")
    parser.add_argument("--run-binary", action="store_true", help="Run Sleep/Wake (Binary) cross-validation.")
    parser.add_argument("--run-staging", action="store_true", help="Run Three-Class (Staging) cross-validation.")
    
    # Plotting targets
    parser.add_argument("--plot-sw-roc", action="store_true", help="Plot Sleep/Wake ROC curves.")
    parser.add_argument("--plot-sw-pr", action="store_true", help="Plot Sleep/Wake PR curves.")
    parser.add_argument("--plot-sw-hist", action="store_true", help="Plot Sleep/Wake Histograms.")
    
    parser.add_argument("--plot-staging-roc", action="store_true", help="Plot Three-Class Staging ROC curves.")
    parser.add_argument("--plot-staging-ovr", action="store_true", help="Plot One-vs-Rest ROC curves.")
    parser.add_argument("--plot-bland-altman", action="store_true", help="Plot Bland-Altman diagrams.")
    
    parser.add_argument("--generate-tables", action="store_true", help="Generate Benchmark Tables.")

    args = parser.parse_args()
    
    # Logic to auto-resolve dependencies
    actions_specified = any([
        args.run_binary, args.run_staging, args.plot_sw_roc, args.plot_sw_pr,
        args.plot_sw_hist, args.plot_staging_roc, args.plot_staging_ovr,
        args.plot_bland_altman, args.generate_tables
    ])
    
    # If no flag is passed, or --run-all is passed, turn everything ON
    if args.run_all or not actions_specified:
        args.run_binary = True
        args.run_staging = True
        args.plot_sw_roc = True
        args.plot_sw_pr = True
        args.plot_sw_hist = True
        args.plot_staging_roc = True
        args.plot_staging_ovr = True
        args.plot_bland_altman = True
        args.generate_tables = True
        
    # Auto-enable underlying cross-validation if plots are requested
    if args.plot_sw_roc or args.plot_sw_pr or args.plot_sw_hist or args.generate_tables:
        args.run_binary = True
    if args.plot_staging_roc or args.plot_staging_ovr or args.plot_bland_altman or args.generate_tables:
        args.run_staging = True
        
    subject_ids = loader.get_all_subject_ids()
    os.makedirs(settings.FIGURE_DIR, exist_ok=True)
    os.makedirs(settings.OUTPUT_DIR / "tables", exist_ok=True)
    
    sw_results = {}
    tc_results = {}
    
    # Outer loop for strategies
    outer_bar = tqdm(CLASSIFIERS.keys(), desc="Overall Benchmark Progress")
    for cls_key in outer_bar:
        strategy_cls, strategy_kwargs, cls_name = CLASSIFIERS[cls_key]
        outer_bar.set_postfix({"model": cls_name})
        
        # Inner loop for features
        sw_perf_dict = {}
        tc_perf_dict = {}
        
        for feature_set in FEATURE_SETS:
            feat_key = FEATURE_TO_KEY[tuple(feature_set)]
            feat_label = FEATURE_LABELS[feat_key]
            
            # --- Sleep/Wake (Binary) ---
            if args.run_binary:
                # Check if TreeSHAP should be run for LightGBM on the last feature set and last split
                run_shap = (cls_key == "lightgbm" and feat_key == "all")
                shap_path = str(settings.FIGURE_DIR / "modern_lightgbm_shap_summary.png") if run_shap else ""
                
                raw_sw = cross_val.run_cross_validation(
                    subject_ids=subject_ids,
                    strategy_cls=strategy_cls,
                    strategy_kwargs=strategy_kwargs,
                    feature_cols=feature_set,
                    cv_type="mc",
                    classification_type="sleep_wake",
                    number_of_splits=args.binary_splits,
                    scoring="roc_auc",
                    compute_shap_last_fold=run_shap,
                    shap_output_path=shap_path
                )
                sw_perf_dict[feat_key] = raw_sw
            
            # --- Three-Class (Staging) ---
            if args.run_staging:
                raw_tc = cross_val.run_cross_validation(
                    subject_ids=subject_ids,
                    strategy_cls=strategy_cls,
                    strategy_kwargs=strategy_kwargs,
                    feature_cols=feature_set,
                    cv_type="mc",
                    classification_type="three_class",
                    number_of_splits=args.three_class_splits,
                    scoring="neg_log_loss"
                )
                tc_perf_dict[feat_key] = raw_tc
            
        if args.run_binary:
            sw_results[cls_key] = sw_perf_dict
        if args.run_staging:
            tc_results[cls_key] = tc_perf_dict
        
        # Generate plot curves for each classifier based on flags
        if args.plot_sw_roc:
            plotters.make_roc_sw(cls_name, sw_perf_dict)
        if args.plot_sw_pr:
            plotters.make_pr_sw(cls_name, sw_perf_dict)
        if args.plot_sw_hist:
            plotters.make_single_threshold_histograms(cls_name, sw_perf_dict)
            
        if args.plot_staging_ovr:
            plotters.make_roc_one_vs_rest(cls_name, tc_perf_dict)
        if args.plot_bland_altman:
            plotters.make_bland_altman(cls_name, tc_perf_dict)
        if args.plot_staging_roc:
            plotters.make_three_class_roc(cls_name, tc_perf_dict)
        
    # Generate combined grid figures across all models
    cls_names = [CLASSIFIERS[k][2] for k in CLASSIFIERS.keys()]
    if args.plot_sw_roc:
        plotters.combine_plots_as_grid(cls_names, args.binary_splits, "sw_roc")
    if args.plot_sw_pr:
        plotters.combine_plots_as_grid(cls_names, args.binary_splits, "sw_pr")
    
    # ----------------------------------------------------
    # Generate Markdown Table Report
    # ----------------------------------------------------
    if args.generate_tables:
        table_path = settings.OUTPUT_DIR / "tables" / "full_paper_benchmark.md"
        
        with open(table_path, "w", encoding="utf-8") as f:
            f.write("# Modern Sleep Classification Master Benchmark Report (Phase 2)\n\n")
            f.write("Generated using Monte Carlo Cross Validation. Strict subject isolation is guaranteed.\n\n")
            
            # Print Tables 2-5 structure (Binary Sleep/Wake Performance for each model)
            f.write("## Part 1: Sleep/Wake (Binary) Performance (Tables 2-5 Equivalent)\n\n")
            
            for cls_key in CLASSIFIERS.keys():
                _, _, cls_name = CLASSIFIERS[cls_key]
                f.write(f"### Sleep/wake differentiation performance by **{cls_name}** ({args.binary_splits} splits)\n\n")
                f.write("| Feature Input | Target TPR | Accuracy | Specificity (Wake Correct) | Sensitivity (Sleep Correct) | Cohen's Kappa | AUC |\n")
                f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
                
                sw_perf_dict = sw_results[cls_key]
                for feature_set in FEATURE_SETS:
                    feat_key = FEATURE_TO_KEY[tuple(feature_set)]
                    feat_label = FEATURE_LABELS[feat_key]
                    raw_sw = sw_perf_dict[feat_key]
                    
                    target_tprs = [0.8, 0.9, 0.93, 0.95]
                    for i, tpr in enumerate(target_tprs):
                        perf = summarize_fold_at_tpr(raw_sw, tpr)
                        if i == 0:
                            f.write(f"| {feat_label} | {tpr:.2f} | {perf.accuracy:.3f} | {perf.wake_correct:.3f} | {perf.sleep_correct:.3f} | {perf.kappa:.3f} | {perf.auc:.3f} |\n")
                        else:
                            f.write(f"| | {tpr:.2f} | {perf.accuracy:.3f} | {perf.wake_correct:.3f} | {perf.sleep_correct:.3f} | {perf.kappa:.3f} | |\n")
                    f.write("| | | | | | | |\n")
                f.write("\n")
                
            # Print Table 6 structure (Three-Class Staging Performance across all models)
            f.write("## Part 2: Three-Class Sleep Staging Performance (Table 6 Equivalent)\n\n")
            f.write("| Classifier | Feature Set | Wake Correct | NREM Correct | REM Correct | Best Accuracy | Cohen's Kappa |\n")
            f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
            
            for cls_key in CLASSIFIERS.keys():
                _, _, cls_name = CLASSIFIERS[cls_key]
                tc_perf_dict = tc_results[cls_key]
                
                # Execute three class search for table values
                for feature_set in FEATURE_SETS:
                    feat_key = FEATURE_TO_KEY[tuple(feature_set)]
                    feat_label = FEATURE_LABELS[feat_key]
                    raw_tc = tc_perf_dict[feat_key]
                    
                    # Run the Equal Accuracy Search across folds and average the metrics
                    from sleep_next.evaluate.metrics import equalize_three_class_thresholds
                    
                    accs, wake_cs, rem_cs, nrem_cs, kappas = [], [], [], [], []
                    for rp in raw_tc:
                        # Calculate thresholds
                        perf = equalize_three_class_thresholds(rp.true_labels, rp.class_probabilities)
                        accs.append(perf.accuracy)
                        wake_cs.append(perf.wake_correct)
                        rem_cs.append(perf.rem_correct)
                        nrem_cs.append(perf.nrem_correct)
                        kappas.append(perf.kappa)
                        
                    f.write(f"| {cls_name} | {feat_label} | {np.mean(wake_cs):.3f} | {np.mean(nrem_cs):.3f} | {np.mean(rem_cs):.3f} | {np.mean(accs):.3f} | {np.mean(kappas):.3f} |\n")
            
            f.write("\n")
            
        print(f"Master benchmark table created successfully at: {table_path}")

if __name__ == "__main__":
    main()
