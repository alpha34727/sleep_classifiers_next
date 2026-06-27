import math
import numpy as np
import polars as pl
import seaborn as sns

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from sklearn.metrics import roc_curve, precision_recall_curve, accuracy_score, cohen_kappa_score, recall_score, precision_score
from sleep_next.config import settings, normalize_feature_label
from sleep_next.evaluate.metrics import get_tst, get_wake_after_sleep_onset, get_sleep_efficiency, get_sleep_onset_latency, get_time_in_rem, get_time_in_nrem

def get_label(feature_set) -> str:
    return normalize_feature_label(feature_set)

def get_color(feature_set) -> str:
    if isinstance(feature_set, str):
        if feature_set == "motion": return sns.xkcd_rgb["denim blue"]
        if feature_set == "hr": return sns.xkcd_rgb["yellow orange"]
        if feature_set == "motion_hr": return sns.xkcd_rgb["medium green"]
        if feature_set == "all": return sns.xkcd_rgb["plum"]
        
    s = set(feature_set)
    if s == {"feature_count"}: return sns.xkcd_rgb["denim blue"]
    if s == {"feature_hr"}: return sns.xkcd_rgb["yellow orange"]
    if s == {"feature_count", "feature_hr"}: return sns.xkcd_rgb["medium green"]
    if s == {"feature_count", "feature_hr", "feature_circadian"}: return sns.xkcd_rgb["medium pink"]
    if s == {"feature_count", "feature_hr", "feature_cosine"}: return sns.xkcd_rgb["plum"]
    if s == {"feature_count", "feature_hr", "feature_time"}: return sns.xkcd_rgb["greyish"]
    return "#3b5b92"

def tidy_plot(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

def build_roc_from_raw(raw_performances, positive_class: int):
    n_pts = 100
    false_positive_spread = np.linspace(0.01, 1.0, n_pts)
    true_positive_spread = np.zeros(n_pts)
    count = 0
    
    for rp in raw_performances:
        # Convert multiclass label if needed
        # positive class: 1 (sleep) or 0 (wake)
        if positive_class == 1:
            true_labels = (rp.true_labels > 0).astype(np.int32)
            probs = rp.class_probabilities[:, 1] if rp.class_probabilities.shape[1] > 1 else rp.class_probabilities[:, 0]
        else:
            # 1 vs rest for positive class
            true_labels = (rp.true_labels == positive_class).astype(np.int32)
            probs = rp.class_probabilities[:, positive_class]
            
        fpr, tpr, _ = roc_curve(true_labels, probs, pos_label=1, drop_intermediate=False)
        true_positive_spread += np.interp(false_positive_spread, fpr, tpr)
        count += 1
        
    true_positive_spread /= count
    false_positive_spread = np.insert(false_positive_spread, 0, 0.0)
    true_positive_spread = np.insert(true_positive_spread, 0, 0.0)
    return false_positive_spread, true_positive_spread

def make_roc_sw(classifier_name: str, performance_dict: dict, description: str = ""):
    plt.figure()
    ax = plt.subplot(111)
    
    for feature_set, raw_performances in performance_dict.items():
        fpr_spread, tpr_spread = build_roc_from_raw(raw_performances, 1)
        plt.plot(fpr_spread, tpr_spread, label=get_label(feature_set), color=get_color(feature_set))
        
    tidy_plot(ax)
    plt.xlabel('Fraction of wake scored as sleep', fontsize=14, fontname='Arial')
    plt.ylabel('Fraction of sleep scored as sleep', fontsize=14, fontname='Arial')
    plt.title(classifier_name, fontsize=18, fontname='Arial', fontweight='bold')
    plt.legend(loc="lower right")
    
    num_trials = len(next(iter(performance_dict.values())))
    out_path = settings.FIGURE_DIR / f"{classifier_name}_{num_trials}_{description}_sw_roc.png"
    plt.savefig(out_path, dpi=300)
    plt.close('all')

def make_pr_sw(classifier_name: str, performance_dict: dict, description: str = ""):
    plt.figure()
    ax = plt.subplot(111)
    
    n_pts = 100
    recall_spread = np.linspace(0.01, 1.0, n_pts)
    
    for feature_set, raw_performances in performance_dict.items():
        precision_spread = np.zeros(n_pts)
        count = 0
        
        for rp in raw_performances:
            count += 1
            # In legacy, sleep/wake labels: wake = 0, sleep = 1.
            # Precision Recall Curve uses Wake as positive class (i.e. pos_label=0)
            true_labels = (rp.true_labels > 0).astype(np.int32)
            # class_probabilities[:, 0] is the probability of wake
            wake_probs = rp.class_probabilities[:, 0]
            prec, rec, _ = precision_recall_curve(true_labels, wake_probs, pos_label=0)
            precision_spread += np.interp(recall_spread, np.flip(rec), np.flip(prec))
            
        precision_spread /= count
        r_spread = np.insert(recall_spread, 0, 0.0)
        p_spread = np.insert(precision_spread, 0, 1.0)
        
        plt.plot(r_spread, p_spread, label=get_label(feature_set), color=get_color(feature_set))
        
    tidy_plot(ax)
    plt.xlabel('Fraction of wake scored as wake', fontsize=14, fontname='Arial')
    plt.ylabel('Fraction of predicted wake correct', fontsize=14, fontname='Arial')
    plt.title(classifier_name, fontsize=18, fontname='Arial', fontweight='bold')
    plt.legend(loc="lower left")
    
    num_trials = len(next(iter(performance_dict.values())))
    out_path = settings.FIGURE_DIR / f"{classifier_name}_{num_trials}_{description}_sw_pr.png"
    plt.savefig(out_path, dpi=300)
    plt.close('all')

def make_roc_one_vs_rest(classifier_name: str, performance_dict: dict, description: str = ""):
    num_trials = len(next(iter(performance_dict.values())))
    
    # Wake ROC (class 0)
    plt.figure()
    ax = plt.subplot(111)
    for feature_set, raw_performances in performance_dict.items():
        fpr_spread, tpr_spread = build_roc_from_raw(raw_performances, 0)
        plt.plot(fpr_spread, tpr_spread, label=get_label(feature_set), color=get_color(feature_set))
    tidy_plot(ax)
    plt.xlabel('Fraction of REM or NREM scored as wake', fontsize=14, fontname='Arial')
    plt.ylabel('Fraction of wake scored as wake', fontsize=14, fontname='Arial')
    plt.title(classifier_name, fontsize=18, fontname='Arial', fontweight='bold')
    plt.legend(loc="lower right")
    plt.savefig(settings.FIGURE_DIR / f"{classifier_name}_{num_trials}_{description}_ovr_wake_roc.png", dpi=300)
    plt.close('all')
    
    # NREM ROC (class 1)
    plt.figure()
    ax = plt.subplot(111)
    for feature_set, raw_performances in performance_dict.items():
        fpr_spread, tpr_spread = build_roc_from_raw(raw_performances, 1)
        plt.plot(fpr_spread, tpr_spread, label=get_label(feature_set), color=get_color(feature_set))
    tidy_plot(ax)
    plt.xlabel('Fraction of wake or REM scored as NREM', fontsize=14, fontname='Arial')
    plt.ylabel('Fraction of NREM scored as NREM', fontsize=14, fontname='Arial')
    plt.title(classifier_name, fontsize=18, fontname='Arial', fontweight='bold')
    plt.legend(loc="lower right")
    plt.savefig(settings.FIGURE_DIR / f"{classifier_name}_{num_trials}_{description}_ovr_nrem_roc.png", dpi=300)
    plt.close('all')
    
    # REM ROC (class 2)
    plt.figure()
    ax = plt.subplot(111)
    for feature_set, raw_performances in performance_dict.items():
        fpr_spread, tpr_spread = build_roc_from_raw(raw_performances, 2)
        plt.plot(fpr_spread, tpr_spread, label=get_label(feature_set), color=get_color(feature_set))
    tidy_plot(ax)
    plt.xlabel('Fraction of wake or NREM scored as REM', fontsize=14, fontname='Arial')
    plt.ylabel('Fraction of REM scored as REM', fontsize=14, fontname='Arial')
    plt.title(classifier_name, fontsize=18, fontname='Arial', fontweight='bold')
    plt.legend(loc="lower right")
    plt.savefig(settings.FIGURE_DIR / f"{classifier_name}_{num_trials}_{description}_ovr_rem_roc.png", dpi=300)
    plt.close('all')

def build_three_class_roc_with_binary_search(raw_performances):
    number_of_wake_scored_as_sleep_bins = 20
    false_positive_buffer = 0.001
    max_attempts_binary_search_wake = 50
    rem_nrem_accuracy_tolerance = 1e-2
    max_attempts_binary_search_rem_nrem = 15
    wake_scored_as_sleep_interpolation_point = 0.4
    
    goal_fraction_wake_scored_as_sleep_spread = np.linspace(0.0, 0.95, number_of_wake_scored_as_sleep_bins)
    
    cumulative_nrem_accuracies = np.zeros_like(goal_fraction_wake_scored_as_sleep_spread)
    cumulative_rem_accuracies = np.zeros_like(goal_fraction_wake_scored_as_sleep_spread)
    cumulative_accuracies = np.zeros_like(goal_fraction_wake_scored_as_sleep_spread)
    
    cumulative_counter = 0
    
    three_class_performances = []
    
    for rp in raw_performances:
        true_labels = rp.true_labels
        class_probabilities = rp.class_probabilities
        
        wake_scored_as_sleep_spread = []
        sleep_accuracy_spread = []
        accuracies = []
        kappas = []
        nrem_class_accuracies = []
        rem_class_accuracies = []
        
        true_wake_indices = np.where(true_labels == 0)[0]
        true_nrem_indices = np.where(true_labels == 1)[0]
        true_rem_indices = np.where(true_labels == 2)[0]
        
        for goal_fraction_wake_scored_as_sleep in goal_fraction_wake_scored_as_sleep_spread:
            fraction_wake_scored_as_sleep = -1.0
            binary_search_counter = 0
            
            while (fraction_wake_scored_as_sleep < goal_fraction_wake_scored_as_sleep - false_positive_buffer
                   or fraction_wake_scored_as_sleep >= goal_fraction_wake_scored_as_sleep + false_positive_buffer) \
                   and binary_search_counter < max_attempts_binary_search_wake:
                
                if binary_search_counter == 0:
                    threshold_for_sleep = 0.5
                    threshold_delta = 0.25
                else:
                    if fraction_wake_scored_as_sleep < goal_fraction_wake_scored_as_sleep - false_positive_buffer:
                        threshold_for_sleep -= threshold_delta
                        threshold_delta /= 2.0
                    else:
                        threshold_for_sleep += threshold_delta
                        threshold_delta /= 2.0
                        
                if goal_fraction_wake_scored_as_sleep == 1.0:
                    threshold_for_sleep = 0.0
                if goal_fraction_wake_scored_as_sleep == 0.0:
                    threshold_for_sleep = 1.0
                    
                predicted_sleep_indices = np.where(1.0 - class_probabilities[:, 0] >= threshold_for_sleep)[0]
                predicted_labels = np.zeros_like(true_labels)
                predicted_labels[predicted_sleep_indices] = 1
                predicted_labels_at_true_wake = predicted_labels[true_wake_indices]
                
                number_wake_correct = len(true_wake_indices) - np.count_nonzero(predicted_labels_at_true_wake)
                fraction_wake_correct = number_wake_correct / max(1.0, len(true_wake_indices))
                fraction_wake_scored_as_sleep = 1.0 - fraction_wake_correct
                
                binary_search_counter += 1
                
            if binary_search_counter < max_attempts_binary_search_wake:
                smallest_accuracy_difference = 2.0
                sleep_accuracy = 0.0
                rem_accuracy = 0.0
                nrem_accuracy = 0.0
                best_accuracy = -1.0
                kappa_at_best_accuracy = -1.0
                
                count_thresh = 0
                threshold_for_rem = 0.5
                threshold_delta_rem = 0.5
                
                while count_thresh < max_attempts_binary_search_rem_nrem and \
                        smallest_accuracy_difference > rem_nrem_accuracy_tolerance:
                    count_thresh += 1
                    
                    for idx in predicted_sleep_indices:
                        if class_probabilities[idx, 2] > threshold_for_rem:
                            predicted_labels[idx] = 2  # REM
                        else:
                            predicted_labels[idx] = 1  # NREM
                            
                    accuracy = accuracy_score(predicted_labels, true_labels)
                    kappa = cohen_kappa_score(predicted_labels, true_labels)
                    
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        kappa_at_best_accuracy = kappa
                        
                    predicted_nrem_indices = np.where(predicted_labels == 1)[0]
                    predicted_rem_indices = np.where(predicted_labels == 2)[0]
                    
                    correct_nrem_indices = np.intersect1d(predicted_nrem_indices, true_nrem_indices)
                    correct_rem_indices = np.intersect1d(predicted_rem_indices, true_rem_indices)
                    
                    nrem_accuracy = len(correct_nrem_indices) / max(1.0, len(true_nrem_indices))
                    rem_accuracy = len(correct_rem_indices) / max(1.0, len(true_rem_indices))
                    
                    sleep_accuracy = (len(correct_nrem_indices) + len(correct_rem_indices)) / max(
                        1.0, len(true_nrem_indices) + len(true_rem_indices)
                    )
                    
                    smallest_accuracy_difference = np.abs(nrem_accuracy - rem_accuracy)
                    
                    if rem_accuracy < nrem_accuracy:
                        threshold_for_rem -= threshold_delta_rem / 2.0
                    else:
                        threshold_for_rem += threshold_delta_rem / 2.0
                    threshold_delta_rem /= 2.0
                    
                wake_scored_as_sleep_spread.append(fraction_wake_scored_as_sleep)
                sleep_accuracy_spread.append(sleep_accuracy)
                nrem_class_accuracies.append(nrem_accuracy)
                rem_class_accuracies.append(rem_accuracy)
                accuracies.append(best_accuracy)
                kappas.append(kappa_at_best_accuracy)
                
        wake_scored_as_sleep_spread = np.array(wake_scored_as_sleep_spread)
        sleep_accuracy_spread = np.array(sleep_accuracy_spread)
        nrem_class_accuracies = np.array(nrem_class_accuracies)
        rem_class_accuracies = np.array(rem_class_accuracies)
        
        wake_scored_as_sleep_spread = np.insert(wake_scored_as_sleep_spread, 0, 0.0)
        sleep_accuracy_spread = np.insert(sleep_accuracy_spread, 0, 0.0)
        nrem_class_accuracies = np.insert(nrem_class_accuracies, 0, 0.0)
        rem_class_accuracies = np.insert(rem_class_accuracies, 0, 0.0)
        
        idx_best = np.argmax(accuracies) if len(accuracies) > 0 else 0
        accuracy = accuracies[idx_best] if len(accuracies) > 0 else 0.0
        kappa = kappas[idx_best] if len(kappas) > 0 else 0.0
        
        cumulative_accuracies += np.interp(
            goal_fraction_wake_scored_as_sleep_spread,
            wake_scored_as_sleep_spread,
            sleep_accuracy_spread
        )
        cumulative_nrem_accuracies += np.interp(
            goal_fraction_wake_scored_as_sleep_spread,
            wake_scored_as_sleep_spread,
            nrem_class_accuracies
        )
        cumulative_rem_accuracies += np.interp(
            goal_fraction_wake_scored_as_sleep_spread,
            wake_scored_as_sleep_spread,
            rem_class_accuracies
        )
        cumulative_counter += 1
        
        rem_correct = float(np.interp(wake_scored_as_sleep_interpolation_point, wake_scored_as_sleep_spread, rem_class_accuracies))
        nrem_correct = float(np.interp(wake_scored_as_sleep_interpolation_point, wake_scored_as_sleep_spread, nrem_class_accuracies))
        
        from sleep_next.evaluate.metrics import ThreeClassPerformance
        three_class_performances.append(
            ThreeClassPerformance(
                accuracy=accuracy,
                wake_correct=1.0 - wake_scored_as_sleep_interpolation_point,
                rem_correct=rem_correct,
                nrem_correct=nrem_correct,
                kappa=kappa
            )
        )
        
    cumulative_accuracies /= max(1, cumulative_counter)
    cumulative_nrem_accuracies /= max(1, cumulative_counter)
    cumulative_rem_accuracies /= max(1, cumulative_counter)
    
    return (
        goal_fraction_wake_scored_as_sleep_spread,
        cumulative_accuracies,
        cumulative_nrem_accuracies,
        cumulative_rem_accuracies,
        three_class_performances
    )

def make_three_class_roc(classifier_name: str, performance_dict: dict, description: str = ""):
    plt.figure()
    ax = plt.subplot(111)
    
    results = {}
    
    for feature_set, raw_performances in performance_dict.items():
        (
            x_spread,
            y_acc,
            y_nrem,
            y_rem,
            three_class_performances
        ) = build_three_class_roc_with_binary_search(raw_performances)
        
        # Keep track of results for table printing
        results[feature_set] = three_class_performances
        
        plot_color = get_color(feature_set)
        legend_text = get_label(feature_set)
        
        plt.plot(x_spread, y_acc, label=legend_text, color=plot_color)
        plt.plot(x_spread, y_nrem, color=plot_color, linestyle=":")
        plt.plot(x_spread, y_rem, color=plot_color, linestyle="--")
        
    tidy_plot(ax)
    plt.xlabel('Fraction of wake scored as REM or NREM', fontsize=14, fontname='Arial')
    plt.ylabel('Fraction of REM, NREM scored correctly', fontsize=14, fontname='Arial')
    plt.title(classifier_name, fontsize=18, fontname='Arial', fontweight='bold')
    plt.legend(loc="lower right")
    
    num_trials = len(next(iter(performance_dict.values())))
    out_path = settings.FIGURE_DIR / f"{classifier_name}_{num_trials}_{description}_three_class_roc.png"
    plt.savefig(out_path, dpi=300)
    plt.close('all')
    
    return results

def make_single_threshold_histograms(classifier_name: str, performance_dict: dict, description: str = ""):
    sleep_threshold = 1.0 - settings.WAKE_THRESHOLD # e.g. 0.5
    
    for feature_set, raw_performances in performance_dict.items():
        num_subjects = len(raw_performances)
        
        all_accuracies = []
        all_fraction_wake_correct = []
        all_fraction_sleep_correct = []
        all_kappas = []
        
        for rp in raw_performances:
            # Sleep/Wake conversion
            true_labels = (rp.true_labels > 0).astype(np.int32)
            probs = rp.class_probabilities[:, 1] if rp.class_probabilities.shape[1] > 1 else rp.class_probabilities[:, 0]
            predicted = (probs >= sleep_threshold).astype(np.int32)
            
            all_accuracies.append(accuracy_score(true_labels, predicted))
            all_fraction_wake_correct.append(recall_score(true_labels, predicted, pos_label=0, zero_division=0))
            all_fraction_sleep_correct.append(recall_score(true_labels, predicted, pos_label=1, zero_division=0))
            all_kappas.append(cohen_kappa_score(true_labels, predicted))
            
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))
        dt = 0.02
        bins = np.arange(0, 1.0 + dt, dt)
        
        ax[0, 0].hist(all_accuracies, bins=bins, color="skyblue", ec="skyblue")
        ax[0, 0].set_xlabel('Accuracy', fontsize=14, fontname='Arial')
        ax[0, 0].set_ylabel('Count', fontsize=14, fontname='Arial')
        ax[0, 0].set_xlim((0, 1))
        
        ax[0, 1].hist(all_kappas, bins=bins, color="skyblue", ec="skyblue")
        ax[0, 1].set_xlabel("Cohen's Kappa", fontsize=14, fontname='Arial')
        ax[0, 1].set_xlim((0, 1))
        
        ax[1, 0].hist(all_fraction_wake_correct, bins=bins, color="skyblue", ec="skyblue")
        ax[1, 0].set_xlabel('Fraction wake correct (specificity)', fontsize=14, fontname='Arial')
        ax[1, 0].set_ylabel('Count', fontsize=14, fontname='Arial')
        ax[1, 0].set_xlim((0, 1))
        
        ax[1, 1].hist(all_fraction_sleep_correct, bins=bins, color="skyblue", ec="skyblue")
        ax[1, 1].set_xlabel('Fraction sleep correct (sensitivity)', fontsize=14, fontname='Arial')
        ax[1, 1].set_xlim((0, 1))
        
        plt.tight_layout()
        out_name = settings.FIGURE_DIR / f"figure_{classifier_name}_{description}_single_threshold_histograms.png"
        plt.savefig(out_name, dpi=300)
        plt.close('all')

def apply_threshold_three_class(rp, wake_threshold, rem_threshold):
    predicted_labels = []
    for probs in rp.class_probabilities:
        if probs[0] >= wake_threshold:
            predicted_labels.append(0) # wake
        else:
            if probs[2] >= rem_threshold:
                predicted_labels.append(2) # rem
            else:
                predicted_labels.append(1) # nrem
    return np.array(predicted_labels)

def make_bland_altman(classifier_name: str, performance_dict: dict, description: str = ""):
    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(10, 10))
    
    wake_threshold = settings.WAKE_THRESHOLD
    rem_threshold = settings.REM_THRESHOLD
    
    for feature_set, raw_performances in performance_dict.items():
        plot_color = get_color(feature_set)
        
        for subject_index, rp in enumerate(raw_performances):
            true_labels = rp.true_labels
            predicted_labels = apply_threshold_three_class(rp, wake_threshold, rem_threshold)
            
            actual_sol = get_sleep_onset_latency(true_labels)
            predicted_sol = get_sleep_onset_latency(predicted_labels)
            sol_diff = actual_sol - predicted_sol
            ax[0, 0].scatter(actual_sol, sol_diff, c=plot_color, alpha=0.6)
            
            actual_waso = get_wake_after_sleep_onset(true_labels)
            predicted_waso = get_wake_after_sleep_onset(predicted_labels)
            waso_diff = actual_waso - predicted_waso
            ax[0, 1].scatter(actual_waso, waso_diff, c=plot_color, alpha=0.6)
            
            actual_tst = get_tst(true_labels)
            predicted_tst = get_tst(predicted_labels)
            tst_diff = actual_tst - predicted_tst
            ax[1, 0].scatter(actual_tst, tst_diff, c=plot_color, alpha=0.6)
            
            actual_se = get_sleep_efficiency(true_labels)
            predicted_se = get_sleep_efficiency(predicted_labels)
            se_diff = actual_se - predicted_se
            ax[1, 1].scatter(actual_se, se_diff, c=plot_color, alpha=0.6)
            
            actual_rem = get_time_in_rem(true_labels)
            predicted_rem = get_time_in_rem(predicted_labels)
            rem_diff = actual_rem - predicted_rem
            ax[2, 0].scatter(actual_rem, rem_diff, c=plot_color, alpha=0.6)
            
            actual_nrem = get_time_in_nrem(true_labels)
            predicted_nrem = get_time_in_nrem(predicted_labels)
            nrem_diff = actual_nrem - predicted_nrem
            
            if subject_index == 0:
                ax[2, 1].scatter(actual_nrem, nrem_diff, c=plot_color, label=get_label(feature_set), alpha=0.6)
            else:
                ax[2, 1].scatter(actual_nrem, nrem_diff, c=plot_color, alpha=0.6)
                
        ax[0, 0].set_xlabel("SOL (min)")
        ax[0, 0].set_ylabel("Difference (min)")
        ax[0, 0].set_title("Sleep Onset Latency")
        
        ax[0, 1].set_xlabel("WASO (min)")
        ax[0, 1].set_ylabel("Difference (min)")
        ax[0, 1].set_title("Wake After Sleep Onset")
        
        ax[1, 0].set_xlabel("TST (min)")
        ax[1, 0].set_ylabel("Difference (min)")
        ax[1, 0].set_title("Total Sleep Time")
        
        ax[1, 1].set_xlabel("Sleep Efficiency")
        ax[1, 1].set_ylabel("Difference")
        ax[1, 1].set_title("Sleep Efficiency")
        
        ax[2, 0].set_xlabel("Time in REM (min)")
        ax[2, 0].set_ylabel("Difference (min)")
        ax[2, 0].set_title("Time in REM")
        
        ax[2, 1].set_xlabel("Time in NREM (min)")
        ax[2, 1].set_ylabel("Difference (min)")
        ax[2, 1].set_title("Time in NREM")
        
    ax[2, 1].legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.tight_layout()
    out_name = settings.FIGURE_DIR / f"figure_{classifier_name}_{description}_bland_altman.png"
    plt.savefig(out_name, dpi=300)
    plt.close('all')

def combine_plots_as_grid(classifiers, number_of_trials, plot_extension):
    combined_filenames = []
    for cls_name in classifiers:
        combined_filenames.append(settings.FIGURE_DIR / f"{cls_name}_{number_of_trials}__{plot_extension}.png")
        
    images = list(map(Image.open, [str(p) for p in combined_filenames]))
    widths, heights = zip(*(i.size for i in images))
    max_width = max(widths)
    max_height = max(heights)
    
    new_image = Image.new('RGB', (2 * max_width, 2 * max_height))
    
    for count, im in enumerate(images):
        x_offset = int((count % 2) * max_width)
        y_offset = int(math.floor(count / 2) * max_height)
        new_image.paste(im, (x_offset, y_offset))
        
    new_image.save(settings.FIGURE_DIR / f"figure_{number_of_trials}{plot_extension}.png")

def plot_tree_shap_summary(model, X_test, feature_names, output_path):
    import shap
    import matplotlib.pyplot as plt
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    plt.figure(figsize=(10, 6))
    
    # Handle list vs array shape
    if isinstance(shap_values, list):
        shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
    elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
        # Multiclass 3D array from some versions of SHAP/XGBoost
        # We can sum across classes or pick class 1
        shap.summary_plot(shap_values[:, :, 1], X_test, feature_names=feature_names, show=False)
    else:
        shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close('all')
