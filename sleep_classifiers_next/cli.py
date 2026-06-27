import os
import gc
import typer
from pathlib import Path
import numpy as np
import polars as pl
from rich.console import Console
from rich.table import Table

from sleep_classifiers_next.config import Settings
from sleep_classifiers_next.etl.feature_pipeline import FeaturePipeline
from sleep_classifiers_next.evaluation import CrossValidationService, MetricsCalculator, FigurePlotter
from sleep_classifiers_next.models import (
    RandomForestSleepClassifier,
    LogisticRegressionSleepClassifier,
    KNNSleepClassifier,
    MLPSleepClassifier,
    LightGBMSleepClassifier,
)

app = typer.Typer(help="sleep_classifiers_next CLI for data preprocessing and classifier benchmarking.")
console = Console()

def get_all_subject_ids(data_dir: Path) -> list[str]:
    labels_dir = data_dir / "labels"
    if not labels_dir.exists():
        return []
    return sorted([
        f.name.split("_")[0] 
        for f in labels_dir.glob("*_labeled_sleep.txt")
    ])

@app.command()
def preprocess(
    data_dir: str = "c:/Users/alpha/Desktop/sleep_classifiers_next/data",
    output_dir: str = "c:/Users/alpha/Desktop/sleep_classifiers_next/outputs",
    legacy_compatibility: bool = False,
    verbose: bool = True
):
    """
    Runs raw data loading, cropping, counts extraction, circadian clock simulation,
    and writes aligned features to compressed Parquet files.
    """
    settings = Settings()
    settings.paths.data_dir = Path(data_dir)
    settings.paths.output_dir = Path(output_dir)
    settings.features.legacy_compatibility = legacy_compatibility
    settings.verbose = verbose

    subject_ids = get_all_subject_ids(settings.paths.data_dir)
    if not subject_ids:
        console.print(f"[bold red]Error: No subjects found in labels folder at {settings.paths.data_dir / 'labels'}[/bold red]")
        raise typer.Exit(1)

    console.print(f"[bold green]Starting preprocessing for {len(subject_ids)} subjects...[/bold green]")
    pipeline = FeaturePipeline(settings)
    
    for idx, sid in enumerate(subject_ids):
        console.print(f"({idx+1}/{len(subject_ids)}) Processing subject {sid}...")
        try:
            pipeline.process_subject(sid)
        except Exception as e:
            console.print(f"[bold red]Failed processing subject {sid}: {e}[/bold red]")
    
    console.print("[bold green]Preprocessing completed successfully![/bold green]")

@app.command()
def benchmark(
    data_dir: str = "c:/Users/alpha/Desktop/sleep_classifiers_next/data",
    output_dir: str = "c:/Users/alpha/Desktop/sleep_classifiers_next/outputs",
    legacy_compatibility: bool = False
):
    """
    Runs cross-validation evaluations across all classifiers and features,
    and prints formatted benchmark tables matching the research paper metrics.
    """
    settings = Settings()
    settings.paths.data_dir = Path(data_dir)
    settings.paths.output_dir = Path(output_dir)
    settings.features.legacy_compatibility = legacy_compatibility
    settings.verbose = False  # Keep progress clean

    features_dir = settings.paths.features_dir
    existing_parquets = list(features_dir.glob("*_features.parquet"))
    if len(existing_parquets) == 0:
        console.print("[bold yellow]No precomputed features found. Running preprocessing first...[/bold yellow]")
        preprocess(data_dir, output_dir, legacy_compatibility, verbose=False)
        existing_parquets = list(features_dir.glob("*_features.parquet"))

    subject_ids = sorted([
        f.name.split("_")[0] 
        for f in existing_parquets
    ])
    
    if not subject_ids:
        console.print("[bold red]Error: No preprocessed subject datasets available for benchmarking.[/bold red]")
        raise typer.Exit(1)

    cv_service = CrossValidationService(settings)
    splits = cv_service.get_loso_splits(subject_ids)

    # 1. Evaluate Sleep/Wake (Binary) performance for classifiers
    # We will evaluate Logistic Regression, KNN, Random Forest, MLP, and LightGBM
    classifiers = [
        LogisticRegressionSleepClassifier(),
        KNNSleepClassifier(),
        RandomForestSleepClassifier(),
        MLPSleepClassifier(max_iter=1000),
        LightGBMSleepClassifier(),
    ]

    feature_sets = {
        "Motion": ["motion_count"],
        "HR": ["heart_rate_std"],
        "Motion, HR": ["motion_count", "heart_rate_std"],
        "Motion, HR, Clock": ["motion_count", "heart_rate_std", "cosine_proxy"],
        "Motion, HR, Circadian": ["motion_count", "heart_rate_std", "circadian_proxy"]
    }

    tprs = [0.8, 0.9, 0.93, 0.95]

    for clf in classifiers:
        console.print(f"\n[bold blue]=== Evaluating {clf.name} Sleep/Wake Performance ===[/bold blue]")
        
        table = Table(title=f"Sleep/Wake Performance by {clf.name} (LOSO CV)")
        table.add_column("Features", justify="left", style="cyan")
        table.add_column("Sleep Sens (TPR)", justify="center")
        table.add_column("Accuracy", justify="center")
        table.add_column("Wake Spec (TNR)", justify="center")
        table.add_column("Kappa (κ)", justify="center")
        table.add_column("AUC", justify="center")

        for f_name, f_cols in feature_sets.items():
            try:
                results = cv_service.run_cross_validation(clf, splits, f_cols, is_three_class=False)
                
                # Combine results across all folds
                all_true = np.concatenate([r["true_labels"] for r in results])
                all_probs = np.concatenate([r["probabilities"] for r in results])

                # Calculate AUC
                from sklearn.metrics import roc_curve, auc
                fpr_curve, tpr_curve, thresholds = roc_curve(all_true, all_probs[:, 1], pos_label=1, drop_intermediate=False)
                auc_val = auc(fpr_curve, tpr_curve)

                # For each target TPR, interpolate threshold and calculate metrics
                for target_tpr in tprs:
                    threshold = np.interp(target_tpr, tpr_curve, thresholds)
                    perf = MetricsCalculator.calculate_sleep_wake(all_true, all_probs, sleep_threshold=threshold)

                    table.add_row(
                        f_name,
                        f"{target_tpr:.2f}",
                        f"{perf.accuracy:.3f}",
                        f"{perf.wake_correct:.3f}",
                        f"{perf.kappa:.3f}",
                        f"{auc_val:.3f}"
                    )
            except Exception as e:
                table.add_row(f_name, "N/A", "N/A", "N/A", "N/A", f"Error: {e}")

        console.print(table)

    # 2. Evaluate 3-Class (Wake/NREM/REM) sleep stage performance
    console.print("\n[bold blue]=== Evaluating Three-Class (Wake/NREM/REM) Performance ===[/bold blue]")
    
    table_3class = Table(title="Three-Class Sleep Stage Accuracy (LOSO CV)")
    table_3class.add_column("Classifier", justify="left", style="cyan")
    table_3class.add_column("Features", justify="left")
    table_3class.add_column("Wake Recall", justify="center")
    table_3class.add_column("NREM Recall", justify="center")
    table_3class.add_column("REM Recall", justify="center")
    table_3class.add_column("Accuracy", justify="center")
    table_3class.add_column("Kappa (κ)", justify="center")

    for clf in classifiers:
        for f_name, f_cols in feature_sets.items():
            try:
                results = cv_service.run_cross_validation(clf, splits, f_cols, is_three_class=True, scoring="neg_log_loss")
                
                all_true = np.concatenate([r["true_labels"] for r in results])
                all_probs = np.concatenate([r["probabilities"] for r in results])

                perf = MetricsCalculator.calculate_three_class(
                    all_true, all_probs, 
                    wake_threshold=settings.models.wake_threshold, 
                    rem_threshold=settings.models.rem_threshold
                )

                table_3class.add_row(
                    clf.name,
                    f_name,
                    f"{perf.wake_correct:.3f}",
                    f"{perf.nrem_correct:.3f}",
                    f"{perf.rem_correct:.3f}",
                    f"{perf.accuracy:.3f}",
                    f"{perf.kappa:.3f}"
                )
            except Exception as e:
                table_3class.add_row(clf.name, f_name, "N/A", "N/A", "N/A", "N/A", f"Error: {e}")

    console.print(table_3class)

@app.command()
def plot(
    fig: str = typer.Option(..., "--fig", help="Figure number to plot (1, 2, 3, 4, 5, 6-7, 8-9)"),
    subject_id: str = "5383425",
    data_dir: str = "c:/Users/alpha/Desktop/sleep_classifiers_next/data",
    output_dir: str = "c:/Users/alpha/Desktop/sleep_classifiers_next/outputs"
):
    """
    Generates paper-aligned figures and saves them in outputs/figures/.
    """
    settings = Settings()
    settings.paths.data_dir = Path(data_dir)
    settings.paths.output_dir = Path(output_dir)
    settings.verbose = False
    
    plotter = FigurePlotter(settings)
    
    if fig == "1":
        path = plotter.plot_figure_1(subject_id)
        console.print(f"[bold green]Figure 1 generated and saved to {path}[/bold green]")
    elif fig in ("2", "3"):
        fig2, fig3 = plotter.plot_figure_2_3()
        console.print(f"[bold green]Figure 2 and 3 generated and saved to {fig2} and {fig3}[/bold green]")
    elif fig == "4":
        path = plotter.plot_figure_4()
        console.print(f"[bold green]Figure 4 generated and saved to {path}[/bold green]")
    elif fig == "5":
        path = plotter.plot_figure_5()
        console.print(f"[bold green]Figure 5 generated and saved to {path}[/bold green]")
    elif fig == "6-7":
        path = plotter.plot_figure_6_7()
        console.print(f"[bold green]Figure 6 & 7 generated and saved to {path}[/bold green]")
    elif fig == "8-9":
        path = plotter.plot_figure_8_9()
        console.print(f"[bold green]Figure 8 & 9 generated and saved to {path}[/bold green]")
    else:
        console.print("[bold red]Error: Invalid figure number. Choose from 1, 2, 3, 4, 5, 6-7, 8-9[/bold red]")

if __name__ == "__main__":
    app()
