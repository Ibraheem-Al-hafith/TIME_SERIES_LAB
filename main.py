"""Application Entry Point for Time Series Analytics Infrastructure.

Provides dual operational execution models using mandatory CLI subcommands:
1. `cli`: Headless Batch Execution Mode using a YAML configuration file.
   Default configuration path: configs/config.py
2. `ui`: Interactive Web Mode booting a Gradio-based interface for dynamic
   visual exploration.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from src.app import (
    build_gradio_interface,
    export_report_to_pdf,
    extract_comparative_table,
    generate_executive_report,
)
from src.config import Config, get_config_from_yaml
from src.data import DataClass
from src.diagnostics import calculate_stationarity, generate_stationarity_report, CombinedStationarityResult
from src.logger import setup_logging
from src.orchestrator import ExperimentOrchestrator
from src.visualizer import Visualizer

logger: logging.Logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH: Path = Path("configs/config.py")


def run_headless_batch_pipeline(config_path: Path) -> None:
    """Executes the automated batch analytics, modeling, and reporting pipeline.

    Args:
        config_path: Path to the YAML configuration file.

    Raises:
        FileNotFoundError: If the configuration file does not exist at `config_path`.
        RuntimeError: If data ingestion or modeling pipeline execution fails.
    """
    if not config_path.exists():
        logger.error("Configuration file not found: %s", config_path)
        raise FileNotFoundError(f"Configuration file missing: {config_path}")

    logger.info("Loading configuration parameters from: %s", config_path)
    config: Optional[Config] = get_config_from_yaml(str(config_path))
    if config is None:
        logger.error("Failed to parse configuration file. Halting batch run.")
        raise RuntimeError(f"Invalid YAML configuration at {config_path}")

    # 1. Data Ingestion
    dataset_path: Path = Path(config.data.path)
    logger.info("Ingesting target dataset from: %s", dataset_path)
    dataset: DataClass = DataClass(config=config.data)

    target_col: str = config.data.target or str(dataset.data.columns[0])
    logger.info("Executing pipeline targeting column: '%s'", target_col)

    # 2. Visualizer Instantiation & Initial Graphics Engine Run
    plot_dir: Path = Path(config.visualizer.plot_path)
    plot_dir.mkdir(parents=True, exist_ok=True)

    vis_engine: Visualizer = Visualizer(dataset=dataset, config=config.visualizer)
    line_plot_path: str = vis_engine.plot_line_series(target_col=target_col)
    envelope_plot_path: str = vis_engine.plot_envelope_components(
        target_col=target_col
    )
    vis_engine.plot_seasonal(target_col=target_col, period=config.models.decompose.period)
    vis_engine.plot_seasonal_decomposition(target_col=target_col, period=config.models.decompose.period)




    # 3. Time Series Diagnostics (Stationarity & ADF Audit)
    logger.info("Computing stationarity diagnostics...")
    # Compute diagnostic stationarity check
    diff_series = dataset.train[target_col].dropna().copy()
    arima_d = config.models.sarima.d
    arima_D = config.models.sarima.D
    arima_s = config.models.sarima.s
    if int(arima_d) > 0:
        for _ in range(int(arima_d)):
            diff_series = diff_series.diff().dropna()
    if int(arima_D) > 0 and arima_s is not None and int(arima_s) > 0:
        for _ in range(int(arima_D)):
            diff_series = diff_series.diff(periods=int(arima_s)).dropna()
    stationarity_results = calculate_stationarity(diff_series)
    adf_summary: str = generate_stationarity_report(stationarity_results)

    # 4. Model Orchestration & Benchmark Calculations
    logger.info("Initiating model training and evaluation suite...")
    orchestrator: ExperimentOrchestrator = ExperimentOrchestrator(
        global_config=config
    )
    run_output: Dict[str, Any] = orchestrator.run_all_models(
        target_column=target_col
    )
    metrics_df = extract_comparative_table(run_output)

    # 5. Model Prediction Plot Generation
    raw_results: Dict[str, Any] = run_output.get("results", {})
    predictions: Dict[str, Any] = {
        model_name.upper(): model_data["predictions"]
        for model_name, model_data in raw_results.items()
        if "predictions" in model_data
    }

    batch_plot_path: str = vis_engine.plot_predictions_vs_actuals(
        predictions=predictions, target_col=target_col
    )

    # 6. Executive Report Synthesis & Markdown Generation
    logger.info("Synthesizing executive evaluation report...")
    report_md: str = generate_executive_report(
        file_name=str(dataset_path),
        target_col=target_col,
        index_col=config.data.index_col or "None",
        split_size=config.data.split_size,
        decomp_mode=config.models.decompose.model,
        decompose_period=config.models.decompose.period,
        hw_trend=str(config.models.exponential_smoothing.trend),
        hw_seasonal=str(config.models.exponential_smoothing.seasonal),
        hw_seasonal_period=config.models.exponential_smoothing.seasonal_periods,
        arima_p=config.models.sarima.p,
        arima_d=config.models.sarima.d,
        arima_q=config.models.sarima.q,
        arima_P=config.models.sarima.P,
        arima_D=config.models.sarima.D,
        arima_Q=config.models.sarima.Q,
        metrics_df=metrics_df,
        ingestion_line_plot=line_plot_path,
        ingestion_envelope_plot=envelope_plot_path,
        batch_plot_output=batch_plot_path,
        adf_summary=adf_summary,
    )

    # 7. Writing Artifacts to Disk
    report_dir: Path = Path("reports")
    report_dir.mkdir(parents=True, exist_ok=True)

    markdown_file_path: Path = report_dir / "performance_report.md"
    markdown_file_path.write_text(report_md, encoding="utf-8")
    logger.info("Markdown report saved successfully to: %s", markdown_file_path)

    # 8. Render PDF Export with Graceful Exception Isolation
    pdf_file_path: Path = report_dir / "performance_report.pdf"
    try:
        exported_pdf: Optional[str] = export_report_to_pdf(
            markdown_report=report_md, output_path=str(pdf_file_path)
        )
        if exported_pdf:
            logger.info("PDF report saved successfully to: %s", exported_pdf)
        else:
            logger.warning(
                "PDF exporter returned an empty path. Output limited to Markdown."
            )
    except Exception as err:
        logger.warning(
            "Skipping PDF compilation due to missing rendering libraries or runtime error: %s",
            err,
        )


def launch_gui() -> None:
    """Launches the interactive Gradio web application UI."""
    logger.info("Launching Interactive Web GUI...")
    try:
        app_interface = build_gradio_interface()
        app_interface.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
        )
    except Exception as err:
        logger.critical(
            "Failed to launch Gradio application interface: %s",
            err,
            exc_info=True,
        )
        sys.exit(1)


def build_parser() -> argparse.ArgumentParser:
    """Constructs the command-line argument parser using subcommands.

    Returns:
        argparse.ArgumentParser: Configured argument parser instance.
    """
    parser = argparse.ArgumentParser(
        description="Time Series Analytics & Modeling Infrastructure Entrypoint"
    )

    subparsers = parser.add_subparsers(
        dest="mode",
        required=True,
        help="Target execution mode: 'cli' for headless batch execution or 'ui' for Web GUI.",
    )

    # Subparser for CLI mode
    cli_parser = subparsers.add_parser(
        "cli",
        help="Run non-interactive headless batch processing pipeline.",
    )
    cli_parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to YAML configuration file (default: {DEFAULT_CONFIG_PATH}).",
    )

    # Subparser for UI mode
    subparsers.add_parser(
        "ui",
        help="Run interactive Gradio Web UI interface.",
    )

    return parser


def main() -> None:
    """Parses command-line inputs and dispatches application execution mode."""
    setup_logging()

    parser = build_parser()
    args = parser.parse_args()

    if args.mode == "cli":
        logger.info(
            "CLI mode selected. Configuration path: %s", args.config
        )
        try:
            run_headless_batch_pipeline(config_path=args.config)
            logger.info("Headless batch execution completed successfully.")
        except Exception as err:
            logger.critical(
                "Fatal error encountered during batch execution: %s",
                err,
                exc_info=True,
            )
            sys.exit(1)

    elif args.mode == "ui":
        launch_gui()


if __name__ == "__main__":
    main()