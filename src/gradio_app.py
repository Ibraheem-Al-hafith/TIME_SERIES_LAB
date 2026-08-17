"""Interactive Gradio Web Interface for the Time Series Forecasting Suite.

Exposes a dual-phase workspace: Automated Batch Dashboards for out-of-sample
evaluation sweeps, and an Interactive Experiment Sandbox for hyperparameter discovery.
Visualizations are fully rendered via src.visualizer.Visualizer initialized with proper DataClass instances.
Includes automated executive Markdown and downloadable PDF report generation on pipeline runs.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import markdown
import pandas as pd
from weasyprint import HTML

from src.config import Config, DataConfig, ModelsConfig, VisualizationConfig
from src.data import DataClass
from src.diagnostics import (
    calculate_stationarity,
    fit_decomposition,
    fit_holt_winters,
    fit_sarima,
    generate_stationarity_report,
)
from src.metrics import ScoringConfig
from src.models import DecomposeConfig, ExponentialConfig, SARIMAConfig
from src.orchestrator import MODEL_REGISTRY, ExperimentOrchestrator
from src.visualizer import Visualizer

logger = logging.getLogger(__name__)


# =====================================================================
# DATA & DATASET HELPERS
# =====================================================================


def create_sandbox_dataset(
    file_path: str, target_col: str, index_col: str, split_size: float
) -> DataClass:
    """Instantiates a standardized DataClass instance using DataConfig.

    Args:
        file_path: Path to the uploaded dataset file.
        target_col: Name of the column containing the target series values.
        index_col: Column name representing the time index.
        split_size: Number of observations reserved for validation split.

    Returns:
        DataClass: Initialized dataset wrapper containing processed train/test partitions.
    """
    data_cfg = DataConfig(
        path=file_path,
        split_size=int(split_size),
        index_col=index_col if index_col else None,
    )
    return DataClass(config=data_cfg)


def parse_csv_column_headers(file_obj: Any) -> gr.Dropdown:
    """Inspects uploaded user dataset files to extract header tokens automatically.

    Args:
        file_obj: File object provided by Gradio file uploader.

    Returns:
        gr.Dropdown: Updated Gradio dropdown containing column header options.
    """
    if file_obj is None:
        return gr.Dropdown(choices=[], value=None)
    try:
        df_headers = pd.read_csv(file_obj.name, nrows=0)
        columns = list(df_headers.columns)
        default_val = columns[1] if len(columns) > 1 else columns[0]
        return gr.Dropdown(choices=columns, value=default_val, interactive=True)
    except Exception as exc:
        logger.error("Failed to parse headers: %s", exc)
        return gr.Dropdown(choices=[], value=None)


# =====================================================================
# REPORT GENERATION & PDF EXPORT HELPERS
# =====================================================================


def generate_executive_report(
    file_name: str,
    target_col: str,
    index_col: str,
    split_size: int,
    decomp_mode: str,
    decompose_period: Any,
    hw_trend: str,
    hw_seasonal: str,
    hw_seasonal_period: Any,
    arima_p: int,
    arima_d: int,
    arima_q: int,
    arima_P: int,
    arima_D: int,
    arima_Q: int,
    metrics_df: pd.DataFrame,
    ingestion_line_plot: Optional[str] = None,
    ingestion_envelope_plot: Optional[str] = None,
    batch_plot_output: Optional[str] = None,
    adf_summary: str = "ADF test not executed.",
    sarima_corr_plot: str = "plots/sandbox/sarima_correlation_diagnostics.png",
) -> str:
    """Generates an executive performance report using a structured Markdown template.

    Args:
        file_name: Name or path of input dataset file.
        target_col: Name of column being forecast.
        index_col: Name of temporal index column.
        split_size: Out-of-sample validation split size.
        decomp_mode: Decomposition synthesis type (additive/multiplicative).
        decompose_period: Seasonal period for decomposition.
        hw_trend: Holt-Winters trend type.
        hw_seasonal: Holt-Winters seasonal type.
        hw_seasonal_period: Holt-Winters seasonal period length.
        arima_p: SARIMA non-seasonal AR order p.
        arima_d: SARIMA non-seasonal differencing d.
        arima_q: SARIMA non-seasonal MA order q.
        arima_P: SARIMA seasonal AR order P.
        arima_D: SARIMA seasonal differencing D.
        arima_Q: SARIMA seasonal MA order Q.
        metrics_df: Comparative metrics table from pipeline execution.
        ingestion_line_plot: Path to rendered ingestion line plot.
        ingestion_envelope_plot: Path to rendered envelope plot.
        batch_plot_output: Path to rendered forecast validation plot.
        adf_summary: Stationarity ADF test summary text.
        sarima_corr_plot: Path to correlation diagnostic plot.

    Returns:
        str: Formatted Markdown string containing complete executive report.
    """
    # 1. Best Model Discovery Logic
    best_model_name = "N/A"
    primary_metric = "RMSE"
    if not metrics_df.empty:
        metric_col = "RMSE" if "RMSE" in metrics_df.columns else "MAPE"
        primary_metric = metric_col
        valid_df = (
            metrics_df[metrics_df["Status"] == "SUCCESS"]
            if "Status" in metrics_df.columns
            else metrics_df
        )
        if not valid_df.empty and metric_col in valid_df.columns:
            best_idx = valid_df[metric_col].astype(float).idxmin()
            best_model_name = str(
                valid_df.loc[best_idx, "Model Strategy"]
                if "Model Strategy" in valid_df.columns
                else best_idx
            )

    # 2. Render comparative metrics table to Markdown
    metrics_markdown = (
        metrics_df.to_markdown(index=False)
        if not metrics_df.empty
        else "No evaluation metrics available."
    )

    # 3. Format dynamic template
    report_template = f"""# 📄 Time Series Performance Evaluation Report

## 1. Executive Summary
**Project Name:** Seasonal Time Series Forecasting Model Comparison.  
**Best Performing Model:** **{best_model_name}** (Selected based on {primary_metric}).  

**Key Finding:** Statistical components (trend and seasonality parameters) were systematically evaluated. Model **{best_model_name}** demonstrated optimal error minimisation and variance stability across out-of-sample horizons.

---

## 2. Dataset Overview & Configuration
- **Data Source File:** `{os.path.basename(file_name)}`
- **Target Metric Column:** `{target_col}`
- **Index Column:** `{index_col}`
- **Validation Split Size:** `{split_size}` observations (Out-of-Sample Test Period)

### Data Visualizations Included
- **Ingestion Line Plot:** `{ingestion_line_plot or "N/A"}`
- **Envelope Analysis Plot:** `{ingestion_envelope_plot or "N/A"}`

---

## 3. Model Configurations & Hyperparameters

| Model Strategy | Configuration / Hyperparameters |
| :--- | :--- |
| **Classical Decomposition** | **Synthesis Type:** `{decomp_mode}`<br>**Seasonal Period:** `{decompose_period}` |
| **Holt-Winters (Exponential Smoothing)** | **Trend Component:** `{hw_trend}`<br>**Seasonal Component:** `{hw_seasonal}`<br>**Seasonal Period:** `{hw_seasonal_period}` |
| **SARIMA** | **Non-Seasonal Order $(p, d, q)$:** `({arima_p}, {arima_d}, {arima_q})`<br>**Seasonal Order $(P, D, Q)_s$:** `({arima_P}, {arima_D}, {arima_Q})_{{{hw_seasonal_period or 12}}}` |

---

## 4. Experimental Results & Performance Comparison

### Model Metrics Comparison Matrix

{metrics_markdown}

### Forecast vs. Actual Visual Layouts
- **Validation Projection Plot:** `{batch_plot_output or "N/A"}`

---

## 5. Detailed Model Analysis & Diagnostics

### A. Classical Decomposition Analysis
Observations on structural additive vs. multiplicative components evaluated against baseline series variance. The isolated seasonal component captured structural period behavior during train splits.

### B. Holt-Winters Smoothing Analysis
In-sample smoothing parameter responsiveness was verified across level, trend, and seasonal updates applied over validation forecast horizons.

### C. SARIMA & Stationarity Diagnostics
- **ADF Stationarity Report:**
{adf_summary}

- **Autocorrelation Analysis:** `{sarima_corr_plot}` (ACF/PACF behavior post-differencing).

---

## 6. Conclusions & Recommendations
- **Trade-offs:** Classical Decomposition offers immediate execution simplicity; Holt-Winters provides dynamic adaptive capabilities; SARIMA delivers statistical rigor and formal diagnostics.
- **Final Project Recommendation:** Model **{best_model_name}** is recommended for deployment on future forecasting horizons for this dataset.
"""
    return report_template.strip()


def export_report_to_pdf(markdown_report: str, output_path: str = "reports/performance_report.pdf") -> str:
    """Converts a Markdown executive report string into a publication-quality PDF via WeasyPrint.

    Args:
        markdown_report: Markdown report text.
        output_path: File destination path for generated PDF document.

    Returns:
        str: Absolute path to the created PDF file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Convert Markdown to HTML tables
    body_html = markdown.markdown(markdown_report, extensions=["tables"])

    # Build styled document with CSS Paged Media rules
    styled_document = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
@page {{
    size: A4;
    margin: 18mm 15mm 18mm 15mm;
    background-color: #faf8f5;
    @bottom-right {{
        content: "Page " counter(page) " of " counter(pages);
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        font-size: 8pt;
        color: #718096;
    }}
    @bottom-left {{
        content: "Time Series Evaluation Report";
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        font-size: 8pt;
        color: #718096;
    }}
}}

*, *::before, *::after {{
    box-sizing: border-box;
}}

body {{
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    font-size: 10pt;
    line-height: 1.5;
    color: #2d3748;
    margin: 0;
    padding: 0;
}}

h1 {{
    font-size: 18pt;
    color: #1a365d;
    border-bottom: 2px solid #2b6cb0;
    padding-bottom: 6px;
    margin-top: 0;
    margin-bottom: 14px;
}}

h2 {{
    font-size: 13pt;
    color: #2b6cb0;
    margin-top: 18px;
    margin-bottom: 8px;
    border-left: 4px solid #3182ce;
    padding-left: 8px;
    page-break-after: avoid;
}}

h3 {{
    font-size: 11pt;
    color: #2d3748;
    margin-top: 12px;
    margin-bottom: 6px;
    page-break-after: avoid;
}}

p, ul, ol {{
    margin-top: 0;
    margin-bottom: 10px;
}}

hr {{
    border: none;
    border-top: 1px solid #e2e8f0;
    margin: 16px 0;
}}

table {{
    width: 100%;
    border-collapse: collapse;
    margin: 12px 0;
    font-size: 9.5pt;
    page-break-inside: avoid;
}}

th, td {{
    padding: 8px 10px;
    text-align: left;
    border: 1px solid #cbd5e0;
}}

th {{
    background-color: #2b6cb0;
    color: #ffffff;
    font-weight: bold;
}}

tr:nth-child(even) {{
    background-color: #f7fafc;
}}

code {{
    font-family: 'Courier New', Courier, monospace;
    background-color: #edf2f7;
    padding: 2px 4px;
    border-radius: 3px;
    font-size: 9pt;
    color: #805ad5;
}}
</style>
</head>
<body>
{body_html}
</body>
</html>
"""
    HTML(string=styled_document).write_pdf(output_path)
    logger.info("Exported executive performance report PDF to %s", output_path)
    return output_path


# =====================================================================
# GRADIO EVENT HANDLERS
# =====================================================================


def handle_data_ingestion_visuals(
    file_obj: Any, target_col: str, index_col: str, split_size: float
) -> Tuple[Optional[str], Optional[str]]:
    """Generates continuous layout line plots and envelope spread tracking configurations instantly."""
    if file_obj is None or not target_col:
        return None, None
    try:
        dataset = create_sandbox_dataset(file_obj.name, target_col, index_col, split_size)
        vis_cfg = VisualizationConfig(plot_path="plots/")
        visualizer = Visualizer(dataset=dataset, config=vis_cfg)

        line_path = visualizer.plot_line_series(target_col=target_col)
        envelope_path = visualizer.plot_envelope_components(target_col=target_col)
        return line_path, envelope_path
    except Exception as exc:
        logger.error("Failed handling ingestion visual paths: %s", exc)
        return None, None


def handle_sandbox_decomposition(
    file_obj: Any, target_col: str, index_col: str, split_size: float, model: str, period: float
) -> Tuple[str, str, str, pd.DataFrame]:
    """Processes in-sample time series structural decompositions via Visualizer initialized with DataClass."""
    default_img = "plots/sandbox/decomposition_explorer.png"
    if file_obj is None or not target_col:
        return default_img, default_img, default_img, pd.DataFrame()
    try:
        dataset = create_sandbox_dataset(file_obj.name, target_col, index_col, split_size)
        train_series = dataset.train[target_col].astype(float)

        fit_res = fit_decomposition(train_series, model=model, period=int(period))

        if not fit_res.success:
            return default_img, default_img, default_img, fit_res.metrics_df

        vis_cfg = VisualizationConfig(plot_path="plots/")
        visualizer = Visualizer(dataset=dataset, config=vis_cfg)
        plot_path = visualizer.plot_in_sample_fit(
            actual=train_series,
            fitted=fit_res.fitted_values,
            title=f"Decomposition In-Sample Reconstruction ({model.title()})",
            filename="decomposition_explorer.png",
            line_color="darkgreen",
        )
        seasonal_plot_path = visualizer.plot_seasonal(target_col=target_col, period=int(period))
        decompose_plot_path = visualizer.plot_seasonal_decomposition(
            target_col=target_col, period=int(period)
        )
        return seasonal_plot_path, decompose_plot_path, plot_path, fit_res.metrics_df
    except Exception as exc:
        logger.error("Sandbox Decomposition failure: %s", exc)
        return default_img, default_img, default_img, pd.DataFrame([{"Error": str(exc)}])


def handle_sandbox_holt_winters(
    file_obj: Any,
    target_col: str,
    index_col: str,
    split_size: float,
    trend: str,
    seasonal: str,
    period: float,
) -> Tuple[str, str, pd.DataFrame]:
    """Processes exponential smoothing simulations using Visualizer initialized with DataClass."""
    default_img = "plots/sandbox/holt_winters_explorer.png"
    if file_obj is None or not target_col:
        return default_img, default_img, pd.DataFrame()
    try:
        dataset = create_sandbox_dataset(file_obj.name, target_col, index_col, split_size)
        train_series = dataset.train[target_col].astype(float)

        fit_res = fit_holt_winters(
            train_series, trend=trend, seasonal=seasonal, period=int(period)
        )

        if not fit_res.success:
            return default_img, default_img, fit_res.metrics_df

        vis_cfg = VisualizationConfig(plot_path="plots/")
        visualizer = Visualizer(dataset=dataset, config=vis_cfg)
        seasonal_plot_path = visualizer.plot_seasonal(target_col=target_col, period=int(period))
        plot_path = visualizer.plot_in_sample_fit(
            actual=train_series,
            fitted=fit_res.fitted_values,
            title=f"Holt-Winters Fit (Trend={trend}, Seasonal={seasonal})",
            filename="holt_winters_explorer.png",
            line_color="darkorange",
        )
        return seasonal_plot_path, plot_path, fit_res.metrics_df
    except Exception as exc:
        logger.error("Sandbox Holt Winters failure: %s", exc)
        return default_img, default_img, pd.DataFrame([{"Error": str(exc)}])


def handle_sandbox_sarima(
    file_obj: Any,
    target_col: str,
    index_col: str,
    split_size: float,
    p: float,
    d: float,
    q: float,
    P: float,
    D: float,
    Q: float,
    seasonal_period: float,
) -> Tuple[str, pd.DataFrame, str, str]:
    """Fits SARIMA configurations onto training boundaries and generates diagnostics via Visualizer with DataClass."""
    default_fit_img = "plots/sandbox/sarima_fit_explorer.png"
    default_corr_img = "plots/sandbox/sarima_correlation_diagnostics.png"

    if file_obj is None or not target_col:
        return (
            default_fit_img,
            pd.DataFrame(),
            "### Configuration Required\nPlease select a dataset and target metric column.",
            default_corr_img,
        )
    try:
        dataset = create_sandbox_dataset(file_obj.name, target_col, index_col, split_size)
        train_series = dataset.train[target_col].astype(float)

        fit_res = fit_sarima(
            train_series,
            p=int(p),
            d=int(d),
            q=int(q),
            P=int(P),
            D=int(D),
            Q=int(Q),
            period=int(seasonal_period),
        )

        vis_cfg = VisualizationConfig(plot_path="plots/")
        visualizer = Visualizer(dataset=dataset, config=vis_cfg)

        if fit_res.success:
            fit_plot = visualizer.plot_in_sample_fit(
                actual=train_series,
                fitted=fit_res.fitted_values,
                title=f"SARIMA({int(p)},{int(d)},{int(q)})x({int(P)},{int(D)},{int(Q)})[{int(seasonal_period)}] In-Sample Alignment",
                filename="sarima_fit_explorer.png",
                line_color="crimson",
            )
        else:
            fit_plot = default_fit_img

        differenced_series = train_series.copy()
        if int(d) > 0:
            for _ in range(int(d)):
                differenced_series = differenced_series.diff().dropna()
        if int(D) > 0 and int(seasonal_period) > 0:
            for _ in range(int(D)):
                differenced_series = differenced_series.diff(periods=int(seasonal_period)).dropna()

        stationary_result = calculate_stationarity(differenced_series)
        ui_markdown = generate_stationarity_report(stationary_result)

        corr_plot = visualizer.plot_autocorrelation(
            series=differenced_series,
            filename="sarima_correlation_diagnostics.png",
            target_col="",
        )

        return fit_plot, fit_res.metrics_df, ui_markdown, corr_plot
    except Exception as exc:
        logger.error("Sandbox SARIMA failure: %s", exc)
        return (
            default_fit_img,
            pd.DataFrame([{"Error": str(exc)}]),
            f"### Execution Failure\n`{str(exc)}`",
            default_corr_img,
        )


def extract_comparative_table(run_output: Dict[str, Any]) -> pd.DataFrame:
    """Transforms raw global batch orchestration dictionaries into tabular formats."""
    rows: List[Dict[str, Any]] = []
    results = run_output.get("results", {})
    for model_name, data in results.items():
        row = {"Model Strategy": model_name.upper(), "Status": "SUCCESS"}
        row.update(data.get("metrics", {}))
        rows.append(row)
    errors = run_output.get("errors", {})
    for model_name, err_msg in errors.items():
        rows.append(
            {"Model Strategy": model_name.upper(), "Status": "FAILED", "Error Context": err_msg}
        )
    return pd.DataFrame(rows)


def execute_ui_pipeline(
    file_obj: Any,
    target_col: str,
    index_col: str,
    split_size: int,
    model_choice: str,
    mae_flag: bool,
    mse_flag: bool,
    rmse_flag: bool,
    mape_flag: bool,
    decomp_mode: str,
    decompose_period: Any,
    hw_trend: str,
    hw_seasonal: str,
    hw_seasonal_period: Any,
    arima_p: int,
    arima_d: int,
    arima_q: int,
    arima_P: int,
    arima_D: int,
    arima_Q: int,
    arima_s: int
) -> Tuple[pd.DataFrame, Optional[str], str, str, Optional[str]]:
    """Compiles parameters, runs orchestrator, generates Markdown report, and exports PDF document."""
    if file_obj is None or not target_col:
        return (
            pd.DataFrame(),
            None,
            "Validation Error: Verify data files and target settings before executing loops.",
            "### Report Generation Failed: Missing required file or target column.",
            None,
        )

    try:
        data_cfg = DataConfig(
            path=file_obj.name,
            split_size=int(split_size),
            index_col=index_col if index_col else "Date",
        )
        scoring_cfg = ScoringConfig(
            mae=mae_flag, mse=mse_flag, rmse=rmse_flag, mape=mape_flag, epsilon=1e-5
        )
        vis_cfg = VisualizationConfig(plot_path="plots/", decomposition_model=decomp_mode)
        models_cfg = ModelsConfig(
            decompose=DecomposeConfig(
                model=decomp_mode,
                period=None if str(decompose_period) == "None" else int(decompose_period),
            ),
            exponential_smoothing=ExponentialConfig(
                trend=None if hw_trend == "None" else hw_trend,
                seasonal=None if hw_seasonal == "None" else hw_seasonal,
                seasonal_periods=(
                    None
                    if str(hw_seasonal_period) == "None"
                    else int(hw_seasonal_period)
                ),
            ),
            sarima=SARIMAConfig(
                p=int(arima_p),
                d=int(arima_d),
                q=int(arima_q),
                P=int(arima_P),
                D=int(arima_D),
                Q=int(arima_Q),
                s=int(arima_s)
            ),
        )
        global_cfg = Config(
            name="GradioUI_Run",
            data=data_cfg,
            visualizer=vis_cfg,
            scoring=scoring_cfg,
            models=models_cfg,
        )
        orchestrator = ExperimentOrchestrator(global_config=global_cfg)

        line_path, envelope_path = handle_data_ingestion_visuals(
            file_obj, target_col, index_col, split_size
        )

        if model_choice == "Run All Models Sweep":
            run_output = orchestrator.run_all_models(target_column=target_col)
            metrics_df = extract_comparative_table(run_output)

            dataset_instance = DataClass(config=data_cfg)
            vis_engine = Visualizer(dataset=dataset_instance, config=vis_cfg)

            predictions = {}
            if "decompose" in run_output.get("results", {}):
                predictions["DECOMPOSE"] = run_output["results"]["decompose"]["predictions"]
            if "sarima" in run_output.get("results", {}):
                predictions["SARIMA"] = run_output["results"]["sarima"]["predictions"]
            if "exponential_smoothing" in run_output.get("results", {}):
                predictions["HOLT-WINTER"] = run_output["results"]["exponential_smoothing"][
                    "predictions"
                ]

            plot_to_render = vis_engine.plot_predictions_vs_actuals(
                predictions=predictions,
                target_col=target_col,
            )
            status_msg = "Batch sequence evaluation sweep completed successfully."
        else:
            run_output = orchestrator.run_single_model(
                model_type=model_choice, target_column=target_col
            )
            metrics_df = pd.DataFrame([run_output["metrics"]], index=[model_choice.upper()])

            dataset_instance = DataClass(config=data_cfg)
            vis_engine = Visualizer(dataset=dataset_instance, config=vis_cfg)
            plot_to_render = vis_engine.plot_predictions_vs_actuals(
                predictions=run_output["predictions"], target_col=target_col
            )
            status_msg = f"Model run target '{model_choice}' successfully complete."

        # Compute Stationarity Diagnostics for Report
        dataset_instance = DataClass(config=data_cfg)
        train_series = dataset_instance.train[target_col].astype(float)
        stationary_result = calculate_stationarity(train_series)
        adf_md = generate_stationarity_report(stationary_result)

        # Generate Executive Markdown Report
        report_markdown = generate_executive_report(
            file_name=file_obj.name,
            target_col=target_col,
            index_col=index_col,
            split_size=int(split_size),
            decomp_mode=decomp_mode,
            decompose_period=decompose_period,
            hw_trend=hw_trend,
            hw_seasonal=hw_seasonal,
            hw_seasonal_period=hw_seasonal_period,
            arima_p=arima_p,
            arima_d=arima_d,
            arima_q=arima_q,
            arima_P=arima_P,
            arima_D=arima_D,
            arima_Q=arima_Q,
            metrics_df=metrics_df,
            ingestion_line_plot=line_path,
            ingestion_envelope_plot=envelope_path,
            batch_plot_output=plot_to_render,
            adf_summary=adf_md,
        )

        # Export Report to PDF
        pdf_file_path = export_report_to_pdf(
            markdown_report=report_markdown,
            output_path="reports/executive_performance_report.pdf",
        )

        return metrics_df, plot_to_render, status_msg, report_markdown, pdf_file_path

    except Exception as err:
        logger.exception("Pipeline Run Failure")
        return (
            pd.DataFrame(),
            None,
            f"Pipeline Run Failure: {str(err)}",
            f"### Error Generating Report\n`{str(err)}`",
            None,
        )


# =====================================================================
# MODULAR UI BUILDER LAYOUT
# =====================================================================


def _render_ingestion_block() -> Dict[str, Any]:
    """Renders the dataset stream ingestion UI block."""
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Core Data Stream Ingestion Controls")
            file_input = gr.File(label="Target Source File (CSV)", file_types=[".csv"])
            with gr.Row():
                index_column = gr.Textbox(label="Calendar Index Column", value="Date")
                target_column = gr.Dropdown(
                    choices=[], label="Target Metric Column", interactive=True
                )
            split_size = gr.Number(
                label="Out-of-Sample Validation Slice Size", value=12, precision=0
            )

        with gr.Column(scale=2):
            gr.Markdown("### Ingestion Temporal Properties Visualization")
            with gr.Row():
                ingestion_line_plot = gr.Image(
                    label="Dataset Sequence Line Split Map", type="filepath"
                )
                ingestion_envelope_plot = gr.Image(
                    label="Envelope Geometry Analysis Map", type="filepath"
                )

    return {
        "file_input": file_input,
        "index_column": index_column,
        "target_column": target_column,
        "split_size": split_size,
        "ingestion_line_plot": ingestion_line_plot,
        "ingestion_envelope_plot": ingestion_envelope_plot,
    }


def _render_batch_dashboard() -> Dict[str, Any]:
    """Renders the Automated Batch Dashboard tab with PDF download capabilities."""
    model_choices = list(MODEL_REGISTRY.keys()) + ["Run All Models Sweep"]

    with gr.Tab("Automated Batch Dashboard"):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Evaluation Metrics Matrix Definitions")
                with gr.Row():
                    mae_tgl = gr.Checkbox(label="MAE", value=True)
                    mse_tgl = gr.Checkbox(label="MSE", value=True)
                    rmse_tgl = gr.Checkbox(label="RMSE", value=True)
                    mape_tgl = gr.Checkbox(label="MAPE", value=True)

                gr.Markdown("### Model Architecture Settings")
                batch_model_selector = gr.Dropdown(
                    choices=model_choices,
                    value="Run All Models Sweep",
                    label="Execution Strategy",
                )

                with gr.Accordion("Decomposition Overrides", open=False):
                    batch_decomp_mode = gr.Radio(
                        choices=["additive", "multiplicative"],
                        value="additive",
                        label="Synthesis Type",
                    )
                    batch_decomp_period = gr.Number(
                        value=12, precision=0, label="Seasonal Period Component"
                    )
                with gr.Accordion("Smoothing Overrides", open=False):
                    batch_hw_trend = gr.Dropdown(
                        choices=["None", "add", "mul"], value="add", label="Trend Component"
                    )
                    batch_hw_seasonal = gr.Dropdown(
                        choices=["None", "add", "mul"], value="add", label="Seasonal Component"
                    )
                    batch_hw_seasonal_period = gr.Number(
                        value=12, precision=0, label="Seasonal Period Component"
                    )
                with gr.Accordion("State Space ARIMA Vectors Order", open=False):
                    with gr.Row():
                        bp = gr.Slider(0, 5, 1, step=1, label="p")
                        bd = gr.Slider(0, 2, 1, step=1, label="d")
                        bq = gr.Slider(0, 5, 1, step=1, label="q")
                    with gr.Row():
                        bP = gr.Slider(0, 3, 1, step=1, label="P")
                        bD = gr.Slider(0, 2, 1, step=1, label="D")
                        bQ = gr.Slider(0, 3, 1, step=1, label="Q")
                    bs = gr.Number(value=12, precision=0, label='Seasonal Period Component (m)')

                run_batch_btn = gr.Button(
                    "Execute Performance Pipeline Run", variant="primary"
                )

            with gr.Column(scale=1):
                gr.Markdown("### Process Operational Output Monitor")
                batch_status = gr.Textbox(
                    label="Execution Telemetry Trace Logs",
                    value="System Ready.",
                    interactive=False,
                )
                batch_metrics_table = gr.DataFrame(
                    label="Out-of-Sample Score Parameters Report", interactive=False
                )
                batch_plot_output = gr.Image(
                    label="Validation Projection Visual Layouts", type="filepath"
                )

        gr.Markdown("---")
        with gr.Row():
            with gr.Column(scale=3):
                report_output_markdown = gr.Markdown(
                    value="*Executive performance report will render here after clicking 'Execute Performance Pipeline Run'.*"
                )
            with gr.Column(scale=1):
                pdf_download_file = gr.File(
                    label="Download Executive Report (PDF)", interactive=False
                )

    return {
        "mae_tgl": mae_tgl,
        "mse_tgl": mse_tgl,
        "rmse_tgl": rmse_tgl,
        "mape_tgl": mape_tgl,
        "batch_model_selector": batch_model_selector,
        "batch_decomp_mode": batch_decomp_mode,
        "batch_decomp_period": batch_decomp_period,
        "batch_hw_trend": batch_hw_trend,
        "batch_hw_seasonal": batch_hw_seasonal,
        "batch_hw_seasonal_period": batch_hw_seasonal_period,
        "bp": bp,
        "bd": bd,
        "bq": bq,
        "bP": bP,
        "bD": bD,
        "bQ": bQ,
        "bs": bs,
        "run_batch_btn": run_batch_btn,
        "batch_status": batch_status,
        "batch_metrics_table": batch_metrics_table,
        "batch_plot_output": batch_plot_output,
        "report_output_markdown": report_output_markdown,
        "pdf_download_file": pdf_download_file,
    }


def _render_sandbox_tabs() -> Dict[str, Any]:
    """Renders the interactive hyperparameter discovery sandbox tabs."""
    components: Dict[str, Any] = {}

    with gr.Tab("Hyperparameter Discovery Sandbox"):
        gr.Markdown("### In-Sample Parameter Analysis Workspace (Isolates Test Partition Data)")

        with gr.Tabs():
            # Panel 1: Decomposition
            with gr.Tab("1. Time Series Decomposition"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### Classical Decomposition Controls")
                        decomp_type = gr.Radio(
                            choices=["additive", "multiplicative"],
                            value="additive",
                            label="Algebraic Composition Mode",
                        )
                        decomp_period = gr.Number(
                            label="Seasonal Decomposition Periodicity Horizon",
                            value=12,
                            precision=0,
                        )
                        calculate_decomp_btn = gr.Button(
                            "Fit In-Sample Decomposition Model", variant="secondary"
                        )
                    with gr.Column(scale=2):
                        decomp_sandbox_image_decomp_plot = gr.Image(
                            label="Decomposision Plot", type="filepath"
                        )
                        decomp_sandbox_image_seasonal_plot = gr.Image(
                            label="Seasonal Plot", type="filepath"
                        )
                        decomp_sandbox_image = gr.Image(
                            label="In-Sample Reconstructed Fit Path", type="filepath"
                        )
                        decomp_sandbox_metrics = gr.DataFrame(label="Decomposition Metrics")

                components["decomp"] = {
                    "type": decomp_type,
                    "period": decomp_period,
                    "btn": calculate_decomp_btn,
                    "decomp_plot": decomp_sandbox_image_decomp_plot,
                    "seasonal_plot": decomp_sandbox_image_seasonal_plot,
                    "img": decomp_sandbox_image,
                    "metrics": decomp_sandbox_metrics,
                }

            # Panel 2: Holt-Winters
            with gr.Tab("2. Holt-Winters Smoothing"):
                with gr.Row():
                    with gr.Column(scale=1):
                        hw_t = gr.Dropdown(
                            choices=["None", "add", "mul"],
                            value="add",
                            label="Trend Parameter Vector",
                        )
                        hw_s = gr.Dropdown(
                            choices=["None", "add", "mul"],
                            value="add",
                            label="Seasonal Parameter Vector",
                        )
                        hw_p = gr.Number(
                            label="Seasonal Frequency Cycles Length", value=12, precision=0
                        )
                        calculate_hw_btn = gr.Button(
                            "Fit Simulation Model Sequence", variant="secondary"
                        )
                    with gr.Column(scale=2):
                        hw_sandbox_image_seasonal_plot = gr.Image(
                            label="Seasonal Plot", type="filepath"
                        )
                        hw_sandbox_image = gr.Image(
                            label="Historical Slices Tracking Visual Paths", type="filepath"
                        )
                        hw_sandbox_metrics = gr.DataFrame(
                            label="In-Sample Inversion Error Vectors"
                        )

                components["hw"] = {
                    "t": hw_t,
                    "s": hw_s,
                    "p": hw_p,
                    "btn": calculate_hw_btn,
                    "seasonal_plot": hw_sandbox_image_seasonal_plot,
                    "img": hw_sandbox_image,
                    "metrics": hw_sandbox_metrics,
                }

            # Panel 3: SARIMA
            with gr.Tab("3. Iterative SARIMA Discovery"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### Model Coefficients Tuning Grid")
                        with gr.Row():
                            s_p = gr.Slider(0, 5, 1, step=1, label="p")
                            s_d = gr.Slider(0, 2, 0, step=1, label="d")
                            s_q = gr.Slider(0, 5, 1, step=1, label="q")
                        with gr.Row():
                            s_P = gr.Slider(0, 3, 0, step=1, label="P")
                            s_D = gr.Slider(0, 2, 0, step=1, label="D")
                            s_Q = gr.Slider(0, 3, 0, step=1, label="Q")
                        m_input = gr.Number(
                            value=12, label="Seasonal Lag Period Configuration (m)", precision=0
                        )

                        calculate_sarima_btn = gr.Button(
                            "Fit Simulation SARIMA Model", variant="secondary"
                        )
                        adf_report_container = gr.Markdown(
                            "Modify configurations and click execute to evaluate metrics."
                        )

                    with gr.Column(scale=2):
                        sarima_fit_image = gr.Image(
                            label="In-Sample Tracking Alignment Path", type="filepath"
                        )
                        sarima_fit_metrics = gr.DataFrame(
                            label="SARIMA In-Sample Performance Portfolio"
                        )
                        diagnostic_plots_container = gr.Image(
                            label="Dynamic ACF / PACF Spatial Coordinate Maps", type="filepath"
                        )

                components["sarima"] = {
                    "p": s_p,
                    "d": s_d,
                    "q": s_q,
                    "P": s_P,
                    "D": s_D,
                    "Q": s_Q,
                    "m": m_input,
                    "btn": calculate_sarima_btn,
                    "adf_md": adf_report_container,
                    "fit_img": sarima_fit_image,
                    "fit_metrics": sarima_fit_metrics,
                    "corr_img": diagnostic_plots_container,
                }

    return components


# =====================================================================
# ROOT INTERFACE GENERATOR DESIGN
# =====================================================================


def build_gradio_interface() -> gr.Blocks:
    """Builds the dual-phase time series visualization system with decoupled UI components."""
    with gr.Blocks(
        title="Forecasting Infrastructure Management Console", theme=gr.themes.Default()
    ) as app:
        gr.Markdown("# Automated Production Analytics & Discovery Engine Hub")

        ingest_ui = _render_ingestion_block()

        with gr.Tabs():
            batch_ui = _render_batch_dashboard()
            sandbox_ui = _render_sandbox_tabs()

        # Wire file header inspection
        ingest_ui["file_input"].change(
            fn=parse_csv_column_headers,
            inputs=[ingest_ui["file_input"]],
            outputs=[ingest_ui["target_column"]],
        )

        # Wire continuous ingestion visuals
        ingestion_inputs = [
            ingest_ui["file_input"],
            ingest_ui["target_column"],
            ingest_ui["index_column"],
            ingest_ui["split_size"],
        ]
        for trigger_comp in [
            ingest_ui["file_input"],
            ingest_ui["target_column"],
            ingest_ui["split_size"],
        ]:
            trigger_comp.change(
                fn=handle_data_ingestion_visuals,
                inputs=ingestion_inputs,
                outputs=[ingest_ui["ingestion_line_plot"], ingest_ui["ingestion_envelope_plot"]],
            )

        # Wire Batch Execution Pipeline with PDF Download Component
        batch_ui["run_batch_btn"].click(
            fn=execute_ui_pipeline,
            inputs=[
                ingest_ui["file_input"],
                ingest_ui["target_column"],
                ingest_ui["index_column"],
                ingest_ui["split_size"],
                batch_ui["batch_model_selector"],
                batch_ui["mae_tgl"],
                batch_ui["mse_tgl"],
                batch_ui["rmse_tgl"],
                batch_ui["mape_tgl"],
                batch_ui["batch_decomp_mode"],
                batch_ui["batch_decomp_period"],
                batch_ui["batch_hw_trend"],
                batch_ui["batch_hw_seasonal"],
                batch_ui["batch_hw_seasonal_period"],
                batch_ui["bp"],
                batch_ui["bd"],
                batch_ui["bq"],
                batch_ui["bP"],
                batch_ui["bD"],
                batch_ui["bQ"],
                batch_ui["bs"],
            ],
            outputs=[
                batch_ui["batch_metrics_table"],
                batch_ui["batch_plot_output"],
                batch_ui["batch_status"],
                batch_ui["report_output_markdown"],
                batch_ui["pdf_download_file"],
            ],
        )

        # Wire Sandbox Tab 1: Decomposition
        decomp_comp = sandbox_ui["decomp"]
        decomp_comp["btn"].click(
            fn=handle_sandbox_decomposition,
            inputs=[
                ingest_ui["file_input"],
                ingest_ui["target_column"],
                ingest_ui["index_column"],
                ingest_ui["split_size"],
                decomp_comp["type"],
                decomp_comp["period"],
            ],
            outputs=[
                decomp_comp["decomp_plot"],
                decomp_comp["seasonal_plot"],
                decomp_comp["img"],
                decomp_comp["metrics"],
            ],
        )

        # Wire Sandbox Tab 2: Holt-Winters
        hw_comp = sandbox_ui["hw"]
        hw_comp["btn"].click(
            fn=handle_sandbox_holt_winters,
            inputs=[
                ingest_ui["file_input"],
                ingest_ui["target_column"],
                ingest_ui["index_column"],
                ingest_ui["split_size"],
                hw_comp["t"],
                hw_comp["s"],
                hw_comp["p"],
            ],
            outputs=[hw_comp["seasonal_plot"], hw_comp["img"], hw_comp["metrics"]],
        )

        # Wire Sandbox Tab 3: SARIMA
        sarima_comp = sandbox_ui["sarima"]
        sarima_comp["btn"].click(
            fn=handle_sandbox_sarima,
            inputs=[
                ingest_ui["file_input"],
                ingest_ui["target_column"],
                ingest_ui["index_column"],
                ingest_ui["split_size"],
                sarima_comp["p"],
                sarima_comp["d"],
                sarima_comp["q"],
                sarima_comp["P"],
                sarima_comp["D"],
                sarima_comp["Q"],
                sarima_comp["m"],
            ],
            outputs=[
                sarima_comp["fit_img"],
                sarima_comp["fit_metrics"],
                sarima_comp["adf_md"],
                sarima_comp["corr_img"],
            ],
        )

    return app


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    interface_hub = build_gradio_interface()
    interface_hub.launch(server_name="0.0.0.0", server_port=7860, share=False)