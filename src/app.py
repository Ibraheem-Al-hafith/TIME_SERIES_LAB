"""Interactive Gradio Web Interface for the Time Series Forecasting Suite.

Acts as a decoupled Presentation Layer that delegates business logic, pipeline orchestration,
model evaluation, and visualization strictly to underlying core modules:
- Data Ingestion & Splitting: `src.data.DataClass`
- Execution Orchestration: `src.orchestrator.ExperimentOrchestrator`
- Statistical Diagnostics: `src.diagnostics`
- Forecasting Models: `src.models`
- Graphics Engine: `src.visualizer.Visualizer`
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import gradio.themes
import markdown
import pandas as pd

from .config import (
    Config,
    DataConfig,
    DecomposeConfig,
    ExponentialConfig,
    ModelsConfig,
    SARIMAConfig,
    ScoringConfig,
    VisualizationConfig,
)
from .data import DataClass
from .diagnostics import (
    calculate_stationarity,
    fit_decomposition,
    generate_stationarity_report,
)
from .models import ExponentialSmoothingModel, SARIMAModel
from .orchestrator import MODEL_REGISTRY, ExperimentOrchestrator
from .visualizer import Visualizer
from .logger import setup_logging

logger = logging.getLogger(__name__)


# =====================================================================
# DATA FACTORY & INGESTION HELPERS
# =====================================================================


def build_dataset_instance(
    file_path: str, target_col: Optional[str], index_col: Optional[str], split_size: int
) -> DataClass:
    """Factory helper to build standardized DataClass instances across handlers.

    Args:
        file_path: File system path to input dataset (.csv, .xlsx, or .parquet).
        target_col: Name of column containing target series values.
        index_col: Column name representing datetime index.
        split_size: Number of row observations reserved for train/test split boundary.

    Returns:
        DataClass: Instantiated dataset container with loaded train/test splits.
    """
    data_cfg = DataConfig(
        path=file_path,
        split_size=int(split_size),
        index_col=index_col if index_col else None,
        target=target_col,
    )
    return DataClass(config=data_cfg)


def parse_csv_column_headers(file_obj: Any) -> Tuple[gr.Dropdown, gr.Dropdown]:
    """Inspects uploaded user dataset files to extract header column tokens.

    Args:
        file_obj: Gradio UploadedFile component instance.

    Returns:
        Tuple[gr.Dropdown, gr.Dropdown]: Dropdown updates for target_col and index_col.
    """
    if file_obj is None or not hasattr(file_obj, "name"):
        return gr.Dropdown(choices=[], value=None), gr.Dropdown(choices=["None"], value="None")
    try:
        df_headers = pd.read_csv(file_obj.name, nrows=0)
        columns = list(df_headers.columns)
        
        # Target column default choice (second column or first if only one)
        default_target = columns[1] if len(columns) > 1 else (columns[0] if columns else None)
        
        # Index column choices (include "None" option)
        index_choices = ["None"] + columns
        default_index = columns[0] if columns else "None"

        return (
            gr.Dropdown(choices=columns, value=default_target, interactive=True),
            gr.Dropdown(choices=index_choices, value=default_index, interactive=True),
        )
    except Exception as exc:
        logger.error("Failed to parse column headers from uploaded file: %s", exc)
        return gr.Dropdown(choices=[], value=None), gr.Dropdown(choices=["None"], value="None")


# =====================================================================
# VISUALIZATION ADAPTERS
# =====================================================================


def handle_data_ingestion_visuals(
    file_obj: Any, target_col: str, index_col: Optional[str], split_size: int, seasonal_period
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Adapter to construct DataClass and render ingestion overview and envelope plots.

    Args:
        file_obj: Uploaded file object wrapper from Gradio.
        target_col: Selected target column name.
        index_col: Selected calendar index column name.
        split_size: Split size parameter value.
        seasonal_period: period of the seasonal period

    Returns:
        Tuple[Optional[str], Optional[str]]: Paths to generated line plot and envelope plot.
    """
    if file_obj is None or not target_col:
        return None, None, None
    if index_col == "None":
        index_col = None
    try:
        dataset = build_dataset_instance(file_obj.name, target_col, index_col, int(split_size))
        vis_cfg = VisualizationConfig(plot_path="plots/")
        visualizer = Visualizer(dataset=dataset, config=vis_cfg)

        line_path = visualizer.plot_line_series(target_col=target_col)
        envelope_path = visualizer.plot_envelope_components(target_col=target_col)
        seasonal_path = visualizer.plot_seasonal(target_col=target_col, period=seasonal_period)
        return line_path, envelope_path, seasonal_path
    except Exception as exc:
        logger.error("Error generating ingestion visuals: %s", exc)
        return None, None, None


# =====================================================================
# SANDBOX DISCOVERY ADAPTERS
# =====================================================================


def handle_sandbox_decomposition(
    file_obj: Any,
    target_col: str,
    index_col: str,
    split_size: int,
    model: str,
    period: float,
) -> Tuple[Optional[str], Optional[str], Optional[str], pd.DataFrame]:
    """Handles classical decomposition simulation and diagnostic plot generation.

    Args:
        file_obj: Source dataset file object.
        target_col: Target variable column name.
        index_col: Calendar index column name.
        split_size: Validation split size.
        model: Decomposition model type ('additive' or 'multiplicative').
        period: Seasonal period length.

    Returns:
        Tuple containing paths to seasonal plot, decomposition plot, fitted fit plot, and metrics.
    """
    if file_obj is None or not target_col:
        return None, None, None, pd.DataFrame()
    try:
        dataset = build_dataset_instance(file_obj.name, target_col, index_col, int(split_size))
        fit_res = fit_decomposition(dataset, target_col=target_col, model=model, period=int(period))

        if not fit_res.success:
            return None, None, None, fit_res.metrics_df

        vis_cfg = VisualizationConfig(plot_path="plots/")
        visualizer = Visualizer(dataset=dataset, config=vis_cfg)

        seasonal_plot = visualizer.plot_seasonal(target_col=target_col, period=int(period))
        decomp_plot = visualizer.plot_seasonal_decomposition(target_col=target_col, period=int(period))
        fit_plot = visualizer.plot_in_sample_fit(
            actual=dataset.train[target_col].dropna(),
            fitted=fit_res.fitted_values,
            title=f"Decomposition Reconstruction ({model.title()})",
            filename="decomposition_explorer.png",
            line_color="darkgreen",
        )
        return seasonal_plot, decomp_plot, fit_plot, fit_res.metrics_df
    except Exception as exc:
        logger.error("Decomposition sandbox execution fault: %s", exc)
        return None, None, None, pd.DataFrame([{"Error": str(exc)}])


def handle_sandbox_holt_winters(
    file_obj: Any,
    target_col: str,
    index_col: str,
    split_size: int,
    trend: str,
    seasonal: str,
    period: float,
) -> Tuple[Optional[str], Optional[str], pd.DataFrame]:
    """Handles Holt-Winters Exponential Smoothing simulation using model abstractions.

    Args:
        file_obj: Source dataset file object.
        target_col: Target variable column name.
        index_col: Calendar index column name.
        split_size: Validation split size.
        trend: Trend component ('None', 'add', 'mul').
        seasonal: Seasonal component ('None', 'add', 'mul').
        period: Seasonal period length.

    Returns:
        Tuple containing seasonal plot path, fitted plot path, and goodness-of-fit metrics.
    """
    if file_obj is None or not target_col:
        return None, None, pd.DataFrame()
    try:
        dataset = build_dataset_instance(file_obj.name, target_col, index_col, int(split_size))
        exp_cfg = ExponentialConfig(
            trend=None if trend == "None" else trend,
            seasonal=None if seasonal == "None" else seasonal,
            seasonal_periods=int(period) if int(period) > 0 else None,
        )

        model = ExponentialSmoothingModel(config=exp_cfg)
        model.fit(data_obj=dataset, target_col=target_col)

        # In-sample fitted values evaluation
        fitted_series = model.fitted_model.fittedvalues
        actual_series = dataset.train[target_col].dropna()

        mae = float((actual_series - fitted_series).abs().mean())
        rmse = float(((actual_series - fitted_series) ** 2).mean() ** 0.5)
        metrics_df = pd.DataFrame(
            [{"Metric": "Training MAE", "Value": f"{mae:.4f}"}, {"Metric": "Training RMSE", "Value": f"{rmse:.4f}"}]
        )

        vis_cfg = VisualizationConfig(plot_path="plots/")
        visualizer = Visualizer(dataset=dataset, config=vis_cfg)

        seasonal_plot = visualizer.plot_seasonal(target_col=target_col, period=int(period))
        fit_plot = visualizer.plot_in_sample_fit(
            actual=actual_series,
            fitted=fitted_series,
            title=f"Holt-Winters Fit (Trend={trend}, Seasonal={seasonal})",
            filename="holt_winters_explorer.png",
            line_color="darkorange",
        )
        return seasonal_plot, fit_plot, metrics_df
    except Exception as exc:
        logger.error("Holt-Winters sandbox execution fault: %s", exc)
        return None, None, pd.DataFrame([{"Error": str(exc)}])


def handle_sandbox_sarima(
    file_obj: Any,
    target_col: str,
    index_col: str,
    split_size: int,
    p: float,
    d: float,
    q: float,
    P: float,
    D: float,
    Q: float,
    seasonal_period: float,
) -> Tuple[Optional[str], pd.DataFrame, str, Optional[str]]:
    """Fits SARIMA model and generates stationarity report and correlation diagnostics.

    Args:
        file_obj: Dataset source file object.
        target_col: Target variable column name.
        index_col: Calendar index column name.
        split_size: Validation split size.
        p: AR order.
        d: Differencing order.
        q: MA order.
        P: Seasonal AR order.
        D: Seasonal differencing order.
        Q: Seasonal MA order.
        seasonal_period: Seasonal period lag m.

    Returns:
        Tuple containing fit plot path, metrics DataFrame, stationarity Markdown report, and ACF plot path.
    """
    if file_obj is None or not target_col:
        return (
            None,
            pd.DataFrame(),
            "### Configuration Required\nPlease upload a dataset and select a valid target column.",
            None,
        )
    try:
        dataset = build_dataset_instance(file_obj.name, target_col, index_col, int(split_size))
        sarima_cfg = SARIMAConfig(
            p=int(p),
            d=int(d),
            q=int(q),
            P=int(P),
            D=int(D),
            Q=int(Q),
            s=int(seasonal_period),
        )

        model = SARIMAModel(config=sarima_cfg)
        model.fit(data_obj=dataset, target_col=target_col)

        actual_series = dataset.train[target_col].dropna().astype(float)
        fitted_series = model.fitted_model.fittedvalues

        mae = float((actual_series - fitted_series).abs().mean())
        rmse = float(((actual_series - fitted_series) ** 2).mean() ** 0.5)
        metrics_df = pd.DataFrame(
            [{"Metric": "Training MAE", "Value": f"{mae:.4f}"}, {"Metric": "Training RMSE", "Value": f"{rmse:.4f}"}]
        )

        # Apply differencing for stationarity diagnostic check
        diff_series = actual_series.copy()
        if int(d) > 0:
            for _ in range(int(d)):
                diff_series = diff_series.diff().dropna()
        if int(D) > 0 and int(seasonal_period) > 0:
            for _ in range(int(D)):
                diff_series = diff_series.diff(periods=int(seasonal_period)).dropna()

        stationarity_res = calculate_stationarity(diff_series)
        stationarity_report = generate_stationarity_report(stationarity_res)

        vis_cfg = VisualizationConfig(plot_path="plots/")
        visualizer = Visualizer(dataset=dataset, config=vis_cfg)

        fit_plot = visualizer.plot_in_sample_fit(
            actual=actual_series,
            fitted=fitted_series,
            title=f"SARIMA({int(p)},{int(d)},{int(q)})x({int(P)},{int(D)},{int(Q)})[{int(seasonal_period)}] Alignment",
            filename="sarima_fit_explorer.png",
            line_color="crimson",
        )
        corr_plot = visualizer.plot_autocorrelation(
            series=diff_series,
            filename="sarima_correlation_diagnostics.png",
            target_col=target_col,
        )

        return fit_plot, metrics_df, stationarity_report, corr_plot
    except Exception as exc:
        logger.error("SARIMA sandbox execution fault: %s", exc)
        return (
            None,
            pd.DataFrame([{"Error": str(exc)}]),
            f"### Execution Failure\n`{str(exc)}`",
            None,
        )


# =====================================================================
# EXECUTIVE REPORT & PDF EXPORT GENERATORS
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
) -> str:
    """Renders structured executive summary in Markdown format.

    Returns:
        str: Renders formatted Markdown executive report.
    """
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

    metrics_markdown = (
        metrics_df.to_markdown(index=False)
        if not metrics_df.empty
        else "No evaluation metrics available."
    )

    report_template = f"""# 📄 Time Series Performance Evaluation Report

## 1. Executive Summary
**Project Name:** Seasonal Time Series Forecasting Model Comparison.  
**Best Performing Model:** **{best_model_name}** (Selected based on {primary_metric}).  

**Key Finding:** Statistical components were systematically evaluated. Model **{best_model_name}** demonstrated optimal error minimization across out-of-sample forecast horizons.

---

## 2. Dataset Overview & Configuration
- **Data Source File:** `{os.path.basename(file_name)}`
- **Target Metric Column:** `{target_col}`
- **Index Column:** `{index_col}`
- **Validation Split Size:** `{split_size}` observations

### Data Visualizations
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

## 5. Detailed Diagnostics
{adf_summary}

---

## 6. Conclusions & Recommendations
Model **{best_model_name}** is recommended for production deployment on future forecasting horizons for this dataset.
"""
    return report_template.strip()


def export_report_to_pdf(markdown_report: str, output_path: str = "reports/performance_report.pdf") -> Optional[str]:
    """Generates PDF report from Markdown report string using WeasyPrint.

    Args:
        markdown_report: Executive report content string.
        output_path: File destination target path.

    Returns:
        Optional[str]: Path to rendered PDF or None if PDF generation dependency failed.
    """
    try:
        from weasyprint import HTML

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        body_html = markdown.markdown(markdown_report, extensions=["tables"])

        styled_document = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
@page {{
    size: A4;
    margin: 18mm 15mm 18mm 15mm;
    @bottom-right {{
        content: "Page " counter(page) " of " counter(pages);
        font-family: Arial, sans-serif;
        font-size: 8pt;
        color: #718096;
    }}
}}
body {{
    font-family: Arial, sans-serif;
    font-size: 10pt;
    line-height: 1.5;
    color: #2d3748;
}}
h1 {{ color: #1a365d; font-size: 18pt; border-bottom: 2px solid #2b6cb0; padding-bottom: 6px; }}
h2 {{ color: #2b6cb0; font-size: 13pt; margin-top: 18px; }}
table {{ width: 100%; border-collapse: collapse; margin: 12px 0; }}
th, td {{ padding: 8px; border: 1px solid #cbd5e0; text-align: left; }}
th {{ background-color: #2b6cb0; color: #ffffff; }}
tr:nth-child(even) {{ background-color: #f7fafc; }}
code {{ font-family: monospace; background-color: #edf2f7; padding: 2px 4px; border-radius: 3px; }}
</style>
</head>
<body>
{body_html}
</body>
</html>
"""
        HTML(string=styled_document).write_pdf(output_path)
        logger.info("Successfully exported executive report PDF to %s", output_path)
        return output_path
    except Exception as exc:
        logger.warning("PDF generation skipped due to error or missing system libraries: %s", exc)
        return None


# =====================================================================
# BATCH PIPELINE ORCHESTRATOR ADAPTER
# =====================================================================


def extract_comparative_table(run_output: Dict[str, Any]) -> pd.DataFrame:
    """Transforms experiment orchestrator output dictionaries into structured tabular DataFrames.

    Args:
        run_output: Output dictionary returned by `ExperimentOrchestrator`.

    Returns:
        pd.DataFrame: Tabular summary of model performances.
    """
    rows: List[Dict[str, Any]] = []
    results = run_output.get("results", {})
    for model_name, data in results.items():
        row = {"Model Strategy": model_name.upper(), "Status": "SUCCESS"}
        row.update(data.get("metrics", {}))
        rows.append(row)
    errors = run_output.get("errors", {})
    for model_name, err_msg in errors.items():
        rows.append({"Model Strategy": model_name.upper(), "Status": "FAILED", "Error Context": err_msg})
    return pd.DataFrame(rows)


def execute_ui_pipeline(
    file_obj: Any,
    target_col: str,
    index_col: str,
    split_size: int,
    seasonality_period: int,
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
    arima_s: int,
) -> Tuple[pd.DataFrame, Optional[str], str, str, Optional[str]]:
    """Coordinates parameters, invokes ExperimentOrchestrator, generates report, and exports PDF.

    Returns:
        Tuple containing metrics DataFrame, forecast plot path, execution log message, Markdown report, and PDF path.
    """
    if file_obj is None or not target_col:
        return (
            pd.DataFrame(),
            None,
            "Validation Failure: Please verify source dataset file and target column choice.",
            "### Report Generation Failed\nMissing required source dataset file or target column.",
            None,
        )

    try:
        data_cfg = DataConfig(
            path=file_obj.name,
            split_size=int(split_size),
            index_col=index_col if index_col else None,
            target=target_col,
        )
        scoring_cfg = ScoringConfig(
            mae=mae_flag, mse=mse_flag, rmse=rmse_flag, mape=mape_flag, epsilon=1e-5
        )
        vis_cfg = VisualizationConfig(plot_path="plots/", decomposition_model=decomp_mode)
        models_cfg = ModelsConfig(
            decompose=DecomposeConfig(
                model=decomp_mode,
                period=int(decompose_period) if str(decompose_period).isdigit() else 12,
            ),
            exponential_smoothing=ExponentialConfig(
                trend=None if hw_trend == "None" else hw_trend,
                seasonal=None if hw_seasonal == "None" else hw_seasonal,
                seasonal_periods=int(hw_seasonal_period) if str(hw_seasonal_period).isdigit() else None,
            ),
            sarima=SARIMAConfig(
                p=int(arima_p),
                d=int(arima_d),
                q=int(arima_q),
                P=int(arima_P),
                D=int(arima_D),
                Q=int(arima_Q),
                s=int(arima_s),
            ),
        )
        global_cfg = Config(
            name="GradioUI_ExecutionRun",
            data=data_cfg,
            visualizer=vis_cfg,
            scoring=scoring_cfg,
            models=models_cfg,
        )

        orchestrator = ExperimentOrchestrator(global_config=global_cfg)
        line_path, envelope_path, _ = handle_data_ingestion_visuals(file_obj, target_col, index_col, split_size, seasonality_period)

        if model_choice == "Run All Models Sweep":
            run_output = orchestrator.run_all_models(target_column=target_col)
            metrics_df = extract_comparative_table(run_output)

            predictions = {}
            results = run_output.get("results", {})
            for m_key, m_val in results.items():
                predictions[m_key.upper()] = m_val["predictions"]

            vis_engine = Visualizer(dataset=orchestrator.data_layer, config=vis_cfg)
            plot_to_render = vis_engine.plot_predictions_vs_actuals(
                predictions=predictions, target_col=target_col
            )
            status_msg = "Global model benchmark sweep completed successfully."
        else:
            run_output = orchestrator.run_single_model(model_type=model_choice, target_column=target_col)
            metrics_df = pd.DataFrame([run_output["metrics"]], index=[model_choice.upper()])

            vis_engine = Visualizer(dataset=orchestrator.data_layer, config=vis_cfg)
            plot_to_render = vis_engine.plot_predictions_vs_actuals(
                predictions=run_output["predictions"], target_col=target_col
            )
            status_msg = f"Single model evaluation '{model_choice}' completed successfully."

        # Compute diagnostic stationarity check
        dataset_instance = build_dataset_instance(file_obj.name, target_col, index_col, split_size)
        diff_series = dataset_instance.train[target_col].dropna().copy()
        if int(arima_d) > 0:
            for _ in range(int(arima_d)):
                diff_series = diff_series.diff().dropna()
        if int(arima_D) > 0 and int(arima_s) > 0:
            for _ in range(int(arima_D)):
                diff_series = diff_series.diff(periods=int(arima_s)).dropna()
        stationary_result = calculate_stationarity(diff_series)
        adf_md = generate_stationarity_report(stationary_result)

        # Build executive markdown report
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

        pdf_path = export_report_to_pdf(markdown_report=report_markdown)
        return metrics_df, plot_to_render, status_msg, report_markdown, pdf_path

    except Exception as err:
        logger.exception("Pipeline execution fault")
        return (
            pd.DataFrame(),
            None,
            f"Pipeline Execution Error: {str(err)}",
            f"### Execution Error\n`{str(err)}`",
            None,
        )


# =====================================================================
# PRESENTATION LAYER: UI BUILDERS
# =====================================================================


def _render_ingestion_block() -> Dict[str, Any]:
    """Constructs stream ingestion UI components."""
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Core Data Stream Ingestion Controls")
            file_input = gr.File(label="Target Source File (.csv, .xlsx, .parquet)")
            with gr.Row():
                index_column = gr.Dropdown(choices=["None"],label="Calendar Index Column", interactive=True)
                target_column = gr.Dropdown(choices=[], label="Target Metric Column", interactive=True)
            split_size = gr.Number(label="Out-of-Sample Validation Slice Size", value=12, precision=0)

        with gr.Column(scale=2):
            gr.Markdown("### Ingestion Temporal Properties Visualization")
            with gr.Row():
                ingestion_line_plot = gr.Image(label="Dataset Sequence Line Split Map", type="filepath")
                ingestion_envelope_plot = gr.Image(label="Envelope Geometry Analysis Map", type="filepath")
            with gr.Row():
                seasonality_period = gr.Number(label="Seasonality period", value=12, precision=0)
                ingestion_seasonality_plot = gr.Image(label="Seasonality plot", type="filepath")


    return {
        "file_input": file_input,
        "index_column": index_column,
        "target_column": target_column,
        "split_size": split_size,
        "ingestion_line_plot": ingestion_line_plot,
        "ingestion_envelope_plot": ingestion_envelope_plot,
        "seasonality_period": seasonality_period,
        "ingestion_seasonality_plot": ingestion_seasonality_plot
    }


def _render_batch_dashboard() -> Dict[str, Any]:
    """Constructs Automated Batch Dashboard tab components."""
    model_choices = list(MODEL_REGISTRY.keys()) + ["Run All Models Sweep"]

    with gr.Tab("Automated Batch Dashboard"):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Evaluation Metrics Selection")
                with gr.Row():
                    mae_tgl = gr.Checkbox(label="MAE", value=True)
                    mse_tgl = gr.Checkbox(label="MSE", value=True)
                    rmse_tgl = gr.Checkbox(label="RMSE", value=True)
                    mape_tgl = gr.Checkbox(label="MAPE", value=True)

                gr.Markdown("### Model Architecture Settings")
                batch_model_selector = gr.Dropdown(
                    choices=model_choices, value="Run All Models Sweep", label="Execution Strategy"
                )

                with gr.Accordion("Decomposition Overrides", open=False):
                    batch_decomp_mode = gr.Radio(
                        choices=["additive", "multiplicative"], value="additive", label="Synthesis Type"
                    )
                    batch_decomp_period = gr.Number(value=12, precision=0, label="Seasonal Period Component")

                with gr.Accordion("Smoothing Overrides", open=False):
                    batch_hw_trend = gr.Dropdown(choices=["None", "add", "mul"], value="add", label="Trend")
                    batch_hw_seasonal = gr.Dropdown(
                        choices=["None", "add", "mul"], value="add", label="Seasonal"
                    )
                    batch_hw_seasonal_period = gr.Number(value=12, precision=0, label="Seasonal Period")

                with gr.Accordion("State Space ARIMA Vectors Order", open=False):
                    with gr.Row():
                        bp = gr.Slider(0, 5, 1, step=1, label="p")
                        bd = gr.Slider(0, 2, 0, step=1, label="d")
                        bq = gr.Slider(0, 5, 1, step=1, label="q")
                    with gr.Row():
                        bP = gr.Slider(0, 3, 0, step=1, label="P")
                        bD = gr.Slider(0, 2, 0, step=1, label="D")
                        bQ = gr.Slider(0, 3, 0, step=1, label="Q")
                    bs = gr.Number(value=12, precision=0, label="Seasonal Period (m)")

                run_batch_btn = gr.Button("Execute Performance Pipeline Run", variant="primary")

            with gr.Column(scale=1):
                gr.Markdown("### Process Operational Output Monitor")
                batch_status = gr.Textbox(label="Execution Telemetry Trace Logs", value="System Ready.", interactive=False)
                batch_metrics_table = gr.DataFrame(label="Out-of-Sample Score Parameters Report", interactive=False)
                batch_plot_output = gr.Image(label="Validation Projection Visual Layouts", type="filepath")

        gr.Markdown("---")
        with gr.Row():
            with gr.Column(scale=3):
                report_output_markdown = gr.Markdown(
                    value="*Executive performance report will render here after executing a pipeline run.*"
                )
            with gr.Column(scale=1):
                pdf_download_file = gr.File(label="Download Executive Report (PDF)", interactive=False)

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
    """Constructs Hyperparameter Discovery Sandbox tab components."""
    components: Dict[str, Any] = {}

    with gr.Tab("Hyperparameter Discovery Sandbox"):
        gr.Markdown("### In-Sample Parameter Analysis Workspace")

        with gr.Tabs():
            # Panel 1: Decomposition
            with gr.Tab("1. Time Series Decomposition"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### Classical Decomposition Controls")
                        decomp_type = gr.Radio(
                            choices=["additive", "multiplicative"], value="additive", label="Synthesis Type"
                        )
                        decomp_period = gr.Number(label="Decomposition Period", value=12, precision=0)
                        calculate_decomp_btn = gr.Button("Fit In-Sample Decomposition", variant="secondary")
                    with gr.Column(scale=2):
                        decomp_plot = gr.Image(label="Decomposition Plot", type="filepath")
                        seasonal_plot = gr.Image(label="Seasonal Plot", type="filepath")
                        fit_plot = gr.Image(label="In-Sample Reconstructed Fit", type="filepath")
                        decomp_metrics = gr.DataFrame(label="Decomposition Metrics")

                components["decomp"] = {
                    "type": decomp_type,
                    "period": decomp_period,
                    "btn": calculate_decomp_btn,
                    "decomp_plot": decomp_plot,
                    "seasonal_plot": seasonal_plot,
                    "fit_plot": fit_plot,
                    "metrics": decomp_metrics,
                }

            # Panel 2: Holt-Winters
            with gr.Tab("2. Holt-Winters Smoothing"):
                with gr.Row():
                    with gr.Column(scale=1):
                        hw_t = gr.Dropdown(choices=["None", "add", "mul"], value="add", label="Trend")
                        hw_s = gr.Dropdown(choices=["None", "add", "mul"], value="add", label="Seasonal")
                        hw_p = gr.Number(label="Seasonal Frequency Cycles", value=12, precision=0)
                        calculate_hw_btn = gr.Button("Fit Simulation Holt-Winters", variant="secondary")
                    with gr.Column(scale=2):
                        hw_seasonal_plot = gr.Image(label="Seasonal Plot", type="filepath")
                        hw_fit_plot = gr.Image(label="In-Sample Tracking Path", type="filepath")
                        hw_metrics = gr.DataFrame(label="Holt-Winters Metrics")

                components["hw"] = {
                    "t": hw_t,
                    "s": hw_s,
                    "p": hw_p,
                    "btn": calculate_hw_btn,
                    "seasonal_plot": hw_seasonal_plot,
                    "fit_plot": hw_fit_plot,
                    "metrics": hw_metrics,
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
                        m_input = gr.Number(value=12, label="Seasonal Lag Period (m)", precision=0)

                        calculate_sarima_btn = gr.Button("Fit Simulation SARIMA Model", variant="secondary")
                        adf_report_container = gr.Markdown("Modify configurations and click execute to evaluate metrics.")

                    with gr.Column(scale=2):
                        sarima_fit_image = gr.Image(label="In-Sample Alignment Path", type="filepath")
                        sarima_fit_metrics = gr.DataFrame(label="SARIMA Metrics")
                        diagnostic_plots_container = gr.Image(label="ACF / PACF Maps", type="filepath")

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
# INTERFACE BUILDER & EVENT WIRING
# =====================================================================


def build_gradio_interface() -> gr.Blocks:
    """Builds the main Gradio application interface and binds event triggers."""
    with gr.Blocks(title="Time Series Analytics Infrastructure", theme=gradio.themes.Default()) as app:
        gr.Markdown("# Time Series Forecasting Infrastructure Management Console")

        ingest_ui = _render_ingestion_block()

        with gr.Tabs():
            batch_ui = _render_batch_dashboard()
            sandbox_ui = _render_sandbox_tabs()

        # Wire CSV Column Header Inspection
        ingest_ui["file_input"].change(
            fn=parse_csv_column_headers,
            inputs=[ingest_ui["file_input"]],
            outputs=[ingest_ui["target_column"], ingest_ui["index_column"]],
        )

        # Wire Data Ingestion Visuals
        ingestion_inputs = [
            ingest_ui["file_input"],
            ingest_ui["target_column"],
            ingest_ui["index_column"],
            ingest_ui["split_size"],
            ingest_ui['seasonality_period']
        ]
        for trigger_comp in [
            ingest_ui["file_input"],
            ingest_ui["target_column"],
            ingest_ui["index_column"],
            ingest_ui["split_size"],
            ingest_ui['seasonality_period']
        ]:
            trigger_comp.change(
                fn=handle_data_ingestion_visuals,
                inputs=ingestion_inputs,
                outputs=[ingest_ui["ingestion_line_plot"], ingest_ui["ingestion_envelope_plot"], ingest_ui["ingestion_seasonality_plot"]],
            )



        # Wire Batch Execution Pipeline Trigger
        batch_ui["run_batch_btn"].click(
            fn=execute_ui_pipeline,
            inputs=[
                ingest_ui["file_input"],
                ingest_ui["target_column"],
                ingest_ui["index_column"],
                ingest_ui["split_size"],
                ingest_ui["seasonality_period"],
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

        # Wire Sandbox Panel 1: Decomposition
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
                decomp_comp["seasonal_plot"],
                decomp_comp["decomp_plot"],
                decomp_comp["fit_plot"],
                decomp_comp["metrics"],
            ],
        )

        # Wire Sandbox Panel 2: Holt-Winters
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
            outputs=[hw_comp["seasonal_plot"], hw_comp["fit_plot"], hw_comp["metrics"]],
        )

        # Wire Sandbox Panel 3: SARIMA
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
    setup_logging()
    app_interface = build_gradio_interface()
    app_interface.launch(server_name="0.0.0.0", server_port=7860, share=False)