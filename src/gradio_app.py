"""Interactive Gradio Web Interface for the Time Series Forecasting Suite.

Exposes a dual-phase workspace: Automated Batch Dashboards for out-of-sample
evaluation sweeps, and an Interactive Experiment Sandbox for hyperparameter discovery.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import pandas as pd

from src.config import Config, DataConfig, ModelsConfig, VisualizationConfig
from src.data import DataClass
from src.diagnostics import (
    calculate_adf_stationarity,
    generate_decomposition_plot,
    generate_holt_winters_plot,
    generate_sarima_sandbox_fit,
    generate_stateless_correlation_plots,
)
from src.metrics import ScoringConfig
from src.models import DecomposeConfig, ExponentialConfig, SARIMAConfig
from src.orchestrator import MODEL_REGISTRY, ExperimentOrchestrator
from src.visualizer import Visualizer

logger = logging.getLogger(__name__)


def extract_training_slice(file_path: str, target_col: str, index_col: str, split_size: float) -> pd.Series:
    """Safely extracts the training split to keep evaluation data isolated."""
    df = pd.read_csv(file_path)
    idx_field = index_col if index_col else df.columns[0]
    
    df[idx_field] = pd.to_datetime(df[idx_field])
    df = df.sort_values(by=idx_field).set_index(idx_field)
    
    target_series = df[target_col].astype(float)
    cutoff = len(target_series) - int(split_size)
    
    if cutoff <= 0:
        raise ValueError(f"Split horizon size ({int(split_size)}) exceeds scale dimensions ({len(target_series)}).")
        
    return target_series.iloc[:cutoff]


# =====================================================================
# EVENT HANDLERS FOR STREAM INGESTION & SANDBOX CONTROLS
# =====================================================================

def handle_data_ingestion_visuals(file_obj: Any, target_col: str, index_col: str, split_size: float) -> Tuple[Optional[str], Optional[str]]:
    """Generates continuous layout line plots and envelope spread tracking configurations instantly."""
    if file_obj is None or not target_col:
        return None, None
    try:
        data_cfg = DataConfig(path=file_obj.name, split_size=int(split_size), index_col=index_col if index_col else None)
        dataset = DataClass(config=data_cfg)
        vis_cfg = VisualizationConfig(plot_path="plots/")
        visualizer = Visualizer(dataset=dataset, config=vis_cfg)
        
        line_path = visualizer.plot_line_series(target_col=target_col)
        envelope_path = visualizer.plot_envelope_components(target_col=target_col)
        return line_path, envelope_path
    except Exception as exc:
        logger.error("Failed handling ingestion visual paths: %s", exc)
        return None, None


def handle_sandbox_decomposition(file_obj: Any, target_col: str, index_col: str, split_size: float, model: str, period: float) -> Tuple[str, pd.DataFrame]:
    """Processes in-sample time series structural decompositions."""
    if file_obj is None or not target_col:
        return "plots/sandbox/decomposition_explorer.png", pd.DataFrame()
    try:
        train_series = extract_training_slice(file_obj.name, target_col, index_col, split_size)
        return generate_decomposition_plot(train_series, model, int(period))
    except Exception as exc:
        logger.error("Sandbox Decomposition failure: %s", exc)
        return "plots/sandbox/decomposition_explorer.png", pd.DataFrame([{"Error": str(exc)}])


def handle_sandbox_holt_winters(file_obj: Any, target_col: str, index_col: str, split_size: float, trend: str, seasonal: str, period: float) -> Tuple[str, pd.DataFrame]:
    """Processes exponential smoothing simulations against historical paths."""
    if file_obj is None or not target_col:
        return "plots/sandbox/holt_winters_explorer.png", pd.DataFrame()
    try:
        train_series = extract_training_slice(file_obj.name, target_col, index_col, split_size)
        return generate_holt_winters_plot(train_series, trend, seasonal, int(period))
    except Exception as exc:
        logger.error("Sandbox Holt Winters failure: %s", exc)
        return "plots/sandbox/holt_winters_explorer.png", pd.DataFrame([{"Error": str(exc)}])


def handle_sandbox_sarima(
    file_obj: Any, target_col: str, index_col: str, split_size: float, 
    p: float, d: float, q: float, P: float, D: float, Q: float, seasonal_period: float
) -> Tuple[str, pd.DataFrame, str, str]:
    """Fits the full SARIMA configurations onto training boundaries and returns diagnostics."""
    if file_obj is None or not target_col:
        return (
            "plots/sandbox/sarima_fit_explorer.png", pd.DataFrame(),
            "### Configuration Required", "plots/sandbox/sarima_correlation_diagnostics.png"
        )
    try:
        train_series = extract_training_slice(file_obj.name, target_col, index_col, split_size)
        
        # 1. Generate fit plot & training score tables
        fit_plot, metrics_df = generate_sarima_sandbox_fit(
            train_series, int(p), int(d), int(q), int(P), int(D), int(Q), int(seasonal_period)
        )
        
        # 2. Perform transformations tracking statistics
        differenced_series = train_series.copy()
        if int(d) > 0:
            for _ in range(int(d)):
                differenced_series = differenced_series.diff().dropna()
        if int(D) > 0 and int(seasonal_period) > 0:
            for _ in range(int(D)):
                differenced_series = differenced_series.diff(periods=int(seasonal_period)).dropna()

        adf_data = calculate_adf_stationarity(differenced_series)
        ui_markdown = f"""
        ### Augmented Dickey-Fuller (ADF) Stationarity Audit Report
        * **Status Message**: {adf_data.message}
        * **Is Stationary?**: `{"TRUE" if adf_data.is_stationary else "FALSE"}`
        * **ADF Test Statistic Value**: `{adf_data.statistic:.4f}`
        * **p-Value Probability**: `{adf_data.p_value:.6f}`
        """
        for pct, val in adf_data.critical_values.items():
            ui_markdown += f"\n* **Critical Boundary ({pct})**: `{val:.4f}`"
            
        corr_plot = generate_stateless_correlation_plots(differenced_series)
        return fit_plot, metrics_df, ui_markdown, corr_plot
    except Exception as exc:
        logger.error("Sandbox SARIMA failure: %s", exc)
        return (
            "plots/sandbox/sarima_fit_explorer.png", pd.DataFrame([{"Error": str(exc)}]),
            f"### Execution Failure\n{str(exc)}", "plots/sandbox/sarima_correlation_diagnostics.png"
        )


# =====================================================================
# SYSTEM CORE INTEGRATIONS (BATCH MODE PORT)
# =====================================================================

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
        rows.append({"Model Strategy": model_name.upper(), "Status": "FAILED", "Error Context": err_msg})
    return pd.DataFrame(rows)


def parse_csv_column_headers(file_obj: Any) -> gr.Dropdown:
    """Inspects uploaded user dataset files to extract header tokens automatically."""
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


def execute_ui_pipeline(
    file_obj: Any, target_col: str, index_col: str, split_size: int, model_choice: str,
    mae_flag: bool, mse_flag: bool, rmse_flag: bool, mape_flag: bool,
    decomp_mode: str, decompose_period,hw_trend: str, hw_seasonal: str, hw_seasonal_period: str,
    arima_p: int, arima_d: int, arima_q: int, arima_P: int, arima_D: int, arima_Q: int
) -> Tuple[pd.DataFrame, Optional[str], str]:
    """Compiles parameters and dispatches execution runs to the central orchestrator."""
    if file_obj is None or not target_col:
        return pd.DataFrame(), None, "Validation Error: Verify data files and target settings before executing loops."

    try:
        data_cfg = DataConfig(path=file_obj.name, split_size=int(split_size), index_col=index_col if index_col else "Date")
        scoring_cfg = ScoringConfig(mae=mae_flag, mse=mse_flag, rmse=rmse_flag, mape=mape_flag, epsilon=1e-5)
        vis_cfg = VisualizationConfig(plot_path="plots/", decomposition_model=decomp_mode)
        models_cfg = ModelsConfig(
            decompose=DecomposeConfig(model=decomp_mode, period=None if decompose_period =="None" else decompose_period),
            exponential_smoothing=ExponentialConfig(
                trend=None if hw_trend == "None" else hw_trend,
                seasonal=None if hw_seasonal == "None" else hw_seasonal,
                seasonal_periods=None if hw_seasonal_period == "None" else int(hw_seasonal_period)
                ),
            sarima=SARIMAConfig(p=int(arima_p), d=int(arima_d), q=int(arima_q), P=int(arima_P), D=int(arima_D), Q=int(arima_Q))
        )
        global_cfg = Config(name="GradioUI_Run", data=data_cfg, visualizer=vis_cfg, scoring=scoring_cfg, models=models_cfg)
        orchestrator = ExperimentOrchestrator(global_config=global_cfg)

        if model_choice == "Run All Models Sweep":
            run_output = orchestrator.run_all_models(target_column=target_col)
            metrics_df = extract_comparative_table(run_output)
            
            # Use unified visualizer engine to chart prediction overlays across sweeps
            dataset_instance = DataClass(config=data_cfg)
            vis_engine = Visualizer(dataset=dataset_instance, config=vis_cfg)
            
            # Locate an available execution layer to map reference overlays
            plot_to_render = None
            if "sarima" in run_output["results"]:
                plot_to_render = vis_engine.plot_predictions_vs_actuals(
                    predictions=run_output["results"]["sarima"]["predictions"], target_col=target_col
                )
            elif "exponential_smoothing" in run_output["results"]:
                plot_to_render = vis_engine.plot_predictions_vs_actuals(
                    predictions=run_output["results"]["exponential_smoothing"]["predictions"], target_col=target_col
                )
            return metrics_df, plot_to_render, "Batch sequence evaluation sweep completed successfully."
        else:
            run_output = orchestrator.run_single_model(model_type=model_choice, target_column=target_col)
            metrics_df = pd.DataFrame([run_output["metrics"]], index=[model_choice.upper()])
            
            dataset_instance = DataClass(config=data_cfg)
            vis_engine = Visualizer(dataset=dataset_instance, config=vis_cfg)
            plot_to_render = vis_engine.plot_predictions_vs_actuals(
                predictions=run_output["predictions"], target_col=target_col
            )
            return metrics_df, plot_to_render, f"Model run target '{model_choice}' successfully complete."
    except Exception as err:
        return pd.DataFrame(), None, f"Pipeline Run Failure: {str(err)}"


# =====================================================================
# ROOT INTERFACE GENERATOR DESIGN
# =====================================================================

def build_gradio_interface() -> gr.Blocks:
    """Builds the comprehensive dual-phase time series visualization system."""
    model_choices = list(MODEL_REGISTRY.keys()) + ["Run All Models Sweep"]

    with gr.Blocks(title="Forecasting Infrastructure Management Console", theme=gr.themes.Default()) as app:
        gr.Markdown("# Automated Production Analytics & Discovery Engine Hub")

        # SHARED TOP-LEVEL INGESTION BLOCK CONFIGURATION WITH DYNAMIC STREAM VISUALIZERS
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Core Data Stream Ingestion Controls")
                file_input = gr.File(label="Target Source File (CSV)", file_types=[".csv"])
                with gr.Row():
                    index_column = gr.Textbox(label="Calendar Index Column", value="Date")
                    target_column = gr.Dropdown(choices=[], label="Target Metric Column", interactive=True)
                split_size = gr.Number(label="Out-of-Sample Validation Slice Size", value=12, precision=0)
            
            with gr.Column(scale=2):
                gr.Markdown("### Ingestion Temporal Properties Visualization")
                with gr.Row():
                    ingestion_line_plot = gr.Image(label="Dataset Sequence Line Split Map", type="filepath")
                    ingestion_envelope_plot = gr.Image(label="Envelope Geometry Analysis Map", type="filepath")

        with gr.Tabs():
            # PHASE 1: AUTOMATED BATCH DASHBOARD
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
                        batch_model_selector = gr.Dropdown(choices=model_choices, value="Run All Models Sweep", label="Execution Strategy")

                        with gr.Accordion("Decomposition Overrides", open=False):
                            batch_decomp_mode = gr.Radio(choices=["additive", "multiplicative"], value="additive", label="Synthesis Type")
                            batch_decomp_period = gr.Number(value=12, precision=0, label="Seasonal Period Component")
                        with gr.Accordion("Smoothing Overrides", open=False):
                            batch_hw_trend = gr.Dropdown(choices=["None", "add", "mul"], value="add", label="Trend Component")
                            batch_hw_seasonal = gr.Dropdown(choices=["None", "add", "mul"], value="add", label="Seasonal Component")
                            batch_hw_seasonal_period = gr.Number(value=12, precision=0, label="Seasonal Period Component")
                        with gr.Accordion("State Space ARIMA Vectors Order", open=False):
                            with gr.Row():
                                bp = gr.Slider(0, 5, 1, step=1, label="p")
                                bd = gr.Slider(0, 2, 1, step=1, label="d")
                                bq = gr.Slider(0, 5, 1, step=1, label="q")
                            with gr.Row():
                                bP = gr.Slider(0, 3, 1, step=1, label="P")
                                bD = gr.Slider(0, 2, 1, step=1, label="D")
                                bQ = gr.Slider(0, 3, 1, step=1, label="Q")

                        run_batch_btn = gr.Button("Execute Performance Pipeline Run", variant="primary")

                    with gr.Column(scale=1):
                        gr.Markdown("### Process Operational Output Monitor")
                        batch_status = gr.Textbox(label="Execution Telemetry Trace Logs", value="System Ready.", interactive=False)
                        batch_metrics_table = gr.DataFrame(label="Out-of-Sample Score Parameters Report", interactive=False)
                        batch_plot_output = gr.Image(label="Validation Projection Visual Layouts", type="filepath")

            # PHASE 2: EXPERIMENTAL EXPLORATION SANDBOX
            with gr.Tab("Hyperparameter Discovery Sandbox"):
                gr.Markdown("### In-Sample Parameter Analysis Workspace (Isolates Test Partition Data)")
                
                with gr.Tabs():
                    # SANDBOX PANEL 1: DECOMPOSITION ANALYSIS
                    with gr.Tab("1. Time Series Decomposition"):
                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("#### Classical Decomposition Controls")
                                decomp_type = gr.Radio(choices=["additive", "multiplicative"], value="additive", label="Algebraic Composition Mode")
                                decomp_period = gr.Number(label="Seasonal Decomposition Periodicity Horizon", value=12, precision=0)
                                calculate_decomp_btn = gr.Button("Fit In-Sample Decomposition Model", variant="secondary")
                            with gr.Column(scale=2):
                                decomp_sandbox_image = gr.Image(label="In-Sample Reconstructed Fit Path", type="filepath")
                                decomp_sandbox_metrics = gr.DataFrame(label="Decomposition Metrics")

                    # SANDBOX PANEL 2: HOLT-WINTERS SMOOTHING PARAMETERS
                    with gr.Tab("2. Holt-Winters Smoothing"):
                        with gr.Row():
                            with gr.Column(scale=1):
                                hw_t = gr.Dropdown(choices=["None", "add", "mul"], value="add", label="Trend Parameter Vector")
                                hw_s = gr.Dropdown(choices=["None", "add", "mul"], value="add", label="Seasonal Parameter Vector")
                                hw_p = gr.Number(label="Seasonal Frequency Cycles Length", value=12, precision=0)
                                calculate_hw_btn = gr.Button("Fit Simulation Model Sequence", variant="secondary")
                            with gr.Column(scale=2):
                                hw_sandbox_image = gr.Image(label="Historical Slices Tracking Visual Paths", type="filepath")
                                hw_sandbox_metrics = gr.DataFrame(label="In-Sample Inversion Error Vectors")

                    # SANDBOX PANEL 3: ITERATIVE SARIMA DISCOVERY STATION
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
                                m_input = gr.Number(value=12, label="Seasonal Lag Period Configuration (m)", precision=0)
                                
                                calculate_sarima_btn = gr.Button("Fit Simulation SARIMA Model", variant="secondary")
                                adf_report_container = gr.Markdown("Modify configurations and click execute to evaluate metrics.")

                            with gr.Column(scale=2):
                                sarima_fit_image = gr.Image(label="In-Sample Tracking Alignment Path", type="filepath")
                                sarima_fit_metrics = gr.DataFrame(label="SARIMA In-Sample Performance Portfolio")
                                diagnostic_plots_container = gr.Image(label="Dynamic ACF / PACF Spatial Coordinate Maps", type="filepath")

        # =====================================================================
        # ASYNCHRONOUS CONNECTOR ROUTINES AND DEPENDENCY REGISTRY
        # =====================================================================
        file_input.change(fn=parse_csv_column_headers, inputs=[file_input], outputs=[target_column])
        
        # Ingestion block continuous plot listeners
        ingestion_inputs = [file_input, target_column, index_column, split_size]
        file_input.change(fn=handle_data_ingestion_visuals, inputs=ingestion_inputs, outputs=[ingestion_line_plot, ingestion_envelope_plot])
        target_column.change(fn=handle_data_ingestion_visuals, inputs=ingestion_inputs, outputs=[ingestion_line_plot, ingestion_envelope_plot])
        split_size.change(fn=handle_data_ingestion_visuals, inputs=ingestion_inputs, outputs=[ingestion_line_plot, ingestion_envelope_plot])

        # Batch Production Swaps Trigger
        run_batch_btn.click(
            fn=execute_ui_pipeline,
            inputs=[
                file_input, target_column, index_column, split_size, batch_model_selector,
                mae_tgl, mse_tgl, rmse_tgl, mape_tgl,
                batch_decomp_mode, batch_decomp_period, batch_hw_trend, batch_hw_seasonal, batch_hw_seasonal_period,
                bp, bd, bq, bP, bD, bQ
            ],
            outputs=[batch_metrics_table, batch_plot_output, batch_status]
        )

        # Sandbox Tab 1 Trigger
        calculate_decomp_btn.click(
            fn=handle_sandbox_decomposition,
            inputs=[file_input, target_column, index_column, split_size, decomp_type, decomp_period],
            outputs=[decomp_sandbox_image, decomp_sandbox_metrics]
        )

        # Sandbox Tab 2 Trigger
        calculate_hw_btn.click(
            fn=handle_sandbox_holt_winters,
            inputs=[file_input, target_column, index_column, split_size, hw_t, hw_s, hw_p],
            outputs=[hw_sandbox_image, hw_sandbox_metrics]
        )

        # Sandbox Tab 3 Trigger
        calculate_sarima_btn.click(
            fn=handle_sandbox_sarima,
            inputs=[file_input, target_column, index_column, split_size, s_p, s_d, s_q, s_P, s_D, s_Q, m_input],
            outputs=[sarima_fit_image, sarima_fit_metrics, adf_report_container, diagnostic_plots_container]
        )

    return app


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    interface_hub = build_gradio_interface()
    interface_hub.launch(server_name="0.0.0.0", server_port=7860, share=False)