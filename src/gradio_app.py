"""Interactive Gradio Web Interface for the Time Series Forecasting Suite.

Exposes a dual-phase workspace: Automated Batch Dashboards for out-of-sample
evaluation sweeps, and an Interactive Experiment Sandbox for hyperparameter discovery.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import pandas as pd

# Import system frameworks and configurations
from src.config import Config, DataConfig, ModelsConfig, VisualizationConfig
from src.config import ModelsConfig  # Double check resolution paths
from src.diagnostics import (
    calculate_adf_stationarity,
    generate_decomposition_plot,
    generate_holt_winters_plot,
    generate_stateless_correlation_plots,
)
from src.metrics import ScoringConfig
from src.models import DecomposeConfig, ExponentialConfig, SARIMAConfig
from src.orchestrator import MODEL_REGISTRY, ExperimentOrchestrator

# Configure module-level logging
logger = logging.getLogger(__name__)


# =====================================================================
# IN-SAMPLE EXPLORATION DATA BOUNDARY SLICERS
# =====================================================================

def extract_training_slice(file_path: str, target_col: str, index_col: str, split_size: float) -> pd.Series:
    """Safely extracts the training split to keep evaluation data isolated."""
    df = pd.read_csv(file_path)
    idx_field = index_col if index_col else df.columns[0]
    
    df[idx_field] = pd.to_datetime(df[idx_field])
    df = df.sort_values(by=idx_field).set_index(idx_field)
    
    target_series = df[target_col].astype(float)
    cutoff = len(target_series) - int(split_size)
    
    if cutoff <= 0:
        raise ValueError(f"Split horizon size ({int(split_size)}) exceeds complete timeline scale dimensions ({len(target_series)}).")
        
    return target_series.iloc[:cutoff]


# =====================================================================
# SANDBOX INTERACTION EVENT HANDLERS
# =====================================================================

def handle_sandbox_decomposition(file_obj: Any, target_col: str, index_col: str, split_size: float, model: str, period: float) -> str:
    """Processes in-sample time series structural decompositions."""
    if file_obj is None or not target_col:
        return "plots/sandbox/decomposition_explorer.png"  # Returns default placeholder if clear data is unpopulated
    try:
        train_series = extract_training_slice(file_obj.name, target_col, index_col, split_size)
        return generate_decomposition_plot(train_series, model, int(period))
    except Exception as exc:
        logger.error("Sandbox Decomposition failure: %s", exc)
        return generate_decomposition_plot(pd.Series(dtype=float), model, int(period))


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


def handle_sarima_stationarity_change(
    file_obj: Any, target_col: str, index_col: str, split_size: float, d: float, D: float, seasonal_period: float
) -> Tuple[str, str]:
    """Processes custom differencing and recalculates core statistical reports."""
    if file_obj is None or not target_col:
        return "### Configuration Required\nUpload data fields to display analytical validation parameters.", "plots/sandbox/sarima_correlation_diagnostics.png"

    try:
        # 1. Parse index constraints and slice up to training split parameters
        train_series = extract_training_slice(file_obj.name, target_col, index_col, split_size)
        
        # 2. Execute regular differencing steps
        differenced_series = train_series.copy()
        if int(d) > 0:
            for _ in range(int(d)):
                differenced_series = differenced_series.diff().dropna()
                
        # Execute seasonal differencing steps
        if int(D) > 0 and int(seasonal_period) > 0:
            for _ in range(int(D)):
                differenced_series = differenced_series.diff(periods=int(seasonal_period)).dropna()

        # 3. Calculate structured statistical metrics using our new engine
        adf_data = calculate_adf_stationarity(differenced_series)
        
        # 4. Generate localized presentation patterns via Markdown boundary rules
        ui_markdown = f"""
        ### Augmented Dickey-Fuller (ADF) Stationarity Audit Report
        * **Status Message**: {adf_data.message}
        * **Is Stationary?**: `{"TRUE" if adf_data.is_stationary else "FALSE"}`
        * **ADF Test Statistic Value**: `{adf_data.statistic:.4f}`
        * **p-Value Probability**: `{adf_data.p_value:.6f}`
        * **Regression Lags Utilized**: `{adf_data.lags_used}`
        * **Effective Data Coordinates Used**: `{adf_data.observations}`
        
        #### Mathematical Alpha Significance Thresholds:
        """
        for pct, val in adf_data.critical_values.items():
            ui_markdown += f"\n* **Critical Boundary ({pct})**: `{val:.4f}`"
            
        # 5. Graph corresponding dependency structures
        plot_path = generate_stateless_correlation_plots(differenced_series)
        return ui_markdown, plot_path

    except Exception as exc:
        logger.error("Failure tracking sandbox transformation arrays: %s", exc)
        return f"### Pipeline Computational Failure\nAn unexpected anomaly halted the diagnostic engine: {str(exc)}", "plots/sandbox/sarima_correlation_diagnostics.png"


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
    decomp_mode: str, hw_trend: str, hw_seasonal: str,
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
            decompose=DecomposeConfig(model=decomp_mode),
            exponential_smoothing=ExponentialConfig(trend=None if hw_trend == "None" else hw_trend, seasonal=None if hw_seasonal == "None" else hw_seasonal),
            sarima=SARIMAConfig(p=int(arima_p), d=int(arima_d), q=int(arima_q), P=int(arima_P), D=int(arima_D), Q=int(arima_Q))
        )
        global_cfg = Config(name="GradioUI_Run", data=data_cfg, visualizer=vis_cfg, scoring=scoring_cfg, models=models_cfg)
        orchestrator = ExperimentOrchestrator(global_config=global_cfg)

        if model_choice == "Run All Models Sweep":
            run_output = orchestrator.run_all_models(target_column=target_col)
            metrics_df = extract_comparative_table(run_output)
            comparison_plot = os.path.join(global_cfg.visualizer.plot_path, "model_comparison_overlay.png")
            plot_to_render = comparison_plot if os.path.exists(comparison_plot) else None
            return metrics_df, plot_to_render, "Batch sequence evaluation sweep completed successfully."
        else:
            run_output = orchestrator.run_single_model(model_type=model_choice, target_column=target_col)
            metrics_df = pd.DataFrame([run_output["metrics"]], index=[model_choice.upper()])
            plot_to_render = os.path.join(global_cfg.visualizer.plot_path, run_output["run_id"], "predictions_vs_actuals.png")
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

        # SHARED TOP-LEVEL INGESTION BLOCK CONFIGURATION
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Core Data Stream Ingestion Controls")
                file_input = gr.File(label="Target Source File (CSV)", file_types=[".csv"])
                with gr.Row():
                    index_column = gr.Textbox(label="Calendar Index Column", value="Date")
                    target_column = gr.Dropdown(choices=[], label="Target Metric Column", interactive=True)
                split_size = gr.Number(label="Out-of-Sample Validation Slice Size", value=12, precision=0)

        # SEPARATION STRATEGY: DUAL PHASE PRIMARY TABS STRUCTURE
        with gr.Tabs():
            
            # PHASE 1: AUTOMATED BATCH DASHBOARD (PRODUCTION HORIZONS)
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
                        with gr.Accordion("Smoothing Overrides", open=False):
                            batch_hw_trend = gr.Dropdown(choices=["None", "add", "mul"], value="add", label="Trend Component")
                            batch_hw_seasonal = gr.Dropdown(choices=["None", "add", "mul"], value="add", label="Seasonal Component")
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

            # PHASE 2: EXPERIMENTAL EXPLORATION SANDBOX (IN-SAMPLE ANALYSIS ONLY)
            with gr.Tab("Hyperparameter Discovery Sandbox"):
                gr.Markdown("### In-Sample Parameter Analysis Workspace (Isolates Test Partition Data)")
                
                with gr.Tabs():
                    # SANDBOX PANEL 1: DECOMPOSITION ANALYSIS
                    with gr.Tab("1. Time Series Decomposition"):
                        with gr.Row():
                            with gr.Column(scale=1):
                                decomp_type = gr.Radio(choices=["additive", "multiplicative"], value="additive", label="Algebraic Composition Mode")
                                decomp_period = gr.Number(label="Seasonal Decomposition Frequency Lag Bounds", value=12, precision=0)
                                calculate_decomp_btn = gr.Button("Extract Trend Matrix Parts", variant="secondary")
                            with gr.Column(scale=2):
                                decomp_sandbox_image = gr.Image(label="Structural Systematic Components", type="filepath")

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
                                gr.Markdown("#### Dynamic Transformations & Testing Metrics")
                                d_slider = gr.Slider(0, 2, value=0, step=1, label="Regular Differencing Multi-Step Operator (d)")
                                D_slider = gr.Slider(0, 2, value=0, step=1, label="Seasonal Differencing Multi-Step Operator (D)")
                                m_input = gr.Number(value=12, label="Seasonal Lag Period Configuration (m)", precision=0)
                                
                                adf_report_container = gr.Markdown("Modify differencing configurations to evaluate ADF statistics instantly.")

                            with gr.Column(scale=2):
                                gr.Markdown("#### Autocorrelation Feature Spaces")
                                diagnostic_plots_container = gr.Image(label="Dynamic ACF / PACF Spatial Coordinate Maps", type="filepath")

        # =====================================================================
        # INTERACTIVE ASYNCHRONOUS DATA EVENT LISTENER MAPS
        # =====================================================================
        # Dynamic header updates upon source data attachment
        file_input.change(fn=parse_csv_column_headers, inputs=[file_input], outputs=[target_column])

        # Batch Run Triggers
        run_batch_btn.click(
            fn=execute_ui_pipeline,
            inputs=[
                file_input, target_column, index_column, split_size, batch_model_selector,
                mae_tgl, mse_tgl, rmse_tgl, mape_tgl,
                batch_decomp_mode, batch_hw_trend, batch_hw_seasonal,
                bp, bd, bq, bP, bD, bQ
            ],
            outputs=[batch_metrics_table, batch_plot_output, batch_status]
        )

        # Sandbox Tab 1: Action Triggers
        calculate_decomp_btn.click(
            fn=handle_sandbox_decomposition,
            inputs=[file_input, target_column, index_column, split_size, decomp_type, decomp_period],
            outputs=[decomp_sandbox_image]
        )

        # Sandbox Tab 2: Action Triggers
        calculate_hw_btn.click(
            fn=handle_sandbox_holt_winters,
            inputs=[file_input, target_column, index_column, split_size, hw_t, hw_s, hw_p],
            outputs=[hw_sandbox_image, hw_sandbox_metrics]
        )

        # Sandbox Tab 3: Asynchronous Event Routing (Sliders update parameters instantly)
        sarima_inputs = [file_input, target_column, index_column, split_size, d_slider, D_slider, m_input]
        d_slider.change(fn=handle_sarima_stationarity_change, inputs=sarima_inputs, outputs=[adf_report_container, diagnostic_plots_container])
        D_slider.change(fn=handle_sarima_stationarity_change, inputs=sarima_inputs, outputs=[adf_report_container, diagnostic_plots_container])
        m_input.change(fn=handle_sarima_stationarity_change, inputs=sarima_inputs, outputs=[adf_report_container, diagnostic_plots_container])

    return app


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    interface_hub = build_gradio_interface()
    interface_hub.launch(server_name="0.0.0.0", server_port=7860, share=False)