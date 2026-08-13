"""Interactive Gradio Web Interface for the Time Series Forecasting Suite.

Exposes modeling, parameter configuration, execution monitoring, 
and performance visual analytics dashboards directly inside the browser.
"""

import logging
import os
from typing import Any, Dict, List, Tuple, Optional

import gradio as gr
import pandas as pd

# Import system dependencies from previous milestones
from src.config import Config, DataConfig, VisualizationConfig, ModelsConfig
from src.models import DecomposeConfig, ExponentialConfig, SARIMAConfig
from src.metrics import ScoringConfig
from src.orchestrator import ExperimentOrchestrator, MODEL_REGISTRY

# Configure module-level logging
logger = logging.getLogger(__name__)


# =====================================================================
# CORE UI DATA PROCESSING BRIDGE
# =====================================================================

def extract_comparative_table(run_output: Dict[str, Any]) -> pd.DataFrame:
    """Transforms raw global batch orchestration dictionaries into tabular formats.

    Args:
        run_output: The structural dictionary returned from run_all_models().

    Returns:
        A formatted pandas DataFrame comparing metrics side-by-side across models.
    """
    rows: List[Dict[str, Any]] = []
    
    # Process successful iterations
    results = run_output.get("results", {})
    for model_name, data in results.items():
        row = {"Model Strategy": model_name.upper(), "Status": "SUCCESS"}
        row.update(data.get("metrics", {}))
        rows.append(row)
        
    # Process failed iterations cleanly to indicate performance drops
    errors = run_output.get("errors", {})
    for model_name, err_msg in errors.items():
        rows.append({
            "Model Strategy": model_name.upper(),
            "Status": "FAILED",
            "Error Context": err_msg
        })
        
    return pd.DataFrame(rows)


def parse_csv_column_headers(file_obj: Any) -> gr.Dropdown:
    """Inspects uploaded user dataset files to extract header tokens automatically.

    Args:
        file_obj: The incoming temporary file payload from the Gradio File element.

    Returns:
        An updated Gradio Dropdown element with populated column choices.
    """
    if file_obj is None:
        return gr.Dropdown(choices=[], value=None)
    
    try:
        # Read only headers line to ensure instant parsing responses
        df_headers = pd.read_csv(file_obj.name, nrows=0)
        columns = list(df_headers.columns)
        logger.info("Successfully discovered %d column paths automatically.", len(columns))
        
        # Default fallback selectors if standard 'Date' or 'Timestamp' headers exist
        default_val = columns[1] if len(columns) > 1 else columns[0]
        return gr.Dropdown(choices=columns, value=default_val, interactive=True)
    except Exception as exc:
        logger.error("Failed to parse headers from structural dataset: %s", exc)
        return gr.Dropdown(choices=[], value=None, label="Parsing Failed. Check CSV structure.")


def execute_ui_pipeline(
    file_obj: Any,
    target_col: str,
    index_col: str,
    split_size: int,
    model_choice: str,
    # Metrics Toggles
    mae_flag: bool,
    mse_flag: bool,
    rmse_flag: bool,
    mape_flag: bool,
    # Model Parameter Framework Blocks
    decomp_mode: str,
    hw_trend: str,
    hw_seasonal: str,
    arima_p: int, arima_d: int, arima_q: int,
    arima_P: int, arima_D: int, arima_Q: int
) -> Tuple[pd.DataFrame, Optional[str], str]:
    """Compiles form variables into configuration parameters and invokes the orchestrator.

    Args:
        file_obj: Temporary uploaded local file handle object.
        target_col: Selected target numerical field array string label.
        index_col: The column tracking datetime intervals.
        split_size: Length of observations allocated out-of-sample for testing.
        model_choice: String identifier dictating explicit routing keys.
        mae_flag, mse_flag, rmse_flag, mape_flag: Metric configuration selectors.
        decomp_mode: Decompose strategy mode setting.
        hw_trend, hw_seasonal: Holt-Winters trend/seasonal modes.
        arima_p, arima_d, arima_q: Non-seasonal ARIMA shapes.
        arima_P, arima_D, arima_Q: Seasonal ARIMA shapes.

    Returns:
        A tuple containing (Metrics DataFrame, Renderable Image File Path, Status Message Box).
    """
    # Defensive early validations
    if file_obj is None:
        return pd.DataFrame(), None, "Validation Error: Please upload a valid CSV dataset before executing runs."
    if not target_col:
        return pd.DataFrame(), None, "Validation Error: Please select an evaluation target variable column."

    try:
        # 1. Map input variables into structured, immutable configuration modules
        data_cfg = DataConfig(
            path=file_obj.name,
            split_size=int(split_size),
            index_col=index_col if index_col else "Date"
        )
        
        scoring_cfg = ScoringConfig(
            mae=mae_flag,
            mse=mse_flag,
            rmse=rmse_flag,
            mape=mape_flag,
            epsilon=1e-5  # Default soft zero safety threshold correction factor
        )
        
        vis_cfg = VisualizationConfig(
            plot_path="plots/",
            decomposition_model=decomp_mode
        )
        
        # Build configuration models block mappings
        models_cfg = ModelsConfig(
            decompose=DecomposeConfig(model=decomp_mode),
            exponential_smoothing=ExponentialConfig(
                trend=None if hw_trend == "None" else hw_trend,
                seasonal=None if hw_seasonal == "None" else hw_seasonal
            ),
            sarima=SARIMAConfig(
                p=int(arima_p), d=int(arima_d), q=int(arima_q),
                P=int(arima_P), D=int(arima_D), Q=int(arima_Q)
            )
        )
        
        global_cfg = Config(
            name="GradioUI_Run",
            data=data_cfg,
            visualizer=vis_cfg,
            scoring=scoring_cfg,
            models=models_cfg
        )
        
        # 2. Instantiate our central orchestrator facade framework layer
        orchestrator = ExperimentOrchestrator(global_config=global_cfg)
        
        # 3. Route execution requests based on explicit model choices
        if model_choice == "Run All Models Sweep":
            logger.info("Triggering comprehensive framework optimization sweep via UI dashboard.")
            run_output = orchestrator.run_all_models(target_column=target_col)
            metrics_df = extract_comparative_table(run_output)
            
            # For batch sweeps, look up the comparison plot if generated by the pipeline
            comparison_plot = os.path.join(global_cfg.visualizer.plot_path, "model_comparison_overlay.png")
            plot_to_render = comparison_plot if os.path.exists(comparison_plot) else None
            
            status_msg = f"Batch benchmark execution complete. Successful models: {run_output['summary']['successful_count']}."
            return metrics_df, plot_to_render, status_msg
            
        else:
            logger.info("Triggering isolated execution routine for model: '%s'", model_choice)
            run_output = orchestrator.run_single_model(model_type=model_choice, target_column=target_col)
            
            # Format single metric returns cleanly
            metrics_df = pd.DataFrame([run_output["metrics"]], index=[model_choice.upper()])
            
            # 4. Resolve exact filepath path to load generated plot directly from the runtime system
            plot_to_render = os.path.join(
                global_cfg.visualizer.plot_path, 
                run_output["run_id"], 
                "predictions_vs_actuals.png"
            )
            
            status_msg = f"Model strategy '{model_choice}' executed successfully. Performance cached in run: {run_output['run_id']}."
            return metrics_df, plot_to_render, status_msg
            
    except Exception as error:
        logger.error("UI orchestration pipe encountered an exception: %s", error, exc_info=True)
        # Catch errors gracefully and pass context back to the user without crashing the server
        return pd.DataFrame(), None, f"Pipeline Execution Failed: {str(error)}"


# =====================================================================
# UI INTERFACE BLOCK LAYOUT GRAPH GENERATION
# =====================================================================

def build_gradio_interface() -> gr.Blocks:
    """Builds the comprehensive tabbed enterprise visualization hub using Blocks."""
    
    # Extract dynamic models dropdown listing from system registry entries
    model_choices = list(MODEL_REGISTRY.keys()) + ["Run All Models Sweep"]
    
    with gr.Blocks(title="Forecasting Engine Hub", theme=gr.themes.Default()) as app:
        gr.Markdown(
            """
            # Production Time Series Analysis & Forecasting Hub
            Exposes stateless forecasting decomposition models, Holt-Winters exponential variations, and state-space SARIMAX vectors natively.
            """
        )
        
        with gr.Row():
            # LEFT HAND PANEL: PARAMETER CONTROLS
            with gr.Column(scale=1):
                gr.Markdown("### 1. Ingestion Data Configuration")
                file_input = gr.File(label="Upload Dataset File (CSV format)", file_types=[".csv"])
                
                with gr.Row():
                    index_column = gr.Textbox(label="Index Datetime Column Name", value="Date", placeholder="e.g., Date")
                    target_column = gr.Dropdown(choices=[], label="Target Evaluation Column", interactive=True)
                
                split_size = gr.Number(label="Out-of-Sample Test Split Horizon Steps", value=12, precision=0)
                
                gr.Markdown("### 2. Global Evaluation Scoring Performance Metrics")
                with gr.Row():
                    mae_toggle = gr.Checkbox(label="MAE", value=True)
                    mse_toggle = gr.Checkbox(label="MSE", value=True)
                    rmse_toggle = gr.Checkbox(label="RMSE", value=True)
                    mape_toggle = gr.Checkbox(label="MAPE", value=True)
                
                gr.Markdown("### 3. Hyperparameter Settings Block")
                model_selector = gr.Dropdown(choices=model_choices, value="Run All Models Sweep", label="Target Modeling Strategy")
                
                # Accordion layout blocks for configuration groupings
                with gr.Accordion("Decomposition Settings Options", open=False):
                    decomp_mode = gr.Radio(choices=["additive", "multiplicative"], value="additive", label="Algebraic Synthesis Composition Mode")
                    
                with gr.Accordion("Holt-Winters Smoothing Parameters", open=False):
                    hw_trend_mode = gr.Dropdown(choices=["None", "add", "mul"], value="add", label="Trend Smoothing Component")
                    hw_seasonal_mode = gr.Dropdown(choices=["None", "add", "mul"], value="add", label="Seasonal Smoothing Component")
                    
                with gr.Accordion("Seasonal ARIMA Parameter Matrices (p,d,q)x(P,D,Q)", open=False):
                    with gr.Row():
                        p_val = gr.Slider(minimum=0, maximum=5, value=1, step=1, label="Trend p")
                        d_val = gr.Slider(minimum=0, maximum=2, value=1, step=1, label="Difference d")
                        q_val = gr.Slider(minimum=0, maximum=5, value=1, step=1, label="Moving Average q")
                    with gr.Row():
                        P_val = gr.Slider(minimum=0, maximum=3, value=1, step=1, label="Seasonal P")
                        D_val = gr.Slider(minimum=0, maximum=2, value=1, step=1, label="Seasonal D")
                        Q_val = gr.Slider(minimum=0, maximum=3, value=1, step=1, label="Seasonal Q")
                
                run_btn = gr.Button("Execute Analytics Run Routine", variant="primary")
            
            # RIGHT HAND PANEL: REAL-TIME ANALYTICAL GRAPHICS AND SCORES
            with gr.Column(scale=1):
                gr.Markdown("### 4. System Operational Status Monitor")
                status_box = gr.Textbox(label="Operational Execution Trace Logs", value="System Idle. Waiting for ingestion instructions...", interactive=False)
                
                gr.Markdown("### 5. Numerical Metrics Evaluation Report")
                results_table = gr.DataFrame(label="Calculated Performance Portfolios Output", interactive=False)
                
                gr.Markdown("### 6. Forecast Trajectory Evaluation Visualization")
                plot_output = gr.Image(label="Rendered Validation Graphic Timeline Slices", type="filepath")
                
        # =====================================================================
        # INTERACTION INTERFACE BINDING EVENT MAPS
        # =====================================================================
        # Bind file uploads to update target column choices instantly
        file_input.change(
            fn=parse_csv_column_headers,
            inputs=[file_input],
            outputs=[target_column]
        )
        
        # Bind the primary operational buttons to execution pipelines
        run_btn.click(
            fn=execute_ui_pipeline,
            inputs=[
                file_input, target_column, index_column, split_size, model_selector,
                mae_toggle, mse_toggle, rmse_toggle, mape_toggle,
                decomp_mode, hw_trend_mode, hw_seasonal_mode,
                p_val, d_val, q_val, P_val, D_val, Q_val
            ],
            outputs=[results_table, plot_output, status_box]
        )
        
    return app


if __name__ == "__main__":
    # Launch application server loop using localized port bounds
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    interface_hub = build_gradio_interface()
    # Setting share=False for strict production isolation containment
    interface_hub.launch(server_name="0.0.0.0", server_port=7860, share=False)