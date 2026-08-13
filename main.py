"""Production Execution Entry Point for the Time Series Forecasting Engine.

Handles CLI argument ingestion, environment bootstrapping, deterministic log routing,
configuration parsing, and maps runtime control streams directly to headless batch
orchestrators or interactive visualization interfaces.
"""

import argparse
import json
import logging
import os
import socket
import sys
from typing import Any, Dict, List, Optional

import yaml

# Import corporate framework layer blueprints
from src.config import Config, get_config_from_yaml
from src.gradio_app import build_gradio_interface
from src.orchestrator import ExperimentOrchestrator
from src.logger import setup_logging

# Setup root-level application execution logger
logger = logging.getLogger("forecasting_platform")
setup_logging()

# =====================================================================
# SYSTEM BOOTSTRAPPING & LOGGING SETUP
# =====================================================================


# =====================================================================
# DETERMINISTIC CONFIGURATION LOADER
# =====================================================================




# =====================================================================
# NETWORKING DEFENSIVE UTILITIES
# =====================================================================

def verify_port_availability(host: str, port: int) -> bool:
    """Checks whether the requested system port is already locked by another process.

    Args:
        host: Target binding network hostname address vector (e.g., '0.0.0.0').
        port: Numerical socket network identity port map key.

    Returns:
        True if the port path can be safely bound, False if a collision is detected.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, port))
            return True
        except socket.error:
            return False


# =====================================================================
# COMMAND LINE INTERFACE IMPLEMENTATION
# =====================================================================

def parse_cli_arguments(args_payload: List[str]) -> argparse.Namespace:
    """Configures the execution schemas and parses terminal workspace flags.

    Args:
        args_payload: Parameter arguments list vector passed from systems layers.

    Returns:
        A Namespace tracker recording verified processing properties.
    """
    parser = argparse.ArgumentParser(
        description="Production Suite Entry Point - Modular Time Series Analysis Platform",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path framework pointing toward the declarative file setup targets."
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        default="ui",
        choices=["cli", "ui"],
        help="Operational run strategy targeting batch analytics evaluations or presentation dashboards."
    )
    
    return parser.parse_args(args_payload)


# =====================================================================
# MAIN ROUTING ENGINE ENTRY POINT
# =====================================================================

def main(cli_args: List[str]) -> None:
    """Coordinates core bootstrapping logic, parses arguments, and routes execution paths."""
    # 1. Parse incoming command-line execution parameters
    parsed_flags = parse_cli_arguments(cli_args)
    
    # 2. Ingest, hydrate, and validate global framework parameter models
    global_config: Config = get_config_from_yaml(parsed_flags.config)
    
    # Extract logging definitions safely with standard system overrides if keys are absent
    log_level = getattr(global_config, "logging_level", "INFO")
    log_file = getattr(global_config, "logging_file", "logs/execution.log")
    
    logger.info("Application context fully initialized. Selected execution path mode: '%s'", parsed_flags.mode.upper())

    # 4. Route traffic dynamically to processing engines or presentation layers
    if parsed_flags.mode == "ui":
        logger.info("Initializing Interactive Gradio Web UI Engine environment...")
        
        server_host = "0.0.0.0"
        preferred_port = 7860
        
        # Guard against port collisions dynamically inside deployment containers
        if not verify_port_availability(server_host, preferred_port):
            logger.warning("Port collision identified on target %d. Scanning fallback options...", preferred_port)
            fallback_found = False
            for target_port in range(7861, 7880):
                if verify_port_availability(server_host, target_port):
                    preferred_port = target_port
                    fallback_found = True
                    break
            
            if not fallback_found:
                logger.critical("System initialization blocked: Network interfaces locked. Unable to claim port links.")
                print("CRITICAL ERROR: Available networking sockets are full. Gradio launch halted.", file=sys.stderr)
                sys.exit(1)
        
        print(f"Launching Server Dashboard Instance safely bound at: http://localhost:{preferred_port}")
        app_block = build_gradio_interface()
        
        # Non-blocking deployment execution server lifecycle initialization
        app_block.launch(
            server_name=server_host,
            server_port=preferred_port,
            share=False,
            prevent_thread_lock=False
        )
        
    elif parsed_flags.mode == "cli":
        logger.info("Executing Headless Automation Batch Sequence Model Sweep...")
        print("Starting batch execution framework pipelines...")
        
        try:
            # Instantiate our central orchestrator facade framework layer
            orchestrator = ExperimentOrchestrator(global_config=global_config)
            
            # Resolve target tracker paths
            target_col = global_config.data.target
            
            # Run sequential loops benchmark hooks across models safely
            summary_report: Dict[str, Any] = orchestrator.run_all_models(target_column=target_col)
            
            # 5. Process tabular evaluations and signal operational completion metrics
            success_count = summary_report["summary"]["successful_count"]
            failed_count = summary_report["summary"]["failed_count"]
            total_runs = summary_report["summary"]["total_attempted"]
            
            print(f"\n--- Batch Sweep Metrics Processing Summary ---")
            print(f"Total Processed Strategies Checked: {total_runs}")
            print(f"Successful Execution Counts:        {success_count}")
            print(f"Failed Model Implementations:       {failed_count}")
            
            # Check tracking constraints to determine OS process signals
            if failed_count > 0:
                logger.error("Headless sweep cycle closed with anomalies flagged in performance suites.")
                print("Batch sequence execution encountered validation failures on specific model matrices.", file=sys.stderr)
                sys.exit(1)
                
            logger.info("Global baseline sweep finished cleanly with zero structural failure traces.")
            print("All forecasting strategies compiled successfully.")
            sys.exit(0)
            
        except Exception as pipe_exc:
            logger.critical("Catastrophic orchestration crash encountered inside CLI execution branch: %s", pipe_exc, exc_info=True)
            print(f"CRITICAL SYSTEM ERROR: Pipeline execution halted: {pipe_exc}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    # Pass environment array values to the entry point
    main(sys.argv[1:])