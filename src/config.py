from dataclasses import dataclass
import os
from dacite import from_dict
import yaml
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """Configuration class for data loading and splitting parameters.

    Attributes:
        path (str): File path to the target dataset (.csv, .xlsx, or .parquet).
        split_size (int): Integer row index split location for train/test segmentation.
        index_col (Optional[str]): Column name to be used as the datetime index.
    """

    path: str
    split_size: int
    index_col: Optional[str] = None


def tuple_constructor(loader, node) -> Tuple:
    """
    A tuple constructor for figsize parameter in visualizer
    """
    return tuple(loader.construct_sequence(node))


yaml.SafeLoader.add_constructor("!tuple", tuple_constructor)


@dataclass
class VisualizationConfig:
    """Configures plot file locations, graphic resolution, and styling choices.

    Attributes:
        plot_path (str): Target directory where plots are saved.
        decomposition_model (str): Type of seasonal decomposition ('additive' or 'multiplicative').
        style_theme (str): Global Matplotlib layout style theme.
        dpi (int): Graphics dots-per-inch sharpness parameter.
        default_figsize (Tuple[int, int]): Dimensions for rendering visual figures.
        acf_lags (int): Max lag index for autocorrelation evaluation.
    """

    plot_path: str = "plots"
    decomposition_model: str = "additive"
    style_theme: str = "seaborn-v0_8-whitegrid"
    dpi: int = 150
    default_figsize: Tuple[int, int] = (14, 7)
    acf_lags: int = 40

    '''
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> VisualizationConfig:
        """Initializes config instances directly from an external dictionary configuration."""
        return cls(
            plot_path=config_dict.get("plot_path", "plots"),
            decomposition_model=config_dict.get("decomposition_model", "additive"),
            style_theme=config_dict.get("style_theme", "seaborn-v0_8-whitegrid"),
            dpi=config_dict.get("dpi", 150),
            default_figsize=tuple(config_dict.get("default_figsize", (14, 7))),  # type: ignore
            acf_lags=config_dict.get("acf_lags", 40)
        )
    '''


# =====================================================================
# IMMUTABLE CONFIGURATION DATA STRUCTURES
# =====================================================================

@dataclass(frozen=True)
class DecomposeConfig:
    """Configuration schema for Classical Decomposition Forecasting Model.

    Attributes:
        model: Type of decomposition structure. Must be 'additive' or 'multiplicative'.
        period: Optional manual definition of seasonal periods. If None, inferred automatically.
    """
    model: str = "additive"
    period: Optional[int] = None

    def __post_init__(self) -> None:
        if self.model not in ("additive", "multiplicative"):
            raise ValueError("Model attribute must be either 'additive' or 'multiplicative'.") 



@dataclass(frozen=True)
class ExponentialConfig:
    """Configuration schema for Holt-Winters Exponential Smoothing Model.

    Attributes:
        trend: Trend component type. E.g., 'add', 'mul', or None.
        seasonal: Seasonal component type. E.g., 'add', 'mul', or None.
        seasonal_periods: Optional manual period definition. If None, inferred automatically.
    """
    trend: Optional[str] = None
    seasonal: Optional[str] = None
    seasonal_periods: Optional[int] = None


@dataclass(frozen=True)
class SARIMAConfig:
    """Configuration schema for Seasonal Autoregressive Integrated Moving Average.

    Attributes:
        p: Trend autoregressive order.
        d: Trend differencing order.
        q: Trend moving average order.
        P: Seasonal autoregressive order.
        D: Seasonal differencing order.
        Q: Seasonal moving average order.
        s: Optional seasonal period lag. If None, inferred automatically.
    """
    p: int = 1
    d: int = 0
    q: int = 1
    P: int = 0
    D: int = 0
    Q: int = 0
    s: Optional[int] = None


@dataclass
class ScoringConfig:
    mae: bool
    mse: bool
    rmse: bool
    mape: bool


@dataclass
class ModelsConfig:
    decompose: DecomposeConfig
    exponential_smoothing: ExponentialConfig
    sarima: SARIMAConfig


@dataclass
class Config:
    name: str
    data: DataConfig
    visualizer: VisualizationConfig
    models: ModelsConfig
    scoring: ScoringConfig


def get_config_from_yaml(yaml_path: str = "configs/config.yaml") -> Optional[Config]:
    """
    Initialize a configuration class from yaml file
    Parameters:
    ---------
        yaml_path (str): the path to the yaml file
    Returns:
        config (Config): initialized config class
    """
    if not os.path.exists(yaml_path):
        logger.error(
            f"Failed to find yaml file on the provided path: {yaml_path}\nPlease ensure path correcteness."
        )
        return
    try:
        with open(yaml_path, "r") as yaml_file:
            data = yaml.safe_load(yaml_file)
            config = from_dict(data_class=Config, data=data)
    except Exception as e:
        logger.error(f"An error occure: {e}")
        raise e

    return config


if __name__ == "__main__":
    print(get_config_from_yaml())
