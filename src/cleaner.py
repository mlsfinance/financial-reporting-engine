"""
this file is for data cleaning and normalization

responsibilities:
    * handle missing values with configurable strategies
    * ensure monotonic DatetimeIndex
    * optional resampling / alignment
    * convert prices to returns (simple / log)
    * provide a standardized output for downstream KPI, viz and reporting
    
assumptions:
    * the input is a DataFrame indexed by datetime (DatetimeIndex)
    * there's at least one numeric price column (validated in loader)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd

from .config import Config


class DataCleanerError(Exception):
    """this error will be raised when either cleaning or normalization"""


NaStrategy = Literal["drop", "ffill", "bfill", "ffill_then_bfill", "interpolate"]
ReturnType = Literal["simple", "log"]
AlignMethod = Literal["inner", "outer"]


@dataclass
class CleanedData:
    prices: pd.DataFrame
    returns: pd.DataFrame



def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise DataCleanerError("Input DataFrame index must be a DatetimeIndex.")
    if df.index.hasnans:
        raise DataCleanerError("DatetimeIndex contains NaT values (bad date parsing upstream).")
    return df



def _apply_na_strategy(df: pd.DataFrame, strategy: NaStrategy) -> pd.DataFrame:
    if strategy == "drop":
        return df.dropna()
    if strategy == "ffill":
        return df.ffill()
    if strategy == "bfill":
        return df.bfill()
    if strategy == "ffill_then_bfill":
        return df.ffill().bfill()
    if strategy == "interpolate":
        #time-based interpolation will only be meaningful with DatetimeIndex
        return df.interpolate(method="time").ffill().bfill()
    raise DataCleanerError(f"Unknown NA strategy: {strategy}")



def _normalize_index(df: pd.DataFrame, timezone: Optional[str]) -> pd.DataFrame:
    """
    to normalize the DatetimeIndex:
        * sort
        * remove duplicates (keep last)
        * optionally localize/convert timezone
        * normalize to midnight only if index has time component? (we keep exact stamps)
    """
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]

    idx = df.index
    if timezone:
        if idx.tz is None:
            df.index = idx.tz_localize(timezone)
        else:
            df.index = idx.tz_convert(timezone)

    return df



def _resample_prices(df: pd.DataFrame, frequency: str) -> pd.DataFrame:
    """
    this function will resample the prices to a given frequency if needed. it uses the last observation in the period (common for prices)
    """
    if not frequency:
        return df
    #if the index is already regular enough, resampling is still deterministic
    return df.resample(frequency).last()



def _align_frames(prices: pd.DataFrame, method: AlignMethod) -> pd.DataFrame:
    if prices.shape[1] <= 1:
        return prices

    if method == "inner":
        return prices.dropna(how="any")
    if method == "outer":
        return prices
    raise DataCleanerError(f"Unknown align method: {method}")



def prices_to_returns(prices: pd.DataFrame, return_type: ReturnType) -> pd.DataFrame:
    """
    convert prices to returns:
        * simple: P_t / P_{t-1} - 1
        * log: ln(P_t / P_{t-1})

    returns are aligned with prices index; first row becomes NaN and is dropped
    """
    if (prices <= 0).any().any() and return_type == "log":
        raise DataCleanerError("Log returns require strictly positive prices.")


    if return_type == "simple":
        rets = prices.pct_change()
    elif return_type == "log":
        rets = np.log(prices / prices.shift(1))
    else:
        raise DataCleanerError(f"Unknown return type: {return_type}")


    rets = rets.replace([np.inf, -np.inf], np.nan).dropna(how="all")
    return rets



def clean_and_normalize(prices: pd.DataFrame, cfg: Config) -> CleanedData:
    """
    the main entrypoint used by the pipeline

    reads (with defaults):
        data.frequency
        data.timezone
        cleaning.na_strategy
        cleaning.return_type
        cleaning.align_method
        cleaning.min_rows
    """
    prices = _ensure_datetime_index(prices)


    frequency = cfg.get("data.frequency", default="D", type_=str) or "D"
    timezone = cfg.get("data.timezone", default=None, type_=str)


    na_strategy: NaStrategy = cfg.get("cleaning.na_strategy", default="ffill_then_bfill", type_=str)
    return_type: ReturnType = cfg.get("cleaning.return_type", default="log", type_=str)  
    align_method: AlignMethod = cfg.get("cleaning.align_method", default="inner", type_=str)  
    min_rows = int(cfg.get("cleaning.min_rows", default=30) or 30)


    #normalize index and resample
    prices_norm = _normalize_index(prices, timezone=timezone)
    prices_rs = _resample_prices(prices_norm, frequency=frequency)


    #NA handling and alignment across assets
    prices_clean = _apply_na_strategy(prices_rs, strategy=na_strategy)
    prices_clean = _align_frames(prices_clean, method=align_method)


    #after alignment, re-apply NA strategy if 'outer' left holes
    prices_clean = _apply_na_strategy(prices_clean, strategy=na_strategy)


    if len(prices_clean) < min_rows:
        raise DataCleanerError(
            f"Not enough data after cleaning. Rows={len(prices_clean)}, min_rows={min_rows}."
        )


    #convert to returns
    returns = prices_to_returns(prices_clean, return_type=return_type)


    if len(returns) < max(2, min_rows - 1):
        raise DataCleanerError("Not enough return observations after conversion.")


    return CleanedData(prices=prices_clean, returns=returns)