"""
this file is for financial KPIs, it computes standard performance/risk metrics from a returns series

it includes:
    * CAGR
    * Annualized volatility
    * Sharpe ratio
    * Sortino ratio
    * Max drawdown
    * Drawdown series (sequential drawdowns)
    * Rolling returns
    * Rolling volatility

things to take into account:
    * inputs are returns (simple or log) as a pandas Series or DataFrame
    * the annualization factor (like 252) is configurable
    * risk_free_rate is annual (for example 0.02 for 2%)
"""


from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Union, cast
import logging

import numpy as np
import pandas as pd
from typing import Literal

from .config import Config


logger = logging.getLogger(__name__)


class KPIError(Exception):
    """raised when KPI computation fails"""


ArrayLike = Union[pd.Series, pd.DataFrame]

#we define the return type using Literal from cleaner for consistency
ReturnType = Literal["simple", "log"]


@dataclass
class KPIResult:
    """container for KPI outputs"""
    summary: pd.DataFrame                            #one row per asset with scalar KPIs
    drawdowns: pd.DataFrame                          #drawdown series per asset
    rolling_returns: Dict[int, pd.DataFrame]         #window -> rolling cumulative return
    rolling_volatility: Dict[int, pd.DataFrame]      #window -> rolling annualized vol


def _to_dataframe(returns: ArrayLike) -> pd.DataFrame:
    if isinstance(returns, pd.Series):

        #ensure name is set if None, important for output structure
        name = returns.name if returns.name is not None else "returns"
        return returns.to_frame(name=name)
    
    if isinstance(returns, pd.DataFrame):
        #ensure we are working on a copy to avoid side effects
        return returns.copy()
    
    raise KPIError(f"Returns must be a pandas Series or DataFrame, got {type(returns).__name__}.")


def equity_curve_from_returns(returns: pd.DataFrame, start_value: float = 1.0) -> pd.DataFrame:
    r = returns.fillna(0.0)
    #prepend a row of 0s to represent the starting state (T-1)
    initial_row = pd.DataFrame(0.0, index=[r.index[0] - pd.Timedelta(days=1)], columns=r.columns)
    r_extended = pd.concat([initial_row, r])
    
    eq = (1.0 + r_extended).cumprod() * start_value
    return eq


def equity_curve_from_log_returns(log_returns: pd.DataFrame, start_value: float = 1.0) -> pd.DataFrame:
    """
    it builds an equity curve from log returns using exponential of cumulative sum (exp(cumsum))
    """
    r = log_returns.fillna(0.0)
    eq = np.exp(r.cumsum()) * start_value
    return eq


def cagr_from_equity(equity: pd.DataFrame, annualization_factor: int) -> pd.Series:
    """
    CAGR (Compound Annual Growth Rate) from equity curve using observation count
    years = n_periods / annualization_factor
    """
    n = len(equity)
    if n < 2:
        #the check happens earlier in cleaner.py, but good failsafe
        raise KPIError("Need at least 2 observations for CAGR.")

    years = n / float(annualization_factor)
    start = equity.iloc[0]
    end = equity.iloc[-1]
   
    #guard against division by zero in the start value (for example: if start_value was 0)
    if (start == 0).any():
         raise KPIError("Starting equity value must be positive to calculate CAGR.")

    cagr = (end / start) ** (1.0 / years) - 1.0
    return cagr



def annualized_volatility(returns: pd.DataFrame, annualization_factor: int) -> pd.Series:
    """we calculate the annualized standard deviation of the returns"""
    vol = returns.std(ddof=1) * np.sqrt(float(annualization_factor))
    return vol


def sharpe_ratio(
    returns: pd.DataFrame,
    annualization_factor: int,
    risk_free_rate_annual: float = 0.0,
) -> pd.Series:
    """
    Sharpe Ratio using periodic excess returns
    rf_per_period = (1 + rf)^(1/ann) - 1 (geometric conversion)
    """
    #we convert the annual risk free rate to per-period rate
    rf_per = (1.0 + risk_free_rate_annual) ** (1.0 / float(annualization_factor)) - 1.0
    excess_returns = returns - rf_per
   
    #the ratio is typically calculated using annualized mean excess return / annualized vol
    mean_excess_annual = excess_returns.mean() * float(annualization_factor)
    vol_annual = annualized_volatility(returns, annualization_factor)
   
    #avoid division by zero volatility
    sharpe = mean_excess_annual / vol_annual.replace(0, np.nan)
    return sharpe


def sortino_ratio(
    returns: pd.DataFrame,
    annualization_factor: int,
    risk_free_rate_annual: float = 0.0,
) -> pd.Series:
    """sortino ratio using downside deviation"""
    rf_per = (1.0 + risk_free_rate_annual) ** (1.0 / float(annualization_factor)) - 1.0
    excess_returns = returns - rf_per

    #only consider negative excess returns (downside)
    downside_returns = excess_returns.clip(upper=0.0)
   
    #calculate annualized downside deviation
    #keep in mind that we use len(downside_returns) - 1 if we want population std, but pandas handles sample std by default
    downside_dev_annual = downside_returns.std(ddof=1) * np.sqrt(float(annualization_factor))
   
    mean_excess_annual = excess_returns.mean() * float(annualization_factor)

    sortino = mean_excess_annual / downside_dev_annual.replace(0, np.nan)
    return sortino



def drawdown_series_from_equity(equity: pd.DataFrame) -> pd.DataFrame:
    """
    to calculate the drawdown series: (equity / running_max) - 1
    the resulting values will either be negative or zero
    """
    running_max = equity.cummax()
    dd = equity.div(running_max).sub(1.0)
    return dd



def max_drawdown(drawdowns: pd.DataFrame) -> pd.Series:
    """
    the max drawdown is the minimum (most negative) drawdown value to ever happen
    """
    return drawdowns.min()



def rolling_cumulative_return(returns: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    rolling cumulative return over a window (column-wise):
        (1+r).rolling(window).apply(prod) - 1
    """
    def _rolling_cumret(s: pd.Series) -> pd.Series:
        # raw=True => gets 1D numpy array for the series window
        return (1.0 + s).rolling(window=window).apply(np.prod, raw=True) - 1.0

    return returns.apply(_rolling_cumret, axis=0)




def rolling_volatility(returns: pd.DataFrame, window: int, annualization_factor: int) -> pd.DataFrame:
    """Calculates rolling annualized volatility."""
    return returns.rolling(window=window).std(ddof=1) * np.sqrt(float(annualization_factor))


def compute_headline_metrics(prices: pd.DataFrame, returns: pd.DataFrame, cfg: Config) -> dict:
    #portfolio period dates from config
    period_start = cfg.get("portfolio.return_period_start", default=None, type_=str)
    as_of = cfg.get("portfolio.as_of", default=None, type_=str)

    px = prices.copy()
    px.index = pd.to_datetime(px.index)

    if as_of:
        px = px.loc[:pd.to_datetime(as_of)]

    #simple "portfolio" (equal-weight for now unless weights provided) the weights will be used from positions later on
    portfolio_px = px.mean(axis=1)

    def total_return(series: pd.Series, start: Optional[pd.Timestamp]=None) -> float:
        s = series.dropna()
        if s.empty:
            return float("nan")
        if start is not None:
            s = s.loc[start:]
        if len(s) < 2:
            return float("nan")
        return (s.iloc[-1] / s.iloc[0]) - 1.0

    period_ret = total_return(portfolio_px, pd.to_datetime(period_start) if period_start else None)

    #YTD (from Jan 1 of last point year)
    if portfolio_px.dropna().empty:
        ytd = float("nan")
    else:
        last_date = portfolio_px.dropna().index[-1]
        ytd_start = pd.Timestamp(year=last_date.year, month=1, day=1)
        ytd = total_return(portfolio_px, ytd_start)

    ann_factor = int(cfg.get("kpis.annualization_factor", default=252) or 252)
    port_ret = returns.mean(axis=1).dropna()
    ann_vol = float(port_ret.std() * (ann_factor ** 0.5)) if not port_ret.empty else float("nan")

    #max drawdown on the portfolio series
    cum = (1.0 + port_ret).cumprod() if not port_ret.empty else pd.Series(dtype=float)
    peak = cum.cummax()
    dd = (cum / peak) - 1.0 if not cum.empty else pd.Series(dtype=float)
    max_dd = float(dd.min()) if not dd.empty else float("nan")

    return {
        "period_return": period_ret,
        "ytd_return": ytd,
        "annualized_vol": ann_vol,
        "max_drawdown": max_dd,
    }



def compute_kpis(returns: ArrayLike, cfg: Config) -> KPIResult:
    """
    the main KPI entrypoint.it reads:
     * kpis.annualization_factor (default 252)
     * kpis.risk_free_rate (default 0.0)
     * kpis.rolling_windows (default [21,63,126,252])
     * cleaning.return_type (simple/log)

    args:
     * returns: Series/DataFrame of returns.
     * cfg: Config object
    """
    r = _to_dataframe(returns).dropna(how="all")


    #determine if returns are simple or log from the config file
    return_type_config = cast(ReturnType, cfg.get("cleaning.return_type", default="log", type_=str))


    annualization_factor = cfg.get("kpis.annualization_factor", default=252, type_=int)
    rf = cfg.get("kpis.risk_free_rate", default=0.0, type_=float)


    rolling_windows = cfg.get("kpis.rolling_windows", default=[21, 63, 126, 252], type_=list)
    #basic validation for rolling windows
    if not all(isinstance(w, int) and w > 1 for w in rolling_windows):
        raise KPIError("kpis.rolling_windows must be a list of integers greater than 1.")


    #equity curve + drawdowns logic based on return type from config
    if return_type_config == "log":
        equity = equity_curve_from_log_returns(r, start_value=1.0)

    elif return_type_config == "simple":
        equity = equity_curve_from_returns(r, start_value=1.0)
    
    else:
        #it should be unreachable due to Type checking and Config validation, but failsafe
        raise KPIError(f"Unknown return_type in config: {return_type_config}")

    dds = drawdown_series_from_equity(equity)
   
    cagr = cagr_from_equity(equity, annualization_factor=annualization_factor)
    vol = annualized_volatility(r, annualization_factor=annualization_factor)
    sharpe = sharpe_ratio(r, annualization_factor=annualization_factor, risk_free_rate_annual=rf)
    sortino = sortino_ratio(r, annualization_factor=annualization_factor, risk_free_rate_annual=rf)
    mdd = max_drawdown(dds)

    #compile summary DataFrame
    summary = pd.DataFrame(
        {
            "CAGR (%)": cagr * 100, #display as percentage for reporting
            "Ann.Vol (%)": vol * 100,
            "Sharpe Ratio": sharpe,
            "Sortino Ratio": sortino,
            "Max.Drawdown (%)": mdd * 100,
        }
    )
    summary.index.name = "Asset"
    #format the numbers properly for the report
    summary = summary.round(2)

    #the olling metrics
    roll_rets: Dict[int, pd.DataFrame] = {}
    roll_vols: Dict[int, pd.DataFrame] = {}
   
    #we check the report config to see if the rolling metrics should be included
    if cfg.get("report.include_rolling", default=True, type_=bool):
        for w in rolling_windows:
            roll_rets[w] = rolling_cumulative_return(r, window=w)
            roll_vols[w] = rolling_volatility(r, window=w, annualization_factor=annualization_factor)


    return KPIResult(
        summary=summary,
        drawdowns=dds,
        rolling_returns=roll_rets,
        rolling_volatility=roll_vols,
    )
