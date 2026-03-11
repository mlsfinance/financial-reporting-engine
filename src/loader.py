"""
data loading utilities. it supports CSV, Excel and Yahoo Finance (via yfinance)

responsibilities:
    * loading raw price data into a standardized pandas DataFrame
    * unifying column names and index
    * basic validations (dates, duplicates, missing critical columns)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Literal, Any

import pandas as pd

from .config import Config
from .messages import RunMessages

def _warn(msgs: Any, text: str) -> None:
    if msgs is None:
        return
    # RunMessages
    if hasattr(msgs, "warn") and callable(getattr(msgs, "warn")):
        msgs.warn(text)
        return
    # list[str]
    if isinstance(msgs, list):
        msgs.append(text)
        return


class DataLoaderError(Exception):
    """raised when loading fails or data is invalid"""


SourceType = Literal["csv", "excel", "yfinance", "positions_excel"]


@dataclass
class LoadedData:
    prices: pd.DataFrame
    price_column: str
    weights: Optional[pd.Series] = None


def _validate_datetime_index(df: pd.DataFrame) -> None:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise DataLoaderError("DataFrame index must be a DatetimeIndex after loading.")
    if df.index.hasnans:
        raise DataLoaderError("DatetimeIndex contains NaT values. Check date parsing.")


def _validate_price_column(df: pd.DataFrame, price_column: str) -> None:
    if price_column not in df.columns:
        raise DataLoaderError(
            f"Missing price column '{price_column}'. Available: {list(df.columns)}"
        )
    if not pd.api.types.is_numeric_dtype(df[price_column]):
        raise DataLoaderError(f"Price column '{price_column}' must be numeric.")


def _basic_validations(
    df: pd.DataFrame,
    drop_duplicates: bool = True,
    sort_by_date: bool = True,
) -> pd.DataFrame:
    if sort_by_date:
        df = df.sort_index()

    if drop_duplicates:
        #keep the last occurrence for duplicate timestamps
        df = df[~df.index.duplicated(keep="last")]

    _validate_datetime_index(df)
    return df


def load_from_csv(path: str | Path, date_column: str, price_column: str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise DataLoaderError(f"CSV file not found: {path}")

    df = pd.read_csv(path)

    if date_column not in df.columns:
        raise DataLoaderError(f"CSV missing date column '{date_column}'.")

    df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
    df = df.set_index(date_column)

    #keep only needed columns if present, if not then keep all and validate later
    if price_column in df.columns:
        df = df[[price_column]].copy()

    return df


def load_from_excel(
    path: str | Path,
    sheet_name: Optional[str],
    date_column: str,
    price_column: str,
) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise DataLoaderError(f"Excel file not found: {path}")

    df = pd.read_excel(path, sheet_name=sheet_name)

    if date_column not in df.columns:
        raise DataLoaderError(f"Excel missing date column '{date_column}'.")

    df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
    df = df.set_index(date_column)

    if price_column in df.columns:
        df = df[[price_column]].copy()

    return df


def load_positions_excel(
    path: str | Path,
    sheet_name: Optional[str] = "positions",
    msgs: RunMessages | None = None,
) -> pd.Series:
    path = Path(path)
    if not path.exists():
        raise DataLoaderError(f"Positions Excel file not found: {path}")

    df = pd.read_excel(path, sheet_name=sheet_name)

    required = {"Asset", "Weight"}
    missing = required - set(df.columns)
    if missing:
        raise DataLoaderError(f"Positions template missing columns: {sorted(missing)}")

    df = df[["Asset", "Weight"]].copy()
    df["Asset"] = df["Asset"].astype(str).str.strip()
    df = df[df["Asset"] != ""]

    df["Weight"] = pd.to_numeric(df["Weight"], errors="coerce")
    df = df.dropna(subset=["Weight"])

    if df.empty:
        raise DataLoaderError("No valid rows found in positions sheet (Asset/Weight).")

    #aggregate duplicates
    df = df.groupby("Asset", as_index=False)["Weight"].sum()

    w = df.set_index("Asset")["Weight"].astype(float)
    total = float(w.sum())
    if total <= 0:
        raise DataLoaderError("Weights sum must be > 0.")

    #percent-like to decimals
    if total > 1.5:
        if msgs is not None:
            _warn(msgs,
                f"Weights look like percentages (sum={total:.4f}). "
                "Converted to decimals by dividing by 100."
            )
        w = w / 100.0

    #   normalize to sum to 1
    s = float(w.sum())
    if s <= 0:
        raise DataLoaderError("Weights sum must be > 0 after cleaning.")
    if abs(s - 1.0) > 1e-6:
        if msgs is not None:
            _warn(msgs, f"Weights were normalized to sum to 1.00 (original sum={s:.6f}).")
        w = w / s

    return w



def _get_ticker_currency(ticker: str) -> Optional[str]:
    
    #tries to detect ticker currency via yfinance. fast_info is usually quicker, but if it fails, the info section has the details we need
    
    import yfinance as yf

    try:
        t = yf.Ticker(ticker)
        cur = None
        if hasattr(t, "fast_info") and t.fast_info:
            cur = t.fast_info.get("currency")
        if not cur:
            info = t.info or {}
            cur = info.get("currency")
        return cur
    except Exception:
        return None


def _fx_pair(primary: str, secondary: str) -> str:
    
    #build a Yahoo FX ticker like 'EURUSD=X'.
    return f"{primary}{secondary}=X"


def load_from_yfinance(
    tickers: list[str],
    start_date: Optional[str],
    end_date: Optional[str],
    price_column: str,
    base_currency: Optional[str] = None,
    fx_enabled: bool = False,
    fx_on_missing: str = "error",  #must use either "error" or "skip"
) -> pd.DataFrame:
    
    try:
        import yfinance as yf
    except ImportError as exc:
        raise DataLoaderError(
            "yfinance is not installed. Add it to requirements.txt or install it."
        ) from exc

    if not tickers:
        raise DataLoaderError("At least one ticker is required for yfinance source_type.")

    #1) download raw prices
    tickers_arg = tickers if len(tickers) > 1 else tickers[0]
    raw = yf.download(
        tickers=tickers_arg,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
    )

    if raw is None or raw.empty:
        raise DataLoaderError("No data returned from yfinance.")

    #normalize to DataFrame with columns = tickers
    if isinstance(raw.columns, pd.MultiIndex):
        if price_column not in raw.columns.levels[0]:
            raise DataLoaderError(
                f"Requested price_column '{price_column}' not found from yfinance. "
                f"Available fields: {list(raw.columns.levels[0])}"
            )
        prices = raw[price_column].copy()  #cols = tickers
    else:
        if price_column not in raw.columns:
            raise DataLoaderError(
                f"Requested price_column '{price_column}' not found from yfinance. "
                f"Available: {list(raw.columns)}"
            )
        prices = raw[[price_column]].copy()
        prices.columns = [tickers[0]]

    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()

    for c in prices.columns:
        prices[c] = pd.to_numeric(prices[c], errors="coerce")

    if prices.dropna(how="all").empty:
        raise DataLoaderError("All price series are empty after numeric conversion.")

    #2) FX conversion
    if fx_enabled:
        if not base_currency:
            raise DataLoaderError("fx.enabled=true requires data.base_currency (e.g., 'USD').")

        base_currency = str(base_currency).upper().strip()

        #detect currencies
        ticker_currency: dict[str, Optional[str]] = {t: _get_ticker_currency(t) for t in prices.columns}

        unknown = [t for t, cur in ticker_currency.items() if not cur]
        if unknown:
            msg = f"Could not detect currency for: {unknown}."
            if fx_on_missing == "skip":
                #leave as-is (but then currencies would be mixed)
                pass
            else:
                raise DataLoaderError(msg)

        #group tickers by currency
        by_cur: dict[str, list[str]] = {}
        for t, cur in ticker_currency.items():
            if not cur:
                continue
            cur = str(cur).upper().strip()
            by_cur.setdefault(cur, []).append(t)

        #if multiple currencies and base conversion are requested, convert non-base
        for cur, cols in by_cur.items():
            if cur == base_currency:
                continue

            #try direct pair CURBASE, if not then inverse BASECUR
            direct = _fx_pair(cur, base_currency)      #for example EURUSD=X
            inverse = _fx_pair(base_currency, cur)     #for example USDEUR=X

            fx = yf.download(
                tickers=[direct, inverse],
                start=start_date,
                end=end_date,
                auto_adjust=True,
                progress=False,
            )

            if fx is None or fx.empty:
                msg = f"FX download failed for currency {cur} -> {base_currency}."
                if fx_on_missing == "skip":
                    continue
                raise DataLoaderError(msg)

            #extract a usable FX series
            fx_series = None
            fx_mode = None  # "multiply" or "divide"

            if isinstance(fx.columns, pd.MultiIndex):
                #fields x tickers
                if "Close" in fx.columns.levels[0]:
                    close = fx["Close"]
                else:
                    #fallback: take first level-0 field available
                    close = fx[fx.columns.levels[0][0]]
            else:
                #single series case
                close = fx

            #choose direct if present
            if direct in close.columns:
                fx_series = close[direct]
                fx_mode = "multiply"
            elif inverse in close.columns:
                fx_series = close[inverse]
                fx_mode = "divide"

            if fx_series is None or fx_series.dropna().empty:
                msg = f"FX pair not available for {cur}->{base_currency} (tried {direct} and {inverse})."
                if fx_on_missing == "skip":
                    continue
                raise DataLoaderError(msg)

            fx_series.index = pd.to_datetime(fx_series.index)
            fx_series = fx_series.sort_index()

            #allign FX to price calendar (forward-fill across holidays)
            fx_series = fx_series.reindex(prices.index).ffill()

            if fx_series.dropna().empty:
                msg = f"FX series empty after alignment for {cur}->{base_currency}."
                if fx_on_missing == "skip":
                    continue
                raise DataLoaderError(msg)

            #apply conversion
            if fx_mode == "multiply":
                prices[cols] = prices[cols].multiply(fx_series, axis=0)
            else:
                prices[cols] = prices[cols].divide(fx_series, axis=0)

    return prices



def load_prices(cfg: Config, msgs: RunMessages | None = None) -> LoadedData:

    weights: Optional[pd.Series] = None

    source_type: SourceType = cfg.get("data.source_type", default="csv", type_=str)
    date_column = cfg.get("data.date_column", default="Date", type_=str) or "Date"
    price_column = cfg.get("data.price_column", default="Close", type_=str) or "Close"


    drop_duplicates = bool(cfg.get("cleaning.drop_duplicates", default=True))
    sort_by_date = bool(cfg.get("cleaning.sort_by_date", default=True))


    if source_type == "csv":
        input_path = cfg.require("data.input_path", str)
        df = load_from_csv(input_path, date_column=date_column, price_column=price_column)


    elif source_type == "excel":
        input_path = cfg.require("data.input_path", str)
        sheet_name = cfg.get("data.sheet_name", default=None, type_=str)
        df = load_from_excel(
            input_path,
            sheet_name=sheet_name,
            date_column=date_column,
            price_column=price_column,
        )


    elif source_type == "yfinance":
        tickers = cfg.get("data.tickers", default=None)
        if not tickers:
            raise DataLoaderError("Missing 'data.tickers' in configuration.")
        if isinstance(tickers, str):
            tickers = [tickers]

        start_date = cfg.get("data.start_date", default=None, type_=str)
        end_date = cfg.get("data.end_date", default=None, type_=str)

        base_currency = cfg.get("data.base_currency", default=None, type_=str)
        fx_enabled = bool(cfg.get("data.fx.enabled", default=False))
        fx_on_missing = cfg.get("data.fx.on_missing", default="error", type_=str) or "error"

        df = load_from_yfinance(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            price_column=price_column,
            base_currency=base_currency,
            fx_enabled=fx_enabled,
            fx_on_missing=fx_on_missing,
        )


    elif source_type == "positions_excel":
        input_path = cfg.require("data.input_path", str)
        sheet_name = cfg.get("data.sheet_name", default="positions", type_=str)

        weights = load_positions_excel(input_path, sheet_name=sheet_name, msgs=msgs)
        tickers = list(weights.index)

        start_date = cfg.get("data.start_date", default=None, type_=str)
        end_date = cfg.get("data.end_date", default=None, type_=str)

        base_currency = cfg.get("data.base_currency", default=None, type_=str)
        fx_enabled = bool(cfg.get("data.fx.enabled", default=False))
        fx_on_missing = cfg.get("data.fx.on_missing", default="error", type_=str) or "error"

        df = load_from_yfinance(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            price_column=price_column,
            base_currency=base_currency,
            fx_enabled=fx_enabled,
            fx_on_missing=fx_on_missing,
        )


    else:
        raise DataLoaderError(f"Unknown data.source_type: '{source_type}'")


    df = _basic_validations(df, drop_duplicates=drop_duplicates, sort_by_date=sort_by_date)


    #final validation: numeric prices
    if df.empty:
        raise DataLoaderError("Loaded price DataFrame is empty.")

    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise DataLoaderError(
                f"Price column '{col}' is not numeric after loading/cleaning."
            )



    return LoadedData(prices=df, price_column=price_column, weights=weights)
