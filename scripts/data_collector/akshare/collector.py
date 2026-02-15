# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import abc
import sys
import copy
import time
import datetime
import functools
from abc import ABC
from pathlib import Path
from typing import Iterable, List

import fire
import numpy as np
import pandas as pd
from loguru import logger

CUR_DIR = Path(__file__).resolve().parent
sys.path.append(str(CUR_DIR.parent.parent))

from data_collector.base import BaseCollector, BaseNormalize, BaseRun, Normalize


# ---- Inlined utilities (avoid importing data_collector.utils which requires yahooquery) ----


def deco_retry(retry: int = 5, retry_sleep: int = 3):
    """Retry decorator: retries the wrapped function up to `retry` times with `retry_sleep` seconds between attempts."""

    def deco_func(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            _retry = 5 if callable(retry) else retry
            _result = None
            for _i in range(1, _retry + 1):
                try:
                    _result = func(*args, **kwargs)
                    break
                except Exception as e:
                    logger.warning(f"{func.__name__}: {_i} :{e}")
                    if _i == _retry:
                        raise
                time.sleep(retry_sleep)
            return _result

        return wrapper

    return deco_func(retry) if callable(retry) else deco_func


_CALENDAR_MAP = {}


def get_calendar_list(bench_code="ALL") -> List[pd.Timestamp]:
    """Get A-share trading calendar list using AkShare.

    Parameters
    ----------
    bench_code: str
        Only "ALL" is supported in this standalone version.

    Returns
    -------
    list of pd.Timestamp
    """
    import akshare as ak

    calendar = _CALENDAR_MAP.get(bench_code, None)
    if calendar is None:
        logger.info(f"get calendar list: {bench_code}......")
        trade_date_df = ak.tool_trade_date_hist_sina()
        trade_date_list = trade_date_df["trade_date"].tolist()
        trade_date_list = [pd.Timestamp(d) for d in trade_date_list]
        dates = pd.DatetimeIndex(trade_date_list)
        filtered_dates = dates[(dates >= "2000-01-04") & (dates <= pd.Timestamp.today().normalize())]
        calendar = filtered_dates.tolist()
        _CALENDAR_MAP[bench_code] = calendar
        logger.info(f"end of get calendar list: {bench_code}, total {len(calendar)} trading days.")
    return calendar


class AkShareCollector(BaseCollector):
    """Base collector using AkShare to fetch A-share (CN) stock data.

    AkShare is an open-source financial data interface library that provides
    convenient access to various Chinese financial data sources.
    """

    def __init__(
        self,
        save_dir: [str, Path],
        start=None,
        end=None,
        interval="1d",
        max_workers=1,
        max_collector_count=2,
        delay=0.5,
        check_data_length: int = None,
        limit_nums: int = None,
    ):
        """
        Parameters
        ----------
        save_dir: str
            stock save dir
        max_workers: int
            workers, default 1
        max_collector_count: int
            default 2
        delay: float
            time.sleep(delay), default 0.5
        interval: str
            freq, value from [1d], default 1d
        start: str
            start datetime, default None
        end: str
            end datetime, default None
        check_data_length: int
            check data length, by default None
        limit_nums: int
            using for debug, by default None
        """
        super(AkShareCollector, self).__init__(
            save_dir=save_dir,
            start=start,
            end=end,
            interval=interval,
            max_workers=max_workers,
            max_collector_count=max_collector_count,
            delay=delay,
            check_data_length=check_data_length,
            limit_nums=limit_nums,
        )

    def get_instrument_list(self):
        """Get A-share stock list via AkShare.

        Uses stock_info_sh_name_code + stock_info_sz_name_code for stability,
        with optional Beijing exchange stocks as fallback.
        """
        import akshare as ak

        logger.info("get A-share stock symbols via AkShare......")
        symbols = []

        # Shanghai (6xxxxx)
        try:
            sh_df = ak.stock_info_sh_name_code(symbol="主板A股")
            symbols.extend(sh_df["证券代码"].tolist())
            logger.info(f"got {len(sh_df)} Shanghai main board symbols.")
        except Exception as e:
            logger.warning(f"failed to get SH main board stocks: {e}")

        try:
            sh_star_df = ak.stock_info_sh_name_code(symbol="科创板")
            symbols.extend(sh_star_df["证券代码"].tolist())
            logger.info(f"got {len(sh_star_df)} Shanghai STAR board symbols.")
        except Exception as e:
            logger.warning(f"failed to get SH STAR board stocks: {e}")

        # Shenzhen (0xxxxx, 3xxxxx)
        try:
            sz_df = ak.stock_info_sz_name_code(symbol="A股列表")
            symbols.extend(sz_df["A股代码"].tolist())
            logger.info(f"got {len(sz_df)} Shenzhen symbols.")
        except Exception as e:
            logger.warning(f"failed to get SZ stocks: {e}")

        # Beijing (4xxxxx, 8xxxxx) - optional, may fail
        try:
            bj_df = ak.stock_info_bj_name_code()
            symbols.extend(bj_df["证券代码"].tolist())
            logger.info(f"got {len(bj_df)} Beijing symbols.")
        except Exception as e:
            logger.warning(f"failed to get BJ stocks (skipping): {e}")

        # Fallback: if all above failed, try the combined function
        if not symbols:
            logger.warning("all individual exchange queries failed, trying stock_info_a_code_name...")
            df = ak.stock_info_a_code_name()
            symbols = df["code"].tolist()

        logger.info(f"get {len(symbols)} symbols in total.")
        return symbols

    def normalize_symbol(self, symbol: str):
        """Normalize symbol to qlib format: sh600519 / sz000001.

        AkShare uses raw 6-digit code. We add sh/sz prefix based on the exchange rules:
          - codes starting with 6 -> sh (Shanghai)
          - codes starting with 0, 3 -> sz (Shenzhen)
          - codes starting with 4, 8 -> bj (Beijing)
        """
        symbol = str(symbol).strip()
        if symbol.startswith("6"):
            return f"sh{symbol}"
        elif symbol.startswith(("0", "3")):
            return f"sz{symbol}"
        elif symbol.startswith(("4", "8")):
            return f"bj{symbol}"
        else:
            return f"sh{symbol}"

    @deco_retry(retry=3, retry_sleep=5)
    def get_data(
        self,
        symbol: str,
        interval: str,
        start_datetime: pd.Timestamp,
        end_datetime: pd.Timestamp,
    ) -> pd.DataFrame:
        """Fetch OHLCV data for a single symbol using AkShare.

        Parameters
        ----------
        symbol: str
            6-digit A-share stock code, e.g. '600519'
        interval: str
            '1d' for daily
        start_datetime: pd.Timestamp
        end_datetime: pd.Timestamp

        Returns
        -------
        pd.DataFrame
            columns: [date, symbol, open, close, high, low, volume, amount]
        """
        import akshare as ak

        _start = start_datetime.strftime("%Y%m%d")
        _end = end_datetime.strftime("%Y%m%d")

        if interval == self.INTERVAL_1d:
            try:
                # stock_zh_a_hist: daily OHLCV data for A-share stocks (post-restoration)
                df = ak.stock_zh_a_hist(
                    symbol=symbol,
                    period="daily",
                    start_date=_start,
                    end_date=_end,
                    adjust="qfq",  # forward-adjusted price
                )
            except Exception as e:
                logger.warning(f"get data error: {symbol}: {e}")
                return pd.DataFrame()
        else:
            raise ValueError(f"akshare collector does not support interval={interval} currently")

        if df is None or df.empty:
            return pd.DataFrame()

        # AkShare stock_zh_a_hist columns (Chinese):
        # 日期, 开盘, 收盘, 最高, 最低, 成交量, 成交额, 振幅, 涨跌幅, 涨跌额, 换手率
        df = df.rename(
            columns={
                "日期": "date",
                "开盘": "open",
                "收盘": "close",
                "最高": "high",
                "最低": "low",
                "成交量": "volume",
                "成交额": "amount",
                "涨跌幅": "change_pct",
                "换手率": "turnover",
            }
        )

        # Keep only the standard OHLCV columns needed by qlib
        keep_cols = ["date", "open", "close", "high", "low", "volume", "amount"]
        keep_cols = [c for c in keep_cols if c in df.columns]
        df = df[keep_cols]

        df["date"] = pd.to_datetime(df["date"])
        df["symbol"] = symbol
        return df


class AkShareCollectorCN1d(AkShareCollector):
    """Collect A-share daily data and also download index benchmark data."""

    def collector_data(self):
        super(AkShareCollectorCN1d, self).collector_data()
        self._download_index_data()

    def _download_index_data(self):
        """Download CSI300 / CSI500 / CSI100 index data via AkShare."""
        import akshare as ak

        _start = self.start_datetime.strftime("%Y%m%d")
        _end = self.end_datetime.strftime("%Y%m%d")

        index_map = {
            "sh000300": "000300",  # CSI300
            "sh000905": "000905",  # CSI500
            "sh000903": "000903",  # CSI100
        }
        for qlib_name, index_code in index_map.items():
            logger.info(f"get bench data: {qlib_name}({index_code})......")
            try:
                df = ak.stock_zh_index_daily(symbol=f"sh{index_code}")
                if df is None or df.empty:
                    logger.warning(f"empty index data for {qlib_name}")
                    continue

                df = df.rename(
                    columns={
                        "date": "date",
                        "open": "open",
                        "close": "close",
                        "high": "high",
                        "low": "low",
                        "volume": "volume",
                    }
                )
                df["date"] = pd.to_datetime(df["date"])
                # Filter by date range
                df = df[(df["date"] >= pd.Timestamp(_start)) & (df["date"] <= pd.Timestamp(_end))]
                df["symbol"] = qlib_name

                keep_cols = ["date", "open", "close", "high", "low", "volume", "symbol"]
                keep_cols = [c for c in keep_cols if c in df.columns]
                df = df[keep_cols]

                _path = self.save_dir.joinpath(f"{qlib_name}.csv")
                if _path.exists():
                    _old_df = pd.read_csv(_path)
                    df = pd.concat([_old_df, df], sort=False)
                df.to_csv(_path, index=False)
                logger.info(f"saved index {qlib_name}, {len(df)} rows.")
            except Exception as e:
                logger.warning(f"get {qlib_name} error: {e}")
                continue


class AkShareNormalize(BaseNormalize):
    """Normalize raw AkShare data to the format expected by Qlib.

    Steps:
      1. Calendar alignment (reindex to trading calendar)
      2. Handle zero/negative volume (set price fields to NaN)
      3. Compute 'change' = close / prev_close - 1
    """

    COLUMNS = ["open", "close", "high", "low", "volume"]
    DAILY_FORMAT = "%Y-%m-%d"

    @staticmethod
    def calc_change(df: pd.DataFrame, last_close: float = None) -> pd.Series:
        """Calculate daily change ratio: close / prev_close - 1."""
        df = df.copy()
        _tmp_series = df["close"].ffill()
        _tmp_shift_series = _tmp_series.shift(1)
        if last_close is not None:
            _tmp_shift_series.iloc[0] = float(last_close)
        change_series = _tmp_series / _tmp_shift_series - 1
        return change_series

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize a single symbol's DataFrame.

        Parameters
        ----------
        df: pd.DataFrame
            Raw data from AkShare collector, must contain 'date', 'symbol', 'open', 'close', 'high', 'low', 'volume'.

        Returns
        -------
        pd.DataFrame
            Normalized DataFrame with calendar-aligned dates and 'change' column.
        """
        if df.empty:
            return df

        symbol = df.loc[df[self._symbol_field_name].first_valid_index(), self._symbol_field_name]
        columns = copy.deepcopy(self.COLUMNS)

        df = df.copy()
        df.set_index(self._date_field_name, inplace=True)
        df.index = pd.to_datetime(df.index)
        # remove timezone if present
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        # remove duplicate dates
        df = df[~df.index.duplicated(keep="first")]

        # reindex to trading calendar
        if self._calendar_list is not None:
            df = df.reindex(
                pd.DataFrame(index=self._calendar_list)
                .loc[
                    pd.Timestamp(df.index.min()).date() : pd.Timestamp(df.index.max()).date()
                    + pd.Timedelta(hours=23, minutes=59)
                ]
                .index
            )
        df.sort_index(inplace=True)

        # set price fields to NaN where volume is 0 or NaN
        df.loc[
            (df["volume"] <= 0) | np.isnan(df["volume"]),
            list(set(df.columns) - {self._symbol_field_name}),
        ] = np.nan

        # compute change
        df["change"] = self.calc_change(df)
        columns += ["change"]
        df.loc[
            (df["volume"] <= 0) | np.isnan(df["volume"]),
            columns,
        ] = np.nan

        df[self._symbol_field_name] = symbol
        df.index.names = [self._date_field_name]
        return df.reset_index()

    def _get_calendar_list(self) -> Iterable[pd.Timestamp]:
        """Get CN A-share trading calendar."""
        return get_calendar_list("ALL")


class AkShareNormalize1d(AkShareNormalize):
    """Normalize daily AkShare data.

    Since AkShare already provides forward-adjusted (qfq) prices,
    we normalize all prices relative to the first day's close (same as Yahoo 1d).
    This makes the data suitable for Qlib's alpha factor computation.
    """

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super(AkShareNormalize1d, self).normalize(df)
        if df.empty:
            return df
        df = self._manual_adj_data(df)
        return df

    def _get_first_close(self, df: pd.DataFrame) -> float:
        """Get first non-NaN close value."""
        df = df.loc[df["close"].first_valid_index() :]
        return df["close"].iloc[0]

    def _manual_adj_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize all fields according to the close of the first day.

        - price fields: price / first_close
        - volume: volume * first_close
        - change: unchanged
        """
        if df.empty:
            return df
        df = df.copy()
        df.sort_values(self._date_field_name, inplace=True)
        df.set_index(self._date_field_name, inplace=True)
        _close = self._get_first_close(df)
        for _col in df.columns:
            if _col in [self._symbol_field_name, "change"]:
                continue
            if _col == "volume":
                df[_col] = df[_col] * _close
            elif _col == "amount":
                # amount doesn't need normalization, keep as is or drop
                continue
            else:
                df[_col] = df[_col] / _close
        return df.reset_index()


class Run(BaseRun):
    """CLI runner for AkShare A-share data collection.

    Usage
    -----
        # Download daily data
        $ python collector.py download_data --source_dir ~/.qlib/stock_data/source --start 2020-01-01 --end 2024-01-01 --delay 0.5

        # Normalize data
        $ python collector.py normalize_data --source_dir ~/.qlib/stock_data/source --normalize_dir ~/.qlib/stock_data/normalize

        # Full pipeline: download + normalize
        $ python collector.py download_and_normalize --source_dir ~/.qlib/stock_data/source --normalize_dir ~/.qlib/stock_data/normalize --start 2020-01-01 --end 2024-01-01
    """

    def __init__(self, source_dir=None, normalize_dir=None, max_workers=1, interval="1d"):
        """
        Parameters
        ----------
        source_dir: str
            The directory where the raw data collected from the Internet is saved,
            default "Path(__file__).parent/source"
        normalize_dir: str
            Directory for normalize data,
            default "Path(__file__).parent/normalize"
        max_workers: int
            Concurrent number, default is 1
        interval: str
            freq, value from [1d], default 1d
        """
        super().__init__(source_dir, normalize_dir, max_workers, interval)

    @property
    def collector_class_name(self):
        return "AkShareCollectorCN1d"

    @property
    def normalize_class_name(self):
        return "AkShareNormalize1d"

    @property
    def default_base_dir(self) -> [Path, str]:
        return CUR_DIR

    def download_data(
        self,
        max_collector_count=2,
        delay=0.5,
        start=None,
        end=None,
        check_data_length: int = None,
        limit_nums=None,
    ):
        """Download A-share data from AkShare.

        Parameters
        ----------
        max_collector_count: int
            default 2
        delay: float
            time.sleep(delay), default 0.5
        start: str
            start datetime, default "2000-01-01"; closed interval(including start)
        end: str
            end datetime, default tomorrow; open interval(excluding end)
        check_data_length: int
            check data length, by default None
        limit_nums: int
            using for debug, by default None

        Examples
        --------
            $ python collector.py download_data --source_dir ~/.qlib/stock_data/source --start 2020-01-01 --end 2024-01-01 --delay 0.5 --interval 1d
        """
        super(Run, self).download_data(max_collector_count, delay, start, end, check_data_length, limit_nums)

    def normalize_data(
        self,
        date_field_name: str = "date",
        symbol_field_name: str = "symbol",
        end_date: str = None,
        **kwargs,
    ):
        """Normalize downloaded data.

        Parameters
        ----------
        date_field_name: str
            date field name, default date
        symbol_field_name: str
            symbol field name, default symbol
        end_date: str
            if not None, data after end_date will be removed

        Examples
        --------
            $ python collector.py normalize_data --source_dir ~/.qlib/stock_data/source --normalize_dir ~/.qlib/stock_data/normalize --interval 1d
        """
        if end_date is not None:
            kwargs["end_date"] = end_date
        super(Run, self).normalize_data(date_field_name, symbol_field_name, **kwargs)

    def download_and_normalize(
        self,
        max_collector_count=2,
        delay=0.5,
        start=None,
        end=None,
        check_data_length: int = None,
        limit_nums=None,
        date_field_name: str = "date",
        symbol_field_name: str = "symbol",
        end_date: str = None,
    ):
        """Full pipeline: download data then normalize.

        Parameters
        ----------
        max_collector_count: int
            default 2
        delay: float
            time.sleep(delay), default 0.5
        start: str
            start datetime, default "2000-01-01"
        end: str
            end datetime, default tomorrow
        check_data_length: int
            check data length, by default None
        limit_nums: int
            using for debug, by default None
        date_field_name: str
            date field name, default date
        symbol_field_name: str
            symbol field name, default symbol
        end_date: str
            if not None, data after end_date will be removed

        Examples
        --------
            $ python collector.py download_and_normalize --source_dir ~/.qlib/stock_data/source --normalize_dir ~/.qlib/stock_data/normalize --start 2020-01-01 --end 2024-01-01 --delay 0.5
        """
        self.download_data(
            max_collector_count=max_collector_count,
            delay=delay,
            start=start,
            end=end,
            check_data_length=check_data_length,
            limit_nums=limit_nums,
        )
        self.normalize_data(
            date_field_name=date_field_name,
            symbol_field_name=symbol_field_name,
            end_date=end_date,
        )


if __name__ == "__main__":
    fire.Fire(Run)
