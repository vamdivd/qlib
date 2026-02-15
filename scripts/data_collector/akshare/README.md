# Collect CN A-Share Data via AkShare

> AkShare is an open-source financial data interface library, providing free access to Chinese A-share stock data without authentication.

## Requirements

```bash
pip install -r requirements.txt
```

## Collector Data

```bash
# Download all A-share daily data
python collector.py download_data --source_dir ~/.qlib/stock_data/source --start 2020-01-01 --end 2024-01-01 --delay 0.5

# Debug: download only 10 stocks
python collector.py download_data --source_dir ~/.qlib/stock_data/source --start 2020-01-01 --end 2024-01-01 --limit_nums 10
```

## Normalize Data

```bash
python collector.py normalize_data --source_dir ~/.qlib/stock_data/source --normalize_dir ~/.qlib/stock_data/normalize
```

## Full Pipeline (Download + Normalize)

```bash
python collector.py download_and_normalize --source_dir ~/.qlib/stock_data/source --normalize_dir ~/.qlib/stock_data/normalize --start 2020-01-01 --end 2024-01-01 --delay 0.5
```

## Convert to Qlib Binary Format

After downloading and normalizing, use `dump_bin.py` to convert data into Qlib binary format:

```bash
cd qlib/scripts
python dump_bin.py dump_all \
    --data_path ~/.qlib/stock_data/normalize \
    --qlib_dir ~/.qlib/qlib_data/cn_data \
    --freq day \
    --date_field_name date \
    --symbol_field_name symbol \
    --include_fields open,close,high,low,volume,amount,change
```

## Notes

- Default data source: `akshare.stock_zh_a_hist` (forward-adjusted daily OHLCV)
- Index data (CSI300/CSI500/CSI100) is automatically downloaded via `akshare.stock_zh_index_daily`
- AkShare has rate limiting; set `--delay 0.5` or higher to avoid being blocked
- Symbol format: AkShare uses 6-digit codes (e.g. `600519`), normalized to qlib format (e.g. `sh600519`)
