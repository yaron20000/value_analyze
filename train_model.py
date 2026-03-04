#!/usr/bin/env python3
"""
Monthly Walk-Forward Outperformance Classifier with Portfolio Backtest
======================================================================
Trains a classification model to predict whether a stock will outperform
the market median next month. Uses predict_proba to produce a 0-1 score
for ranking stocks.

Each month:
1. Assemble features: latest available fundamentals (no look-ahead),
   month-end price features, trailing 12-month holdings
2. Train on all prior months (expanding window)
3. Predict outperformance probability for all stocks
4. Rank stocks by probability and pick top 20 / top decile
5. Measure actual 1-month portfolio return

Results are aggregated into yearly performance (compounded monthly returns).

Outputs:
- Excel debug files per month (model inputs + outputs only)
- Excel report per year (compounded monthly performance, portfolio picks)
- Summary report across all years (cumulative performance)

Usage:
    python train_model.py --db-password yourpass
    python train_model.py --db-password yourpass --min-train-months 36 --top-n 20
"""

import pandas as pd
import numpy as np
import psycopg2
import argparse
import os
import sys
import json
import urllib.request
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

from dotenv import load_dotenv
load_dotenv()


NORDIC_INDEXES = {
    'OMXS30': {'orderbookId': 19002, 'country': 'Sweden', 'name': 'OMX Stockholm 30'},
    'OMXC25': {'orderbookId': 731293, 'country': 'Denmark', 'name': 'OMX Copenhagen 25'},
    'OMXH25': {'orderbookId': 53295, 'country': 'Finland', 'name': 'OMX Helsinki 25'},
    'OSEBX':  {'orderbookId': 53294, 'country': 'Norway', 'name': 'Oslo Børs Benchmark'},
}


def fetch_nordic_index_returns() -> dict:
    """
    Fetch monthly close prices from Avanza for each Nordic index
    and compute monthly returns.

    Returns dict: {ticker: {(year, month): monthly_return_pct, ...}, ...}
    """
    base_url = "https://www.avanza.se/_api/price-chart/stock/{}?timePeriod=five_years&resolution=month"
    headers = {'User-Agent': 'Mozilla/5.0', 'Accept': 'application/json'}
    index_returns = {}

    for ticker, info in NORDIC_INDEXES.items():
        url = base_url.format(info['orderbookId'])
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode())

            ohlc = data.get('ohlc', [])
            if len(ohlc) < 2:
                print(f"  Warning: {ticker} returned {len(ohlc)} data points, skipping")
                continue

            monthly = {}
            for i in range(1, len(ohlc)):
                prev_close = ohlc[i - 1]['close']
                cur_close = ohlc[i]['close']
                ts = datetime.fromtimestamp(ohlc[i]['timestamp'] / 1000)
                ret = (cur_close / prev_close - 1) * 100
                monthly[(ts.year, ts.month)] = ret

            index_returns[ticker] = monthly
            print(f"  {ticker}: {len(monthly)} monthly returns "
                  f"({min(monthly.keys())} to {max(monthly.keys())})")

        except Exception as e:
            print(f"  Warning: Failed to fetch {ticker}: {e}")

    return index_returns


class MonthlyWalkForwardTrainer:
    """Monthly walk-forward expanding-window classifier for outperformance prediction."""

    def __init__(self, db_config: dict, output_dir: str = 'results',
                 min_train_months: int = 36, top_n: int = 20,
                 initial_capital: float = 100_000, commission_rate: float = 0.0015,
                 max_stale_days: int = 10):
        self.db_config = db_config
        self.conn = None
        self.output_dir = output_dir
        self.min_train_months = min_train_months
        self.top_n = top_n
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.max_stale_days = max_stale_days
        self.index_returns = {}  # populated before walk-forward

        # Fundamental feature columns (from annual reports)
        self.fundamental_cols = [
            # Valuation
            'dividend_yield', 'pe_ratio', 'ps_ratio', 'pb_ratio',
            'ev_ebit', 'ev_ebitda', 'ev_fcf', 'peg_ratio', 'ev_sales',
            # Profitability & Returns
            'roe', 'roa', 'roc', 'roic',
            'ebitda_margin', 'operating_margin', 'gross_margin', 'net_margin',
            'fcf_margin_pct', 'ocf_margin',
            # Financial Health
            'debt_equity', 'equity_ratio', 'current_ratio',
            'net_debt_pct', 'net_debt_ebitda', 'cash_pct',
            # Growth
            'revenue_growth', 'earnings_growth',
            'ebit_growth', 'book_value_growth', 'assets_growth',
            # Cash Flow (ratio)
            'fcf_margin', 'earnings_fcf',
            # Dividend
            'dividend_payout', 'dividend_growth',
        ]

        # Monthly price features (from ml_monthly_price_features)
        self.price_cols = [
            'price_change_5d', 'price_change_10d', 'price_change_20d', 'price_change_30d',
            'volume_ratio_5d_20d',
            'volatility_5d', 'volatility_20d',
            'was_rising_5d', 'was_rising_10d', 'was_rising_20d',
            'pct_from_20d_high', 'pct_from_20d_low',
        ]

        # Holdings features (from ml_holdings_features, yearly aggregation)
        self.holdings_cols = [
            'insider_buy_count', 'insider_sell_count',
            'insider_transaction_count', 'insider_buy_ratio',
            'buyback_count',
        ]

        # Normalized features (created in _create_normalized_features)
        self.normalized_cols = [
            'total_assets_to_mcap', 'net_debt_to_mcap', 'ocf_to_mcap', 'capex_to_mcap',
            'insider_amount_to_mcap',
            'insider_shares_to_total', 'buyback_shares_to_total',
        ]

        # Dividend yield history features (computed from multi-year fundamentals)
        self.dividend_history_cols = [
            'div_yield_2y_avg',   # average of current + prior year dividend yield
            'div_yield_3y_avg',   # average of current + 2 prior years' dividend yield
            'div_yield_trend',    # current yield minus prior year yield (+ = yield rising)
        ]

        # Long-term price behaviour features (computed from month_end_price history)
        self.long_price_cols = [
            'price_change_3m',    # 3-month return
            'price_change_6m',    # 6-month return
            'price_change_12m',   # 12-month return
            'momentum_2_12m',     # months 2-12 momentum (excludes last month to avoid reversal)
            'price_change_24m',   # 2-year return
            'pct_from_52w_high',  # % below 52-week high
            'pct_from_52w_low',   # % above 52-week low
            'volatility_52w',     # 52-week return volatility
        ]

        # Asset quality features (computed from annual fundamentals)
        # Requires: intangible_assets (KPI 126, incl. goodwill), total_equity in ml_features
        # Note: Börsdata KPI 126 is total intangibles incl. goodwill (no separate goodwill KPI)
        self.asset_cols = [
            'intangibles_to_total_assets',  # intangible-heaviness of balance sheet
            'price_to_tangible_book',       # market_cap / (equity - intangibles)
            'intangibles_growth',           # YoY change in intangibles (capitalisation trend)
        ]

        # Size feature (log of current market cap = num_shares * month_end_price)
        self.size_cols = ['log_market_cap']

        self.feature_cols = (self.fundamental_cols + self.price_cols + self.holdings_cols
                             + self.normalized_cols + self.dividend_history_cols
                             + self.long_price_cols + self.asset_cols + self.size_cols)

        self.target_col = 'next_month_excess_return'

    def connect(self):
        self.conn = psycopg2.connect(
            host=self.db_config['host'],
            database=self.db_config['database'],
            user=self.db_config['user'],
            password=self.db_config['password'],
            port=self.db_config.get('port', 5432)
        )
        print(f"Connected to database: {self.db_config['database']}")

    def close(self):
        if self.conn:
            self.conn.close()

    def _load_monthly_targets(self) -> pd.DataFrame:
        """Load monthly targets (returns). Includes latest partial month even if target not yet known."""
        print("Loading monthly targets...")
        df = pd.read_sql("""
            SELECT instrument_id, year, month, month_end_date, month_end_price,
                   next_month_return, market_median_monthly_return, next_month_excess_return
            FROM ml_monthly_targets
            ORDER BY year, month, instrument_id
        """, self.conn)
        complete = df['next_month_excess_return'].notna().sum()
        latest = df[['year', 'month']].drop_duplicates().sort_values(['year', 'month']).iloc[-1]
        print(f"  Monthly targets: {len(df)} rows ({complete} with complete targets), "
              f"{df['instrument_id'].nunique()} instruments, "
              f"latest month: {int(latest['year'])}-{int(latest['month']):02d}")
        return df

    def _load_fundamentals_with_dates(self) -> pd.DataFrame:
        """Load fundamental features with report_date for look-ahead prevention."""
        print("Loading fundamentals with report dates...")
        df = pd.read_sql("""
            SELECT
                f.instrument_id, f.year as report_year,
                f.company_name, f.sector, f.market,
                -- Use report_date to know when fundamentals became public
                -- Fall back to April of next year if report_date missing
                COALESCE(pr.report_date, make_date(f.year + 1, 4, 1)) as report_date,
                -- All fundamental features
                f.dividend_yield, f.pe_ratio, f.ps_ratio, f.pb_ratio,
                f.ev_ebit, f.ev_ebitda, f.ev_fcf, f.peg_ratio, f.ev_sales,
                f.roe, f.roa, f.roc, f.roic,
                f.ebitda_margin, f.operating_margin, f.gross_margin, f.net_margin,
                f.fcf_margin_pct, f.ocf_margin,
                f.debt_equity, f.equity_ratio, f.current_ratio,
                f.net_debt_pct, f.net_debt_ebitda, f.cash_pct,
                f.revenue_growth, f.earnings_growth,
                f.ebit_growth, f.book_value_growth, f.assets_growth,
                f.fcf_margin, f.earnings_fcf,
                f.dividend_payout, f.dividend_growth,
                -- Absolute metrics for normalization
                f.market_cap, f.num_shares, f.total_assets, f.net_debt,
                f.ocf, f.capex,
                -- Asset quality (KPI 126 = intangibles incl. goodwill; total_equity = KPI 58)
                f.intangible_assets, f.total_equity
            FROM ml_features f
            LEFT JOIN ml_pre_report_features pr
                ON f.instrument_id = pr.instrument_id
                AND f.year = pr.report_year
                AND f.period = pr.report_period
            WHERE f.period = 5
            ORDER BY f.instrument_id, f.year
        """, self.conn)
        df['report_date'] = pd.to_datetime(df['report_date'])
        print(f"  Fundamentals: {len(df)} rows, {df['instrument_id'].nunique()} instruments")
        return df

    def _load_monthly_price_features(self) -> pd.DataFrame:
        """Load monthly price features."""
        print("Loading monthly price features...")
        df = pd.read_sql("""
            SELECT instrument_id, year, month,
                   price_change_5d, price_change_10d, price_change_20d, price_change_30d,
                   volume_ratio_5d_20d,
                   volatility_5d, volatility_20d,
                   was_rising_5d, was_rising_10d, was_rising_20d,
                   pct_from_20d_high, pct_from_20d_low
            FROM ml_monthly_price_features
            ORDER BY instrument_id, year, month
        """, self.conn)
        print(f"  Monthly price features: {len(df)} rows")
        return df

    def _load_holdings(self) -> pd.DataFrame:
        """Load holdings features (yearly, will be mapped to months)."""
        print("Loading holdings features...")
        df = pd.read_sql("""
            SELECT instrument_id, year,
                   insider_net_shares, insider_net_amount,
                   insider_buy_count, insider_sell_count,
                   insider_transaction_count, insider_buy_ratio,
                   buyback_total_shares, buyback_total_amount,
                   buyback_count, buyback_shares_pct
            FROM ml_holdings_features
            ORDER BY instrument_id, year
        """, self.conn)
        print(f"  Holdings: {len(df)} rows")
        return df

    def _add_dividend_history_features(self, fundamentals_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute trailing dividend yield features from the multi-year fundamentals DataFrame.

        Each row in fundamentals_df is one annual report per instrument.  We shift within
        each instrument to compute:
          - div_yield_2y_avg: mean of current + prior-year dividend yield
          - div_yield_3y_avg: mean of current + 2 prior years
          - div_yield_trend:  current yield minus prior-year yield (positive = yield rising)
        """
        df = fundamentals_df.copy().sort_values(['instrument_id', 'report_year'])

        prev1 = df.groupby('instrument_id')['dividend_yield'].shift(1)
        prev2 = df.groupby('instrument_id')['dividend_yield'].shift(2)

        df['div_yield_2y_avg'] = df[['dividend_yield']].assign(p1=prev1).mean(axis=1)
        df['div_yield_3y_avg'] = df[['dividend_yield']].assign(p1=prev1, p2=prev2).mean(axis=1)
        df['div_yield_trend']  = df['dividend_yield'] - prev1

        return df

    def _compute_long_term_price_features(self, targets_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute long-term price momentum and volatility features from month_end_price history.

        All features are look-ahead safe: they use only prices up to and including
        the current month-end, which is the prediction time.
        """
        df = targets_df[['instrument_id', 'year', 'month', 'month_end_price']].copy()
        df = df.sort_values(['instrument_id', 'year', 'month']).reset_index(drop=True)

        price = df.groupby('instrument_id')['month_end_price']

        # Multi-month price changes
        for n_months, col in [
            (3,  'price_change_3m'),
            (6,  'price_change_6m'),
            (12, 'price_change_12m'),
            (24, 'price_change_24m'),
        ]:
            df[col] = (df['month_end_price'] / price.shift(n_months) - 1) * 100

        # Momentum months 2-12: skip most recent month to avoid short-term reversal
        df['momentum_2_12m'] = (price.shift(1) / price.shift(12) - 1) * 100

        # 52-week high/low proximity
        rolling_high = price.transform(lambda x: x.rolling(12, min_periods=6).max())
        rolling_low  = price.transform(lambda x: x.rolling(12, min_periods=6).min())
        df['pct_from_52w_high'] = (df['month_end_price'] / rolling_high - 1) * 100
        df['pct_from_52w_low']  = (df['month_end_price'] / rolling_low  - 1) * 100

        # 52-week return volatility (std of monthly returns)
        monthly_ret = price.pct_change()
        df['volatility_52w'] = monthly_ret.transform(
            lambda x: x.rolling(12, min_periods=6).std()
        ) * 100

        result_cols = ['instrument_id', 'year', 'month'] + self.long_price_cols
        available = [c for c in result_cols if c in df.columns]
        print(f"  Long-term price features: {len(df)} rows, "
              f"{len(available) - 3} features computed")
        return df[available]

    def _add_asset_ratio_features(self, fundamentals_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute tangible/intangible asset features from annual fundamentals.

        Requires intangible_assets, goodwill, total_equity columns in ml_features.
        Features are computed per-instrument per-year; look-ahead is prevented via
        the same report_date join used for all fundamentals.
        """
        df = fundamentals_df.copy().sort_values(['instrument_id', 'report_year'])

        for col in ['total_assets', 'total_equity']:
            if col in df.columns:
                df[col] = df[col].astype(float)
        ta = df['total_assets'].replace(0, np.nan)

        if 'intangible_assets' in df.columns:
            df['intangible_assets'] = df['intangible_assets'].astype(float)
            df['intangibles_to_total_assets'] = df['intangible_assets'] / ta

            prev_intangibles = df.groupby('instrument_id')['intangible_assets'].shift(1)
            df['intangibles_growth'] = (
                (df['intangible_assets'] - prev_intangibles)
                / prev_intangibles.abs().replace(0, np.nan) * 100
            )

            # Tangible book = equity minus all intangibles (KPI 126 includes goodwill)
            if 'total_equity' in df.columns:
                tangible_book = df['total_equity'] - df['intangible_assets'].fillna(0)
                df['price_to_tangible_book'] = df['market_cap'] / tangible_book.replace(0, np.nan)

        return df

    def _assemble_monthly_features(self, targets_df: pd.DataFrame,
                                    fundamentals_df: pd.DataFrame,
                                    price_features_df: pd.DataFrame,
                                    holdings_df: pd.DataFrame,
                                    long_price_features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Assemble the full feature matrix for all (stock, month) pairs.

        For each (instrument, year, month):
        - Fundamentals: latest annual report published BEFORE the 1st of that month
        - Price features: as of that month-end
        - Holdings: most recent completed year before that month
        """
        print("Assembling monthly feature matrix...")

        # Create a date column for each monthly target row (1st of month)
        df = targets_df.copy()
        df['month_start'] = pd.to_datetime(
            df['year'].astype(str) + '-' + df['month'].astype(str).str.zfill(2) + '-01'
        )

        # --- Join fundamentals: latest report published before month_start ---
        # Sort fundamentals by report_date for merge_asof
        fund_sorted = fundamentals_df.sort_values('report_date')
        df_sorted = df.sort_values('month_start')

        merged = pd.merge_asof(
            df_sorted,
            fund_sorted,
            by='instrument_id',
            left_on='month_start',
            right_on='report_date',
            direction='backward',
            suffixes=('', '_fund')
        )

        # --- Join monthly price features ---
        merged = merged.merge(
            price_features_df,
            on=['instrument_id', 'year', 'month'],
            how='left',
            suffixes=('', '_price')
        )

        # --- Join holdings: use the most recent year with data before the target month ---
        # For month M in year Y: use holdings from year Y-1 (or Y if month >= July)
        # Simple approach: use holdings year = target year - 1
        holdings_mapped = holdings_df.copy()
        holdings_mapped['holdings_for_year'] = holdings_mapped['year'] + 1
        merged = merged.merge(
            holdings_mapped.drop(columns=['year']),
            left_on=['instrument_id', 'year'],
            right_on=['instrument_id', 'holdings_for_year'],
            how='left',
            suffixes=('', '_hold')
        )
        if 'holdings_for_year' in merged.columns:
            merged.drop(columns=['holdings_for_year'], inplace=True)

        # --- Join long-term price features ---
        merged = merged.merge(
            long_price_features_df,
            on=['instrument_id', 'year', 'month'],
            how='left'
        )

        # Create normalized features
        merged = self._create_normalized_features(merged)

        # Log market cap: use current num_shares * month_end_price (more accurate than stale annual)
        if 'num_shares' in merged.columns and 'month_end_price' in merged.columns:
            merged['log_market_cap'] = np.log1p(
                merged['num_shares'] * merged['month_end_price']
            )

        print(f"  Assembled: {len(merged)} rows, {merged['instrument_id'].nunique()} instruments")
        n_with_fundamentals = merged['report_date'].notna().sum()
        print(f"  With fundamentals: {n_with_fundamentals} ({n_with_fundamentals/len(merged)*100:.1f}%)")

        return merged

    def _create_normalized_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create market-cap and share-count normalized features."""
        df = df.copy()

        if 'market_cap' in df.columns:
            mcap = df['market_cap'].replace(0, np.nan)
            for raw_col, new_col in [
                ('total_assets', 'total_assets_to_mcap'),
                ('net_debt', 'net_debt_to_mcap'),
                ('ocf', 'ocf_to_mcap'),
                ('capex', 'capex_to_mcap'),
                ('insider_net_amount', 'insider_amount_to_mcap'),
            ]:
                if raw_col in df.columns:
                    df[new_col] = df[raw_col] / mcap

        if 'num_shares' in df.columns:
            shares = df['num_shares'].replace(0, np.nan)
            for raw_col, new_col in [
                ('insider_net_shares', 'insider_shares_to_total'),
                ('buyback_total_shares', 'buyback_shares_to_total'),
            ]:
                if raw_col in df.columns:
                    df[new_col] = df[raw_col] / shares

        return df

    def _prepare_fold(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        """
        Prepare features for one walk-forward fold.
        Imputes and scales (fit on train only).

        Returns: X_train, X_test, y_train, y_test, available_cols, train_medians
        """
        available_cols = [c for c in self.feature_cols if c in train_df.columns]

        X_train = train_df[available_cols].copy()
        X_test = test_df[available_cols].copy()

        # Binary target: 1 if stock outperformed market median that month
        y_train = (train_df[self.target_col] > 0).astype(int)
        y_test = (test_df[self.target_col] > 0).astype(int)

        # Impute with training medians only
        train_medians = X_train.median()
        X_train = X_train.fillna(train_medians)
        X_test = X_test.fillna(train_medians)

        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        return X_train_scaled, X_test_scaled, y_train, y_test, available_cols, train_medians

    def _predict_current_month(self, train_df: pd.DataFrame, current_df: pd.DataFrame,
                               output_dir: str):
        """
        Train on all complete historical months and predict on the current (partial) month.

        Produces a ranked list of stocks by outperformance probability and saves to Excel.
        No target variable is available — this is a live forward-looking prediction.
        """
        year = int(current_df['year'].iloc[0])
        month = int(current_df['month'].iloc[0])
        latest_date = (current_df['month_end_date'].max()
                       if 'month_end_date' in current_df.columns else 'N/A')

        print(f"\n{'='*60}")
        print(f"CURRENT MONTH PREDICTION: {year}-{month:02d}  (data as of {latest_date})")
        print(f"  Training on {len(train_df)} rows from all complete months")
        print(f"  Predicting for {len(current_df)} stocks")

        available_cols = [c for c in self.feature_cols if c in train_df.columns]

        X_train = train_df[available_cols].copy()
        X_current = current_df[available_cols].copy()
        y_train = (train_df[self.target_col] > 0).astype(int)

        train_medians = X_train.median()
        X_train = X_train.fillna(train_medians)
        X_current = X_current.fillna(train_medians)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_current_scaled = scaler.transform(X_current)

        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_split=10,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train_scaled, y_train)

        proba = model.predict_proba(X_current_scaled)[:, 1]
        ranked = current_df.copy()
        ranked['score'] = proba
        ranked = ranked.sort_values('score', ascending=False).reset_index(drop=True)
        ranked['rank'] = range(1, len(ranked) + 1)

        feature_importance = pd.DataFrame({
            'feature': available_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        # Save to Excel
        filepath = os.path.join(output_dir, f'current_month_{year}_{month:02d}_predictions.xlsx')
        pick_cols = [c for c in ['rank', 'instrument_id', 'company_name', 'sector', 'market',
                                  'score', 'month_end_date', 'month_end_price',
                                  'report_date', 'report_year'] if c in ranked.columns]
        decile_size = max(1, len(ranked) // 10)

        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            ranked[pick_cols].to_excel(writer, sheet_name='all_ranked', index=False)
            ranked.head(self.top_n)[pick_cols].to_excel(
                writer, sheet_name=f'top_{self.top_n}_picks', index=False)
            ranked.head(30)[pick_cols].to_excel(
                writer, sheet_name='top_30_picks', index=False)
            ranked.head(decile_size)[pick_cols].to_excel(
                writer, sheet_name='top_decile_picks', index=False)
            feature_importance.head(30).to_excel(
                writer, sheet_name='feature_importance', index=False)

            # Prediction input: scaled feature values for transparency
            id_data = {col: current_df[col].values
                       for col in ['instrument_id', 'company_name', 'sector']
                       if col in current_df.columns}
            pred_input_df = pd.DataFrame(id_data, index=current_df.index)
            for j, col in enumerate(available_cols):
                pred_input_df[col] = X_current_scaled[:, j]
            pred_input_df['score'] = proba
            pred_input_df.sort_values('score', ascending=False).to_excel(
                writer, sheet_name='prediction_input', index=False)

            # Model info
            pd.DataFrame([
                {'metric': 'Prediction month', 'value': f'{year}-{month:02d}'},
                {'metric': 'Data as of (latest price date)', 'value': str(latest_date)},
                {'metric': 'N stocks predicted', 'value': len(ranked)},
                {'metric': 'Training samples (complete months)', 'value': len(train_df)},
                {'metric': 'Features used', 'value': len(available_cols)},
                {'metric': f'Top {self.top_n} size', 'value': self.top_n},
                {'metric': 'Top decile size', 'value': decile_size},
            ]).to_excel(writer, sheet_name='model_info', index=False)

        print(f"\n  Top {self.top_n} picks for {year}-{month:02d}:")
        for _, row in ranked.head(self.top_n).iterrows():
            name = str(row.get('company_name', row['instrument_id']))
            sector = str(row.get('sector', 'N/A'))
            print(f"    #{int(row['rank']):2d}  {name:<35s}  {sector:<25s}  score={row['score']:.3f}")
        print(f"\n  Saved: {filepath}")
        print(f"{'='*60}")

    def _save_prediction_timestamp(self, year: int, month: int):
        """Save a timestamp file recording when the last prediction was run."""
        import json as _json
        filepath = os.path.join(self.output_dir, 'last_prediction.json')
        data = {
            'timestamp': datetime.now().isoformat(timespec='seconds'),
            'year': year,
            'month': month,
        }
        os.makedirs(self.output_dir, exist_ok=True)
        with open(filepath, 'w') as f:
            _json.dump(data, f, indent=2)
        print(f"  Prediction timestamp saved: {filepath}")

    def run_walk_forward(self, predict_only: bool = False):
        """Run the full monthly walk-forward expanding-window backtest.

        If predict_only=True, skip the backtest loop and only run the
        current-month prediction using all complete months as training data.
        """

        # Load all data
        targets_df = self._load_monthly_targets()
        fundamentals_df = self._load_fundamentals_with_dates()
        fundamentals_df = self._add_dividend_history_features(fundamentals_df)
        fundamentals_df = self._add_asset_ratio_features(fundamentals_df)
        price_features_df = self._load_monthly_price_features()
        holdings_df = self._load_holdings()

        # Compute long-term price features from full price history (before any filtering)
        print("Computing long-term price features...")
        long_price_features_df = self._compute_long_term_price_features(targets_df)

        # Assemble full feature matrix
        df = self._assemble_monthly_features(
            targets_df, fundamentals_df, price_features_df, holdings_df, long_price_features_df
        )

        # Fetch Nordic index returns from Avanza
        print("\nFetching Nordic index monthly returns from Avanza...")
        self.index_returns = fetch_nordic_index_returns()

        # Filter to rows with valid target and fundamentals
        df_valid = df[df[self.target_col].notna() & df['report_date'].notna()].copy()

        # Rows with no target yet — candidate for current-month live prediction
        df_current_candidates = df[df[self.target_col].isna() & df['report_date'].notna()].copy()

        # Exclude index instruments (OMX indexes, Nordic indexes, sector indexes)
        # These have instrument_type 2, 4, or 13 in the source API and are not stocks
        index_name_pattern = r'^(OMX|OMXH|OMXS|OMXC|OBX|OSEBX|OB Oslo|First North)'
        if 'company_name' in df_valid.columns:
            is_index = df_valid['company_name'].str.match(index_name_pattern, case=False, na=False)
            n_excluded = is_index.sum()
            if n_excluded > 0:
                excluded = df_valid.loc[is_index, 'company_name'].unique()
                print(f"  Excluding {n_excluded} rows from {len(excluded)} index instrument(s): "
                      f"{sorted(excluded)}")
            df_valid = df_valid[~is_index]
        if 'company_name' in df_current_candidates.columns:
            is_index_cur = df_current_candidates['company_name'].str.match(
                index_name_pattern, case=False, na=False)
            df_current_candidates = df_current_candidates[~is_index_cur]

        # Keep only the single latest (year, month) as the current-month prediction target
        df_current = pd.DataFrame()
        if len(df_current_candidates) > 0:
            latest_ym = (df_current_candidates[['year', 'month']]
                         .drop_duplicates()
                         .sort_values(['year', 'month'])
                         .iloc[-1])
            df_current = df_current_candidates[
                (df_current_candidates['year'] == latest_ym['year']) &
                (df_current_candidates['month'] == latest_ym['month'])
            ].copy()
            print(f"  Current (partial) month for live prediction: "
                  f"{int(latest_ym['year'])}-{int(latest_ym['month']):02d} "
                  f"({len(df_current)} stocks)")

            # --- Filter out stocks with stale price data ---
            # Companies that haven't traded recently (e.g. suspended/delisted) would
            # have an old month_end_date while active stocks have prices up to yesterday.
            # Drop any stock whose last price is more than max_stale_days behind the
            # freshest price in the current-month batch.
            if 'month_end_date' in df_current.columns and len(df_current) > 0:
                dates = pd.to_datetime(df_current['month_end_date'], errors='coerce')
                freshest = dates.max()
                stale_cutoff = freshest - pd.Timedelta(days=self.max_stale_days)
                is_stale = dates < stale_cutoff
                n_stale = int(is_stale.sum())
                if n_stale > 0:
                    stale_names = df_current.loc[is_stale, 'company_name'].tolist() \
                        if 'company_name' in df_current.columns else []
                    preview = stale_names[:8]
                    ellipsis = '...' if n_stale > 8 else ''
                    print(f"  Excluding {n_stale} stale stock(s) (last price >{self.max_stale_days}d "
                          f"before {freshest.date()}): {preview}{ellipsis}")
                    df_current = df_current[~is_stale].copy()

        print(f"\nValid rows for training: {len(df_valid)}")

        # Create (year, month) index for walk-forward
        month_keys = df_valid[['year', 'month']].drop_duplicates().sort_values(['year', 'month'])
        months_list = list(month_keys.itertuples(index=False, name=None))
        print(f"Available months: {months_list[0][0]}-{months_list[0][1]:02d} to "
              f"{months_list[-1][0]}-{months_list[-1][1]:02d} ({len(months_list)} months)")
        print(f"Min training months: {self.min_train_months}")
        print(f"Walk-forward folds: {len(months_list) - self.min_train_months}")

        # Create output directories
        debug_dir = os.path.join(self.output_dir, 'debug')
        reports_dir = os.path.join(self.output_dir, 'reports')
        os.makedirs(debug_dir, exist_ok=True)
        os.makedirs(reports_dir, exist_ok=True)

        # --predict-only: skip backtest, jump straight to current-month prediction
        if predict_only:
            print("\n[predict-only mode] Skipping backtest walk-forward.")
            if len(df_current) > 0:
                self._predict_current_month(df_valid, df_current, reports_dir)
                latest_ym = df_current[['year', 'month']].iloc[0]
                self._save_prediction_timestamp(int(latest_ym['year']), int(latest_ym['month']))
            else:
                print("  No current-month data available for prediction.")
            return

        all_monthly_results = []

        for i in range(self.min_train_months, len(months_list)):
            train_months = months_list[:i]
            test_year, test_month = months_list[i]

            # Build train/test masks using vectorized year*100+month key
            train_set = set(y * 100 + m for y, m in train_months)
            ym_key = df_valid['year'] * 100 + df_valid['month']
            train_mask = ym_key.isin(train_set)
            test_mask = (df_valid['year'] == test_year) & (df_valid['month'] == test_month)

            train_df = df_valid[train_mask].copy()
            test_df = df_valid[test_mask].copy()

            if len(test_df) < 5:
                continue

            first_train = train_months[0]
            print(f"\n  {test_year}-{test_month:02d}: "
                  f"Train {first_train[0]}-{first_train[1]:02d} to "
                  f"{train_months[-1][0]}-{train_months[-1][1]:02d} "
                  f"({len(train_df)} rows) -> Test {len(test_df)} stocks", end='')

            # Prepare features
            X_train, X_test, y_train, y_test, available_cols, _train_medians = \
                self._prepare_fold(train_df, test_df)

            # Train classifier
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=8,
                min_samples_split=10,
                min_samples_leaf=10,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train, y_train)

            # Predict
            proba = model.predict_proba(X_test)[:, 1]
            test_df = test_df.copy()
            test_df['score'] = proba
            predictions = model.predict(X_test)

            # Evaluate
            accuracy = accuracy_score(y_test, predictions)
            auc = roc_auc_score(y_test, proba) if len(y_test.unique()) > 1 else 0.0
            precision = precision_score(y_test, predictions, zero_division=0)
            recall = recall_score(y_test, predictions, zero_division=0)
            f1 = f1_score(y_test, predictions, zero_division=0)

            print(f" | AUC={auc:.3f}", end='')

            # Rank and pick
            ranked = test_df.sort_values('score', ascending=False).copy()
            ranked['rank'] = range(1, len(ranked) + 1)

            top_n_picks = ranked.head(self.top_n).copy()
            top_30_picks = ranked.head(30).copy()
            decile_size = max(1, len(ranked) // 10)
            top_decile_picks = ranked.head(decile_size).copy()

            # Portfolio returns (actual next-month returns)
            benchmark_return = test_df['market_median_monthly_return'].iloc[0]
            top_n_return = top_n_picks['next_month_return'].mean()
            top_n_excess = top_n_picks[self.target_col].mean()
            top_30_return = top_30_picks['next_month_return'].mean()
            top_30_excess = top_30_picks[self.target_col].mean()
            decile_return = top_decile_picks['next_month_return'].mean()
            decile_excess = top_decile_picks[self.target_col].mean()

            # Look up Nordic index returns for this month
            month_index_returns = {}
            for ticker, returns_dict in self.index_returns.items():
                month_index_returns[ticker] = returns_dict.get((test_year, test_month))

            idx_str = ''.join(
                f" {t}={r:+.1f}%" for t, r in month_index_returns.items() if r is not None
            )
            print(f" | Top{self.top_n}={top_n_return:+.2f}% Top30={top_30_return:+.2f}% Bench={benchmark_return:+.2f}%{idx_str}")

            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': available_cols,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)

            # Store monthly result
            monthly_result = {
                'year': test_year,
                'month': test_month,
                'n_train': len(train_df),
                'n_test': len(test_df),
                'accuracy': accuracy,
                'auc': auc,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'benchmark_return': benchmark_return,
                'top_n_return': top_n_return,
                'top_n_excess': top_n_excess,
                'top_30_return': top_30_return,
                'top_30_excess': top_30_excess,
                'decile_return': decile_return,
                'decile_excess': decile_excess,
                'decile_size': decile_size,
                'index_returns': month_index_returns,
                'feature_importance': feature_importance,
                'top_n_picks': top_n_picks,
                'top_30_picks': top_30_picks,
                'top_decile_picks': top_decile_picks,
            }
            all_monthly_results.append(monthly_result)

            # Save debug Excel
            self._save_debug_excel(
                debug_dir, test_year, test_month,
                train_df, test_df, ranked,
                X_test, available_cols, _train_medians,
                proba, predictions
            )

        if not all_monthly_results:
            print("\nNo walk-forward folds were run. Check your data.")
            return

        # Aggregate monthly results to yearly
        yearly_results = self._aggregate_to_yearly(all_monthly_results)

        # Save reports
        self._save_yearly_reports(reports_dir, yearly_results)
        self._save_summary_report(reports_dir, yearly_results, all_monthly_results)

        # Account simulation
        print("\nSimulating trading account...")
        tx_df, period_df = self._simulate_account(
            all_monthly_results,
            initial_capital=self.initial_capital,
            commission_rate=self.commission_rate,
        )
        self._save_account_report(
            reports_dir, tx_df, period_df,
            self.initial_capital, self.commission_rate,
        )

        # Print summary
        self._print_summary(yearly_results, all_monthly_results)

        # Predict on the current (possibly incomplete) month using all historical data
        if len(df_current) > 0:
            self._predict_current_month(df_valid, df_current, reports_dir)
            latest_ym = df_current[['year', 'month']].iloc[0]
            self._save_prediction_timestamp(int(latest_ym['year']), int(latest_ym['month']))

    def _simulate_account(self, monthly_results: list,
                          initial_capital: float = 100_000,
                          commission_rate: float = 0.0015) -> tuple:
        """
        Simulate a trading account following the model's top-N picks.

        Logic:
          - Each month SELL all stocks that left the top-N list.
          - BUY all stocks newly entering the list, splitting sell proceeds equally.
          - Staying stocks are left untouched (their value drifts with returns).
          - Commission (commission_rate) is charged on every individual buy/sell.

        Sell price for leaving stocks is estimated as:
            previous_month_end_price × (1 + next_month_return / 100)

        A parallel OMXS30 buy-and-hold account (same initial capital, same initial
        buy commission) is simulated over the identical holding periods (month T+1
        for picks made at end of month T).

        Returns:
            (transactions_df, period_summary_df)
        """
        omxs30_monthly = self.index_returns.get('OMXS30', {})

        model_value: float = initial_capital
        omxs30_value: float = initial_capital * (1 - commission_rate)

        current_shares: dict = {}   # {iid: shares held}
        current_prices: dict = {}   # {iid: price last used for this position}
        prev_price_map: dict = {}   # price_map from the previous period
        prev_return_map: dict = {}  # return_map from the previous period
        all_names: dict = {}        # cumulative {iid: company_name} across all periods

        tx_rows: list = []
        period_rows: list = []

        for i, mr in enumerate(monthly_results):
            year, month = int(mr['year']), int(mr['month'])
            picks_df = mr['top_n_picks'].copy()
            if len(picks_df) == 0:
                continue

            new_ids_set = set(picks_df['instrument_id'].tolist())
            old_ids_set = set(current_shares.keys())
            period_str = f'{year}-{month:02d}'

            # ---- build lookup maps for this period ----
            price_map: dict = {}
            if 'month_end_price' in picks_df.columns:
                for _, row in picks_df.iterrows():
                    p = row.get('month_end_price')
                    if pd.notna(p) and float(p) > 0:
                        price_map[row['instrument_id']] = float(p)

            if 'company_name' in picks_df.columns:
                all_names.update(zip(picks_df['instrument_id'], picks_df['company_name']))

            return_map = dict(zip(
                picks_df['instrument_id'],
                picks_df['next_month_return'].fillna(0)
            ))

            leaving  = old_ids_set - new_ids_set
            entering = new_ids_set - old_ids_set
            period_commission = 0.0
            period_tx: list = []

            def _name(iid):
                return all_names.get(iid, str(iid))

            # ============================================================
            # INITIAL PURCHASE (period 0)
            # ============================================================
            if i == 0:
                init_commission = model_value * commission_rate
                period_commission = init_commission
                model_value -= init_commission
                target_per_stock = model_value / max(len(new_ids_set), 1)

                for iid in sorted(new_ids_set, key=_name):
                    price = price_map.get(iid, 1.0) or 1.0
                    shares = target_per_stock / price
                    commission = target_per_stock * commission_rate
                    current_shares[iid] = shares
                    current_prices[iid] = price
                    period_tx.append({
                        'period':                  period_str,
                        'action':                  'BUY',
                        'company':                 _name(iid),
                        'instrument_id':           iid,
                        'shares':                  round(shares, 2),
                        'price_sek':               round(price, 2),
                        'trade_value_sek':         round(target_per_stock, 2),
                        'commission_sek':          round(commission, 2),
                        'model_account_value_sek': None,   # filled after returns
                        'omxs30_account_value_sek': None,
                    })

            # ============================================================
            # REBALANCING (period i > 0): SELL leaving, BUY entering
            # ============================================================
            else:
                # Sell prices: previous price × (1 + previous return)
                # For staying stocks prefer the current period's month_end_price.
                sell_price_for: dict = {}
                for iid in old_ids_set:
                    prev_p = prev_price_map.get(iid) or current_prices.get(iid, 1.0) or 1.0
                    prev_r = prev_return_map.get(iid, 0.0)
                    sell_price_for[iid] = prev_p * (1 + prev_r / 100)
                for iid in (old_ids_set & new_ids_set):   # staying: use direct price if available
                    if iid in price_map:
                        sell_price_for[iid] = price_map[iid]

                # --- SELL leaving stocks ---
                sell_proceeds = 0.0
                for iid in sorted(leaving, key=_name):
                    shares = current_shares.pop(iid, 0.0)
                    sell_price = sell_price_for.get(iid, 1.0) or 1.0
                    trade_value = shares * sell_price
                    commission = trade_value * commission_rate
                    period_commission += commission
                    sell_proceeds += trade_value - commission
                    current_prices.pop(iid, None)
                    period_tx.append({
                        'period':                  period_str,
                        'action':                  'SELL',
                        'company':                 _name(iid),
                        'instrument_id':           iid,
                        'shares':                  round(shares, 2),
                        'price_sek':               round(sell_price, 2),
                        'trade_value_sek':         round(trade_value, 2),
                        'commission_sek':          round(commission, 2),
                        'model_account_value_sek': None,
                        'omxs30_account_value_sek': None,
                    })

                # --- BUY entering stocks (split sell proceeds equally) ---
                if entering and sell_proceeds > 0:
                    alloc = sell_proceeds / len(entering)
                    for iid in sorted(entering, key=_name):
                        buy_price = price_map.get(iid, 1.0) or 1.0
                        commission = alloc * commission_rate
                        period_commission += commission
                        net_invest = alloc - commission
                        shares = net_invest / buy_price
                        current_shares[iid] = shares
                        current_prices[iid] = buy_price
                        period_tx.append({
                            'period':                  period_str,
                            'action':                  'BUY',
                            'company':                 _name(iid),
                            'instrument_id':           iid,
                            'shares':                  round(shares, 2),
                            'price_sek':               round(buy_price, 2),
                            'trade_value_sek':         round(alloc, 2),
                            'commission_sek':          round(commission, 2),
                            'model_account_value_sek': None,
                            'omxs30_account_value_sek': None,
                        })

            # ============================================================
            # APPLY RETURNS for the holding period
            # ============================================================
            model_value_before_returns = sum(
                current_shares.get(iid, 0) * current_prices.get(iid, 1.0)
                for iid in current_shares
            )
            new_model_value = 0.0
            for iid in list(current_shares.keys()):
                r = return_map.get(iid, 0.0)
                current_prices[iid] = (current_prices.get(iid, 1.0) or 1.0) * (1 + r / 100)
                new_model_value += current_shares[iid] * current_prices[iid]

            model_month_return = (
                (new_model_value / model_value_before_returns - 1) * 100
                if model_value_before_returns > 0 else 0.0
            )
            model_value = new_model_value

            # ---- OMXS30 for the same holding period (month T+1) ----
            next_year_omx  = year if month < 12 else year + 1
            next_month_omx = month + 1 if month < 12 else 1
            omxs30_ret = omxs30_monthly.get((next_year_omx, next_month_omx))
            if omxs30_ret is not None:
                omxs30_value *= (1 + omxs30_ret / 100)

            # Fill account values on all transactions for this period
            for t in period_tx:
                t['model_account_value_sek']  = round(model_value, 2)
                t['omxs30_account_value_sek'] = round(omxs30_value, 2)
            tx_rows.extend(period_tx)

            # Period-level summary row (one per month)
            period_rows.append({
                'period':                  period_str,
                'year':                    year,
                'month':                   month,
                'model_account_value_sek': round(model_value, 2),
                'omxs30_account_value_sek': round(omxs30_value, 2),
                'model_month_return_pct':  round(model_month_return, 2),
                'omxs30_month_return_pct': round(omxs30_ret, 2) if omxs30_ret is not None else None,
                'model_vs_omxs30_pct':     round(model_month_return - (omxs30_ret or 0), 2)
                                           if omxs30_ret is not None else None,
                'commission_sek':          round(period_commission, 2),
                'n_sells':                 len(leaving),
                'n_buys':                  len(entering),
            })

            # Store for next iteration
            prev_price_map  = dict(price_map)
            prev_return_map = dict(return_map)

        return pd.DataFrame(tx_rows), pd.DataFrame(period_rows)

    def _save_account_report(self, reports_dir: str,
                              tx_df: pd.DataFrame,
                              period_df: pd.DataFrame,
                              initial_capital: float,
                              commission_rate: float):
        """Save the account simulation Excel report (transaction log + summaries)."""
        filepath = os.path.join(reports_dir, 'account_simulation.xlsx')

        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:

            if len(tx_df) == 0:
                print("  Account simulation: no transactions to save.")
                pd.DataFrame().to_excel(writer, sheet_name='transactions', index=False)
                return

            # ---- Sheet 1: transaction log ----
            tx_df.to_excel(writer, sheet_name='transactions', index=False)

            # ---- Sheet 2: monthly side-by-side summary ----
            period_df.to_excel(writer, sheet_name='monthly_summary', index=False)

            # ---- Sheet 3: yearly summary ----
            yearly_rows = []
            for year in sorted(period_df['year'].unique()):
                yr   = period_df[period_df['year'] == year]
                prev = period_df[period_df['year'] < year]

                model_start  = (prev['model_account_value_sek'].iloc[-1]
                                if len(prev) else initial_capital)
                model_end    = yr['model_account_value_sek'].iloc[-1]
                omxs30_start = (prev['omxs30_account_value_sek'].iloc[-1]
                                if len(prev) else initial_capital * (1 - commission_rate))
                omxs30_end   = yr['omxs30_account_value_sek'].iloc[-1]

                model_annual  = (model_end  / model_start  - 1) * 100
                omxs30_annual = (omxs30_end / omxs30_start - 1) * 100

                yearly_rows.append({
                    'year':                   year,
                    'model_start_sek':        round(model_start, 2),
                    'model_end_sek':          round(model_end, 2),
                    'model_annual_return_%':  round(model_annual, 2),
                    'omxs30_start_sek':       round(omxs30_start, 2),
                    'omxs30_end_sek':         round(omxs30_end, 2),
                    'omxs30_annual_return_%': round(omxs30_annual, 2),
                    'excess_return_%':        round(model_annual - omxs30_annual, 2),
                    'commission_sek':         round(yr['commission_sek'].sum(), 2),
                    'beat_omxs30':            'YES' if model_annual > omxs30_annual else 'no',
                })

            pd.DataFrame(yearly_rows).to_excel(writer, sheet_name='yearly_summary', index=False)

            # ---- Sheet 4: overall summary ----
            final_model   = period_df['model_account_value_sek'].iloc[-1]
            final_omxs30  = period_df['omxs30_account_value_sek'].iloc[-1]
            total_comm    = period_df['commission_sek'].sum()
            omxs30_init   = initial_capital * (1 - commission_rate)
            beat_years    = sum(1 for r in yearly_rows if r['beat_omxs30'] == 'YES')
            total_years   = len(yearly_rows)

            summary_data = [
                ('Initial Capital (SEK)',               initial_capital),
                ('Commission Rate',                     f'{commission_rate * 100:.2f}%'),
                ('N Months Simulated',                  len(period_df)),
                ('Final Model Account Value (SEK)',     round(final_model, 2)),
                ('Final OMXS30 Account Value (SEK)',    round(final_omxs30, 2)),
                ('Model Total Return (%)',              round((final_model  / initial_capital - 1) * 100, 2)),
                ('OMXS30 Total Return (%)',             round((final_omxs30 / omxs30_init - 1) * 100, 2)),
                ('Total Commission Paid (SEK)',         round(total_comm, 2)),
                ('Commission as % of Initial Capital', round(total_comm / initial_capital * 100, 2)),
                ('Years Beat OMXS30',                  f'{beat_years}/{total_years}'),
            ]
            pd.DataFrame(summary_data, columns=['metric', 'value']).to_excel(
                writer, sheet_name='summary', index=False)

        print(f"  Saved account simulation: {filepath}")

    def _save_debug_excel(self, debug_dir: str, year: int, month: int,
                          train_df: pd.DataFrame, test_df: pd.DataFrame,
                          ranked: pd.DataFrame, X_test_scaled: np.ndarray,
                          available_cols: list, train_medians: pd.Series,
                          proba: np.ndarray, predictions: np.ndarray):
        """Save debug data for one fold to Excel."""
        filepath = os.path.join(debug_dir, f'month_{year}_{month:02d}.xlsx')

        id_cols = ['instrument_id', 'year', 'month', 'company_name', 'sector']
        target_cols = [self.target_col, 'next_month_return', 'market_median_monthly_return']

        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # Train data (features + targets)
            train_export_cols = [c for c in id_cols + available_cols + target_cols if c in train_df.columns]
            train_df[train_export_cols].to_excel(writer, sheet_name='train_data', index=False)

            # Test data with predictions
            test_export_cols = [c for c in id_cols + available_cols + target_cols if c in test_df.columns]
            test_export = test_df[test_export_cols].copy()
            if 'score' in test_df.columns:
                test_export['score'] = test_df['score'].values
            test_export.to_excel(writer, sheet_name='test_data', index=False)

            # Feature statistics (train set)
            feature_stats = train_df[available_cols].describe().T
            feature_stats['missing_pct'] = (train_df[available_cols].isna().sum() / len(train_df) * 100)
            feature_stats['median_impute_value'] = train_medians[available_cols]
            feature_stats.to_excel(writer, sheet_name='feature_stats')

            # Top N picks
            pick_cols = [c for c in ['rank', 'instrument_id', 'company_name', 'sector',
                                      'score', self.target_col,
                                      'next_month_return', 'market_median_monthly_return'] if c in ranked.columns]
            ranked.head(self.top_n)[pick_cols].to_excel(writer, sheet_name='top_n_picks', index=False)

            # Top 30 picks
            ranked.head(30)[pick_cols].to_excel(writer, sheet_name='top_30_picks', index=False)

            # Top decile picks
            decile_size = max(1, len(ranked) // 10)
            ranked.head(decile_size)[pick_cols].to_excel(writer, sheet_name='top_decile_picks', index=False)

            # All predictions ranked
            ranked[pick_cols].to_excel(writer, sheet_name='all_ranked', index=False)

            # Prediction input: exact scaled feature values sent to the model
            id_data = {}
            for col in ['instrument_id', 'company_name', 'sector']:
                if col in test_df.columns:
                    id_data[col] = test_df[col].values
            pred_input_df = pd.DataFrame(id_data, index=test_df.index)
            for j, col in enumerate(available_cols):
                pred_input_df[col] = X_test_scaled[:, j]
            pred_input_df['score'] = proba
            pred_input_df['predicted_outperform'] = predictions
            pred_input_df.sort_values('score', ascending=False).to_excel(
                writer, sheet_name='prediction_input', index=False
            )

        print(f"    Debug: {filepath}")

    def _aggregate_to_yearly(self, monthly_results: list) -> list:
        """Compound monthly returns into yearly results."""
        yearly = {}
        for mr in monthly_results:
            y = mr['year']
            if y not in yearly:
                yearly[y] = {
                    'year': y,
                    'months': [],
                    'benchmark_monthly': [],
                    'top_n_monthly': [],
                    'top_30_monthly': [],
                    'decile_monthly': [],
                    'index_monthly': {t: [] for t in NORDIC_INDEXES},
                    'accuracies': [],
                    'aucs': [],
                    'precisions': [],
                    'recalls': [],
                    'f1s': [],
                    'n_trains': [],
                    'n_tests': [],
                    'decile_sizes': [],
                    'feature_importances': [],
                    'all_top_n_picks': [],
                    'all_top_30_picks': [],
                    'all_decile_picks': [],
                }
            yearly[y]['months'].append(mr['month'])
            yearly[y]['benchmark_monthly'].append(mr['benchmark_return'])
            yearly[y]['top_n_monthly'].append(mr['top_n_return'])
            yearly[y]['top_30_monthly'].append(mr['top_30_return'])
            yearly[y]['decile_monthly'].append(mr['decile_return'])
            for ticker in NORDIC_INDEXES:
                val = mr.get('index_returns', {}).get(ticker)
                yearly[y]['index_monthly'][ticker].append(val)
            yearly[y]['accuracies'].append(mr['accuracy'])
            yearly[y]['aucs'].append(mr['auc'])
            yearly[y]['precisions'].append(mr['precision'])
            yearly[y]['recalls'].append(mr['recall'])
            yearly[y]['f1s'].append(mr['f1'])
            yearly[y]['n_trains'].append(mr['n_train'])
            yearly[y]['n_tests'].append(mr['n_test'])
            yearly[y]['decile_sizes'].append(mr['decile_size'])
            yearly[y]['feature_importances'].append(mr['feature_importance'])
            yearly[y]['all_top_n_picks'].append(mr['top_n_picks'])
            yearly[y]['all_top_30_picks'].append(mr['top_30_picks'])
            yearly[y]['all_decile_picks'].append(mr['top_decile_picks'])

        # Compound monthly returns for each year
        results = []
        for y in sorted(yearly.keys()):
            yd = yearly[y]
            # Compound: product of (1 + r/100) - 1, times 100
            benchmark_compound = (np.prod([1 + r/100 for r in yd['benchmark_monthly']]) - 1) * 100
            top_n_compound = (np.prod([1 + r/100 for r in yd['top_n_monthly']]) - 1) * 100
            top_30_compound = (np.prod([1 + r/100 for r in yd['top_30_monthly']]) - 1) * 100
            decile_compound = (np.prod([1 + r/100 for r in yd['decile_monthly']]) - 1) * 100

            # Compound index returns (skip months with missing data)
            index_compound = {}
            for ticker in NORDIC_INDEXES:
                vals = [r for r in yd['index_monthly'][ticker] if r is not None]
                if vals:
                    index_compound[ticker] = (np.prod([1 + r/100 for r in vals]) - 1) * 100
                else:
                    index_compound[ticker] = None

            # Average feature importance across months
            all_fi = pd.concat(yd['feature_importances'])
            avg_fi = all_fi.groupby('feature')['importance'].mean().reset_index()
            avg_fi = avg_fi.sort_values('importance', ascending=False)

            results.append({
                'year': y,
                'n_months': len(yd['months']),
                'months': yd['months'],
                'n_train_avg': int(np.mean(yd['n_trains'])),
                'n_test_avg': int(np.mean(yd['n_tests'])),
                'accuracy': np.mean(yd['accuracies']),
                'auc': np.mean(yd['aucs']),
                'precision': np.mean(yd['precisions']),
                'recall': np.mean(yd['recalls']),
                'f1': np.mean(yd['f1s']),
                'benchmark_return': benchmark_compound,
                'top_n_return': top_n_compound,
                'top_n_excess': top_n_compound - benchmark_compound,
                'top_30_return': top_30_compound,
                'top_30_excess': top_30_compound - benchmark_compound,
                'decile_return': decile_compound,
                'decile_excess': decile_compound - benchmark_compound,
                'decile_size_avg': int(np.mean(yd['decile_sizes'])),
                'index_returns': index_compound,
                'index_monthly': yd['index_monthly'],
                'feature_importance': avg_fi,
                'monthly_benchmark': yd['benchmark_monthly'],
                'monthly_top_n': yd['top_n_monthly'],
                'monthly_top_30': yd['top_30_monthly'],
                'monthly_decile': yd['decile_monthly'],
                'all_top_n_picks': yd['all_top_n_picks'],
                'all_top_30_picks': yd['all_top_30_picks'],
                'all_decile_picks': yd['all_decile_picks'],
            })

        return results

    def _save_yearly_reports(self, reports_dir: str, yearly_results: list):
        """Save per-year performance reports."""
        for yr in yearly_results:
            filepath = os.path.join(reports_dir, f'year_{yr["year"]}_report.xlsx')

            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                # Model performance
                perf_row = {
                    'year': yr['year'],
                    'months_tested': yr['n_months'],
                    'avg_train_samples': yr['n_train_avg'],
                    'avg_test_samples': yr['n_test_avg'],
                    'avg_accuracy': round(yr['accuracy'], 3),
                    'avg_auc': round(yr['auc'], 3),
                    'avg_precision': round(yr['precision'], 3),
                    'avg_recall': round(yr['recall'], 3),
                    'avg_f1': round(yr['f1'], 3),
                    'benchmark_return_compounded': round(yr['benchmark_return'], 2),
                    f'top_{self.top_n}_return_compounded': round(yr['top_n_return'], 2),
                    f'top_{self.top_n}_excess': round(yr['top_n_excess'], 2),
                    'top_30_return_compounded': round(yr['top_30_return'], 2),
                    'top_30_excess': round(yr['top_30_excess'], 2),
                    'decile_return_compounded': round(yr['decile_return'], 2),
                    'decile_excess': round(yr['decile_excess'], 2),
                }
                for ticker in NORDIC_INDEXES:
                    val = yr.get('index_returns', {}).get(ticker)
                    perf_row[f'{ticker}_return'] = round(val, 2) if val is not None else None
                perf = pd.DataFrame([perf_row])
                perf.to_excel(writer, sheet_name='performance', index=False)

                # Monthly breakdown
                monthly_dict = {
                    'month': yr['months'],
                    'benchmark_return': [round(r, 2) for r in yr['monthly_benchmark']],
                    f'top_{self.top_n}_return': [round(r, 2) for r in yr['monthly_top_n']],
                    'top_30_return': [round(r, 2) for r in yr['monthly_top_30']],
                    'decile_return': [round(r, 2) for r in yr['monthly_decile']],
                }
                for ticker in NORDIC_INDEXES:
                    vals = yr.get('index_monthly', {}).get(ticker, [])
                    monthly_dict[ticker] = [round(v, 2) if v is not None else None for v in vals]
                monthly_data = pd.DataFrame(monthly_dict)
                monthly_data.to_excel(writer, sheet_name='monthly_breakdown', index=False)

                # Feature importance (avg across months)
                yr['feature_importance'].head(20).to_excel(
                    writer, sheet_name='feature_importance', index=False
                )

                # Top N picks — show which stocks were picked each month
                pick_display_cols = ['instrument_id', 'company_name', 'sector',
                                     'score', 'next_month_return', 'next_month_excess_return']
                for m, picks_df in zip(yr['months'], yr['all_top_n_picks']):
                    display_cols = [c for c in pick_display_cols if c in picks_df.columns]
                    sheet_name = f'picks_month_{m:02d}'
                    picks_df[display_cols].to_excel(writer, sheet_name=sheet_name, index=False)

            print(f"  Saved report: {filepath}")

    def _save_summary_report(self, reports_dir: str, yearly_results: list,
                              monthly_results: list):
        """Save the cross-year summary report."""
        filepath = os.path.join(reports_dir, 'summary_report.xlsx')

        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # Yearly overview
            yearly_rows = []
            for r in yearly_results:
                row = {
                    'year': r['year'],
                    'months': r['n_months'],
                    'avg_auc': round(r['auc'], 3),
                    'avg_accuracy': round(r['accuracy'], 3),
                    'benchmark_return': round(r['benchmark_return'], 2),
                    f'top_{self.top_n}_return': round(r['top_n_return'], 2),
                    f'top_{self.top_n}_excess': round(r['top_n_excess'], 2),
                    'top_30_return': round(r['top_30_return'], 2),
                    'top_30_excess': round(r['top_30_excess'], 2),
                    'decile_return': round(r['decile_return'], 2),
                    'decile_excess': round(r['decile_excess'], 2),
                    'decile_size_avg': r['decile_size_avg'],
                }
                for ticker in NORDIC_INDEXES:
                    val = r.get('index_returns', {}).get(ticker)
                    row[ticker] = round(val, 2) if val is not None else None
                yearly_rows.append(row)
            yearly_df = pd.DataFrame(yearly_rows)
            yearly_df.to_excel(writer, sheet_name='yearly_overview', index=False)

            # Cumulative performance (invest 100 at start)
            benchmark_cum = 100.0
            top_n_cum = 100.0
            top_30_cum = 100.0
            decile_cum = 100.0
            index_cum = {t: 100.0 for t in NORDIC_INDEXES}
            cum_rows = []

            for r in yearly_results:
                benchmark_cum *= (1 + r['benchmark_return'] / 100)
                top_n_cum *= (1 + r['top_n_return'] / 100)
                top_30_cum *= (1 + r['top_30_return'] / 100)
                decile_cum *= (1 + r['decile_return'] / 100)
                row = {
                    'year': r['year'],
                    'benchmark_cumulative': round(benchmark_cum, 2),
                    f'top_{self.top_n}_cumulative': round(top_n_cum, 2),
                    'top_30_cumulative': round(top_30_cum, 2),
                    'decile_cumulative': round(decile_cum, 2),
                }
                for ticker in NORDIC_INDEXES:
                    val = r.get('index_returns', {}).get(ticker)
                    if val is not None:
                        index_cum[ticker] *= (1 + val / 100)
                    row[f'{ticker}_cumulative'] = round(index_cum[ticker], 2)
                cum_rows.append(row)

            pd.DataFrame(cum_rows).to_excel(writer, sheet_name='cumulative_performance', index=False)

            # All months detail
            monthly_detail_rows = []
            for r in monthly_results:
                row = {
                    'year': r['year'],
                    'month': r['month'],
                    'n_test': r['n_test'],
                    'auc': round(r['auc'], 3),
                    'benchmark': round(r['benchmark_return'], 2),
                    f'top_{self.top_n}': round(r['top_n_return'], 2),
                    'top_30': round(r['top_30_return'], 2),
                    'decile': round(r['decile_return'], 2),
                }
                for ticker in NORDIC_INDEXES:
                    val = r.get('index_returns', {}).get(ticker)
                    row[ticker] = round(val, 2) if val is not None else None
                monthly_detail_rows.append(row)
            monthly_detail = pd.DataFrame(monthly_detail_rows)
            monthly_detail.to_excel(writer, sheet_name='all_months', index=False)

            # Model stability - feature importance across years
            importance_data = {}
            for r in yearly_results:
                fi = r['feature_importance'].set_index('feature')['importance']
                importance_data[r['year']] = fi
            if importance_data:
                stability = pd.DataFrame(importance_data)
                stability['avg_importance'] = stability.mean(axis=1)
                stability['std_importance'] = stability.std(axis=1)
                stability = stability.sort_values('avg_importance', ascending=False)
                stability.to_excel(writer, sheet_name='model_stability')

            # Summary metrics
            metrics_rows = [{
                'metric': 'Avg AUC',
                'value': round(yearly_df['avg_auc'].mean(), 3),
            }, {
                'metric': 'Avg Accuracy',
                'value': round(yearly_df['avg_accuracy'].mean(), 3),
            }, {
                'metric': f'Avg Top {self.top_n} Annual Return',
                'value': round(yearly_df[f'top_{self.top_n}_return'].mean(), 2),
            }, {
                'metric': f'Avg Top {self.top_n} Annual Excess',
                'value': round(yearly_df[f'top_{self.top_n}_excess'].mean(), 2),
            }, {
                'metric': 'Avg Top 30 Annual Return',
                'value': round(yearly_df['top_30_return'].mean(), 2),
            }, {
                'metric': 'Avg Top 30 Annual Excess',
                'value': round(yearly_df['top_30_excess'].mean(), 2),
            }, {
                'metric': 'Avg Decile Annual Return',
                'value': round(yearly_df['decile_return'].mean(), 2),
            }, {
                'metric': 'Avg Benchmark Annual Return',
                'value': round(yearly_df['benchmark_return'].mean(), 2),
            }, {
                'metric': 'Final Cumulative - Benchmark',
                'value': cum_rows[-1]['benchmark_cumulative'] if cum_rows else 0,
            }, {
                'metric': f'Final Cumulative - Top {self.top_n}',
                'value': cum_rows[-1][f'top_{self.top_n}_cumulative'] if cum_rows else 0,
            }, {
                'metric': 'Final Cumulative - Top 30',
                'value': cum_rows[-1]['top_30_cumulative'] if cum_rows else 0,
            }, {
                'metric': 'Final Cumulative - Decile',
                'value': cum_rows[-1]['decile_cumulative'] if cum_rows else 0,
            }]
            for ticker, info in NORDIC_INDEXES.items():
                metrics_rows.append({
                    'metric': f'Final Cumulative - {ticker} ({info["country"]})',
                    'value': cum_rows[-1].get(f'{ticker}_cumulative', 0) if cum_rows else 0,
                })
            metrics_rows.extend([{
                'metric': 'Years Outperformed (Top N vs Benchmark)',
                'value': sum(1 for r in yearly_results if r['top_n_return'] > r['benchmark_return']),
            }, {
                'metric': 'Total Test Years',
                'value': len(yearly_results),
            }, {
                'metric': 'Total Test Months',
                'value': len(monthly_results),
            }])
            pd.DataFrame(metrics_rows).to_excel(writer, sheet_name='summary_metrics', index=False)

            # Year-by-year overview (Beat? based on OMXS30)
            yby_rows = []
            for r in yearly_results:
                omxs30_val = r.get('index_returns', {}).get('OMXS30')
                if omxs30_val is not None:
                    beat = 'YES' if r['top_n_return'] > omxs30_val else 'no'
                else:
                    beat = 'N/A'
                row = {
                    'Year': r['year'],
                    'Months': r['n_months'],
                    'Benchmark': round(r['benchmark_return'], 2),
                    f'Top {self.top_n}': round(r['top_n_return'], 2),
                    'Top 30': round(r['top_30_return'], 2),
                    'Decile': round(r['decile_return'], 2),
                    'Beat? (vs OMXS30)': beat,
                }
                for ticker in NORDIC_INDEXES:
                    val = r.get('index_returns', {}).get(ticker)
                    row[ticker] = round(val, 2) if val is not None else None
                yby_rows.append(row)
            pd.DataFrame(yby_rows).to_excel(writer, sheet_name='year_by_year', index=False)

        print(f"\nSaved summary report: {filepath}")

    def _print_summary(self, yearly_results: list, monthly_results: list):
        """Print final summary to console."""
        print(f"\n{'='*70}")
        print(f"MONTHLY WALK-FORWARD BACKTEST SUMMARY")
        print(f"{'='*70}")
        print(f"Total months tested: {len(monthly_results)}")
        print(f"Years: {yearly_results[0]['year']} - {yearly_results[-1]['year']}")

        avg_auc = np.mean([r['auc'] for r in yearly_results])
        avg_accuracy = np.mean([r['accuracy'] for r in yearly_results])
        avg_benchmark = np.mean([r['benchmark_return'] for r in yearly_results])
        avg_top_n = np.mean([r['top_n_return'] for r in yearly_results])
        avg_top_30 = np.mean([r['top_30_return'] for r in yearly_results])
        avg_decile = np.mean([r['decile_return'] for r in yearly_results])

        print(f"\nModel Quality (avg across years):")
        print(f"  Avg AUC:       {avg_auc:.3f}")
        print(f"  Avg Accuracy:  {avg_accuracy:.1%}")

        print(f"\nPortfolio Performance (avg compounded annual return):")
        print(f"  Benchmark (market median):  {avg_benchmark:+.2f}%")
        print(f"  Top {self.top_n} portfolio:         {avg_top_n:+.2f}%")
        print(f"  Top 30 portfolio:           {avg_top_30:+.2f}%")
        print(f"  Top decile portfolio:       {avg_decile:+.2f}%")

        # Avg index returns
        for ticker, info in NORDIC_INDEXES.items():
            vals = [r['index_returns'].get(ticker) for r in yearly_results
                    if r.get('index_returns', {}).get(ticker) is not None]
            if vals:
                print(f"  {ticker} ({info['country']}):  {np.mean(vals):>+14.2f}%")

        # Cumulative
        benchmark_cum = 100.0
        top_n_cum = 100.0
        top_30_cum = 100.0
        decile_cum = 100.0
        index_cum = {t: 100.0 for t in NORDIC_INDEXES}
        for r in yearly_results:
            benchmark_cum *= (1 + r['benchmark_return'] / 100)
            top_n_cum *= (1 + r['top_n_return'] / 100)
            top_30_cum *= (1 + r['top_30_return'] / 100)
            decile_cum *= (1 + r['decile_return'] / 100)
            for ticker in NORDIC_INDEXES:
                val = r.get('index_returns', {}).get(ticker)
                if val is not None:
                    index_cum[ticker] *= (1 + val / 100)

        print(f"\nCumulative Growth (100 invested at start):")
        print(f"  Benchmark:         {benchmark_cum:.2f}")
        print(f"  Top {self.top_n} portfolio:  {top_n_cum:.2f}")
        print(f"  Top 30 portfolio:  {top_30_cum:.2f}")
        print(f"  Top decile:        {decile_cum:.2f}")
        for ticker, info in NORDIC_INDEXES.items():
            print(f"  {ticker:17s}  {index_cum[ticker]:.2f}")

        outperform_count = sum(1 for r in yearly_results if r['top_n_return'] > r['benchmark_return'])
        print(f"\nTop {self.top_n} beat benchmark: {outperform_count}/{len(yearly_results)} years")

        # Build dynamic header for index columns
        idx_tickers = [t for t in NORDIC_INDEXES
                       if any(r.get('index_returns', {}).get(t) is not None for r in yearly_results)]
        idx_header = ''.join(f' {t:>8}' for t in idx_tickers)
        idx_sep = '-' * (8 * len(idx_tickers) + len(idx_tickers))

        print(f"\nYear-by-year (compounded monthly returns):")
        print(f"  {'Year':<6} {'Months':>6} {'Benchmark':>10} {'Top '+str(self.top_n):>10} {'Top 30':>10} {'Decile':>10} {'Beat?':>6}{idx_header}")
        print(f"  {'-'*60}{idx_sep}")
        for r in yearly_results:
            beat = 'YES' if r['top_n_return'] > r['benchmark_return'] else 'no'
            idx_vals = ''
            for t in idx_tickers:
                v = r.get('index_returns', {}).get(t)
                idx_vals += f' {v:>+7.1f}%' if v is not None else f' {"N/A":>8}'
            print(f"  {r['year']:<6} {r['n_months']:>6} {r['benchmark_return']:>+9.2f}% "
                  f"{r['top_n_return']:>+9.2f}% {r['top_30_return']:>+9.2f}% {r['decile_return']:>+9.2f}% {beat:>6}{idx_vals}")

        print(f"\nOutputs saved to: {os.path.abspath(self.output_dir)}/")
        print(f"  debug/   - Model inputs + outputs per month")
        print(f"  reports/ - Yearly performance reports + summary")


def main():
    parser = argparse.ArgumentParser(
        description='Monthly walk-forward outperformance model with portfolio backtest')
    parser.add_argument('--db-host', default=os.getenv('DB_HOST', 'localhost'))
    parser.add_argument('--db-name', default=os.getenv('DB_NAME', 'borsdata'))
    parser.add_argument('--db-user', default=os.getenv('DB_USER', 'postgres'))
    parser.add_argument('--db-password', default=os.getenv('DB_PASSWORD'))
    parser.add_argument('--db-port', type=int, default=int(os.getenv('DB_PORT', 5432)))
    parser.add_argument('--min-train-months', type=int, default=36,
                        help='Minimum months of training data before first prediction (default: 36)')
    parser.add_argument('--top-n', type=int, default=20,
                        help='Number of top stocks to pick each month (default: 20)')
    parser.add_argument('--output-dir', default='results',
                        help='Output directory for debug and report files (default: results)')
    parser.add_argument('--predict-only', action='store_true',
                        help='Skip backtest, only run current-month prediction using all history as training data')
    parser.add_argument('--initial-capital', type=float, default=100_000,
                        help='Starting capital in SEK for account simulation (default: 100000)')
    parser.add_argument('--commission', type=float, default=0.0015,
                        help='Commission rate per transaction (default: 0.0015 = 0.15%%)')
    parser.add_argument('--max-stale-days', type=int, default=10,
                        help='Exclude stocks from current-month prediction whose last price '
                             'is more than this many days older than the freshest price '
                             'in the batch (default: 10). Prevents picking suspended/delisted stocks.')

    args = parser.parse_args()

    if not args.db_password:
        print("Error: Database password required (--db-password or DB_PASSWORD env var)")
        sys.exit(1)

    db_config = {
        'host': args.db_host,
        'database': args.db_name,
        'user': args.db_user,
        'password': args.db_password,
        'port': args.db_port
    }

    trainer = MonthlyWalkForwardTrainer(
        db_config,
        output_dir=args.output_dir,
        min_train_months=args.min_train_months,
        top_n=args.top_n,
        initial_capital=args.initial_capital,
        commission_rate=args.commission,
        max_stale_days=args.max_stale_days,
    )

    try:
        trainer.connect()
        trainer.run_walk_forward(predict_only=args.predict_only)
    finally:
        trainer.close()


if __name__ == '__main__':
    main()
