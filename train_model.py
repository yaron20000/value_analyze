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
                 min_train_months: int = 36, top_n: int = 20):
        self.db_config = db_config
        self.conn = None
        self.output_dir = output_dir
        self.min_train_months = min_train_months
        self.top_n = top_n
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
            'dividend_payout',
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

        self.feature_cols = self.fundamental_cols + self.price_cols + self.holdings_cols + self.normalized_cols

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
        """Load monthly targets (returns)."""
        print("Loading monthly targets...")
        df = pd.read_sql("""
            SELECT instrument_id, year, month, month_end_date, month_end_price,
                   next_month_return, market_median_monthly_return, next_month_excess_return
            FROM ml_monthly_targets
            WHERE next_month_excess_return IS NOT NULL
            ORDER BY year, month, instrument_id
        """, self.conn)
        print(f"  Monthly targets: {len(df)} rows, "
              f"{df['instrument_id'].nunique()} instruments, "
              f"{df['year'].min()}-{df['month'].iloc[0]:02d} to "
              f"{df['year'].max()}-{df['month'].iloc[-1]:02d}")
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
                f.dividend_payout,
                -- Absolute metrics for normalization
                f.market_cap, f.num_shares, f.total_assets, f.net_debt,
                f.ocf, f.capex
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

    def _assemble_monthly_features(self, targets_df: pd.DataFrame,
                                    fundamentals_df: pd.DataFrame,
                                    price_features_df: pd.DataFrame,
                                    holdings_df: pd.DataFrame) -> pd.DataFrame:
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

        # Create normalized features
        merged = self._create_normalized_features(merged)

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

    def run_walk_forward(self):
        """Run the full monthly walk-forward expanding-window backtest."""

        # Load all data
        targets_df = self._load_monthly_targets()
        fundamentals_df = self._load_fundamentals_with_dates()
        price_features_df = self._load_monthly_price_features()
        holdings_df = self._load_holdings()

        # Assemble full feature matrix
        df = self._assemble_monthly_features(
            targets_df, fundamentals_df, price_features_df, holdings_df
        )

        # Fetch Nordic index returns from Avanza
        print("\nFetching Nordic index monthly returns from Avanza...")
        self.index_returns = fetch_nordic_index_returns()

        # Filter to rows with valid target and fundamentals
        df_valid = df[df[self.target_col].notna() & df['report_date'].notna()].copy()
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
            decile_size = max(1, len(ranked) // 10)
            top_decile_picks = ranked.head(decile_size).copy()

            # Portfolio returns (actual next-month returns)
            benchmark_return = test_df['market_median_monthly_return'].iloc[0]
            top_n_return = top_n_picks['next_month_return'].mean()
            top_n_excess = top_n_picks[self.target_col].mean()
            decile_return = top_decile_picks['next_month_return'].mean()
            decile_excess = top_decile_picks[self.target_col].mean()

            # Look up Nordic index returns for this month
            month_index_returns = {}
            for ticker, returns_dict in self.index_returns.items():
                month_index_returns[ticker] = returns_dict.get((test_year, test_month))

            idx_str = ''.join(
                f" {t}={r:+.1f}%" for t, r in month_index_returns.items() if r is not None
            )
            print(f" | Top{self.top_n}={top_n_return:+.2f}% Bench={benchmark_return:+.2f}%{idx_str}")

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
                'decile_return': decile_return,
                'decile_excess': decile_excess,
                'decile_size': decile_size,
                'index_returns': month_index_returns,
                'feature_importance': feature_importance,
                'top_n_picks': top_n_picks,
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

        # Print summary
        self._print_summary(yearly_results, all_monthly_results)

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
                    'all_decile_picks': [],
                }
            yearly[y]['months'].append(mr['month'])
            yearly[y]['benchmark_monthly'].append(mr['benchmark_return'])
            yearly[y]['top_n_monthly'].append(mr['top_n_return'])
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
            yearly[y]['all_decile_picks'].append(mr['top_decile_picks'])

        # Compound monthly returns for each year
        results = []
        for y in sorted(yearly.keys()):
            yd = yearly[y]
            # Compound: product of (1 + r/100) - 1, times 100
            benchmark_compound = (np.prod([1 + r/100 for r in yd['benchmark_monthly']]) - 1) * 100
            top_n_compound = (np.prod([1 + r/100 for r in yd['top_n_monthly']]) - 1) * 100
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
                'decile_return': decile_compound,
                'decile_excess': decile_compound - benchmark_compound,
                'decile_size_avg': int(np.mean(yd['decile_sizes'])),
                'index_returns': index_compound,
                'index_monthly': yd['index_monthly'],
                'feature_importance': avg_fi,
                'monthly_benchmark': yd['benchmark_monthly'],
                'monthly_top_n': yd['top_n_monthly'],
                'monthly_decile': yd['decile_monthly'],
                'all_top_n_picks': yd['all_top_n_picks'],
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
            decile_cum = 100.0
            index_cum = {t: 100.0 for t in NORDIC_INDEXES}
            cum_rows = []

            for r in yearly_results:
                benchmark_cum *= (1 + r['benchmark_return'] / 100)
                top_n_cum *= (1 + r['top_n_return'] / 100)
                decile_cum *= (1 + r['decile_return'] / 100)
                row = {
                    'year': r['year'],
                    'benchmark_cumulative': round(benchmark_cum, 2),
                    f'top_{self.top_n}_cumulative': round(top_n_cum, 2),
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
        avg_decile = np.mean([r['decile_return'] for r in yearly_results])

        print(f"\nModel Quality (avg across years):")
        print(f"  Avg AUC:       {avg_auc:.3f}")
        print(f"  Avg Accuracy:  {avg_accuracy:.1%}")

        print(f"\nPortfolio Performance (avg compounded annual return):")
        print(f"  Benchmark (market median):  {avg_benchmark:+.2f}%")
        print(f"  Top {self.top_n} portfolio:         {avg_top_n:+.2f}%")
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
        decile_cum = 100.0
        index_cum = {t: 100.0 for t in NORDIC_INDEXES}
        for r in yearly_results:
            benchmark_cum *= (1 + r['benchmark_return'] / 100)
            top_n_cum *= (1 + r['top_n_return'] / 100)
            decile_cum *= (1 + r['decile_return'] / 100)
            for ticker in NORDIC_INDEXES:
                val = r.get('index_returns', {}).get(ticker)
                if val is not None:
                    index_cum[ticker] *= (1 + val / 100)

        print(f"\nCumulative Growth (100 invested at start):")
        print(f"  Benchmark:         {benchmark_cum:.2f}")
        print(f"  Top {self.top_n} portfolio:  {top_n_cum:.2f}")
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
        print(f"  {'Year':<6} {'Months':>6} {'Benchmark':>10} {'Top '+str(self.top_n):>10} {'Decile':>10} {'Beat?':>6}{idx_header}")
        print(f"  {'-'*50}{idx_sep}")
        for r in yearly_results:
            beat = 'YES' if r['top_n_return'] > r['benchmark_return'] else 'no'
            idx_vals = ''
            for t in idx_tickers:
                v = r.get('index_returns', {}).get(t)
                idx_vals += f' {v:>+7.1f}%' if v is not None else f' {"N/A":>8}'
            print(f"  {r['year']:<6} {r['n_months']:>6} {r['benchmark_return']:>+9.2f}% "
                  f"{r['top_n_return']:>+9.2f}% {r['decile_return']:>+9.2f}% {beat:>6}{idx_vals}")

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
        top_n=args.top_n
    )

    try:
        trainer.connect()
        trainer.run_walk_forward()
    finally:
        trainer.close()


if __name__ == '__main__':
    main()
