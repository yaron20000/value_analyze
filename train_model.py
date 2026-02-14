#!/usr/bin/env python3
"""
Walk-Forward Outperformance Classifier with Portfolio Backtest
===============================================================
Trains a classification model to predict whether a stock will outperform
the market median next year. Uses predict_proba to produce a 0-1 score
for ranking stocks.

Each year:
1. Train on all prior years
2. Predict outperformance probability for all stocks in the test year
3. Rank stocks by probability and pick top 10 / top decile
4. Measure actual portfolio performance vs benchmark

Outputs:
- Excel debug files per year (train/test data sent to model)
- Excel report per year (model performance, portfolio picks)
- Summary report across all years (cumulative performance)

Usage:
    python train_model.py --db-password yourpass
    python train_model.py --db-password yourpass --min-train-years 5 --top-n 15
"""

import pandas as pd
import numpy as np
import psycopg2
import argparse
import os
import sys
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

from dotenv import load_dotenv
load_dotenv()


class WalkForwardTrainer:
    """Walk-forward expanding-window classifier for outperformance prediction."""

    def __init__(self, db_config: dict, output_dir: str = 'results',
                 min_train_years: int = 3, top_n: int = 10):
        self.db_config = db_config
        self.conn = None
        self.output_dir = output_dir
        self.min_train_years = min_train_years
        self.top_n = top_n

        # Feature columns — ratios only, no absolute/per-share scale-dependent features
        self.feature_cols = [
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
            # Pre-report price features
            'price_change_5d', 'price_change_10d', 'price_change_20d', 'price_change_30d',
            'volume_ratio_5d_20d',
            'volatility_5d', 'volatility_20d',
            'was_rising_5d', 'was_rising_10d', 'was_rising_20d',
            'pct_from_20d_high', 'pct_from_20d_low',
            # Holdings (count-based — scale-independent)
            'insider_buy_count', 'insider_sell_count',
            'insider_transaction_count', 'insider_buy_ratio',
            'buyback_count',
            # Normalized by market_cap (created in _prepare_features)
            'total_assets_to_mcap', 'net_debt_to_mcap', 'ocf_to_mcap', 'capex_to_mcap',
            'insider_amount_to_mcap',
            # Normalized by num_shares (created in _prepare_features)
            'insider_shares_to_total', 'buyback_shares_to_total',
        ]

        self.target_col = 'next_year_excess_return'  # used to derive binary target

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

    def load_data(self) -> pd.DataFrame:
        """Load all training data from the ml_training_data view."""
        print("Loading data from ml_training_data...")
        query = """
        SELECT *
        FROM ml_training_data
        ORDER BY instrument_id, year
        """
        df = pd.read_sql(query, self.conn)
        print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
        print(f"  Instruments: {df['instrument_id'].nunique()}")
        print(f"  Years: {df['year'].min()} - {df['year'].max()}")
        print(f"  Columns: {list(df.columns)}")
        return df

    def _create_normalized_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create market-cap and share-count normalized features for absolute metrics
        that don't already have ratio equivalents."""
        df = df.copy()

        # Normalize by market_cap (only features without existing ratio equivalents)
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

        # Normalize by num_shares
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
        Creates normalized features, then imputes and scales (fit on train only).

        Returns: X_train, X_test, y_train, y_test, scaler, available_cols, train_medians
        """
        available_cols = [c for c in self.feature_cols if c in train_df.columns]

        X_train = train_df[available_cols].copy()
        X_test = test_df[available_cols].copy()

        # Binary target: 1 if stock outperformed market median, 0 otherwise
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

        return X_train_scaled, X_test_scaled, y_train, y_test, scaler, available_cols, train_medians

    def run_walk_forward(self):
        """Run the full walk-forward expanding-window backtest."""
        df = self.load_data()

        # Filter to rows with valid target
        df_valid = df[df[self.target_col].notna()].copy()
        print(f"  Rows with valid {self.target_col}: {len(df_valid)}")

        years = sorted(df_valid['year'].unique())
        print(f"  Available years: {years[0]} - {years[-1]} ({len(years)} years)")
        print(f"  Min training years: {self.min_train_years}")
        print(f"  Walk-forward folds: {len(years) - self.min_train_years}")

        # Create output directories
        debug_dir = os.path.join(self.output_dir, 'debug')
        reports_dir = os.path.join(self.output_dir, 'reports')
        os.makedirs(debug_dir, exist_ok=True)
        os.makedirs(reports_dir, exist_ok=True)

        all_results = []

        for i in range(self.min_train_years, len(years)):
            train_years = years[:i]
            test_year = years[i]

            print(f"\n{'='*70}")
            print(f"FOLD: Train on {train_years[0]}-{train_years[-1]} ({len(train_years)} years) -> Test on {test_year}")
            print(f"{'='*70}")

            train_df = df_valid[df_valid['year'].isin(train_years)].copy()
            test_df = df_valid[df_valid['year'] == test_year].copy()

            if len(test_df) < 5:
                print(f"  Skipping: only {len(test_df)} stocks in test year")
                continue

            # Create normalized features on the dataframes before anything else
            train_df = self._create_normalized_features(train_df)
            test_df = self._create_normalized_features(test_df)

            print(f"  Train: {len(train_df)} rows, Test: {len(test_df)} stocks")

            # Prepare features
            X_train, X_test, y_train, y_test, _scaler, available_cols, train_medians = \
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

            # Predict probability of outperformance (0-1 score)
            proba = model.predict_proba(X_test)[:, 1]  # probability of class 1 (outperform)
            test_df['score'] = proba
            predictions = model.predict(X_test)

            # Build prediction input snapshot: the exact scaled values sent to the model
            id_data = {}
            for col in ['instrument_id', 'company_name', 'sector']:
                if col in test_df.columns:
                    id_data[col] = test_df[col].values
            prediction_input_df = pd.DataFrame(id_data, index=test_df.index)
            prediction_input_df[available_cols] = X_test
            prediction_input_df['score'] = proba
            prediction_input_df['predicted_outperform'] = predictions

            # Evaluate classification quality
            accuracy = accuracy_score(y_test, predictions)
            auc = roc_auc_score(y_test, proba) if len(y_test.unique()) > 1 else 0.0
            precision = precision_score(y_test, predictions, zero_division=0)
            recall = recall_score(y_test, predictions, zero_division=0)
            f1 = f1_score(y_test, predictions, zero_division=0)

            print(f"  Accuracy: {accuracy:.1%}, AUC: {auc:.3f}, Precision: {precision:.1%}, Recall: {recall:.1%}, F1: {f1:.3f}")

            # Rank stocks by outperformance probability
            ranked = test_df.sort_values('score', ascending=False).copy()
            ranked['rank'] = range(1, len(ranked) + 1)

            # Pick top N and top decile
            top_n_picks = ranked.head(self.top_n).copy()
            decile_size = max(1, len(ranked) // 10)
            top_decile_picks = ranked.head(decile_size).copy()

            # Calculate portfolio returns
            # Use actual next_year_return (not excess) for real portfolio performance
            benchmark_return = test_df['market_median_return'].iloc[0] if 'market_median_return' in test_df.columns else 0

            top_n_actual_return = top_n_picks['next_year_return'].mean() if 'next_year_return' in top_n_picks.columns else np.nan
            top_n_excess = top_n_picks[self.target_col].mean()
            decile_actual_return = top_decile_picks['next_year_return'].mean() if 'next_year_return' in top_decile_picks.columns else np.nan
            decile_excess = top_decile_picks[self.target_col].mean()

            print(f"  Benchmark (market median): {benchmark_return:.2f}%")
            print(f"  Top {self.top_n} portfolio: {top_n_actual_return:.2f}% return ({top_n_excess:+.2f}% excess)")
            print(f"  Top decile ({decile_size}): {decile_actual_return:.2f}% return ({decile_excess:+.2f}% excess)")

            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': available_cols,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)

            print(f"  Top 5 features: {', '.join(feature_importance.head(5)['feature'].tolist())}")

            # Store results
            fold_result = {
                'test_year': test_year,
                'train_years': f"{train_years[0]}-{train_years[-1]}",
                'n_train': len(train_df),
                'n_test': len(test_df),
                'accuracy': accuracy,
                'auc': auc,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'benchmark_return': benchmark_return,
                'top_n_return': top_n_actual_return,
                'top_n_excess': top_n_excess,
                'decile_return': decile_actual_return,
                'decile_excess': decile_excess,
                'decile_size': decile_size,
                'feature_importance': feature_importance,
                'ranked_df': ranked,
                'top_n_picks': top_n_picks,
                'top_decile_picks': top_decile_picks,
            }
            all_results.append(fold_result)

            # Save debug Excel for this year
            self._save_debug_excel(debug_dir, test_year, train_df, test_df,
                                   ranked, available_cols, train_medians,
                                   prediction_input_df)

            # Save year report
            self._save_year_report(reports_dir, test_year, fold_result,
                                   train_df, available_cols)

        if not all_results:
            print("\nNo walk-forward folds were run. Check your data.")
            return

        # Generate summary report
        self._save_summary_report(reports_dir, all_results)

        # Print final summary
        self._print_summary(all_results)

    def _save_debug_excel(self, debug_dir: str, test_year: int,
                          train_df: pd.DataFrame, test_df: pd.DataFrame,
                          ranked: pd.DataFrame, available_cols: list,
                          train_medians: pd.Series,
                          prediction_input_df: pd.DataFrame = None):
        """Save debug data for one fold to Excel."""
        filepath = os.path.join(debug_dir, f'year_{test_year}.xlsx')

        # Columns to include for readability
        id_cols = ['instrument_id', 'year', 'company_name', 'sector', 'market']
        target_cols = [self.target_col, 'next_year_return', 'market_median_return']

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
                                      'next_year_return', 'market_median_return'] if c in ranked.columns]
            ranked.head(self.top_n)[pick_cols].to_excel(writer, sheet_name='top_n_picks', index=False)

            # Top decile picks
            decile_size = max(1, len(ranked) // 10)
            ranked.head(decile_size)[pick_cols].to_excel(writer, sheet_name='top_decile_picks', index=False)

            # All predictions ranked
            ranked[pick_cols].to_excel(writer, sheet_name='all_ranked', index=False)

            # Prediction input: exact scaled feature values sent to the model per stock
            if prediction_input_df is not None:
                prediction_input_df.sort_values('score', ascending=False).to_excel(
                    writer, sheet_name='prediction_input', index=False)

        print(f"  Saved debug: {filepath}")

    def _save_year_report(self, reports_dir: str, test_year: int, result: dict,
                          train_df: pd.DataFrame = None, available_cols: list = None):
        """Save a per-year performance report."""
        filepath = os.path.join(reports_dir, f'year_{test_year}_report.xlsx')

        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # Model performance
            perf = pd.DataFrame([{
                'test_year': result['test_year'],
                'train_period': result['train_years'],
                'train_samples': result['n_train'],
                'test_samples': result['n_test'],
                'accuracy': round(result['accuracy'], 3),
                'auc': round(result['auc'], 3),
                'precision': round(result['precision'], 3),
                'recall': round(result['recall'], 3),
                'f1': round(result['f1'], 3),
            }])
            perf.to_excel(writer, sheet_name='model_performance', index=False)

            # Feature importances (top 20)
            result['feature_importance'].head(20).to_excel(
                writer, sheet_name='feature_importance', index=False)

            # Top N portfolio
            top_n = result['top_n_picks'].copy()
            display_cols = [c for c in ['rank', 'instrument_id', 'company_name', 'sector',
                                         'score', self.target_col,
                                         'next_year_return', 'market_median_return']
                           if c in top_n.columns]
            top_n_display = top_n[display_cols].copy()

            # Add summary row
            summary_row = pd.DataFrame([{
                'rank': '',
                'instrument_id': '',
                'company_name': f'PORTFOLIO AVG (top {self.top_n})',
                'sector': '',
                'score': top_n['score'].mean() if 'score' in top_n.columns else np.nan,
                self.target_col: result['top_n_excess'],
                'next_year_return': result['top_n_return'],
                'market_median_return': result['benchmark_return'],
            }])
            top_n_out = pd.concat([top_n_display, summary_row], ignore_index=True)
            top_n_out.to_excel(writer, sheet_name='portfolio_top_n', index=False)

            # Top decile portfolio
            top_dec = result['top_decile_picks'].copy()
            top_dec_display = top_dec[[c for c in display_cols if c in top_dec.columns]].copy()
            dec_summary = pd.DataFrame([{
                'rank': '',
                'instrument_id': '',
                'company_name': f'PORTFOLIO AVG (top decile, {result["decile_size"]})',
                'sector': '',
                'score': top_dec['score'].mean() if 'score' in top_dec.columns else np.nan,
                self.target_col: result['decile_excess'],
                'next_year_return': result['decile_return'],
                'market_median_return': result['benchmark_return'],
            }])
            top_dec_out = pd.concat([top_dec_display, dec_summary], ignore_index=True)
            top_dec_out.to_excel(writer, sheet_name='portfolio_decile', index=False)

            # All stocks ranked
            ranked = result['ranked_df']
            all_cols = [c for c in display_cols if c in ranked.columns]
            ranked[all_cols].to_excel(writer, sheet_name='all_predictions', index=False)

            # Training data per instrument
            if train_df is not None and available_cols is not None:
                id_cols = ['instrument_id', 'year', 'company_name', 'sector', 'market']
                target_cols = [self.target_col, 'next_year_return', 'market_median_return']
                export_cols = [c for c in id_cols + available_cols + target_cols if c in train_df.columns]
                train_export = train_df[export_cols].sort_values(['instrument_id', 'year'])
                train_export.to_excel(writer, sheet_name='training_data', index=False)

        print(f"  Saved report: {filepath}")

    def _save_summary_report(self, reports_dir: str, all_results: list):
        """Save the cross-year summary report."""
        filepath = os.path.join(reports_dir, 'summary_report.xlsx')

        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # Yearly overview
            yearly = pd.DataFrame([{
                'year': r['test_year'],
                'train_period': r['train_years'],
                'n_train': r['n_train'],
                'n_test': r['n_test'],
                'accuracy': round(r['accuracy'], 3),
                'auc': round(r['auc'], 3),
                'precision': round(r['precision'], 3),
                'recall': round(r['recall'], 3),
                'f1': round(r['f1'], 3),
                'benchmark_return': round(r['benchmark_return'], 2),
                f'top_{self.top_n}_return': round(r['top_n_return'], 2),
                f'top_{self.top_n}_excess': round(r['top_n_excess'], 2),
                'decile_return': round(r['decile_return'], 2),
                'decile_excess': round(r['decile_excess'], 2),
                'decile_size': r['decile_size'],
            } for r in all_results])
            yearly.to_excel(writer, sheet_name='yearly_overview', index=False)

            # Cumulative performance (invest 100 at start)
            cumulative = pd.DataFrame()
            cumulative['year'] = yearly['year']
            cumulative['benchmark_return_pct'] = yearly['benchmark_return']
            cumulative[f'top_{self.top_n}_return_pct'] = yearly[f'top_{self.top_n}_return']
            cumulative['decile_return_pct'] = yearly['decile_return']

            # Calculate cumulative growth of 100
            benchmark_cum = 100.0
            top_n_cum = 100.0
            decile_cum = 100.0
            cum_benchmark = []
            cum_top_n = []
            cum_decile = []

            for _, row in cumulative.iterrows():
                benchmark_cum *= (1 + row['benchmark_return_pct'] / 100)
                top_n_cum *= (1 + row[f'top_{self.top_n}_return_pct'] / 100)
                decile_cum *= (1 + row['decile_return_pct'] / 100)
                cum_benchmark.append(round(benchmark_cum, 2))
                cum_top_n.append(round(top_n_cum, 2))
                cum_decile.append(round(decile_cum, 2))

            cumulative['benchmark_cumulative'] = cum_benchmark
            cumulative[f'top_{self.top_n}_cumulative'] = cum_top_n
            cumulative['decile_cumulative'] = cum_decile
            cumulative.to_excel(writer, sheet_name='cumulative_performance', index=False)

            # Model stability - feature importance across years
            importance_data = {}
            for r in all_results:
                fi = r['feature_importance'].set_index('feature')['importance']
                importance_data[r['test_year']] = fi

            if importance_data:
                stability = pd.DataFrame(importance_data)
                stability['avg_importance'] = stability.mean(axis=1)
                stability['std_importance'] = stability.std(axis=1)
                stability = stability.sort_values('avg_importance', ascending=False)
                stability.to_excel(writer, sheet_name='model_stability')

            # Average metrics
            avg_metrics = pd.DataFrame([{
                'metric': 'Avg Accuracy',
                'value': round(yearly['accuracy'].mean(), 3),
            }, {
                'metric': 'Avg AUC',
                'value': round(yearly['auc'].mean(), 3),
            }, {
                'metric': 'Avg Precision',
                'value': round(yearly['precision'].mean(), 3),
            }, {
                'metric': 'Avg Recall',
                'value': round(yearly['recall'].mean(), 3),
            }, {
                'metric': 'Avg F1',
                'value': round(yearly['f1'].mean(), 3),
            }, {
                'metric': f'Avg Top {self.top_n} Return',
                'value': round(yearly[f'top_{self.top_n}_return'].mean(), 2),
            }, {
                'metric': f'Avg Top {self.top_n} Excess Return',
                'value': round(yearly[f'top_{self.top_n}_excess'].mean(), 2),
            }, {
                'metric': 'Avg Decile Return',
                'value': round(yearly['decile_return'].mean(), 2),
            }, {
                'metric': 'Avg Decile Excess Return',
                'value': round(yearly['decile_excess'].mean(), 2),
            }, {
                'metric': 'Avg Benchmark Return',
                'value': round(yearly['benchmark_return'].mean(), 2),
            }, {
                'metric': f'Final Cumulative (100 invested) - Benchmark',
                'value': cum_benchmark[-1] if cum_benchmark else 0,
            }, {
                'metric': f'Final Cumulative (100 invested) - Top {self.top_n}',
                'value': cum_top_n[-1] if cum_top_n else 0,
            }, {
                'metric': f'Final Cumulative (100 invested) - Decile',
                'value': cum_decile[-1] if cum_decile else 0,
            }, {
                'metric': 'Years Outperformed (Top N vs Benchmark)',
                'value': sum(1 for r in all_results if r['top_n_return'] > r['benchmark_return']),
            }, {
                'metric': 'Total Test Years',
                'value': len(all_results),
            }])
            avg_metrics.to_excel(writer, sheet_name='summary_metrics', index=False)

        print(f"\nSaved summary report: {filepath}")

    def _print_summary(self, all_results: list):
        """Print final summary to console."""
        print(f"\n{'='*70}")
        print(f"WALK-FORWARD BACKTEST SUMMARY")
        print(f"{'='*70}")
        print(f"Folds: {len(all_results)}")
        print(f"Years: {all_results[0]['test_year']} - {all_results[-1]['test_year']}")

        avg_accuracy = np.mean([r['accuracy'] for r in all_results])
        avg_auc = np.mean([r['auc'] for r in all_results])
        avg_precision = np.mean([r['precision'] for r in all_results])
        avg_recall = np.mean([r['recall'] for r in all_results])
        avg_f1 = np.mean([r['f1'] for r in all_results])
        avg_benchmark = np.mean([r['benchmark_return'] for r in all_results])
        avg_top_n = np.mean([r['top_n_return'] for r in all_results])
        avg_decile = np.mean([r['decile_return'] for r in all_results])

        print(f"\nModel Quality:")
        print(f"  Avg Accuracy:  {avg_accuracy:.1%}")
        print(f"  Avg AUC:       {avg_auc:.3f}")
        print(f"  Avg Precision:  {avg_precision:.1%}")
        print(f"  Avg Recall:     {avg_recall:.1%}")
        print(f"  Avg F1:         {avg_f1:.3f}")

        print(f"\nPortfolio Performance (avg annual return):")
        print(f"  Benchmark (market median):  {avg_benchmark:+.2f}%")
        print(f"  Top {self.top_n} portfolio:          {avg_top_n:+.2f}%")
        print(f"  Top decile portfolio:       {avg_decile:+.2f}%")

        # Cumulative
        benchmark_cum = 100.0
        top_n_cum = 100.0
        decile_cum = 100.0
        for r in all_results:
            benchmark_cum *= (1 + r['benchmark_return'] / 100)
            top_n_cum *= (1 + r['top_n_return'] / 100)
            decile_cum *= (1 + r['decile_return'] / 100)

        print(f"\nCumulative Growth (100 invested at start):")
        print(f"  Benchmark:         {benchmark_cum:.2f}")
        print(f"  Top {self.top_n} portfolio:  {top_n_cum:.2f}")
        print(f"  Top decile:        {decile_cum:.2f}")

        outperform_count = sum(1 for r in all_results if r['top_n_return'] > r['benchmark_return'])
        print(f"\nTop {self.top_n} beat benchmark: {outperform_count}/{len(all_results)} years")

        print(f"\nYear-by-year:")
        print(f"  {'Year':<6} {'Benchmark':>10} {'Top '+str(self.top_n):>10} {'Decile':>10} {'Beat?':>6}")
        print(f"  {'-'*44}")
        for r in all_results:
            beat = 'YES' if r['top_n_return'] > r['benchmark_return'] else 'no'
            print(f"  {r['test_year']:<6} {r['benchmark_return']:>+9.2f}% {r['top_n_return']:>+9.2f}% {r['decile_return']:>+9.2f}% {beat:>6}")

        print(f"\nOutputs saved to: {os.path.abspath(self.output_dir)}/")
        print(f"  debug/   - Excel files with raw data per year")
        print(f"  reports/ - Performance reports per year + summary")


def main():
    parser = argparse.ArgumentParser(
        description='Walk-forward excess return model with portfolio backtest')
    parser.add_argument('--db-host', default=os.getenv('DB_HOST', 'localhost'))
    parser.add_argument('--db-name', default=os.getenv('DB_NAME', 'borsdata'))
    parser.add_argument('--db-user', default=os.getenv('DB_USER', 'postgres'))
    parser.add_argument('--db-password', default=os.getenv('DB_PASSWORD'))
    parser.add_argument('--db-port', type=int, default=int(os.getenv('DB_PORT', 5432)))
    parser.add_argument('--min-train-years', type=int, default=3,
                        help='Minimum years of training data before first prediction (default: 3)')
    parser.add_argument('--top-n', type=int, default=10,
                        help='Number of top stocks to pick each year (default: 10)')
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

    trainer = WalkForwardTrainer(
        db_config,
        output_dir=args.output_dir,
        min_train_years=args.min_train_years,
        top_n=args.top_n
    )

    try:
        trainer.connect()
        trainer.run_walk_forward()
    finally:
        trainer.close()


if __name__ == '__main__':
    main()
