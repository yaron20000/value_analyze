#!/usr/bin/env python3
"""
ML Training Script - Stock Prediction & Dividend Growth
========================================================
Example script showing how to use the ML-optimized schema for training models.

Models included:
1. Dividend Growth Prediction (Regression)
2. Dividend Increase Classifier (Binary Classification)
3. Stock Return Prediction (when implemented)

Usage:
    python train_model.py --db-password yourpass --model dividend-growth
    python train_model.py --db-password yourpass --model dividend-classifier
"""

import pandas as pd
import numpy as np
import psycopg2
import argparse
import os
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')


class MLTrainer:
    """Train ML models for stock prediction and dividend growth."""

    def __init__(self, db_config: dict):
        self.db_config = db_config
        self.conn = None

        # Feature columns for training (exclude identifiers and targets)
        self.feature_cols = [
            # Valuation
            'pe_ratio', 'ps_ratio', 'pb_ratio', 'ev_ebitda', 'peg_ratio', 'ev_sales',
            # Profitability
            'roe', 'roi', 'roa', 'ebitda_margin', 'operating_margin', 'gross_margin', 'net_margin',
            # Financial Health
            'debt_equity', 'equity_ratio', 'current_ratio', 'quick_ratio', 'interest_coverage',
            # Growth (exclude dividend_growth - it's in the target!)
            'revenue_growth', 'earnings_growth',
            # Per Share
            'eps', 'dividend_per_share', 'book_value_per_share', 'fcf_per_share', 'ocf_per_share',
            # Cash Flow
            'fcf_margin', 'earnings_fcf', 'ocf', 'capex',
            # Dividend
            'dividend_payout',
            # Absolute (scaled)
            'earnings', 'revenue', 'ebitda', 'total_assets', 'total_equity', 'net_debt', 'num_shares',
            # Pre-report price features
            'price_change_5d', 'price_change_10d', 'price_change_20d', 'price_change_30d',
            'volume_ratio_5d_20d',
            'volatility_5d', 'volatility_20d',
            'was_rising_5d', 'was_rising_10d', 'was_rising_20d',
            'pct_from_20d_high', 'pct_from_20d_low',
        ]

    def connect(self):
        """Connect to database."""
        self.conn = psycopg2.connect(
            host=self.db_config['host'],
            database=self.db_config['database'],
            user=self.db_config['user'],
            password=self.db_config['password'],
            port=self.db_config.get('port', 5432)
        )
        print(f"✓ Connected to database: {self.db_config['database']}")

    def load_training_data(self) -> pd.DataFrame:
        """Load data from ml_training_data view."""
        print("Loading training data from database...")

        query = """
        SELECT *
        FROM ml_training_data
        WHERE period = 5  -- Only full-year data
        ORDER BY instrument_id, year
        """

        df = pd.read_sql(query, self.conn)
        print(f"✓ Loaded {len(df)} rows, {len(df.columns)} columns")
        print(f"  Instruments: {df['instrument_id'].nunique()}")
        print(f"  Years: {df['year'].min()} - {df['year'].max()}")

        return df

    def prepare_features(self, df: pd.DataFrame, target_col: str):
        """
        Prepare features and target for training.

        Returns:
            X_train, X_test, y_train, y_test, scaler
        """
        print(f"\nPreparing features for target: {target_col}")

        # Filter: only rows where target is not null
        df_clean = df[df[target_col].notna()].copy()
        print(f"  Rows with valid target: {len(df_clean)}")

        # Select features (drop any not present in the dataframe)
        available_cols = [c for c in self.feature_cols if c in df_clean.columns]
        X = df_clean[available_cols].copy()
        y = df_clean[target_col].copy()

        # Split by time (not random!) - use early years for train, later for test
        # This prevents data leakage
        split_year = df_clean['year'].quantile(0.8)
        train_mask = df_clean['year'] < split_year
        test_mask = df_clean['year'] >= split_year

        X_train = X[train_mask]
        X_test = X[test_mask]
        y_train = y[train_mask]
        y_test = y[test_mask]

        print(f"  Train set: {len(X_train)} rows (years < {split_year:.0f})")
        print(f"  Test set: {len(X_test)} rows (years >= {split_year:.0f})")

        # Impute missing values using training set medians only (avoid leakage)
        train_medians = X_train.median()
        print(f"  Missing values before imputation: {X_train.isna().sum().sum()} (train), {X_test.isna().sum().sum()} (test)")
        X_train = X_train.fillna(train_medians)
        X_test = X_test.fillna(train_medians)
        print(f"  Missing values after imputation: {X_train.isna().sum().sum()} (train), {X_test.isna().sum().sum()} (test)")

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        return X_train_scaled, X_test_scaled, y_train, y_test, scaler, df_clean, available_cols

    def train_dividend_growth_model(self):
        """Train regression model to predict next year dividend growth %."""
        print("\n" + "="*70)
        print("TRAINING: Dividend Growth Prediction (Regression)")
        print("="*70)

        df = self.load_training_data()

        # Prepare data
        X_train, X_test, y_train, y_test, scaler, df_clean, used_cols = self.prepare_features(
            df, 'next_year_dividend_growth'
        )

        # Train Random Forest Regressor
        print("\nTraining Random Forest Regressor...")
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train, y_train)
        print("✓ Model trained")

        # Evaluate
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)

        print("\n" + "="*70)
        print("RESULTS: Dividend Growth Prediction")
        print("="*70)
        print(f"Training RMSE:   {train_rmse:.2f}%")
        print(f"Test RMSE:       {test_rmse:.2f}%")
        print(f"Training R²:     {train_r2:.3f}")
        print(f"Test R²:         {test_r2:.3f}")

        # Feature importance
        print("\nTop 10 Most Important Features:")
        feature_importance = pd.DataFrame({
            'feature': used_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        for idx, row in feature_importance.head(10).iterrows():
            print(f"  {row['feature']:25s} {row['importance']:.4f}")

        # Sample predictions
        print("\nSample Predictions (Test Set):")
        sample_df = df_clean.iloc[len(X_train):len(X_train)+10][
            ['instrument_id', 'year', 'company_name', 'dividend_per_share', 'next_year_dividend_growth']
        ].copy()
        sample_df['predicted_growth'] = test_pred[:10]
        sample_df['error'] = sample_df['predicted_growth'] - sample_df['next_year_dividend_growth']

        print(sample_df.to_string(index=False))

        return model, scaler, feature_importance

    def train_dividend_classifier(self):
        """Train binary classifier: will dividend increase next year?"""
        print("\n" + "="*70)
        print("TRAINING: Dividend Increase Classifier (Binary)")
        print("="*70)

        df = self.load_training_data()

        # Prepare data
        X_train, X_test, y_train, y_test, scaler, df_clean, used_cols = self.prepare_features(
            df, 'dividend_increased'
        )

        # Convert boolean to int
        y_train = y_train.astype(int)
        y_test = y_test.astype(int)

        # Check class balance
        print(f"\nClass distribution (train):")
        print(f"  Decreased/Same (0): {(y_train == 0).sum()} ({(y_train == 0).mean()*100:.1f}%)")
        print(f"  Increased (1):      {(y_train == 1).sum()} ({(y_train == 1).mean()*100:.1f}%)")

        # Train Random Forest Classifier
        print("\nTraining Random Forest Classifier...")
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'  # Handle imbalanced classes
        )

        model.fit(X_train, y_train)
        print("✓ Model trained")

        # Evaluate
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        test_proba = model.predict_proba(X_test)[:, 1]

        train_acc = (train_pred == y_train).mean()
        test_acc = (test_pred == y_test).mean()

        print("\n" + "="*70)
        print("RESULTS: Dividend Increase Classifier")
        print("="*70)
        print(f"Training Accuracy: {train_acc:.3f}")
        print(f"Test Accuracy:     {test_acc:.3f}")

        print("\nClassification Report (Test Set):")
        print(classification_report(y_test, test_pred,
                                   target_names=['No Increase', 'Increase']))

        print("\nConfusion Matrix (Test Set):")
        cm = confusion_matrix(y_test, test_pred)
        print(f"  True Negatives:  {cm[0,0]:3d}  |  False Positives: {cm[0,1]:3d}")
        print(f"  False Negatives: {cm[1,0]:3d}  |  True Positives:  {cm[1,1]:3d}")

        # Feature importance
        print("\nTop 10 Most Important Features:")
        feature_importance = pd.DataFrame({
            'feature': used_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        for idx, row in feature_importance.head(10).iterrows():
            print(f"  {row['feature']:25s} {row['importance']:.4f}")

        return model, scaler, feature_importance

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()


def main():
    parser = argparse.ArgumentParser(description='Train ML models for stock prediction')
    parser.add_argument('--db-host', default=os.getenv('DB_HOST', 'localhost'))
    parser.add_argument('--db-name', default=os.getenv('DB_NAME', 'borsdata'))
    parser.add_argument('--db-user', default=os.getenv('DB_USER', 'postgres'))
    parser.add_argument('--db-password', default=os.getenv('DB_PASSWORD'))
    parser.add_argument('--db-port', type=int, default=int(os.getenv('DB_PORT', 5432)))
    parser.add_argument('--model', choices=['dividend-growth', 'dividend-classifier', 'all'],
                       default='all', help='Which model to train')

    args = parser.parse_args()

    if not args.db_password:
        print("Error: Database password required (--db-password or DB_PASSWORD env var)")
        import sys
        sys.exit(1)

    db_config = {
        'host': args.db_host,
        'database': args.db_name,
        'user': args.db_user,
        'password': args.db_password,
        'port': args.db_port
    }

    trainer = MLTrainer(db_config)

    try:
        trainer.connect()

        if args.model in ['dividend-growth', 'all']:
            trainer.train_dividend_growth_model()

        if args.model in ['dividend-classifier', 'all']:
            trainer.train_dividend_classifier()

    finally:
        trainer.close()


if __name__ == '__main__':
    main()
