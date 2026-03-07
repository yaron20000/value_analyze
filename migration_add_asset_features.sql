-- Migration: Add intangible_assets to ml_features
-- KPI 126 (ImmateriellaTillgangar) = total intangible assets incl. goodwill.
-- Börsdata does not expose goodwill as a separate KPI.
-- total_equity (KPI 58) already exists, no change needed for it.

ALTER TABLE ml_features
    ADD COLUMN IF NOT EXISTS intangible_assets NUMERIC(15,2);

COMMENT ON COLUMN ml_features.intangible_assets IS 'Total intangible assets incl. goodwill (Börsdata KPI 126)';
