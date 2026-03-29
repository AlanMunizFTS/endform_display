-- 003_add_local_model_results_staging.sql
-- Local staging table used by utilities/classifier.py to simulate the remote
-- Vision-Standard metadata stream before MainController syncs rows into the
-- app classification tables.

CREATE TABLE IF NOT EXISTS model_results_local (
    id INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    img_name TEXT NOT NULL,
    class_name TEXT NOT NULL,
    confidence DECIMAL(5,4) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_model_results_local_img_name
    ON model_results_local (img_name);

CREATE INDEX IF NOT EXISTS idx_model_results_local_created_at
    ON model_results_local (created_at);
