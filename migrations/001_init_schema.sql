-- 001_init_schema.sql
-- Initial schema for results and image classification

CREATE TABLE IF NOT EXISTS piece_result (
    id INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    jsn TEXT,
    operator_result TEXT CHECK (operator_result IN ('OK','NOK')),
    model_result TEXT CHECK (model_result IN ('OK','NOK')),
    final_result TEXT GENERATED ALWAYS AS (
        CASE
            WHEN operator_result = model_result THEN operator_result
            WHEN operator_result = 'OK' AND model_result = 'NOK' THEN 'FNOK'
            WHEN operator_result = 'NOK' AND model_result = 'OK' THEN 'FOK'
        END
    ) STORED
);

CREATE TABLE IF NOT EXISTS classified_images (
    id INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    img_name TEXT,
    operator_result TEXT CHECK (operator_result IN ('OK','NOK')),
    model_result TEXT CHECK (model_result IN ('OK','NOK')),
    piece_id INTEGER,
    CONSTRAINT fk_piece
        FOREIGN KEY (piece_id)
        REFERENCES piece_result(id)
        ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS img_results (
    id INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    img_name TEXT,
    result TEXT
);
