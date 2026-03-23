-- 002_add_remote_metadata_to_classified_images.sql
-- Keep created_at on classified_images, move defect metadata to child tables,
-- and extend piece_result with local creation timestamp.

ALTER TABLE classified_images
    ADD COLUMN IF NOT EXISTS created_at TIMESTAMP;

ALTER TABLE classified_images
    ALTER COLUMN created_at SET DEFAULT CURRENT_TIMESTAMP;

CREATE TABLE IF NOT EXISTS classified_image_defects (
    id INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    classified_image_id INTEGER NOT NULL,
    class_name TEXT NOT NULL,
    confidence DECIMAL(5,4) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_classified_image
        FOREIGN KEY (classified_image_id)
        REFERENCES classified_images(id)
        ON DELETE CASCADE
);

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'uq_classified_image_defects_image_class_confidence'
    ) THEN
        ALTER TABLE classified_image_defects
            ADD CONSTRAINT uq_classified_image_defects_image_class_confidence
            UNIQUE (classified_image_id, class_name, confidence);
    END IF;
END $$;

ALTER TABLE piece_result
    ADD COLUMN IF NOT EXISTS created_at TIMESTAMP;

ALTER TABLE piece_result
    ALTER COLUMN created_at SET DEFAULT CURRENT_TIMESTAMP;

CREATE TABLE IF NOT EXISTS piece_result_defects (
    id INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    piece_result_id INTEGER NOT NULL,
    class_name TEXT NOT NULL,
    confidence DECIMAL(5,4) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_piece_result_defect
        FOREIGN KEY (piece_result_id)
        REFERENCES piece_result(id)
        ON DELETE CASCADE
);

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'uq_piece_result_defects_piece_class'
    ) THEN
        ALTER TABLE piece_result_defects
            ADD CONSTRAINT uq_piece_result_defects_piece_class
            UNIQUE (piece_result_id, class_name);
    END IF;
END $$;
