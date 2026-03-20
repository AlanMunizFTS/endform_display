-- 002_add_remote_metadata_to_classified_images.sql
-- Extend classified_images with remote metadata and local creation timestamp

ALTER TABLE classified_images
    ADD COLUMN IF NOT EXISTS class_name TEXT;

ALTER TABLE classified_images
    ADD COLUMN IF NOT EXISTS confidence DECIMAL(5,4);

ALTER TABLE classified_images
    ADD COLUMN IF NOT EXISTS created_at TIMESTAMP;

ALTER TABLE classified_images
    ALTER COLUMN created_at SET DEFAULT CURRENT_TIMESTAMP;
