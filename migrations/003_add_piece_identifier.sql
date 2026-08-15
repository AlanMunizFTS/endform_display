-- 003_add_piece_identifier.sql
-- Optional operator-managed numeric identifier and its automatic sequence state.

ALTER TABLE piece_result
    ADD COLUMN IF NOT EXISTS piece_identifier BIGINT;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'ck_piece_result_piece_identifier_positive'
    ) THEN
        ALTER TABLE piece_result
            ADD CONSTRAINT ck_piece_result_piece_identifier_positive
            CHECK (piece_identifier IS NULL OR piece_identifier > 0);
    END IF;

    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'uq_piece_result_piece_identifier'
    ) THEN
        ALTER TABLE piece_result
            ADD CONSTRAINT uq_piece_result_piece_identifier
            UNIQUE (piece_identifier);
    END IF;
END $$;

CREATE TABLE IF NOT EXISTS piece_identifier_state (
    singleton BOOLEAN PRIMARY KEY DEFAULT TRUE CHECK (singleton),
    next_identifier BIGINT NULL CHECK (next_identifier IS NULL OR next_identifier > 0)
);

INSERT INTO piece_identifier_state (singleton, next_identifier)
VALUES (TRUE, NULL)
ON CONFLICT (singleton) DO NOTHING;
