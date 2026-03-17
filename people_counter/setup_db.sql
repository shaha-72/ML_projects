-- ─────────────────────────────────────────
--  People Counter — MySQL Setup Script
--  Run this ONCE before starting the app.
-- ─────────────────────────────────────────

CREATE DATABASE IF NOT EXISTS people_counter
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;

USE people_counter;

CREATE TABLE IF NOT EXISTS daily_counts (
    id          INT AUTO_INCREMENT PRIMARY KEY,
    count_date  DATE         NOT NULL,
    camera_id   VARCHAR(64)  NOT NULL DEFAULT 'default',
    people_in   INT          NOT NULL DEFAULT 0,
    people_out  INT          NOT NULL DEFAULT 0,
    last_update DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP
                             ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY uq_date_cam (count_date, camera_id)
);

-- Optional: view today's data quickly
CREATE OR REPLACE VIEW v_today AS
    SELECT camera_id, people_in, people_out, last_update
    FROM daily_counts
    WHERE count_date = CURDATE();
