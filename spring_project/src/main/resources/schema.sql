-- schema.sql

-- Create table
CREATE TABLE IF NOT EXISTS devices (
    id BIGSERIAL PRIMARY KEY,
    battery_power INTEGER NOT NULL,
    bluetooth BOOLEAN NOT NULL,
    clock_speed DOUBLE PRECISION NOT NULL,
    dual_sim BOOLEAN NOT NULL,
    front_camera INTEGER NOT NULL,
    four_g BOOLEAN NOT NULL,
    internal_memory INTEGER NOT NULL,
    mobile_depth DOUBLE PRECISION NOT NULL,
    mobile_weight INTEGER NOT NULL,
    num_cores INTEGER NOT NULL,
    primary_camera INTEGER NOT NULL,
    pixel_height INTEGER NOT NULL,
    pixel_width INTEGER NOT NULL,
    ram INTEGER NOT NULL,
    screen_height INTEGER NOT NULL,
    screen_width INTEGER NOT NULL,
    talk_time INTEGER NOT NULL,
    three_g BOOLEAN NOT NULL,
    touch_screen BOOLEAN NOT NULL,
    wifi BOOLEAN NOT NULL,
    price_range INTEGER
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_price_range ON devices(price_range);
CREATE INDEX IF NOT EXISTS idx_ram ON devices(ram);