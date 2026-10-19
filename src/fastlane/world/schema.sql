-- FastLane practice booking world. One database for all three domains.
-- Dates are ISO 'YYYY-MM-DD'; times are minutes after midnight (INTEGER) unless named *_at.

PRAGMA foreign_keys = ON;

CREATE TABLE meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE customers (
    phone      TEXT PRIMARY KEY,          -- 10-digit Indian mobile number
    name       TEXT NOT NULL,
    city       TEXT NOT NULL,
    created_on TEXT NOT NULL
);

-- ----------------------------------------------------------------- movies
CREATE TABLE cinemas (
    id       TEXT PRIMARY KEY,            -- e.g. 'galaxy-kukatpally'
    name     TEXT NOT NULL,
    area     TEXT NOT NULL,
    address  TEXT NOT NULL
);

CREATE TABLE screens (
    id         TEXT PRIMARY KEY,          -- e.g. 'galaxy-kukatpally-s1'
    cinema_id  TEXT NOT NULL REFERENCES cinemas(id),
    name       TEXT NOT NULL,
    rows       TEXT NOT NULL,             -- row letters, e.g. 'ABCDEFGH'
    seats_per_row INTEGER NOT NULL
);

CREATE TABLE seat_categories (
    screen_id TEXT NOT NULL REFERENCES screens(id),
    row       TEXT NOT NULL,
    category  TEXT NOT NULL CHECK (category IN ('classic', 'prime', 'recliner')),
    PRIMARY KEY (screen_id, row)
);

CREATE TABLE movies (
    id           TEXT PRIMARY KEY,
    title        TEXT NOT NULL,
    language     TEXT NOT NULL,
    genre        TEXT NOT NULL,
    certificate  TEXT NOT NULL CHECK (certificate IN ('U', 'UA', 'A')),
    runtime_min  INTEGER NOT NULL,
    release_date TEXT NOT NULL
);

CREATE TABLE shows (
    id         TEXT PRIMARY KEY,          -- e.g. 'SH-0412'
    screen_id  TEXT NOT NULL REFERENCES screens(id),
    movie_id   TEXT NOT NULL REFERENCES movies(id),
    date       TEXT NOT NULL,
    start_min  INTEGER NOT NULL,
    price_classic  INTEGER NOT NULL,
    price_prime    INTEGER NOT NULL,
    price_recliner INTEGER NOT NULL
);
CREATE INDEX shows_by_date ON shows(date, movie_id);

-- One row per *taken* seat; free seats are the screen layout minus these.
CREATE TABLE show_seats_taken (
    show_id    TEXT NOT NULL REFERENCES shows(id),
    seat       TEXT NOT NULL,             -- e.g. 'F7'
    booking_id TEXT REFERENCES bookings(id),   -- NULL = sold through another channel
    PRIMARY KEY (show_id, seat)
);

-- ------------------------------------------------------------------ buses
CREATE TABLE bus_routes (
    id          TEXT PRIMARY KEY,         -- e.g. 'HYD-BLR'
    origin      TEXT NOT NULL,
    destination TEXT NOT NULL,
    distance_km INTEGER NOT NULL,
    duration_min INTEGER NOT NULL,
    boarding_point TEXT NOT NULL
);

CREATE TABLE bus_trips (
    id         TEXT PRIMARY KEY,          -- e.g. 'TR-0123'
    route_id   TEXT NOT NULL REFERENCES bus_routes(id),
    date       TEXT NOT NULL,
    depart_min INTEGER NOT NULL,
    bus_type   TEXT NOT NULL CHECK (bus_type IN ('ac_sleeper', 'nonac_sleeper', 'ac_seater', 'nonac_seater')),
    fare       INTEGER NOT NULL,          -- per seat/berth, INR
    capacity   INTEGER NOT NULL
);
CREATE INDEX trips_by_date ON bus_trips(date, route_id);

CREATE TABLE bus_seats_taken (
    trip_id    TEXT NOT NULL REFERENCES bus_trips(id),
    seat       TEXT NOT NULL,             -- seater '1'..'40'; sleeper 'L1'..'L15', 'U1'..'U15'
    booking_id TEXT REFERENCES bookings(id),
    passenger_name   TEXT,
    passenger_age    INTEGER,
    passenger_gender TEXT CHECK (passenger_gender IN ('M', 'F', 'O')),
    PRIMARY KEY (trip_id, seat)
);

-- ------------------------------------------------------------ restaurants
CREATE TABLE restaurants (
    id          TEXT PRIMARY KEY,         -- e.g. 'tandoor-jubilee'
    name        TEXT NOT NULL,
    area        TEXT NOT NULL,
    address     TEXT NOT NULL,
    max_party   INTEGER NOT NULL,         -- largest party bookable by phone
    deposit_min_party INTEGER NOT NULL,   -- parties this size or larger pay a deposit
    deposit_per_guest INTEGER NOT NULL,
    lunch_open  INTEGER NOT NULL, lunch_last_seating  INTEGER NOT NULL,
    dinner_open INTEGER NOT NULL, dinner_last_seating INTEGER NOT NULL,
    closed_weekday INTEGER                -- 0 = Monday ... 6 = Sunday, NULL = open daily
);

CREATE TABLE restaurant_tables (
    id            TEXT PRIMARY KEY,       -- e.g. 'tandoor-jubilee-T05'
    restaurant_id TEXT NOT NULL REFERENCES restaurants(id),
    seats         INTEGER NOT NULL,
    zone          TEXT NOT NULL CHECK (zone IN ('indoor', 'outdoor', 'private'))
);

-- Each reservation occupies one table for a 90-minute seating.
CREATE TABLE table_reservations (
    table_id   TEXT NOT NULL REFERENCES restaurant_tables(id),
    date       TEXT NOT NULL,
    start_min  INTEGER NOT NULL,
    end_min    INTEGER NOT NULL,
    booking_id TEXT REFERENCES bookings(id),
    PRIMARY KEY (table_id, date, start_min)
);
CREATE INDEX reservations_by_date ON table_reservations(date, table_id);

-- --------------------------------------------------------------- bookings
CREATE TABLE bookings (
    id          TEXT PRIMARY KEY,         -- e.g. 'MV7K2Q9' / 'BS...' / 'RT...'
    domain      TEXT NOT NULL CHECK (domain IN ('movies', 'buses', 'restaurants')),
    phone       TEXT NOT NULL REFERENCES customers(phone),
    name        TEXT NOT NULL,
    status      TEXT NOT NULL CHECK (status IN ('confirmed', 'cancelled', 'pending_deposit', 'completed')),
    -- Domain item: exactly one of these groups is filled.
    show_id     TEXT REFERENCES shows(id),
    seat_category TEXT,
    trip_id     TEXT REFERENCES bus_trips(id),
    restaurant_id TEXT REFERENCES restaurants(id),
    res_date    TEXT,
    res_start_min INTEGER,
    party_size  INTEGER,
    quantity    INTEGER NOT NULL,         -- tickets, berths/seats or guests
    amount      INTEGER NOT NULL,         -- INR; deposit for restaurants
    special_request TEXT,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    cancelled_at TEXT,
    refund_amount INTEGER,
    source      TEXT NOT NULL DEFAULT 'seed'   -- 'seed' or 'agent'
);
CREATE INDEX bookings_by_phone ON bookings(phone);

-- Past bookings older than the inventory window (for caller history only).
CREATE TABLE booking_history (
    id        TEXT PRIMARY KEY,
    domain    TEXT NOT NULL,
    phone     TEXT NOT NULL REFERENCES customers(phone),
    date      TEXT NOT NULL,
    summary   TEXT NOT NULL,              -- one-line human readable
    details   TEXT NOT NULL,              -- JSON: venue, item, time, quantity, category...
    status    TEXT NOT NULL CHECK (status IN ('completed', 'cancelled', 'no_show'))
);
CREATE INDEX history_by_phone ON booking_history(phone);

-- ----------------------------------------------------------- side effects
CREATE TABLE sms_log (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    phone   TEXT NOT NULL,
    body    TEXT NOT NULL,
    sent_at TEXT NOT NULL
);

CREATE TABLE handoffs (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    phone    TEXT,
    reason   TEXT NOT NULL,
    summary  TEXT NOT NULL,
    created_at TEXT NOT NULL
);
