#!/usr/bin/env bash
set -e

# =========================
# CONFIG
# =========================
DB_NAME="gisdb"
DB_USER="gisuser"
DB_PASSWORD="gispass"

echo "=== PostgreSQL + PostGIS setup (SAFE MODE) ==="

# =========================
# Step 1: Enable & start PostgreSQL
# =========================
echo "[1/5] Enabling PostgreSQL service..."
sudo systemctl enable postgresql

echo "[2/5] Starting PostgreSQL service..."
sudo systemctl start postgresql
sleep 2

# =========================
# Step 2: Create user
# =========================
echo "[3/5] Creating database user (if not exists)..."
sudo -iu postgres psql <<EOF
DO \$\$
BEGIN
   IF NOT EXISTS (
      SELECT FROM pg_roles WHERE rolname = '$DB_USER'
   ) THEN
      CREATE ROLE $DB_USER
      LOGIN
      PASSWORD '$DB_PASSWORD'
      CREATEDB;
   END IF;
END
\$\$;
EOF

# =========================
# Step 3: Create database
# =========================
echo "[4/5] Creating database if it does not exist..."
DB_EXISTS=$(sudo -iu postgres psql -tAc "SELECT 1 FROM pg_database WHERE datname='$DB_NAME'")

if [ "$DB_EXISTS" != "1" ]; then
  sudo -iu postgres createdb -O "$DB_USER" "$DB_NAME"
  echo "Database '$DB_NAME' created."
else
  echo "Database '$DB_NAME' already exists."
fi

# =========================
# Step 4: Enable PostGIS
# =========================
echo "[5/5] Enabling PostGIS extensions..."
sudo -iu postgres psql -d "$DB_NAME" <<EOF
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS postgis_topology;
EOF

# =========================
# Step 5: Test connection
# =========================
echo "Testing connection..."
psql "postgresql://$DB_USER:$DB_PASSWORD@localhost:5432/$DB_NAME" \
  -c "SELECT PostGIS_Version();"

echo "=== Setup complete ==="
