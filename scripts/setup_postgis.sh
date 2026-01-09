#!/bin/bash
# Setup PostgreSQL, PostGIS, pgRouting, and osmium-tool for Arch Linux

set -e

# =========================
# CONFIG
# =========================
DB_NAME="gisdb"
DB_USER="gisuser"
DB_PASSWORD="gispass"

# =========================
# Setup PostgreSQL, PostGIS, and pgRouting
# =========================

echo "[0/6] Installing PostgreSQL, PostGIS, and pgRouting..."

# Install required dependencies
sudo pacman -S --noconfirm base-devel postgresql cmake make postgis

# Initialize PostgreSQL if not already initialized
if [ ! -d "/var/lib/postgres/data" ]; then
    echo "Initializing PostgreSQL database cluster..."
    sudo -u postgres initdb --locale=en_US.UTF-8 -D /var/lib/postgres/data
fi

# Enable and start PostgreSQL service
echo "[1/6] Enabling PostgreSQL service..."
sudo systemctl enable postgresql

echo "[2/6] Starting PostgreSQL service..."
sudo systemctl start postgresql
sleep 2

# Install pgRouting via AUR using yay
if command -v yay &> /dev/null; then
    yay -S --noconfirm pgrouting
else
    echo "ERROR: pgRouting is in the AUR. Please install yay first:"
    echo "  sudo pacman -S --needed base-devel git"
    echo "  git clone https://aur.archlinux.org/yay.git"
    echo "  cd yay && makepkg -si"
    exit 1
fi

echo "Ensuring pgRouting extension is present..."
sudo -iu postgres psql -d "$DB_NAME" -c "CREATE EXTENSION IF NOT EXISTS pgrouting;" || true

echo "Verify pgRouting version (optional):"
psql "postgresql://$DB_USER:$DB_PASSWORD@localhost:5432/$DB_NAME" -c "SELECT pgr_version();" || true

echo "=== PostgreSQL + PostGIS setup (SAFE MODE) ==="

# =========================
# Step 2: Create user (if not exists)
# =========================
echo "[3/6] Creating database user (if not exists)..."
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
# Step 3: Create database (if not exists)
# =========================
echo "[4/6] Creating database if it does not exist..."
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
echo "[5/6] Enabling PostGIS extensions..."
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

# =========================
# Install osmium-tool
# =========================

echo "[6/6] Installing osmium-tool for efficient PBF filtering..."

if command -v osmium &> /dev/null; then
    echo "osmium-tool is already installed"
    osmium --version
    exit 0
fi

# Check if the system is Arch Linux
if [ -f /etc/arch-release ] || grep -q "ID=arch" /etc/os-release 2>/dev/null; then
    echo "Detected Arch Linux"

    # Check if yay is installed and use it to install osmium-tool
    if command -v yay &> /dev/null; then
        yay -S --noconfirm osmium-tool
    else
        echo "ERROR: osmium-tool is in AUR. Please install yay first:"
        echo "  sudo pacman -S --needed base-devel git"
        echo "  git clone https://aur.archlinux.org/yay.git"
        echo "  cd yay && makepkg -si"
        exit 1
    fi
else
    echo "ERROR: This script only supports Arch Linux. Please install osmium-tool manually."
    echo "See: https://osmcode.org/osmium-tool/"
    exit 1
fi

echo "osmium-tool installed successfully!"
osmium --version
