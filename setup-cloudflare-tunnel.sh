#!/usr/bin/env zsh
# setup-cloudflare-tunnel.sh — Automates setting up a Cloudflare Tunnel for Odysseus.
# Designed for RJ Business Solutions | Rick Jefferson

set -e

echo "▶ Starting Cloudflare Tunnel Setup for Odysseus..."

# 1. Check if cloudflared is installed
if ! command -v cloudflared &> /dev/null; then
  echo "✗ cloudflared CLI is not installed. Please install it first:"
  echo "    brew install cloudflared"
  exit 1
fi

# 2. Login to Cloudflare
echo "▶ Running Cloudflare Login..."
echo "  This will open a browser window to authorize cloudflared."
cloudflared tunnel login

# 3. Prompt user for their Cloudflare Domain Name
echo -n "▶ Enter your Cloudflare registered domain name (e.g., rjsolutions.com): "
read DOMAIN_NAME
if [[ -z "$DOMAIN_NAME" ]]; then
  echo "✗ Domain name cannot be empty."
  exit 1
fi

SUBDOMAIN="odysseus.${DOMAIN_NAME}"

# 4. Create the tunnel
TUNNEL_NAME="odysseus"
echo "▶ Creating Cloudflare Tunnel named '${TUNNEL_NAME}'..."
TUNNEL_INFO=$(cloudflared tunnel create ${TUNNEL_NAME})
echo "$TUNNEL_INFO"

# Extract Tunnel ID
TUNNEL_ID=$(echo "$TUNNEL_INFO" | grep -oE "Created tunnel odysseus with id [a-f0-9-]+" | awk '{print $NF}')

if [[ -z "$TUNNEL_ID" ]]; then
  # Try fallback extraction
  TUNNEL_ID=$(echo "$TUNNEL_INFO" | grep -oE "[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}")
fi

if [[ -z "$TUNNEL_ID" ]]; then
  echo "✗ Failed to extract Tunnel ID. Please check the output above."
  exit 1
fi

echo "✔ Tunnel created successfully with ID: ${TUNNEL_ID}"

# 5. Write the Tunnel configuration file
CONFIG_PATH="${HOME}/.cloudflared/odysseus-config.yml"
echo "▶ Writing configuration to ${CONFIG_PATH}..."

cat <<EOF > "${CONFIG_PATH}"
tunnel: ${TUNNEL_ID}
credentials-file: ${HOME}/.cloudflared/${TUNNEL_ID}.json

ingress:
  - hostname: ${SUBDOMAIN}
    service: http://127.0.0.1:7860
  - service: http_status:404
EOF

# 6. Route DNS
echo "▶ Routing DNS for ${SUBDOMAIN} through the tunnel..."
cloudflared tunnel route dns ${TUNNEL_NAME} ${SUBDOMAIN}

# 7. Start the Tunnel
echo "▶ Starting the Cloudflare Tunnel in the background..."
cloudflared tunnel --config "${CONFIG_PATH}" run ${TUNNEL_NAME} &

echo "=================================================="
echo "✔ CLOUDFLARE TUNNEL SETUP COMPLETED SUCCESSFULLY!"
echo "=================================================="
echo "  - Access URL: https://${SUBDOMAIN}"
echo "  - Local Port: http://127.0.0.1:7860"
echo "  - Config File: ${CONFIG_PATH}"
echo "=================================================="
