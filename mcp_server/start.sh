# Start the local MCP server

set -e

echo "========================================="
echo "AscendC MCP Server"
echo "========================================="

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is required but not installed."
    exit 1
fi

echo "Python version: $(python3 --version)"

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

# Set remote server URL
export REMOTE_SERVER_URL=${REMOTE_SERVER_URL:-http://localhost:8089}
echo "Remote Server URL: $REMOTE_SERVER_URL"

# Start MCP server
echo ""
echo "Starting MCP Server..."
echo "Configure Claude Code to use this server."
echo "========================================="

python3 server.py
