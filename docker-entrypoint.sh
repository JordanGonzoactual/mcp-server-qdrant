#!/usr/bin/env sh
set -eu

if [ "$#" -gt 0 ]; then
  exec "$@"
fi

transport="${MCP_TRANSPORT:-streamable-http}"
case "$transport" in
  stdio|sse|streamable-http) ;;
  *)
    echo "Invalid MCP_TRANSPORT: '$transport'. Expected one of: stdio, sse, streamable-http." >&2
    exit 1
    ;;
esac

exec mcp-server-qdrant --transport "$transport"
