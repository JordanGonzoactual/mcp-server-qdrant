FROM python:3.11-slim

WORKDIR /app

# Install uv for package management.
RUN pip install --no-cache-dir uv

# Copy local source so container builds include branch-specific changes.
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh

# Install the package from the locked dependency set.
RUN uv sync --frozen --no-dev \
    && chmod +x /usr/local/bin/docker-entrypoint.sh

# Expose the default port for HTTP MCP transports.
EXPOSE 8000

# Listen on all interfaces so Docker port publishing works.
ENV PATH="/app/.venv/bin:${PATH}"
ENV FASTMCP_SERVER_HOST="0.0.0.0"
ENV FASTMCP_SERVER_PORT="8000"
# Legacy envs kept for compatibility with older FastMCP variants.
ENV FASTMCP_HOST="0.0.0.0"
ENV FASTMCP_PORT="8000"
ENV MCP_TRANSPORT="streamable-http"

ENTRYPOINT ["docker-entrypoint.sh"]
