import argparse
import os


def main():
    """
    Main entry point for the mcp-server-qdrant script defined
    in pyproject.toml. It runs the MCP server with a specific transport
    protocol.
    """

    # Parse the command-line arguments to determine the transport protocol.
    parser = argparse.ArgumentParser(description="mcp-server-qdrant")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default="stdio",
    )
    # Host/port for the http transports. The bundled FastMCP console path does
    # not honour FASTMCP_HOST/FASTMCP_PORT (FastMCP is constructed before the
    # env is read), so accept them here and set the settings explicitly. This
    # is what lets one shared daemon bind a chosen port (e.g. 8077) without an
    # external launcher. CLI flag wins, then QDRANT_MCP_*, then FASTMCP_*.
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()

    # Import is done here to make sure environment variables are loaded
    # only after we make the changes.
    from mcp_server_qdrant.server import mcp

    if args.transport != "stdio":
        host = (
            args.host
            or os.environ.get("QDRANT_MCP_HOST")
            or os.environ.get("FASTMCP_HOST")
        )
        port = (
            args.port
            or os.environ.get("QDRANT_MCP_PORT")
            or os.environ.get("FASTMCP_PORT")
        )
        if host:
            mcp.settings.host = host
        if port:
            mcp.settings.port = int(port)

    mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()
