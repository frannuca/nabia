"""MCP servers exposing the agents' tools: news, stats and portfolio.

Each module runs as a stdio MCP server (``python -m nabia.ai.servers.<name>
--config ai_config.json``), so the agents here, Claude Code or any other MCP
client can use the same tools.
"""
