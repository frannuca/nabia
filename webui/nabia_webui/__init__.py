"""nabia_webui: the agents, their MCP tool servers and the web app, on top of the nabia package.

- ``nabia_webui.ai``: three agents (news, statistics, execution) that set the risk budgets, the
  MCP servers that give them their tools (``nabia_webui.ai.servers``), and a backtest mode.
- ``nabia_webui.web``: the local web app (FastAPI back end in ``web/app.py``, static front end in
  ``web/static``) that runs and inspects all of it.

Every computation goes through the ``nabia`` package of the Python project (../python):
``nabia.backtest`` / ``nabia.latest_weights`` (C++ engine), ``nabia.analytics`` and
``nabia.marketdata``.
"""
