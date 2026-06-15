"""
AgentMine × Agent Frameworks — 一行代码导出日志

LangChain:
    from agentmine.integrations.langchain_callback import AgentMineCallback
    agent = create_agent(..., callbacks=[AgentMineCallback("logs.jsonl")])

LlamaIndex:
    from agentmine.integrations.llamaindex_observer import AgentMineObserver
    Settings.callback_manager = CallbackManager([AgentMineObserver("logs.jsonl")])

通用（任何框架）:
    from agentmine.integrations.langchain_callback import AgentMineLogger
    logger = AgentMineLogger("logs.jsonl")
    logger.log(query="...", output="...", error=None)
"""
