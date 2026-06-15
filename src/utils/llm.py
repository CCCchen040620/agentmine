"""
LLM 工厂（轻量版）
支持 OpenAI / Claude / 通义千问
"""
import os
import logging

logger = logging.getLogger(__name__)


def create_llm(model: str = None):
    """
    创建 LLM 实例用于 AgentMine 的深度分析

    使用 LiteLLM 实现多模型统一接口。
    如果 LiteLLM 不可用，回退到 langchain-openai。

    Args:
        model: 模型名 (如 gpt-4o, claude-sonnet-4-6, qwen-max)
               不传则从环境变量 AGENTMINE_LLM_MODEL 读取

    Returns:
        LangChain BaseChatModel 实例
    """
    model = model or os.getenv("AGENTMINE_LLM_MODEL", "gpt-4o-mini")

    # 尝试用 LiteLLM（更通用）
    try:
        from langchain_community.chat_models import ChatLiteLLM
        logger.info(f"Using ChatLiteLLM with model={model}")
        return ChatLiteLLM(
            model=model,
            temperature=0.1,
            max_tokens=2048,
        )
    except (ImportError, Exception):
        pass

    # Fallback: OpenAI
    try:
        from langchain_openai import ChatOpenAI
        logger.info(f"Using ChatOpenAI with model={model}")
        return ChatOpenAI(
            model=model,
            temperature=0.1,
            max_tokens=2048,
        )
    except ImportError:
        logger.error(
            "No LLM backend available. "
            "Install one: pip install langchain-openai or pip install langchain-community[litellm]"
        )
        raise
