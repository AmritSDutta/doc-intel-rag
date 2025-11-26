import logging
from typing import Optional

from google import genai
from google.genai import types
from google.genai.chats import Chat
from llama_index.core.agent import FunctionAgent
from llama_index.core.evaluation import FaithfulnessEvaluator
from llama_index.llms.google_genai import GoogleGenAI

from services.llm.base import LLMService

logger = logging.getLogger(__name__)


class GenAILLMService(LLMService):
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = "gemini-2.5-flash-lite"):
        """Long-lived GenAI client for synthesis (Gemini)."""
        self.client = genai.Client(api_key=api_key) if api_key else genai.Client()
        self.model = model
        self.safety_settings = [
            types.SafetySetting(
                category="HARM_CATEGORY_HATE_SPEECH",
                threshold="BLOCK_LOW_AND_ABOVE",
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_DANGEROUS_CONTENT",
                threshold="BLOCK_LOW_AND_ABOVE",
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_HARASSMENT",
                threshold="BLOCK_LOW_AND_ABOVE",
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                threshold="BLOCK_LOW_AND_ABOVE",
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_CIVIC_INTEGRITY",
                threshold="BLOCK_LOW_AND_ABOVE",
            )
        ]
        self.system_prompt = (
            "You are a helpful agent. "
            "You answer questions using retrieved context (RAG). "
            "Never hallucinate facts not present in the provided information."
        )

    def _get_response_text(self, agent: Chat, user_prompt: str) -> str:
        logging.info(f'user query: {user_prompt[:100]}')
        response = agent.send_message(user_prompt)
        synthesis: str = 'not available'
        if response and response.text:
            logging.info(f'Agent summarization response: {response.text[:100]}')
            logging.info(f'Agent token usage: {response.usage_metadata.total_token_count}')
            synthesis = response.text

        return synthesis

    def _get_synthesizer_agent(self) -> Chat:
        return self.client.chats.create(
            model=self.model,
            config=types.GenerateContentConfig(
                system_instruction=self.system_prompt,
                safety_settings=self.safety_settings
            )
        )

    def synthesize(self, user_prompt: str, max_output_tokens: int = 512) -> str:
        """Synthesize a short answer from a free-form prompt."""
        logger.info(f"synthesizing query: {user_prompt[:50] if user_prompt else 'NA'}")
        evaluator = FaithfulnessEvaluator(llm=GoogleGenAI(
            model="gemini-2.5-flash-lite",
        ))
        agent: Chat = self._get_synthesizer_agent()
        resp: str = self._get_response_text(agent, user_prompt)
        eval_result = evaluator.evaluate_response(response=resp)
        logger.info(f'FaithfulnessEvaluator: {str(eval_result.passing)}')
        return resp

    async def synthesize_agentic(self, user_prompt: str, max_output_tokens: int = 512) -> str:
        """
        LlamaIndex-style generation using Gemini LLM.
        """
        logger.info(f"Synthesizing query: {user_prompt[:80]}")
        _genai_llm = GoogleGenAI(
            model="gemini-2.5-flash-lite",
        )
        evaluator = FaithfulnessEvaluator(llm=_genai_llm)
        workflow = FunctionAgent(
            llm=_genai_llm,
            system_prompt=self.system_prompt
        )
        response = await workflow.run(user_msg=user_prompt)
        if not hasattr(response, "source_nodes"):
            response.source_nodes = hits  # whatever your retrieval returned
        eval_result = evaluator.evaluate_response(response=response)
        logger.info(f'FunctionAgent: {response.text}')
        logger.info(f'FaithfulnessEvaluator: {str(eval_result.passing)}')
        return response.text
