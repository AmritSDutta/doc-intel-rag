class LLMService:
    def synthesize(self, user_prompt: str, max_output_tokens: int = 512) -> str:
        raise NotImplementedError

    def synthesize_agentic(self, prompt):
        raise NotImplementedError
