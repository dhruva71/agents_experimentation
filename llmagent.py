import math
import os
import logging
import yaml
from openai import OpenAI, Stream
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.chat.chat_completion import ChoiceLogprobs


class LLMAgent:
    def __init__(self, api_key: str, model_id: str = "deepseek/deepseek-v4-flash",
                 prompts_yaml_path: str = 'prompts/prompts.yaml'):
        self.api_key = api_key
        self.model_id = model_id
        self.response = None
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        self.logger.info(f'Using model: {self.model_id}')

        with open(prompts_yaml_path, 'r') as f:
            config = yaml.safe_load(f.read())

        if config is not None:
            self.system_prompt = config["system_prompt_2"]
        else:
            # TODO log this better
            self.logger.info("No system_prompt provided")
            self.system_prompt = ""

        print(f'{self.system_prompt=}\n')

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )

    def query_llm(self, user_query: str, reasoning_enabled: bool = False, **kwargs) -> ChatCompletion | Stream[
        ChatCompletionChunk]:
        """
        Get response from model `model_id` for `user_query`.
        :param user_query:
        :param reasoning_enabled: Whether reasoning is enabled or not.
        :return: JSON response
        """
        self.logger.debug(f'Using kwargs: {kwargs}')

        # noinspection bad-argument-type
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": user_query
                }
            ],
            extra_body={"reasoning": {"enabled": reasoning_enabled}},
            **kwargs
        )

        self.response = response
        return response

    def process_logprobs(self) -> ChoiceLogprobs | None:
        """
        Process log probabilities from the last response.
        :return: Log probabilities or None if not available.
        """
        logprobs = self.response.choices[0].logprobs
        if logprobs is not None and logprobs.content is not None:
            for logprob in logprobs.content:
                probability = math.exp(logprob.logprob)
                confidence_label = "high" if probability > 0.9 else "low"
                considered_tokens = []
                if logprob.top_logprobs is not None:
                    for toplogprob_token in logprob.top_logprobs:
                        considered_tokens.append(toplogprob_token.token)
                self.logger.debug(f'Token: {logprob.token}, logprob: {logprob.logprob}, considered_tokens: {considered_tokens}, probability: {probability}, confidence: {confidence_label}')
        else:
            self.logger.debug("Log probabilities not available in the response.")
        return logprobs
