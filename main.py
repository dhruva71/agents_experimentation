import logging
import os

from dotenv import load_dotenv

import llmagent
import toolbox

if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s"
    )

    logger.info("Loading environment variables")
    load_dotenv()

    models = ["openai/gpt-5.6-luna", "nvidia/nemotron-3-ultra-550b-a55b:free", "deepseek/deepseek-v4-flash",
              "stealth/ox-alpha"]
    selected_model_index = -1

    # Intialize agent
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("API key not set")
    agent = llmagent.LLMAgent(api_key=api_key, model_id=models[selected_model_index])

    query = "Display budget vs actual for FY25 Q4 for department IT"
    logger.info(f"Executing with query: {query}")
    logger.debug(f"Selected model: {models[selected_model_index]}")
    reasoning_enabled = True
    response = agent.execute_query(user_query=query, reasoning_enabled=reasoning_enabled, max_completion_tokens=1000,
                                   stop=['Observation'], logprobs=True,
                                   top_logprobs=10)

    logger.debug(response)
    logprobs = agent.process_logprobs()
    content = response.choices[0].message.content

    if content is not None:  # split for readability
        if "action" in content:
            logger.debug("Response contains action")
            logger.debug(f'{content=}')
            action = agent.extract_action()
            logger.info(f'{action=}')
            print(f'Received action: {action}')
