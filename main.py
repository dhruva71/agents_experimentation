import os

from dotenv import load_dotenv

import llmagent

load_dotenv()

models = ["openai/gpt-5.6-luna", "nvidia/nemotron-3-ultra-550b-a55b:free", "deepseek/deepseek-v4-flash"]
selected_model_index = 2

api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise ValueError("API key not set")
llm = llmagent.LLMAgent(api_key=api_key, model_id=models[selected_model_index])

query = "Display budget vs actual for FY25 for department IT"
response = llm.query_llm(user_query=query, max_completion_tokens=100, stop=['Observation'], logprobs=True,
                         top_logprobs=10)

print(response)
logprobs = llm.process_logprobs()
content = response.choices[0].message.content

if content is not None:  # split for readability
    if "action" in content:
        print("Response contains action")
        print(f'{content=}')
