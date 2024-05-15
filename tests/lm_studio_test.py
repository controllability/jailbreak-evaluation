from jailbreak_evaluation import MultifacetedEvaluation

# Initialize the jailbreak evaluation
openai_api_key = "lm-studio"
lm_studio_url = "http://localhost:5000/v1"
multifaceted_evaluation = MultifacetedEvaluation( openai_url=lm_studio_url, openai_api_key=openai_api_key)

# Construct a jailbreak attempt example
intent = "intent"
response = "response"

# Evaluate the above jailbreak attempt example
result = multifaceted_evaluation.evaluate(intent, response)

print("The evaluation result for all levels:")
print(result)

print("The evaluation result for combination level, which is our best accurate result:")
print(result.combination_level)