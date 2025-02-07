from datasets import load_dataset

dataset_id = 'AI-MO/NuminaMath-TIR'
train_dataset, test_dataset = load_dataset(dataset_id, split=['train[:5%]', 'test[:5%]'])

print(train_dataset)

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

def make_conversation(example):
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["problem"]},
        ],
    }

train_dataset = train_dataset.map(make_conversation)
test_dataset = test_dataset.map(make_conversation)

import torch
from transformers import AutoModelForCausalLM

import re
def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content) for content in completion_contents]
    rewards_list = [1.0 if match else 0.0 for match in matches]
    return [1.0 if match else 0.0 for match in matches]

# 2. **Solution Accuracy:** Verifies whether the solution to the problem is correct.

from math_verify import LatexExtractionConfig, parse, verify
def accuracy_reward(completions, **kwargs):
    """Reward function that checks if the completion is the same as the ground truth."""
    solutions = kwargs['solution']
    completion_contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, solution in zip(completion_contents, solutions):
        gold_parsed = parse(solution, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
        answer_parsed = parse(content, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
        if len(gold_parsed) != 0:
            try:
                rewards.append(float(verify(answer_parsed, gold_parsed)))
            except Exception:
                rewards.append(0.0)
        else:
            rewards.append(1.0)
    return rewards

# ## 4. Check the Model Performance
# 
# We've kept things simple so far, but now let's check if the model has already learned to reason. We'll load the saved model and run an evaluation on a test sample.

from transformers import AutoTokenizer

# model_id = "sergiopaniego/Qwen2-0.5B-GRPO"

### between 60 and 200 inclusive, seems pretty good, then it seems to get worse in the reasoning
### 90 may have been best ??
model = "Qwen2-0.5B-GRPO-test/checkpoint-90"  # 60 and 70 worked ; 50 did poorly

trained_model = AutoModelForCausalLM.from_pretrained(
    model,
    torch_dtype="auto",
    device_map="auto",
)
trained_tokenizer = AutoTokenizer.from_pretrained(model)

# Let's check one sample from the test set!

print("CHECKING")
print(test_dataset['prompt'][0])
print("-"*50)

# We'll create a function to interact with the model. In addition to generating the answer, we'll measure the inference duration and count the number of generated tokens. This will give us insights into how much the model has reasoned during generation.

import time

def generate_with_reasoning(prompt):
  # Build the prompt from the dataset
  prompt = " ".join(entry['content'] for entry in prompt)

  # Tokenize and move to the same device as the model
  inputs = trained_tokenizer(prompt, return_tensors="pt").to(trained_model.device)

  # Generate text without gradients
  start_time = time.time()
  with torch.no_grad():
      output_ids = trained_model.generate(**inputs, max_length=500)
  end_time = time.time()

  # Decode and extract model response
  generated_text = trained_tokenizer.decode(output_ids[0], skip_special_tokens=True)

  # Get inference time
  inference_duration = end_time - start_time

  # Get number of generated tokens
  num_input_tokens = inputs['input_ids'].shape[1]
  num_generated_tokens = output_ids.shape[1] - num_input_tokens

  return generated_text, inference_duration, num_generated_tokens

# Let's generate the answer for that test sample!

prompt = test_dataset['prompt'][0]
prompt[1]["content"] = "If John has 5 apples and gives 2 to Mary, how many apples does John have left?"
print("PROMPT",prompt)
generated_text, inference_duration, num_generated_tokens = generate_with_reasoning(prompt)
print("\nGENERATED")
print(generated_text)
print("-"*50)

# The model already demonstrates the ability to generate the correct `<think>` and `<answer>` tags, even though the solution itself is incorrect.
# 
# Given the inference time and the number of generated tokens, this approach shows potential benefits:

print(f"Inference time: {inference_duration:.2f} seconds")
print(f"Generated tokens: {num_generated_tokens}")

# Let’s review the generated response to better visualize this behavior:

prompt_text = " ".join(entry['content'] for entry in prompt)
response_text = generated_text[len(prompt_text):].strip()
print("\nRESPONSE TEXT")
print(response_text)

