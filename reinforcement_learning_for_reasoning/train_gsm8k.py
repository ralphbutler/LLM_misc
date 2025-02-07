
from datasets import load_dataset, Dataset

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

# (train_dataset,test_dataset) = load_dataset('openai/gsm8k', split=['train[:5%]', 'test[:5%]'])
## train_dataset = load_dataset('openai/gsm8k', 'main')['train']
## print(train_dataset)
## print(train_dataset[0])

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

def get_gsm8k_questions(split = "train") -> Dataset:
    data = load_dataset('openai/gsm8k', 'main')[split] # type: ignore
    print(data)
    print(data[0])
    data = data.map(lambda x: { # type: ignore
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['question']}
        ],
        'answer': extract_hash_answer(x['answer'])
    }) # type: ignore
    data = data.select(list(range(0,1500)))  ### RMB just use a few to test
    return data # type: ignore

train_dataset = get_gsm8k_questions()

print(train_dataset)
print(train_dataset[0])

# We'll remove the `messages` and `problem` columns, as we only need the custom `prompt` column and `solution` to verify the generated answer.  

train_dataset = train_dataset.remove_columns(['question'])
train_dataset = train_dataset.rename_column('answer', 'solution')
print(train_dataset)
print(train_dataset[0])

# ## 3. Post-Training the Base Model Using GRPO

# ### 3.1 Loading the Baseline Model
# 
# To begin, we'll load [Qwen/Qwen2-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct) as the baseline model (`Policy Model` in the diagram above). With only 0.5 billion parameters, it is lightweight and fits within the available resources. However, for better results, a larger [alternative](https://x.com/jiayi_pirate/status/1882839487417561307) should be considered.  
# 

import torch
from transformers import AutoModelForCausalLM

model_id = "Qwen/Qwen2-0.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto",
)

# ### 3.3 Loading Reward Functions
# 
# For the reward component of the system, we can use either pretrained reward models or reward functions defined directly in code. For training, the DeepSeek-R1 authors used an accuracy-based reward model evaluates whether the response is correct, alongside a format-based reward that ensures the model places its reasoning process between `<think> </think>` tags. You can find more details [here](https://github.com/huggingface/open-r1/blob/main/src/open_r1/grpo.py). We can simply define and implement these reward functions as generic Python functions.
# 
# In this case, we will utilize these reward functions:
# 
# 1. **Format Enforcement:** Ensures that the generation follows a specific format using `<think> </think> <answer> </answer>` tags for reasoning.  

import re
def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content) for content in completion_contents]
    rewards_list = [1.0 if match else 0.0 for match in matches]
    return [1.0 if match else 0.0 for match in matches]

def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

# Reward functions
# def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
def correctness_reward_func(prompts, completions, **kwargs) -> list[float]:
    answer = kwargs["solution"]
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1])*0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]

# OLD reward stuff
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

# ### 3.4 Configuring GRPO Training Parameters
# 
# Next, let's configure the training parameters for GRPO. We recommend experimenting with the `max_completion_length`, `num_generations`, and `max_prompt_length` parameters (refer to the image at the beginning for details about each of them).
# 
# To keep things simple, we’ll start by training for just one epoch and reducing the `max_completion_length`, `num_generations`, and `max_prompt_length` from their default values.

from trl import GRPOConfig

# Configure training arguments using GRPOConfig
training_args = GRPOConfig(
    output_dir="Qwen2-0.5B-GRPO-test",
    learning_rate=1e-5,
    remove_unused_columns=False, # to access the solution column in accuracy_reward
    gradient_accumulation_steps=16,
    num_train_epochs=1,
    bf16=True,

    # Parameters that control de data preprocessing
    max_completion_length=256,  # 64, # default: 256
    num_generations=4, # default: 8
    max_prompt_length=256,  # 128, # default: 512

    # Parameters related to reporting and saving
    report_to="none",  # RMB ["tensorboard"],
    logging_steps=10,
    # RMB push_to_hub=True,
    save_strategy="steps",
    save_steps=10,
)

# ### 3.5 Training the Model 🏃
# 
# Now, let's configure the trainer and start training the model!
# 
# In this case, we pass the two reward functions we previously defined to the trainer
# 

from trl import GRPOTrainer

trainer = GRPOTrainer(
    model=model,
    reward_funcs=[format_reward, accuracy_reward],
    # processing_class = tokenizer,  # RMB new 
    ## reward_funcs = [
        ## xmlcount_reward_func,
        ## soft_format_reward_func,
        ## strict_format_reward_func,
        ## int_reward_func,
        ## correctness_reward_func,
    ## ],
    args=training_args,
    train_dataset=train_dataset
)

print("BEGIN TRAIN")
trainer.train()
print("DONE TRAIN")

trainer.save_model(training_args.output_dir)
# RMB trainer.push_to_hub(dataset_name=dataset_id)

print("**** EXITING") ; exit(0)

