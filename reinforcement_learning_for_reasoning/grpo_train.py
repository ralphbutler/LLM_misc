from datasets import load_dataset

dataset_id = 'AI-MO/NuminaMath-TIR'
train_dataset, test_dataset = load_dataset(dataset_id, split=['train[:5%]', 'test[:5%]'])

print(train_dataset)

print(train_dataset[0])

# In the **DeepSeek-R1** training procedure, a specific system prompt was used to generate a conversational pipeline that includes reasoning steps. We'll adapt our dataset to follow this approach, where the model is guided to first think through the problem and then present its answer.
# 
# The system prompt used is:

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

# an example:

print(train_dataset[0]['prompt'])

# remove messages and problem columns; we only need the custom prompt column and solution to verify the generated answer.  

train_dataset = train_dataset.remove_columns(['messages', 'problem'])
print(train_dataset)

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
# use these reward functions:
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
    max_completion_length=64, # default: 256
    num_generations=4, # default: 8
    max_prompt_length=128, # default: 512

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
    args=training_args,
    train_dataset=train_dataset
)

trainer.train()

trainer.save_model(training_args.output_dir)
# RMB trainer.push_to_hub(dataset_name=dataset_id)

print("**** EXITING") ; exit(0)
