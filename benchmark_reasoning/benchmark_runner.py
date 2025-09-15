#!/usr/bin/env python3
"""
Benchmark Runner for Local LLM Evaluation

This program reads a YAML benchmark configuration and runs tests on a specified model.
Currently prints placeholder output - LLM integration to be added later.
"""

import yaml
import argparse
import sys
import os
import time
import signal
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from datasets import load_dataset
import pandas as pd
import random
import json
import requests
from urllib.parse import urlparse

# LiteLLM imports
from litellm import completion
import litellm

# Disable excessive logging unless debug mode
if os.environ.get('BENCHMARK_DEBUG') != '1':
    os.environ['LITELLM_LOG'] = 'DEBUG'

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("LLM call timed out")

class BenchmarkRunner:
    def __init__(self, config_path: str, model_name: str, timeout_seconds: int = 60):
        self.config_path = config_path
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds
        self.config = None
        self.test_problems = []
        self.model_config = None
        self.api_base = None
        self.results = []
        
    def load_model_config(self, config_file="model_config.tsv") -> Dict[str, str]:
        """Load model configuration from TSV file (same format as demo_litellm.py)"""
        model2apibase = {}
        
        if not Path(config_file).exists():
            print(f"Warning: Model config file '{config_file}' not found. Using model name directly.")
            return {self.model_name: ""}
            
        try:
            with open(config_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    # Skip comments and empty lines
                    if not line or line.startswith('#'):
                        continue
                    
                    # Parse TSV format
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        model_name = parts[0]
                        api_base = parts[1] if parts[1] else ""
                        model2apibase[model_name] = api_base
                    elif len(parts) == 1:
                        # Model with empty api_base
                        model_name = parts[0]
                        model2apibase[model_name] = ""
        
        except Exception as e:
            print(f"Error reading model config file: {e}")
            return {self.model_name: ""}
        
        return model2apibase
    
    def check_model_availability(self, allow_unknown: bool = False):
        """Check if the model is available and reachable"""
        self.model_config = self.load_model_config()

        if self.model_name not in self.model_config:
            if self.model_config:  # We have a config file with models
                print(f"Warning: Model '{self.model_name}' not found in config. Available models:")
                for model in self.model_config.keys():
                    print(f"  - {model}")

                # Always fail on obvious placeholders (unless explicitly allowed)
                placeholder_words = ['test', 'invalid', 'fake', 'model-name', 'example', 'placeholder', 'demo']
                if any(word in self.model_name.lower() for word in placeholder_words):
                    if allow_unknown:
                        print(f"WARNING: Model name '{self.model_name}' appears to be a placeholder, but proceeding due to --allow-unlisted-vendor flag.")
                    else:
                        print(f"ERROR: Model name '{self.model_name}' appears to be a placeholder.")
                        print("Please specify a valid model name from the config above.")
                        print("Or use --allow-unlisted-vendor if this is intentional for testing.")
                        sys.exit(1)

                if not allow_unknown:
                    # For dry-run, be more strict about unknown models
                    print(f"ERROR: Unknown model '{self.model_name}'. Use --allow-unlisted-vendor to proceed anyway.")
                    sys.exit(1)
                else:
                    print(f"Proceeding with '{self.model_name}' as cloud API or direct model name...")

            self.api_base = ""
        else:
            self.api_base = self.model_config[self.model_name]

        # Check if local server is running (but not if model is loaded)
        if self.api_base:
            try:
                parsed_url = urlparse(self.api_base)
                health_url = f"{parsed_url.scheme}://{parsed_url.netloc}/health"
                response = requests.get(health_url, timeout=5)
                print(f"✓ Local server at {self.api_base} is running")
            except Exception as e:
                print(f"✗ ERROR: Cannot reach local server at {self.api_base}")
                print("Please start your local LLM server before running the benchmark.")
                sys.exit(1)
        else:
            print(f"Using cloud API or direct model name: {self.model_name}")

        # Test that the specific model is loaded and working
        print(f"Testing model '{self.model_name}' with a simple completion...")

        def test_completion():
            test_messages = [{"role": "user", "content": "What is 2+2?"}]
            completion_kwargs = {
                "model": self.model_name,
                "messages": test_messages,
                "max_tokens": 10,
                "temperature": 0.0,
            }
            if self.api_base:
                completion_kwargs["api_base"] = self.api_base

            # Debug: Print what we're sending
            print(f"  Debug: Sending model name: '{self.model_name}'")
            print(f"  Debug: API base: '{self.api_base}'")

            return completion(**completion_kwargs)

        try:
            # Use ThreadPoolExecutor for reliable timeout
            from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(test_completion)
                test_response = future.result(timeout=10)

            # Debug: Show what response we got
            response_content = test_response.choices[0].message.content if test_response.choices else "No content"
            print(f"✓ Model '{self.model_name}' is loaded and responding")
            print(f"  Debug: Test response was: '{response_content}'")

        except FuturesTimeoutError:
            print(f"✗ ERROR: Model '{self.model_name}' test timed out after 10 seconds")
            if self.api_base:
                print("The model may not be loaded in LM Studio. Please load the model and try again.")
            else:
                print("The model may be slow to respond or unavailable.")
            sys.exit(1)
        except Exception as e:
            error_str = str(e).lower()
            if "connection" in error_str or "refused" in error_str:
                print(f"✗ ERROR: Cannot connect to model '{self.model_name}'")
                if self.api_base:
                    print("Please check that LM Studio is running and the model is loaded.")
                else:
                    print("Please check your internet connection and API credentials.")
                sys.exit(1)
            elif "not found" in error_str or "does not exist" in error_str:
                print(f"✗ ERROR: Model '{self.model_name}' not found or not loaded")
                if self.api_base:
                    print("Please load this model in LM Studio before running the benchmark.")
                else:
                    print("Please check the model name or ensure you have access to this model.")
                sys.exit(1)
            else:
                print(f"✗ ERROR testing model '{self.model_name}': {e}")
                sys.exit(1)
    
    def format_problem_prompt(self, dataset_name: str, problem_data: Dict[str, Any]) -> str:
        """Format a problem into a prompt for the LLM based on dataset type"""
        

        if dataset_name == 'puzzte':
            # Logic puzzle from puzzte dataset
            puzzle_text = problem_data.get('puzzle_text', '')
            question = problem_data.get('question', '')
            return f"""Solve this logic puzzle:

{puzzle_text}

Question: {question}

Think step by step and provide your final answer."""

        elif dataset_name == 'gsm8k':
            # Math word problem
            question = problem_data.get('question', '')
            return f"""Solve this math problem step by step:

{question}

Show your work and provide the final numerical answer."""

        elif dataset_name == 'numina_math':
            # Advanced math problem
            problem = problem_data.get('problem', '')
            return f"""Solve this advanced math problem:

{problem}

Show your detailed solution and provide the final answer."""

        elif dataset_name in ['aime_2024', 'aime_2025']:
            # AIME competition problem
            problem = problem_data.get('Problem', problem_data.get('problem', ''))
            return f"""Solve this AIME competition math problem:

{problem}

Provide a detailed solution and the final numerical answer."""

        elif dataset_name == 'arc_challenge':
            # Science reasoning question
            question = problem_data.get('question', '')
            choices = problem_data.get('choices', {})
            choice_text = choices.get('text', [])
            choice_labels = choices.get('label', [])
            
            choices_str = ""
            for label, text in zip(choice_labels, choice_text):
                choices_str += f"{label}. {text}\n"
            
            return f"""Answer this science question:

{question}

{choices_str}

Think through the problem and provide the letter of the correct answer."""

        elif dataset_name == 'winogrande':
            # Commonsense reasoning - multiple choice
            sentence = problem_data.get('sentence', '')
            option1 = problem_data.get('option1', '')
            option2 = problem_data.get('option2', '')

            return f"""Complete the sentence by choosing the correct option:

{sentence}

A. {option1}
B. {option2}

Choose A or B and explain your reasoning."""

        elif dataset_name == 'piqa':
            # Physical reasoning - A/B choice
            goal = problem_data.get('goal', '')
            sol1 = problem_data.get('sol1', '')
            sol2 = problem_data.get('sol2', '')

            return f"""Choose the better solution for this goal:

Goal: {goal}

A. {sol1}
B. {sol2}

Choose A or B and explain your reasoning."""

        elif dataset_name == 'hellaswag':
            # Commonsense reasoning - multiple choice
            ctx = problem_data.get('ctx', '')
            endings = problem_data.get('endings', [])

            choices_text = ""
            for i, ending in enumerate(endings):
                choices_text += f"{chr(65+i)}. {ending}\n"

            return f"""Complete the scenario by choosing the most likely continuation:

{ctx}

{choices_text.strip()}

Choose the letter of the best continuation and explain your reasoning."""

        elif dataset_name == 'logiqa':
            # Logical reasoning - multiple choice
            context = problem_data.get('context', '')
            query = problem_data.get('query', '')
            options = problem_data.get('options', [])

            choices_text = ""
            for i, option in enumerate(options):
                choices_text += f"{chr(65+i)}. {option}\n"

            return f"""Answer this logical reasoning question:

Context: {context}

Question: {query}

{choices_text.strip()}

Choose the letter of the correct answer and explain your reasoning."""

        elif dataset_name == 'mmlu':
            # MMLU: Academic knowledge across 57 subjects - A/B/C/D choice
            question = problem_data.get('question', '')
            choices = problem_data.get('choices', [])
            subject = problem_data.get('subject', 'unknown')

            choices_text = ""
            for i, choice in enumerate(choices):
                choices_text += f"{chr(65+i)}. {choice}\n"

            # Format subject name for display
            subject_display = subject.replace('_', ' ').title()

            return f"""Answer this {subject_display} question:

{question}

{choices_text.strip()}

Choose the letter of the correct answer."""

        elif dataset_name == 'competition_math':
            # MATH: Advanced mathematics competition problems
            problem = problem_data.get('problem', '')
            subject = problem_data.get('type', 'Mathematics')
            level = problem_data.get('level', 'Unknown')

            return f"""Solve this {subject} problem ({level}):

{problem}

Provide your solution with clear reasoning and give the final answer."""

        elif dataset_name == 'rmb_puzzles':
            # RMB custom puzzles - multiple choice
            question = problem_data.get('question', '')
            options = problem_data.get('options', [])

            options_str = ""
            for i, option in enumerate(options, 1):
                options_str += f"{i}. {option}\n"

            return f"""Answer this puzzle question:

{question}

{options_str}
Think through the problem step by step and provide the number of your answer (1, 2, 3, or 4)."""
        else:
            # Generic fallback
            return f"Solve this problem: {str(problem_data)}"

    def evaluate_answer(self, dataset_name: str, problem_data: Dict[str, Any], llm_response: str) -> Dict[str, Any]:
        """Evaluate LLM response against ground truth for the given dataset"""

        def extract_multiple_choice_answer(response: str, valid_letters: str) -> str:
            """Extract multiple choice answer with robust patterns for Sonnet-4's format"""
            import re

            # Pattern 1: Bold answer like "**B. crack**" or "**Choose B:**"
            bold_pattern = rf'\*\*[^*]*?([{valid_letters}])[^*]*?\*\*'
            bold_match = re.search(bold_pattern, response.upper(), re.IGNORECASE)
            if bold_match:
                return bold_match.group(1)

            # Pattern 2: "The answer is X" or "The correct answer is X"
            answer_pattern = rf'(?:the (?:correct )?answer is|choose|answer:?)\s*([{valid_letters}])'
            answer_match = re.search(answer_pattern, response.upper(), re.IGNORECASE)
            if answer_match:
                return answer_match.group(1)

            # Pattern 3: Letter followed by period "B. option"
            period_pattern = rf'\b([{valid_letters}])\.'
            period_matches = re.findall(period_pattern, response.upper())
            if period_matches:
                return period_matches[-1]  # Use last one

            # Pattern 4: Original fallback - isolated letters
            letter_matches = re.findall(rf'\b([{valid_letters}])\b', response.upper())
            if letter_matches:
                return letter_matches[-1]  # Use last mentioned letter

            return None

        if dataset_name in ['aime_2024', 'aime_2025']:
            # AIME problems: extract numerical answer (0-999)
            ground_truth = problem_data.get('Answer', problem_data.get('answer'))
            if ground_truth is None:
                return {"correct": False, "error": "No ground truth found", "extracted_answer": None}

            # Extract numbers from LLM response (look for final numerical answer)
            import re
            numbers = re.findall(r'\b\d{1,3}\b', llm_response)

            # Try to find the answer - often it's the last number mentioned
            if numbers:
                # Convert ground truth to int for comparison
                try:
                    gt_int = int(ground_truth)
                    # Check if any extracted number matches
                    extracted_answers = [int(num) for num in numbers if 0 <= int(num) <= 999]
                    if extracted_answers:
                        # Use the last valid answer as the final answer
                        final_answer = extracted_answers[-1]
                        is_correct = final_answer == gt_int
                        return {
                            "correct": is_correct,
                            "ground_truth": gt_int,
                            "extracted_answer": final_answer,
                            "all_extracted": extracted_answers
                        }
                except (ValueError, TypeError):
                    pass

            return {"correct": False, "error": "Could not extract valid numerical answer",
                   "ground_truth": ground_truth, "extracted_answer": None}

        elif dataset_name == 'gsm8k':
            # GSM8K: extract final numerical answer
            ground_truth = problem_data.get('answer', '')
            # GSM8K answers are in format like "#### 42"
            import re
            gt_match = re.search(r'#### *(\d+)', ground_truth)
            if not gt_match:
                return {"correct": False, "error": "Could not parse ground truth", "extracted_answer": None}

            gt_number = int(gt_match.group(1))

            # Extract numbers from response
            numbers = re.findall(r'\b\d+(?:\.\d+)?\b', llm_response)
            if numbers:
                try:
                    # Use the last number as the final answer
                    final_answer = float(numbers[-1])
                    # Allow for small floating point differences
                    is_correct = abs(final_answer - gt_number) < 0.01
                    return {
                        "correct": is_correct,
                        "ground_truth": gt_number,
                        "extracted_answer": final_answer
                    }
                except ValueError:
                    pass

            return {"correct": False, "error": "Could not extract numerical answer",
                   "ground_truth": gt_number, "extracted_answer": None}

        elif dataset_name == 'arc_challenge':
            # ARC Challenge: multiple choice (A, B, C, D)
            ground_truth = problem_data.get('answerKey', '')

            # Extract answer using robust patterns
            final_answer = extract_multiple_choice_answer(llm_response, "ABCD")
            if final_answer:
                is_correct = final_answer == ground_truth.upper()
                return {
                    "correct": is_correct,
                    "ground_truth": ground_truth,
                    "extracted_answer": final_answer
                }

            return {"correct": False, "error": "Could not extract letter choice",
                   "ground_truth": ground_truth, "extracted_answer": None}

        elif dataset_name == 'winogrande':
            # Winogrande: A/B choice (answer is "1" for A or "2" for B)
            ground_truth = problem_data.get('answer')
            if ground_truth is None:
                return {"correct": False, "error": "No ground truth found", "extracted_answer": None}

            # Convert ground truth to letter (1->A, 2->B)
            gt_letter = "A" if str(ground_truth) == "1" else "B"

            # Extract answer using robust patterns
            final_answer = extract_multiple_choice_answer(llm_response, "AB")
            if final_answer:
                is_correct = final_answer == gt_letter
                return {
                    "correct": is_correct,
                    "ground_truth": gt_letter,
                    "extracted_answer": final_answer
                }

            return {"correct": False, "error": "Could not extract A/B choice",
                   "ground_truth": gt_letter, "extracted_answer": None}

        elif dataset_name == 'piqa':
            # PIQA: A/B choice (label is 0 for A or 1 for B)
            ground_truth = problem_data.get('label')
            if ground_truth is None:
                return {"correct": False, "error": "No ground truth found", "extracted_answer": None}

            # Convert ground truth to letter (0->A, 1->B)
            gt_letter = "A" if ground_truth == 0 else "B"

            # Extract answer using robust patterns
            final_answer = extract_multiple_choice_answer(llm_response, "AB")
            if final_answer:
                is_correct = final_answer == gt_letter
                return {
                    "correct": is_correct,
                    "ground_truth": gt_letter,
                    "extracted_answer": final_answer
                }

            return {"correct": False, "error": "Could not extract A/B choice",
                   "ground_truth": gt_letter, "extracted_answer": None}

        elif dataset_name == 'hellaswag':
            # HellaSwag: multiple choice (label is string index "0", "1", "2", "3")
            ground_truth = problem_data.get('label')
            if ground_truth is None:
                return {"correct": False, "error": "No ground truth found", "extracted_answer": None}

            # Convert ground truth to letter ("0"->A, "1"->B, etc.)
            try:
                gt_index = int(ground_truth)
                gt_letter = chr(65 + gt_index)  # 65 is ASCII for 'A'
            except (ValueError, TypeError):
                return {"correct": False, "error": "Invalid ground truth format", "extracted_answer": None}

            # Extract answer using robust patterns
            final_answer = extract_multiple_choice_answer(llm_response, "ABCD")
            if final_answer:
                is_correct = final_answer == gt_letter
                return {
                    "correct": is_correct,
                    "ground_truth": gt_letter,
                    "extracted_answer": final_answer
                }

            return {"correct": False, "error": "Could not extract letter choice",
                   "ground_truth": gt_letter, "extracted_answer": None}

        elif dataset_name == 'logiqa':
            # LogiQA: multiple choice (correct_option is 0, 1, 2, 3)
            ground_truth = problem_data.get('correct_option')
            if ground_truth is None:
                return {"correct": False, "error": "No ground truth found", "extracted_answer": None}

            # Convert ground truth to letter (0->A, 1->B, etc.)
            try:
                gt_letter = chr(65 + ground_truth)  # 65 is ASCII for 'A'
            except (TypeError, ValueError):
                return {"correct": False, "error": "Invalid ground truth format", "extracted_answer": None}

            # Extract answer using robust patterns
            final_answer = extract_multiple_choice_answer(llm_response, "ABCD")
            if final_answer:
                is_correct = final_answer == gt_letter
                return {
                    "correct": is_correct,
                    "ground_truth": gt_letter,
                    "extracted_answer": final_answer
                }

            return {"correct": False, "error": "Could not extract letter choice",
                   "ground_truth": gt_letter, "extracted_answer": None}

        elif dataset_name == 'mmlu':
            # MMLU: multiple choice (answer is 0, 1, 2, 3)
            ground_truth = problem_data.get('answer')
            if ground_truth is None:
                return {"correct": False, "error": "No ground truth found", "extracted_answer": None}

            # Convert ground truth to letter (0->A, 1->B, etc.)
            try:
                gt_letter = chr(65 + ground_truth)  # 65 is ASCII for 'A'
            except (TypeError, ValueError):
                return {"correct": False, "error": "Invalid ground truth format", "extracted_answer": None}

            # Extract answer using robust patterns
            final_answer = extract_multiple_choice_answer(llm_response, "ABCD")
            if final_answer:
                is_correct = final_answer == gt_letter
                return {
                    "correct": is_correct,
                    "ground_truth": gt_letter,
                    "extracted_answer": final_answer
                }

            return {"correct": False, "error": "Could not extract letter choice",
                   "ground_truth": gt_letter, "extracted_answer": None}

        elif dataset_name == 'competition_math':
            # MATH: Competition mathematics problems with solution comparison
            ground_truth_solution = problem_data.get('solution', '')
            if not ground_truth_solution:
                return {"correct": False, "error": "No ground truth solution found", "extracted_answer": None}

            # Normalize LaTeX and whitespace (based on friend's logic)
            def normalize(text):
                if not text:
                    return ""
                return text.replace(" ", "").replace("\\", "").replace("$", "").replace("{","").replace("}","")

            # Normalize both prediction and ground truth
            normalized_response = normalize(llm_response.strip())
            normalized_ground_truth = normalize(ground_truth_solution.strip())

            # Direct comparison after normalization
            is_correct = normalized_response == normalized_ground_truth

            return {
                "correct": is_correct,
                "ground_truth": ground_truth_solution,
                "extracted_answer": llm_response.strip(),
                "normalized_gt": normalized_ground_truth[:100] + "..." if len(normalized_ground_truth) > 100 else normalized_ground_truth,
                "normalized_pred": normalized_response[:100] + "..." if len(normalized_response) > 100 else normalized_response
            }

        elif dataset_name == 'rmb_puzzles':
            # RMB custom puzzles: multiple choice 1-4
            ground_truth = problem_data.get('answer')
            if ground_truth is None:
                return {"correct": False, "error": "No ground truth found", "extracted_answer": None}

            # Extract numerical choice (1, 2, 3, 4) from response
            import re
            # Look for numbers 1-4 in various contexts
            number_patterns = [
                r'(?:answer is|choose|option)\s*(\d)',  # "answer is 2"
                r'\b(\d)\b',  # isolated digits
            ]

            extracted_number = None
            for pattern in number_patterns:
                matches = re.findall(pattern, llm_response, re.IGNORECASE)
                # Filter to valid choices (1-4)
                valid_matches = [m for m in matches if m in ['1', '2', '3', '4']]
                if valid_matches:
                    extracted_number = valid_matches[-1]  # Use last valid match
                    break

            if extracted_number:
                is_correct = extracted_number == str(ground_truth)
                return {
                    "correct": is_correct,
                    "ground_truth": ground_truth,
                    "extracted_answer": extracted_number
                }
            return {"correct": False, "error": "Could not extract choice 1-4",
                   "ground_truth": ground_truth, "extracted_answer": None}
        else:
            # For datasets without implemented evaluation
            return {"correct": None, "error": f"Evaluation not implemented for {dataset_name}",
                   "extracted_answer": None}

    def call_llm(self, prompt: str, max_retries: int = 1) -> Optional[Dict[str, Any]]:
        """Call the LLM with timeout and retry logic"""

        for attempt in range(max_retries):
            try:
                start_time = time.time()

                # Prepare completion arguments
                # Use more tokens for logic puzzles that need longer reasoning
                is_logic_puzzle = any(dataset in prompt.lower() for dataset in ['knights', 'knaves', 'logic puzzle'])
                max_tokens = 6000 if is_logic_puzzle else 2048

                if is_logic_puzzle and os.environ.get('BENCHMARK_DEBUG'):
                    print(f"DEBUG: Using {max_tokens} tokens for logic puzzle")

                completion_kwargs = {
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": 0.0,  # Deterministic for benchmarking
                    "timeout": self.timeout_seconds,  # Simple LiteLLM timeout
                }

                if self.api_base:
                    completion_kwargs["api_base"] = self.api_base

                # Make the LLM call
                response = completion(**completion_kwargs)

                end_time = time.time()
                response_time = end_time - start_time

                # Extract response content
                content = response.choices[0].message.content

                return {
                    "content": content,
                    "response_time": response_time,
                    "attempt": attempt + 1,
                    "success": True,
                    "error": None
                }

            except Exception as e:
                end_time = time.time()
                response_time = end_time - start_time
                error_str = str(e).lower()

                # Check if it's a timeout error
                is_timeout = ("timeout" in error_str or
                             "timed out" in error_str or
                             response_time >= self.timeout_seconds * 0.9)

                if is_timeout:
                    print(f"  ⏰ Timeout after {response_time:.1f}s (attempt {attempt + 1}/{max_retries})")
                    if attempt == max_retries - 1:
                        return {
                            "content": None,
                            "response_time": response_time,
                            "attempt": attempt + 1,
                            "success": False,
                            "error": "timeout"
                        }
                else:
                    print(f"  ❌ Error: {str(e)} (attempt {attempt + 1}/{max_retries})")
                    if attempt == max_retries - 1:
                        return {
                            "content": None,
                            "response_time": response_time,
                            "attempt": attempt + 1,
                            "success": False,
                            "error": str(e)
                        }

                # Wait before retry
                time.sleep(1)

        return None
        
    def load_config(self) -> Dict[str, Any]:
        """Load and validate the YAML configuration file."""
        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            
            print(f"Loaded config from: {self.config_path}")
            print(f"Random seed: {self.config['global_options']['random_seed']}")
            print(f"Datasets to process: {len(self.config['datasets'])}")
            print()
            
            # Set random seed from config
            random.seed(self.config['global_options']['random_seed'])
            
            return self.config
            
        except FileNotFoundError:
            print(f"Error: Config file not found: {self.config_path}")
            sys.exit(1)
        except yaml.YAMLError as e:
            print(f"Error parsing YAML config: {e}")
            sys.exit(1)
    
    def validate_dataset_config(self, dataset_name: str, dataset_config: Dict[str, Any]) -> bool:
        """Validate individual dataset configuration."""
        required_fields = ['path', 'format', 'mode']
        for field in required_fields:
            if field not in dataset_config:
                print(f"Error: Dataset '{dataset_name}' missing required field: {field}")
                return False
                
        # Validate mode-specific requirements
        mode = dataset_config['mode']
        if mode == 'counts':
            if 'counts' not in dataset_config:
                print(f"Error: Dataset '{dataset_name}' mode 'counts' requires 'counts' section")
                return False
            if 'specific_problems' in dataset_config:
                print(f"Error: Dataset '{dataset_name}' mode 'counts' cannot have 'specific_problems'")
                return False
        elif mode == 'subset':
            if 'specific_problems' not in dataset_config:
                print(f"Error: Dataset '{dataset_name}' mode 'subset' requires 'specific_problems'")
                return False
            if 'counts' in dataset_config:
                print(f"Error: Dataset '{dataset_name}' mode 'subset' cannot have 'counts'")
                return False
        elif mode == 'all':
            if 'counts' in dataset_config or 'specific_problems' in dataset_config:
                print(f"Error: Dataset '{dataset_name}' mode 'all' cannot have 'counts' or 'specific_problems'")
                return False
        elif mode == 'none':
            # Skip this dataset
            return True
        else:
            print(f"Error: Dataset '{dataset_name}' has invalid mode: {mode}")
            return False
            
        return True
    
    def load_dataset_problems(self, dataset_name: str, dataset_config: Dict[str, Any]) -> List[Tuple[str, str, Any]]:
        """Load problems from a single dataset based on its configuration."""
        problems = []
        
        if dataset_config['mode'] == 'none':
            return problems
            
        print(f"Loading dataset: {dataset_name}")
        
        try:
            # Load dataset based on format
            if dataset_config['format'] == 'huggingface':
                # Load from HuggingFace
                dataset_path = dataset_config['path']
                config = dataset_config.get('config')
                split = dataset_config.get('split', 'train')
                
                if config:
                    dataset = load_dataset(dataset_path, config)
                else:
                    dataset = load_dataset(dataset_path)
                
                data = dataset[split]
                
            elif dataset_config['format'] == 'parquet':
                # Load from local parquet file
                data = pd.read_parquet(dataset_config['path'])

            elif dataset_config['format'] == 'json':
                # Load from local JSON file
                dataset_path = dataset_config['path']
                split = dataset_config.get('split', 'train')
                dataset = load_dataset("json", data_files=dataset_path)
                data = dataset[split]

            else:
                print(f"  Unsupported format: {dataset_config['format']}")
                return problems
                
            print(f"  Loaded {len(data)} total problems")
            
            # Sample problems based on mode
            if dataset_config['mode'] == 'all':
                # Use all problems
                for i, item in enumerate(data):
                    problems.append((dataset_name, 'all', item))
                    
            elif dataset_config['mode'] == 'counts':
                # Sample based on difficulty counts
                counts = dataset_config['counts']
                total_requested = sum(counts.values())
                
                # Handle Puzzte with ambiguity score mapping
                if dataset_name == 'puzzte' and 'difficulty_mapping' in dataset_config:
                    difficulty_mapping = dataset_config['difficulty_mapping']
                    
                    # Group problems by ambiguity score ranges
                    difficulty_groups = {difficulty: [] for difficulty in counts.keys()}
                    
                    for item in data:
                        if isinstance(data, pd.DataFrame):
                            item_dict = item if isinstance(item, dict) else item.to_dict()
                        else:
                            item_dict = item
                            
                        ambiguity = item_dict.get('ambiguity', 0.0)
                        
                        # Map ambiguity score to difficulty
                        if ambiguity < 5.0:
                            difficulty = 'easy'
                        elif ambiguity < 20.0:
                            difficulty = 'medium'
                        else:
                            difficulty = 'hard'
                            
                        if difficulty in difficulty_groups:
                            difficulty_groups[difficulty].append(item_dict)
                    
                    # Sample from each difficulty group
                    for difficulty, requested_count in counts.items():
                        available = difficulty_groups.get(difficulty, [])
                        if len(available) >= requested_count:
                            sampled = random.sample(available, requested_count)
                        else:
                            print(f"  Warning: Requested {requested_count} {difficulty} problems, only {len(available)} available")
                            sampled = available
                        
                        for item in sampled:
                            problems.append((dataset_name, difficulty, item))
                
                else:
                    # Default: random sampling (for datasets without real difficulty mapping)
                    if len(data) >= total_requested:
                        if isinstance(data, pd.DataFrame):
                            sampled_data = data.sample(n=total_requested).to_dict('records')
                        else:
                            sampled_indices = random.sample(range(len(data)), total_requested)
                            sampled_data = [data[i] for i in sampled_indices]
                            
                        difficulty_labels = []
                        for difficulty, count in counts.items():
                            difficulty_labels.extend([difficulty] * count)
                        random.shuffle(difficulty_labels)
                        
                        for item, difficulty in zip(sampled_data, difficulty_labels):
                            problems.append((dataset_name, difficulty, item))
                    else:
                        print(f"  Warning: Requested {total_requested} problems but only {len(data)} available")
                        for i, item in enumerate(data):
                            if isinstance(data, pd.DataFrame):
                                item = data.iloc[i].to_dict()
                            problems.append((dataset_name, 'unknown', item))
                        
            elif dataset_config['mode'] == 'subset':
                # Use specific problems (placeholder - would need ID matching)
                specific_ids = dataset_config['specific_problems']
                print(f"  Specific problem selection not yet implemented, requested: {len(specific_ids)} problems")
                
            print(f"  Selected {len(problems) - len([p for p in problems if p[0] != dataset_name])} problems")
            
        except Exception as e:
            print(f"  Error loading dataset {dataset_name}: {e}")
            
        return problems
    
    def load_all_problems(self):
        """Load problems from all configured datasets."""
        self.test_problems = []
        
        for dataset_name, dataset_config in self.config['datasets'].items():
            if not self.validate_dataset_config(dataset_name, dataset_config):
                continue

            dataset_problems = self.load_dataset_problems(dataset_name, dataset_config)
            self.test_problems.extend(dataset_problems)
        
        print(f"\nTotal problems loaded: {len(self.test_problems)}")
        
        # Show breakdown by dataset and difficulty
        dataset_counts = {}
        difficulty_counts = {}
        
        for dataset_name, difficulty, _ in self.test_problems:
            dataset_counts[dataset_name] = dataset_counts.get(dataset_name, 0) + 1
            difficulty_counts[difficulty] = difficulty_counts.get(difficulty, 0) + 1
        
        print("\nDataset breakdown:")
        for dataset, count in dataset_counts.items():
            print(f"  {dataset}: {count} problems")
            
        print("\nDifficulty breakdown:")
        for difficulty, count in difficulty_counts.items():
            print(f"  {difficulty}: {count} problems")
        print()
    
    def run_benchmark(self):
        """Run the benchmark on all loaded problems."""
        print(f"Starting benchmark run with model: {self.model_name}")
        print(f"Timeout: {self.timeout_seconds}s per problem")
        print("=" * 60)
        
        start_time = time.time()
        successful_completions = 0
        correct_answers = 0

        for i, (dataset_name, difficulty, problem_data) in enumerate(self.test_problems, 1):
            # Generate a simple test identifier
            test_id = f"{dataset_name}_{difficulty}_{i:03d}"

            print(f"[{i:3d}/{len(self.test_problems)}] {test_id}")

            # Format the problem into a prompt
            prompt = self.format_problem_prompt(dataset_name, problem_data)

            # Call the LLM
            result = self.call_llm(prompt)

            # Initialize evaluation result
            evaluation = None

            if result and result['success']:
                # Evaluate the answer
                evaluation = self.evaluate_answer(dataset_name, problem_data, result['content'])

                # Update output based on completion and correctness
                if evaluation['correct'] is True:
                    print(f"  ✓ Correct in {result['response_time']:.1f}s")
                    correct_answers += 1
                elif evaluation['correct'] is False:
                    print(f"  ○ Completed in {result['response_time']:.1f}s (incorrect)")
                else:
                    print(f"  ○ Completed in {result['response_time']:.1f}s (not evaluated)")
                    print(f"    ⚠ Warning: Evaluation not implemented for dataset '{dataset_name}'")

                successful_completions += 1
            else:
                error_type = result['error'] if result else 'unknown'
                print(f"  ✗ Failed ({error_type})")

            # Store the result
            result_entry = {
                "test_id": test_id,
                "dataset": dataset_name,
                "difficulty": difficulty,
                "problem_data": problem_data,
                "prompt": prompt,
                "llm_result": result,
                "evaluation": evaluation,
                "timestamp": time.time()
            }
            self.results.append(result_entry)

            # Add delay between requests to avoid overloading Anthropic servers
            if i < len(self.test_problems):  # Don't delay after the last problem
                time.sleep(1.5)

        end_time = time.time()
        total_time = end_time - start_time
        
        print("=" * 60)
        print(f"Benchmark completed!")
        print(f"Total time: {total_time:.1f}s")
        print(f"Successful completions: {successful_completions}/{len(self.test_problems)} ({100*successful_completions/len(self.test_problems):.1f}%)")
        print(f"Correct answers: {correct_answers}/{len(self.test_problems)} ({100*correct_answers/len(self.test_problems):.1f}%)")
        if successful_completions > 0:
            print(f"Accuracy (of completed): {correct_answers}/{successful_completions} ({100*correct_answers/successful_completions:.1f}%)")
        print(f"Average time per problem: {total_time/len(self.test_problems):.1f}s")
        
        # Save results
        self.save_results()
    
    def save_results(self):
        """Save benchmark results to a JSON file"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # Clean model name for filename (convert / to _ for readability)
        clean_model_name = self.model_name.replace('/', '_')
        clean_model_name = "".join(c for c in clean_model_name if c.isalnum() or c in ('-', '_'))
        filename = f"benchmark_results_{clean_model_name}_{timestamp}.json"

        # Save to JSON_RESULTS directory
        results_dir = Path("JSON_RESULTS")
        results_dir.mkdir(exist_ok=True)
        filepath = results_dir / filename
        
        # Calculate statistics
        successful_completions = sum(1 for r in self.results if r['llm_result'] and r['llm_result']['success'])
        correct_answers = sum(1 for r in self.results if r['evaluation'] and r['evaluation'].get('correct') is True)

        results_data = {
            "metadata": {
                "model_name": self.model_name,
                "config_path": self.config_path,
                "timeout_seconds": self.timeout_seconds,
                "total_problems": len(self.test_problems),
                "successful_completions": successful_completions,
                "correct_answers": correct_answers,
                "completion_rate": round(100 * successful_completions / len(self.test_problems), 1),
                "accuracy_overall": round(100 * correct_answers / len(self.test_problems), 1),
                "accuracy_of_completed": round(100 * correct_answers / successful_completions, 1) if successful_completions > 0 else 0,
                "timestamp": timestamp
            },
            "results": self.results
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(results_data, f, indent=2, default=str)
            print(f"Results saved to: {filepath}")
        except Exception as e:
            print(f"Error saving results: {e}")

def main():
    parser = argparse.ArgumentParser(description='Run LLM benchmark evaluation')
    parser.add_argument('config', help='Path to YAML benchmark configuration file')
    parser.add_argument('model', help='Model name/path to evaluate')
    parser.add_argument('--dry-run', action='store_true', help='Only validate config without running')
    parser.add_argument('--timeout', type=int, default=60, help='Timeout in seconds per LLM call (default: 60)')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--allow-unlisted-vendor', action='store_true', help='Allow models not in config (for cloud APIs or testing)')
    
    args = parser.parse_args()
    
    # Set debug environment variable
    if args.debug:
        os.environ['BENCHMARK_DEBUG'] = '1'
        os.environ['LITELLM_LOG'] = 'DEBUG'
    
    # Validate config file exists
    if not Path(args.config).exists():
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)
    
    # Create benchmark runner
    runner = BenchmarkRunner(args.config, args.model, timeout_seconds=args.timeout)
    
    try:
        # Load configuration
        runner.load_config()

        # Check model availability FIRST (fail fast if model is invalid)
        runner.check_model_availability(allow_unknown=args.allow_unlisted_vendor)

        # Only load datasets if model is valid
        runner.load_all_problems()

        if args.dry_run:
            print("Dry run completed successfully. Configuration and model are valid.")
        else:
            # Run the benchmark
            runner.run_benchmark()
            
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
        print("Partial results may have been saved.")
        sys.exit(1)
    except Exception as e:
        print(f"Error running benchmark: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
