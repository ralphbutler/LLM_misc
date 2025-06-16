import sys
import glob
import time
import random
import json # Needed for json.loads in main
import os # Needed to check for features file existence

# Import core logic components
from core_arc import (
    PuzzleResult,
    # arc_prompt, # arc_prompt is not used directly here, it's built dynamically
    call_llm,
    create_augmented_prompt,
    create_comparison_table,
    format_time,
)

# Import the XML formatting function
from format_xml import format_as_xml

# Removed Rich imports - using plain output for workx8.py

# Global cost tracking
total_cost = 0.0
MAX_COST = 1.0 # Set your desired max cost here

# --- Configuration ---
# Dictionary mapping model names to API bases
model2apibase = {
    # "openai/DeepSeek-R1-Distill-Qwen-32B" : "http://localhost:1234/v1",  # lmstudio
    # "openai/qwen2.5-7b-instruct-mlx"      : "http://localhost:1234/v1",  # lmstudio
    # "openai/qwen2.5-32b-instruct"         : "http://localhost:1234/v1",  # lmstudio
    # "openai/phi-4"                        : "http://localhost:1234/v1",  # lmstudio
    # "openai/phi-4-reasoning-plus-mlx"     : "http://localhost:1234/v1",  # lmstudio
    # "openai/gemma-3-27b-it"               : "http://localhost:1234/v1",  # lmstudio
    # "openai/qwq-32b"                      : "http://localhost:1234/v1",  # lmstudio
    # "openai/qwen3-32b-mlx"                      : "http://localhost:1234/v1",  # lmstudio

    # "ollama_chat/llama3.2:3b"             : "http://localhost:11434",    # ollama

    # "gpt-4o-mini"                         : "",  # known to litellm
    # "gpt-4o"                              : "",  # known to litellm
    # "o3-mini"                             : "",  # known to litellm
    # "o4-mini"                             : "",  # known to litellm

    # "gemini/gemini-2.0-flash"             : "",  # known to litellm
    "gemini/gemini-2.5-flash-preview-04-17"             : "",  # known to litellm
    # "gemini/gemini-2.5-pro-exp-03-25"     : "",  # known to litellm
    # "gemini/gemini-2.5-pro-preview-05-06"     : "",  # known to litellm

    # "claude-3-5-haiku-latest"             : "",  # known to litellm
    # "claude-3-7-sonnet-latest"            : "",  # known to litellm
}

# Default to first model in list from core
model_name = list(model2apibase.keys())[0]
api_base = model2apibase[model_name]

# Maximum tokens for generation
max_tokens = 8_000  # 10_000  # some models only support smaller numbers

# Get list of puzzle files - can be customized or passed as arguments
random.seed(222)
puzzle_filenames = glob.glob("DATA1/training/*.json")
random.shuffle(puzzle_filenames)
files_to_use = puzzle_filenames[0:50]
print("FILES TO USE:", files_to_use)

files_to_use = ["DATA1/training/d037b0a7.json"]
# files_to_use = ["tempd0hacked.json"]   ## has two test inputs ##
# files_to_use = ["DATA1/training/3aa6fb7a.json"]
# files_to_use = ["DATA1/training/007bbfb7.json"]
# files_to_use = ["DATA1/training/d4469b4b.json"]
# ---------------------
if len(sys.argv) > 1:
    files_to_use = [sys.argv[1]]
    print("DOING FILE",sys.argv[1])


def main():
    global total_cost # Declare intent to modify global variable

    # Print model configuration
    print(f"\nUsing model: {model_name}")
    print(f"API base: {api_base}")
    print(f"Max Tokens: {max_tokens}")
    print(f"Max Cost: ${MAX_COST:.2f}")


    # Track overall success
    total_puzzles = 0
    puzzles_solved = 0
    total_test_cases = 0 # New counter for total test cases
    test_cases_passed = 0 # New counter for passed test cases

    start_time_total = time.time()
    puzzle_times = []

    for (idx,puzzle_file) in enumerate(files_to_use):
        if len(files_to_use) > 5  and  (idx % 5) == 0:
            print("SLEEPING")
            time.sleep(62)
            print("DONE SLEEPING")
        total_puzzles += 1
        # print("DBG", puzzle_file) # Debug print
        start_time_puzzle = time.time()
        print(f"\nProcessing puzzle file: {puzzle_file}")

        # Load the puzzle
        try:
            with open(puzzle_file) as f:
                puzzle_json = json.loads(f.read().strip())
                # Store all test cases, not just the first one
                test_cases = puzzle_json["test"]
                total_test_cases += len(test_cases) # Add test cases from this puzzle to total
        except Exception as e:
            print(f"Error loading puzzle file {puzzle_file}: {e}")
            continue # Skip to the next puzzle file

        # Load and format features file
        features_xml_string = ""
        # --- START CHANGE ---
        # Determine features file path based on puzzle filename
        puzzle_basename = os.path.basename(puzzle_file)
        features_file = os.path.join("FEATURES", puzzle_basename)
        # --- END CHANGE ---

        if os.path.exists(features_file):
            try:
                with open(features_file) as f:
                    features_json = json.loads(f.read().strip())
                features_xml_string = format_as_xml(features_json)
                print(f"Loaded and formatted features from: {features_file}")
            except Exception as e:
                print(f"Error loading or formatting features file {features_file}: {e}")
                sys.exit(1) # Exit if features file is corrupt or cannot be formatted
        else:
            print(f"Error: Features file not found: {features_file}")
            sys.exit(1) # Exit if features file is missing


        # Track attempts for this puzzle
        attempts = 0
        max_attempts = 2  ## RMB  3
        found_solution = False
        previous_attempts = []

        while attempts < max_attempts and not found_solution:
            attempts += 1
            print(f"\nAttempt {attempts} of {max_attempts}")

            # Create the prompt with features XML and previous attempt information
            # Pass puzzle_json directly, not arc_prompt
            prompt = create_augmented_prompt(features_xml_string, puzzle_json, previous_attempts)

            # Display the prompt
            if True: # Set to False to hide prompt
                print("=" * 50)
                print(f"Request to {model_name}")
                print("=" * 50)
                print(f"Prompt: {prompt}")
                print("=" * 50)

            # Call the LLM
            response_model = None
            call_cost = 0.0
            try:
                # with console.status(
                        # f"[bold red]Waiting for response: attempt {attempts} from {model_name} ...",
                    # spinner="bouncingBar",
                # ):
                    # response_model, call_cost = call_llm(model_name, api_base, max_tokens, prompt, PuzzleResult)
                response_model, call_cost = call_llm(model_name, api_base, max_tokens, prompt, PuzzleResult)

                # Track cost
                call_cost = call_cost if call_cost else 0.0
                total_cost += call_cost
                print(f"Response cost: ${call_cost:.08f} | Total cost: ${total_cost:.08f}")

                # Check if we've exceeded the cost limit
                if total_cost > MAX_COST:
                    print(f"*** EXITING: Cost limit of ${MAX_COST:.2f} exceeded (${total_cost:.2f})")
                    sys.exit(-1)

            except Exception as e:
                 # call_llm already printed error, just break attempt loop
                 print(f"LLM call failed for attempt {attempts}. Abandoning puzzle.")
                 print("DBG",e)
                 break # Exit attempt loop for this puzzle and move to the next puzzle file

            # Display the response
            print("\n" + "=" * 30)
            print("API Response")
            print("=" * 30)
            print(json.dumps(response_model.model_dump(), indent=4))
            print("=" * 30)

            # Execute the generated code
            attempt_feedback = None # Store feedback for this attempt if it fails

            try:
                # Display the code
                print("\n" + "=" * 30)
                print("Generated Code")
                print("=" * 30)
                print(response_model.code)
                print("=" * 30)

                # Execute the code
                # Define transform_grid in a local scope for safety
                local_scope = {}
                exec(response_model.code, {}, local_scope)
                transform_grid = local_scope.get('transform_grid')

                if not callable(transform_grid):
                     raise ValueError("Generated code did not define a callable 'transform_grid' function.")

                # Test on training examples
                print("\nTesting on training examples:")
                all_training_correct = True
                failed_training_indices = [] # List to store indices of failed examples

                for i, example in enumerate(puzzle_json["train"]):
                    print(f"\nTraining example {i+1}:")
                    print(f"Expected Output: {example['output']}")
                    try:
                        result = transform_grid(example['input'])
                        print(f"Got Output: {result}")

                        if result != example['output']:
                            all_training_correct = False
                            failed_training_indices.append(i + 1) # Store 1-based index
                            print("❌ Failed on this example")
                        else:
                            print("✓ Passed")
                    except Exception as e:
                        print(f"\nError executing generated code on training example {i+1}: {e}")
                        all_training_correct = False
                        failed_training_indices.append(i + 1) # Mark as failed due to execution error


                # If all training examples pass, test on the test case(s)
                if all_training_correct:
                    print("\nAll training examples passed, testing on test case(s)...")

                    all_test_cases_correct = True
                    failed_test_indices = [] # List to store indices of failed test cases
                    current_puzzle_test_cases_passed = 0 # Counter for test cases passed in this puzzle attempt

                    for i, test_case in enumerate(test_cases): # Iterate through all test cases
                        test_input = test_case['input']
                        correct_test_output = test_case['output']

                        print(f"\nTest case {i+1}:")
                        # print(f"Input: {test_input}") # Optional: display input grid
                        print(f"Expected Output: {correct_test_output}")

                        try:
                            test_result = transform_grid(test_input)
                            print(f"Got Output: {test_result}")

                            is_this_test_correct = test_result == correct_test_output

                            print(create_comparison_table(correct_test_output, test_result))

                            match_result = "Yes" if is_this_test_correct else "No"
                            print(f"Match for test case {i+1}: {match_result}")

                            if not is_this_test_correct:
                                all_test_cases_correct = False
                                failed_test_indices.append(i + 1)
                                print("❌ Failed on this test case")
                            else:
                                print("✓ Passed this test case")
                                current_puzzle_test_cases_passed += 1 # Increment if this specific test case passed

                        except Exception as e:
                             print(f"\nError executing generated code on test case {i+1}: {e}")
                             all_test_cases_correct = False
                             failed_test_indices.append(i + 1) # Mark as failed due to execution error

                    # Determine if the puzzle is solved based on ALL test cases passing
                    if all_test_cases_correct:
                        found_solution = True
                        puzzles_solved += 1 # Increment puzzle solved count
                        test_cases_passed += current_puzzle_test_cases_passed # Add passed test cases from this puzzle
                        print(f"\nAll {len(test_cases)} test cases passed!")
                        print(f"Reasoning: {response_model.reasoning}") # Print reasoning once after all tests
                    else:
                        # Puzzle not solved, prepare feedback
                        feedback_result_str = 'failed test cases'
                        if failed_test_indices:
                             feedback_result_str += f" (failed on examples: {', '.join(map(str, failed_test_indices))})"
                        attempt_feedback = {
                            'reasoning': response_model.reasoning,
                            'result': feedback_result_str
                        }
                        print(f"\nFailed on {len(failed_test_indices)} out of {len(test_cases)} test cases.")
                        print(f"Reasoning: {response_model.reasoning}") # Print reasoning even if failed

                else: # Training examples failed
                    print("\nNot all training examples passed, skipping test case(s).")
                    # Puzzle not solved, prepare feedback
                    feedback_result_str = 'failed training examples'
                    if failed_training_indices:
                         feedback_result_str += f" (failed on examples: {', '.join(map(str, failed_training_indices))})"
                    attempt_feedback = {
                        'reasoning': response_model.reasoning,
                        'result': feedback_result_str
                    }


            except Exception as e: # This catch block handles errors during code execution *before* test cases are run (e.g., transform_grid not defined)
                print(f"\nError executing generated code: {e}")
                # Puzzle not solved, prepare feedback
                attempt_feedback = {
                    'reasoning': response_model.reasoning,
                    'result': f'execution error: {e}'
                }

            # Add feedback to previous_attempts ONLY if the puzzle was not solved in this attempt
            if not found_solution and attempt_feedback:
                 previous_attempts.append(attempt_feedback)


        puzzle_time = time.time() - start_time_puzzle
        puzzle_times.append(puzzle_time)

        # Report final status for this puzzle
        time_display = format_time(puzzle_time)
        if found_solution:
            print(f"REPORT: Successful solve {puzzle_basename} in {attempts} attempts. Time: {time_display}")
            # puzzles_solved += 1 # Moved increment inside the all_test_cases_correct block
        else:
            print(f"REPORT: Failure to solve {puzzle_basename} in {attempts} attempts. Time: {time_display}")

    total_time = time.time() - start_time_total

    # Print overall summary
    print("\nOverall Results:")

    # Calculate rates and format times/costs
    puzzle_success_rate = f"{(puzzles_solved / total_puzzles) * 100:.1f}%" if total_puzzles > 0 else "N/A"
    test_case_success_rate = f"{(test_cases_passed / total_test_cases) * 100:.1f}%" if total_test_cases > 0 else "N/A"
    avg_time = format_time(sum(puzzle_times) / len(puzzle_times)) if puzzle_times else "N/A"
    total_time_formatted = format_time(total_time)
    total_cost_formatted = f"${total_cost:.4f}"

    # Print formatted summary tables
    print("\n" + "=" * 60)
    print("PUZZLE SOLVING PERFORMANCE (SUMMARY)")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Puzzles Solved: {puzzles_solved}")
    print(f"Total Puzzles: {total_puzzles}")
    print(f"Puzzle Success Rate: {puzzle_success_rate}")
    
    print("\n" + "=" * 60)
    print("PERFORMANCE DETAILS")
    print("=" * 60)
    print(f"Test Cases Passed: {test_cases_passed}")
    print(f"Total Test Cases: {total_test_cases}")
    print(f"Test Case Success Rate: {test_case_success_rate}")
    print(f"Avg Time/Puzzle: {avg_time}")
    print(f"Total Time: {total_time_formatted}")
    print(f"Total Cost: {total_cost_formatted}")
    print("=" * 60)


if __name__ == "__main__":
    main()
