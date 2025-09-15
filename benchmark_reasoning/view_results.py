#!/usr/bin/env python3
"""
Results Viewer for Benchmark Runner

View and analyze benchmark results from JSON files.
Supports interactive file selection, filtering, and comparison.
"""

import json
import argparse
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import re

def load_result_file(filepath: Path) -> Dict[str, Any]:
    """Load and parse a result JSON file"""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        sys.exit(1)

def get_file_info(filepath: Path) -> Dict[str, str]:
    """Extract summary info from a result file"""
    try:
        data = load_result_file(filepath)
        metadata = data.get('metadata', {})

        # Extract model name (remove long prefixes for display)
        model_name = metadata.get('model_name', 'unknown')
        if model_name.startswith('openai/'):
            model_name = model_name.replace('openai/', '')

        # Get timing info
        timestamp = metadata.get('timestamp', '')
        try:
            file_time = datetime.strptime(timestamp, "%Y%m%d_%H%M%S")
            time_ago = datetime.now() - file_time
            if time_ago.days > 0:
                time_str = f"{time_ago.days}d ago"
            elif time_ago.seconds > 3600:
                time_str = f"{time_ago.seconds//3600}h ago"
            else:
                time_str = f"{time_ago.seconds//60}m ago"
        except:
            time_str = "unknown"

        return {
            'model': model_name,
            'problems': str(metadata.get('total_problems', 0)),
            'accuracy': f"{metadata.get('accuracy_overall', 0):.0f}%",
            'time_ago': time_str
        }
    except:
        return {'model': 'error', 'problems': '?', 'accuracy': '?', 'time_ago': '?'}

def list_result_files() -> List[Path]:
    """Get all JSON result files from JSON_RESULTS directory"""
    results_dir = Path("JSON_RESULTS")
    if not results_dir.exists():
        print("No JSON_RESULTS directory found.")
        sys.exit(1)

    json_files = list(results_dir.glob("*.json"))
    if not json_files:
        print("No JSON result files found in JSON_RESULTS/")
        sys.exit(1)

    # Sort by modification time (newest first)
    json_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return json_files

def interactive_file_selection() -> List[Path]:
    """Interactive file selection interface"""
    files = list_result_files()

    print("Available result files:")
    for i, filepath in enumerate(files, 1):
        info = get_file_info(filepath)
        print(f"{i:2}. {info['model']} ({info['problems']} problems, {info['accuracy']}, {info['time_ago']})")

    print()
    while True:
        try:
            selection = input("Select file(s) [1-{}, or 1,3 for compare]: ".format(len(files))).strip()
            if not selection:
                continue

            # Parse selection
            if ',' in selection:
                # Multiple files for comparison
                indices = [int(x.strip()) for x in selection.split(',')]
                if len(indices) > 2:
                    print("Can only compare 2 files at a time.")
                    continue
            else:
                # Single file or range
                indices = [int(selection)]

            # Validate indices
            if all(1 <= i <= len(files) for i in indices):
                return [files[i-1] for i in indices]
            else:
                print(f"Please enter numbers between 1 and {len(files)}")

        except (ValueError, KeyboardInterrupt):
            print("\nExiting.")
            sys.exit(0)

def print_summary(data: Dict[str, Any], filepath: Path):
    """Print summary view of benchmark results"""
    metadata = data.get('metadata', {})
    results = data.get('results', [])

    print(f"\n📊 Results Summary: {filepath.name}")
    print("=" * 60)

    # Basic info
    print(f"Model: {metadata.get('model_name', 'Unknown')}")
    print(f"Config: {metadata.get('config_path', 'Unknown')}")
    print(f"Timestamp: {metadata.get('timestamp', 'Unknown')}")
    print(f"Timeout: {metadata.get('timeout_seconds', 'Unknown')}s")
    print()

    # Overall stats
    total = metadata.get('total_problems', 0)
    completed = metadata.get('successful_completions', 0)
    correct = metadata.get('correct_answers', 0)

    print(f"📈 Overall Performance:")
    print(f"  Total problems: {total}")
    print(f"  Completed: {completed} ({metadata.get('completion_rate', 0):.1f}%)")
    print(f"  Correct: {correct} ({metadata.get('accuracy_overall', 0):.1f}%)")
    if completed > 0:
        print(f"  Accuracy of completed: {metadata.get('accuracy_of_completed', 0):.1f}%")
    print()

    # Dataset breakdown
    dataset_stats = {}
    difficulty_stats = {}

    for result in results:
        dataset = result.get('dataset', 'unknown')
        difficulty = result.get('difficulty', 'unknown')
        evaluation = result.get('evaluation', {})
        llm_result = result.get('llm_result', {})

        # Dataset stats
        if dataset not in dataset_stats:
            dataset_stats[dataset] = {'total': 0, 'completed': 0, 'correct': 0}
        dataset_stats[dataset]['total'] += 1
        if llm_result.get('success'):
            dataset_stats[dataset]['completed'] += 1
            if evaluation and evaluation.get('correct'):
                dataset_stats[dataset]['correct'] += 1

        # Difficulty stats
        if difficulty not in difficulty_stats:
            difficulty_stats[difficulty] = {'total': 0, 'completed': 0, 'correct': 0}
        difficulty_stats[difficulty]['total'] += 1
        if llm_result.get('success'):
            difficulty_stats[difficulty]['completed'] += 1
            if evaluation and evaluation.get('correct'):
                difficulty_stats[difficulty]['correct'] += 1

    if dataset_stats:
        print("📚 By Dataset:")
        for dataset, stats in dataset_stats.items():
            comp_pct = (stats['completed'] / stats['total'] * 100) if stats['total'] > 0 else 0
            corr_pct = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
            print(f"  {dataset}: {stats['correct']}/{stats['total']} correct ({corr_pct:.0f}%), {stats['completed']}/{stats['total']} completed ({comp_pct:.0f}%)")
        print()

    if difficulty_stats:
        print("🎯 By Difficulty:")
        for difficulty, stats in difficulty_stats.items():
            comp_pct = (stats['completed'] / stats['total'] * 100) if stats['total'] > 0 else 0
            corr_pct = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
            print(f"  {difficulty}: {stats['correct']}/{stats['total']} correct ({corr_pct:.0f}%), {stats['completed']}/{stats['total']} completed ({comp_pct:.0f}%)")

def filter_results(results: List[Dict], dataset: Optional[str] = None, show_failures: bool = False, show_successes: bool = False) -> List[Dict]:
    """Filter results based on criteria"""
    filtered = results

    # Dataset filter
    if dataset:
        filtered = [r for r in filtered if r.get('dataset', '').lower() == dataset.lower()]

    # Success/failure filter
    if show_failures:
        filtered = [r for r in filtered if not (r.get('evaluation', {}).get('correct') == True)]
    elif show_successes:
        filtered = [r for r in filtered if r.get('evaluation', {}).get('correct') == True]

    return filtered

def analyze_failure(result: Dict) -> str:
    """Analyze why a problem failed and return analysis text"""
    dataset = result.get('dataset', 'unknown')
    llm_result = result.get('llm_result', {})
    evaluation = result.get('evaluation', {})

    if not llm_result.get('success'):
        return "ROOT CAUSE: LLM failed to generate response"

    if not evaluation or evaluation.get('correct') is None:
        return "ROOT CAUSE: No evaluation data available"

    if evaluation.get('correct') == True:
        return "This problem was actually correct"

    # Get the key data
    ground_truth = str(evaluation.get('ground_truth', ''))
    extracted_answer = str(evaluation.get('extracted_answer', ''))
    model_response = llm_result.get('content', '')

    analysis_parts = []

    # Look for what the LLM actually calculated/said
    llm_calculated = None
    if dataset in ['gsm8k', 'competition_math', 'aime_2024']:
        # Look for mathematical calculations in response
        # Find patterns like "= 2180", "Total = 2180", "answer is 2180"
        math_patterns = [
            r'(?:=|is|equals?)\s*\$?(\d+(?:,\d+)*(?:\.\d+)?)',
            r'(?:total|answer|result).*?(\d+(?:,\d+)*(?:\.\d+)?)',
            r'\$?(\d+(?:,\d+)*(?:\.\d+)?)(?:\s*dollars?|\s*total|\s*altogether)',
        ]

        for pattern in math_patterns:
            matches = re.findall(pattern, model_response.lower(), re.IGNORECASE)
            if matches:
                # Take the last/most likely final answer
                llm_calculated = matches[-1].replace(',', '')
                break

    elif dataset in ['winogrande', 'piqa', 'hellaswag', 'logiqa', 'arc_challenge', 'mmlu']:
        # Look for multiple choice answers
        mc_patterns = [
            r'\*\*([A-D])\.\s',  # **A. option**
            r'answer is\s*([A-D])',  # answer is A
            r'choose\s*([A-D])',  # choose A
            r'option\s*([A-D])',  # option A
            r'\b([A-D])\b(?=\s*$|\s*[.!])',  # isolated letter
        ]

        for pattern in mc_patterns:
            matches = re.findall(pattern, model_response, re.IGNORECASE | re.MULTILINE)
            if matches:
                llm_calculated = matches[-1].upper()
                break

        # For numerical MC like piqa (0/1) or hellaswag (0/1/2/3)
        if not llm_calculated and dataset in ['piqa', 'hellaswag']:
            num_patterns = [
                r'answer is\s*(\d)',
                r'choose\s*(\d)',
                r'\b(\d)\b(?=\s*$|\s*[.!])',
            ]
            for pattern in num_patterns:
                matches = re.findall(pattern, model_response)
                if matches:
                    llm_calculated = matches[-1]
                    break

    # Analyze what went wrong
    if llm_calculated:
        analysis_parts.append(f"✓ LLM calculated/said: {llm_calculated}")

        if llm_calculated.lower().strip() == ground_truth.lower().strip():
            analysis_parts.append(f"✓ LLM's answer matches expected: {ground_truth}")
            analysis_parts.append(f"✗ But extracted: {extracted_answer}")
            analysis_parts.append("→ ROOT CAUSE: Answer extraction failure")
        else:
            analysis_parts.append(f"✗ Expected answer: {ground_truth}")

            # Check if extraction was correct from what LLM said
            if extracted_answer.lower().strip() == llm_calculated.lower().strip():
                analysis_parts.append(f"✓ Extraction correct from LLM response")
                analysis_parts.append("→ ROOT CAUSE: LLM calculation/reasoning error")
            else:
                analysis_parts.append(f"✗ But extracted: {extracted_answer}")
                analysis_parts.append("→ ROOT CAUSE: Both LLM error AND extraction failure")
    else:
        analysis_parts.append(f"✗ Could not find clear answer in LLM response")
        analysis_parts.append(f"✗ Expected: {ground_truth}")
        analysis_parts.append(f"✗ Extracted: {extracted_answer}")
        analysis_parts.append("→ ROOT CAUSE: Unclear/verbose LLM response + extraction difficulty")

    return "\n".join(analysis_parts) if analysis_parts else "Unable to determine cause"

def print_detailed_results(results: List[Dict], title: str, show_analysis: bool = True):
    """Print detailed view of specific results"""
    if not results:
        print(f"\n{title}: No results match criteria")
        return

    print(f"\n{title}: {len(results)} problems")
    print("=" * 80)

    for i, result in enumerate(results, 1):
        test_id = result.get('test_id', 'unknown')
        dataset = result.get('dataset', 'unknown')
        difficulty = result.get('difficulty', 'unknown')
        llm_result = result.get('llm_result', {})
        evaluation = result.get('evaluation', {})
        problem_data = result.get('problem_data', {})
        prompt = result.get('prompt', '')

        # Status
        if not llm_result.get('success'):
            status = f"✗ Failed ({llm_result.get('error', 'unknown')})"
        elif evaluation.get('correct') == True:
            status = "✓ Correct"
        elif evaluation.get('correct') == False:
            status = "○ Incorrect"
        else:
            status = "○ Not evaluated"

        print(f"\nProblem {i}: [{test_id}] {status}")
        print(f"Dataset: {dataset} ({difficulty})")

        if llm_result.get('success'):
            response_time = llm_result.get('response_time', 0)
            print(f"Time: {response_time:.1f}s")

        print("-" * 40)

        # Show the actual problem/prompt
        if prompt:
            print("PROMPT:")
            print(prompt[:800] + "..." if len(prompt) > 800 else prompt)
        elif problem_data:
            print("PROBLEM DATA:")
            # Show key fields based on dataset type
            if dataset == 'winogrande':
                sentence = problem_data.get('sentence', 'No sentence found')
                option1 = problem_data.get('option1', 'No option1')
                option2 = problem_data.get('option2', 'No option2')
                answer = problem_data.get('answer', 'No answer')
                print(f"Sentence: {sentence}")
                print(f"A. {option1}")
                print(f"B. {option2}")
                print(f"Correct answer: {'A' if str(answer) == '1' else 'B'}")
            elif dataset == 'gsm8k':
                question = problem_data.get('question', 'No question found')
                print(f"Question: {question[:300]}..." if len(question) > 300 else f"Question: {question}")
            elif dataset == 'aime_2024' or dataset == 'aime_2025':
                problem = problem_data.get('Problem', problem_data.get('problem', 'No problem found'))
                print(f"Problem: {problem[:300]}..." if len(problem) > 300 else f"Problem: {problem}")
            elif dataset == 'arc_challenge':
                question = problem_data.get('question', 'No question found')
                choices = problem_data.get('choices', {})
                print(f"Question: {question}")
                if choices.get('text') and choices.get('label'):
                    for label, text in zip(choices['label'], choices['text']):
                        print(f"  {label}. {text}")
            elif dataset == 'piqa':
                goal = problem_data.get('goal', 'No goal found')
                sol1 = problem_data.get('sol1', 'No solution1')
                sol2 = problem_data.get('sol2', 'No solution2')
                label = problem_data.get('label', 'No label')
                print(f"Goal: {goal}")
                print(f"A. {sol1}")
                print(f"B. {sol2}")
                print(f"Correct answer: {'A' if label == 0 else 'B'}")
            elif dataset == 'hellaswag':
                ctx = problem_data.get('ctx', 'No context found')
                endings = problem_data.get('endings', [])
                label = problem_data.get('label', 'No label')
                print(f"Context: {ctx}")
                for i, ending in enumerate(endings):
                    print(f"  {chr(65+i)}. {ending}")
                try:
                    correct_idx = int(label)
                    correct_letter = chr(65 + correct_idx)
                    print(f"Correct answer: {correct_letter}")
                except (ValueError, TypeError):
                    print(f"Correct answer: {label}")
            elif dataset == 'logiqa':
                context = problem_data.get('context', 'No context found')
                query = problem_data.get('query', 'No query found')
                options = problem_data.get('options', [])
                correct_option = problem_data.get('correct_option', 'No correct option')
                print(f"Context: {context[:200]}..." if len(context) > 200 else f"Context: {context}")
                print(f"Query: {query}")
                for i, option in enumerate(options):
                    print(f"  {chr(65+i)}. {option}")
                try:
                    correct_letter = chr(65 + correct_option)
                    print(f"Correct answer: {correct_letter}")
                except (TypeError, ValueError):
                    print(f"Correct answer: {correct_option}")
            elif dataset == 'mmlu':
                question = problem_data.get('question', 'No question found')
                choices = problem_data.get('choices', [])
                answer = problem_data.get('answer', 'No answer')
                subject = problem_data.get('subject', 'unknown')
                subject_display = subject.replace('_', ' ').title()
                print(f"Subject: {subject_display}")
                print(f"Question: {question}")
                for i, choice in enumerate(choices):
                    print(f"  {chr(65+i)}. {choice}")
                try:
                    correct_letter = chr(65 + answer)
                    print(f"Correct answer: {correct_letter}")
                except (TypeError, ValueError):
                    print(f"Correct answer: {answer}")
            elif dataset == 'competition_math':
                problem = problem_data.get('problem', 'No problem found')
                solution = problem_data.get('solution', 'No solution found')
                subject = problem_data.get('type', 'Unknown')
                level = problem_data.get('level', 'Unknown')
                print(f"Subject: {subject} ({level})")
                print(f"Problem: {problem[:300]}..." if len(problem) > 300 else f"Problem: {problem}")
                print(f"Solution: {solution[:200]}..." if len(solution) > 200 else f"Solution: {solution}")

        print("-" * 40)

        # Show model response and evaluation
        if llm_result.get('success'):
            model_response = llm_result.get('content', 'No response content')
            print("MODEL RESPONSE:")
            print(model_response)  # Show full response without truncation

            print("-" * 40)
            print("EVALUATION:")
            if evaluation:
                ground_truth = evaluation.get('ground_truth')
                extracted_answer = evaluation.get('extracted_answer')
                if ground_truth is not None and extracted_answer is not None:
                    print(f"Expected: {ground_truth}")
                    print(f"Extracted: {extracted_answer}")
                elif evaluation.get('error'):
                    print(f"Evaluation error: {evaluation.get('error')}")

                # Show analysis for failures
                if show_analysis and evaluation.get('correct') == False:
                    print("-" * 40)
                    print("ANALYSIS:")
                    analysis = analyze_failure(result)
                    print(analysis)

            else:
                print("No evaluation data")
        else:
            print(f"MODEL FAILED: {llm_result.get('error', 'unknown error')}")

        print("=" * 80)

def compare_results(file1: Path, file2: Path):
    """Compare two result files side by side"""
    data1 = load_result_file(file1)
    data2 = load_result_file(file2)

    meta1 = data1.get('metadata', {})
    meta2 = data2.get('metadata', {})

    print(f"\n🔍 Comparison: {file1.name} vs {file2.name}")
    print("=" * 80)

    print(f"{'Metric':<25} {'File 1':<25} {'File 2':<25}")
    print("-" * 75)
    print(f"{'Model':<25} {meta1.get('model_name', 'Unknown'):<25} {meta2.get('model_name', 'Unknown'):<25}")
    print(f"{'Total Problems':<25} {meta1.get('total_problems', 0):<25} {meta2.get('total_problems', 0):<25}")
    print(f"{'Completion Rate':<25} {meta1.get('completion_rate', 0):.1f}%{'':<21} {meta2.get('completion_rate', 0):.1f}%{'':<21}")
    print(f"{'Overall Accuracy':<25} {meta1.get('accuracy_overall', 0):.1f}%{'':<21} {meta2.get('accuracy_overall', 0):.1f}%{'':<21}")
    print(f"{'Accuracy (completed)':<25} {meta1.get('accuracy_of_completed', 0):.1f}%{'':<21} {meta2.get('accuracy_of_completed', 0):.1f}%{'':<21}")
    print(f"{'Timeout':<25} {meta1.get('timeout_seconds', 0)}s{'':<22} {meta2.get('timeout_seconds', 0)}s{'':<22}")

def main():
    parser = argparse.ArgumentParser(description='View benchmark results')
    parser.add_argument('files', nargs='*', help='Result JSON file(s)')
    parser.add_argument('--failures', action='store_true', help='Show only incorrect problems and answers')
    parser.add_argument('--successes', action='store_true', help='Show only correct problems and answers')
    parser.add_argument('--compare', action='store_true', help='Compare two result files')
    parser.add_argument('--dataset', help='Filter by dataset (e.g., gsm8k)')

    args = parser.parse_args()

    # Get files (interactive or from command line)
    if not args.files:
        files = interactive_file_selection()
    else:
        files = [Path("JSON_RESULTS") / f if not f.startswith('/') else Path(f) for f in args.files]
        # Validate files exist
        for f in files:
            if not f.exists():
                print(f"File not found: {f}")
                sys.exit(1)

    # Handle comparison mode
    if len(files) == 2 or args.compare:
        if len(files) != 2:
            print("Comparison requires exactly 2 files")
            sys.exit(1)
        compare_results(files[0], files[1])
        return

    # Single file analysis
    if len(files) != 1:
        print("Please specify 1 file for analysis or 2 files for comparison")
        sys.exit(1)

    data = load_result_file(files[0])
    results = data.get('results', [])

    # Apply filters
    filtered_results = filter_results(results, args.dataset, args.failures, args.successes)

    # Show summary unless filtering for specific results
    if not args.failures and not args.successes:
        print_summary(data, files[0])

    # Show detailed results if filtering
    if args.failures:
        print_detailed_results(filtered_results, "❌ Failed/Incorrect Problems", show_analysis=True)
    elif args.successes:
        print_detailed_results(filtered_results, "✅ Correct Problems", show_analysis=False)
    elif args.dataset:
        print_detailed_results(filtered_results, f"📚 {args.dataset.upper()} Problems", show_analysis=True)

if __name__ == "__main__":
    main()
