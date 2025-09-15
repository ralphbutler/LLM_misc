#!/usr/bin/env python3
"""
Dataset Viewer for Benchmark Runner

View actual problems and answers from benchmark datasets.
Helps understand dataset content before creating benchmark configurations.
"""

import argparse
import sys
import pandas as pd
from datasets import load_dataset
from pathlib import Path
from typing import Any, Dict, List

def load_gsm8k_data() -> List[Dict[str, Any]]:
    """Load GSM8K dataset using datasets library"""
    try:
        dataset = load_dataset("openai/gsm8k", "main")
        return list(dataset['test'])
    except Exception as e:
        print(f"Error loading GSM8K: {e}")
        sys.exit(1)

def load_numina_math_data() -> List[Dict[str, Any]]:
    """Load Numina Math dataset using datasets library"""
    try:
        dataset = load_dataset("AI-MO/NuminaMath-TIR")
        return list(dataset['test'])
    except Exception as e:
        print(f"Error loading Numina Math: {e}")
        sys.exit(1)

def load_dataset_data(dataset_name: str) -> List[Dict[str, Any]]:
    """Load data from specified dataset"""
    dataset_name = dataset_name.lower()

    if dataset_name == 'gsm8k':
        return load_gsm8k_data()

    elif dataset_name == 'numina_math':
        return load_numina_math_data()

    elif dataset_name == 'aime_2024':
        try:
            dataset = load_dataset("Maxwell-Jia/AIME_2024")
            return list(dataset['train'])
        except Exception as e:
            print(f"Error loading AIME 2024: {e}")
            sys.exit(1)

    elif dataset_name == 'aime_2025':
        try:
            dataset = load_dataset("yentinglin/aime_2025")
            return list(dataset['train'])
        except Exception as e:
            print(f"Error loading AIME 2025: {e}")
            sys.exit(1)

    elif dataset_name == 'arc_challenge':
        try:
            dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge")
            return list(dataset['test'])
        except Exception as e:
            print(f"Error loading ARC Challenge: {e}")
            sys.exit(1)


    elif dataset_name == 'puzzte':
        try:
            dataset = load_dataset("tasksource/puzzte")
            return list(dataset['validation'])
        except Exception as e:
            print(f"Error loading Puzzte: {e}")
            sys.exit(1)

    elif dataset_name == 'winogrande':
        try:
            dataset = load_dataset("winogrande", "winogrande_xl")
            return list(dataset['validation'])
        except Exception as e:
            print(f"Error loading Winogrande: {e}")
            sys.exit(1)

    elif dataset_name == 'piqa':
        try:
            dataset = load_dataset("piqa", trust_remote_code=True)
            return list(dataset['validation'])
        except Exception as e:
            print(f"Error loading PIQA: {e}")
            sys.exit(1)

    elif dataset_name == 'hellaswag':
        try:
            dataset = load_dataset("hellaswag")
            return list(dataset['validation'])
        except Exception as e:
            print(f"Error loading HellaSwag: {e}")
            sys.exit(1)

    elif dataset_name == 'logiqa':
        try:
            dataset = load_dataset("lucasmccabe/logiqa")
            return list(dataset['validation'])
        except Exception as e:
            print(f"Error loading LogiQA: {e}")
            sys.exit(1)

    elif dataset_name == 'mmlu':
        try:
            dataset = load_dataset("cais/mmlu", "all")
            return list(dataset['test'])
        except Exception as e:
            print(f"Error loading MMLU: {e}")
            sys.exit(1)

    elif dataset_name == 'competition_math':
        try:
            dataset = load_dataset("qwedsacf/competition_math")
            return list(dataset['train'])
        except Exception as e:
            print(f"Error loading MATH: {e}")
            sys.exit(1)

    else:
        print(f"Unknown dataset: {dataset_name}")
        print("Available datasets: gsm8k, numina_math, aime_2024, aime_2025, arc_challenge, puzzte, winogrande, piqa, hellaswag, logiqa, mmlu, competition_math")
        sys.exit(1)

def format_problem(dataset_name: str, problem: Dict[str, Any], index: int) -> str:
    """Format a problem for display based on dataset type"""
    separator = "=" * 80
    header = f"Problem {index + 1} - {dataset_name.upper()}"

    if dataset_name == 'gsm8k':
        question = problem.get('question', 'No question found')
        answer = problem.get('answer', 'No answer found')
        return f"{separator}\n{header}\n{separator}\n\nQuestion:\n{question}\n\nAnswer:\n{answer}\n"

    elif dataset_name == 'numina_math':
        problem_text = problem.get('problem', 'No problem found')
        solution = problem.get('solution', 'No solution found')
        return f"{separator}\n{header}\n{separator}\n\nProblem:\n{problem_text}\n\nSolution:\n{solution}\n"

    elif dataset_name in ['aime_2024', 'aime_2025']:
        problem_text = problem.get('Problem', problem.get('problem', 'No problem found'))
        answer = problem.get('Answer', problem.get('answer', 'No answer found'))
        return f"{separator}\n{header}\n{separator}\n\nProblem:\n{problem_text}\n\nAnswer:\n{answer}\n"

    elif dataset_name == 'arc_challenge':
        question = problem.get('question', 'No question found')
        choices = problem.get('choices', {})
        choice_text = choices.get('text', [])
        choice_labels = choices.get('label', [])
        answer_key = problem.get('answerKey', 'No answer key found')

        choices_str = ""
        for label, text in zip(choice_labels, choice_text):
            choices_str += f"{label}. {text}\n"

        return f"{separator}\n{header}\n{separator}\n\nQuestion:\n{question}\n\nChoices:\n{choices_str}\nAnswer: {answer_key}\n"


    elif dataset_name == 'puzzte':
        puzzle_text = problem.get('puzzle_text', 'No puzzle text found')
        question = problem.get('question', 'No question found')
        answer = problem.get('answer', 'No answer found')
        ambiguity = problem.get('ambiguity', 'No ambiguity score')

        return f"{separator}\n{header} (Ambiguity: {ambiguity})\n{separator}\n\nPuzzle:\n{puzzle_text}\n\nQuestion:\n{question}\n\nAnswer:\n{answer}\n"

    elif dataset_name == 'winogrande':
        sentence = problem.get('sentence', 'No sentence found')
        option1 = problem.get('option1', 'No option1 found')
        option2 = problem.get('option2', 'No option2 found')
        answer = problem.get('answer', 'No answer found')
        correct_letter = 'A' if str(answer) == '1' else 'B'

        return f"{separator}\n{header}\n{separator}\n\nSentence:\n{sentence}\n\nA. {option1}\nB. {option2}\n\nAnswer: {correct_letter}\n"

    elif dataset_name == 'piqa':
        goal = problem.get('goal', 'No goal found')
        sol1 = problem.get('sol1', 'No solution1 found')
        sol2 = problem.get('sol2', 'No solution2 found')
        label = problem.get('label', 'No label found')
        correct_letter = 'A' if label == 0 else 'B'

        return f"{separator}\n{header}\n{separator}\n\nGoal:\n{goal}\n\nA. {sol1}\nB. {sol2}\n\nAnswer: {correct_letter}\n"

    elif dataset_name == 'hellaswag':
        ctx = problem.get('ctx', 'No context found')
        endings = problem.get('endings', [])
        label = problem.get('label', 'No label found')
        try:
            correct_idx = int(label)
            correct_letter = chr(65 + correct_idx)
        except (ValueError, TypeError):
            correct_letter = str(label)

        choices_str = ""
        for i, ending in enumerate(endings):
            choices_str += f"{chr(65+i)}. {ending}\n"

        return f"{separator}\n{header}\n{separator}\n\nContext:\n{ctx}\n\nChoices:\n{choices_str}\nAnswer: {correct_letter}\n"

    elif dataset_name == 'logiqa':
        context = problem.get('context', 'No context found')
        query = problem.get('query', 'No query found')
        options = problem.get('options', [])
        correct_option = problem.get('correct_option', 'No correct option found')
        try:
            correct_letter = chr(65 + correct_option)
        except (TypeError, ValueError):
            correct_letter = str(correct_option)

        choices_str = ""
        for i, option in enumerate(options):
            choices_str += f"{chr(65+i)}. {option}\n"

        return f"{separator}\n{header}\n{separator}\n\nContext:\n{context}\n\nQuery:\n{query}\n\nChoices:\n{choices_str}\nAnswer: {correct_letter}\n"

    elif dataset_name == 'mmlu':
        question = problem.get('question', 'No question found')
        choices = problem.get('choices', [])
        answer = problem.get('answer', 'No answer found')
        subject = problem.get('subject', 'Unknown subject')
        try:
            correct_letter = chr(65 + answer)
        except (TypeError, ValueError):
            correct_letter = str(answer)

        choices_str = ""
        for i, choice in enumerate(choices):
            choices_str += f"{chr(65+i)}. {choice}\n"

        subject_display = subject.replace('_', ' ').title()
        return f"{separator}\n{header} - {subject_display}\n{separator}\n\nQuestion:\n{question}\n\nChoices:\n{choices_str}\nAnswer: {correct_letter}\n"

    elif dataset_name == 'competition_math':
        problem_text = problem.get('problem', 'No problem found')
        solution = problem.get('solution', 'No solution found')
        subject = problem.get('type', 'Unknown subject')
        level = problem.get('level', 'Unknown level')

        return f"{separator}\n{header} - {subject} ({level})\n{separator}\n\nProblem:\n{problem_text}\n\nSolution:\n{solution}\n"

    else:
        # Generic fallback
        return f"{separator}\n{header}\n{separator}\n\nRaw data:\n{problem}\n"

def main():
    parser = argparse.ArgumentParser(description='View dataset problems and answers')
    parser.add_argument('dataset', help='Dataset name (gsm8k, aime_2024, arc_challenge, etc.)')
    parser.add_argument('--limit', type=int, default=10, help='Number of problems to show (default: 10)')
    parser.add_argument('--start', type=int, default=0, help='Starting index (default: 0)')
    parser.add_argument('--all', action='store_true', help='Show all problems (ignores --limit)')

    args = parser.parse_args()

    print(f"Loading {args.dataset} dataset...")
    try:
        data = load_dataset_data(args.dataset)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    total_problems = len(data)
    print(f"Loaded {total_problems} problems from {args.dataset}")
    print()

    # Determine range to show
    if args.all:
        start_idx = 0
        end_idx = total_problems
        print("Showing ALL problems:")
    else:
        start_idx = args.start
        end_idx = min(args.start + args.limit, total_problems)
        print(f"Showing problems {start_idx + 1}-{end_idx} of {total_problems}:")
        if end_idx < total_problems:
            print("(Use --all to see all problems, or --start N --limit M for different range)")

    print()

    # Display problems
    for i in range(start_idx, end_idx):
        formatted_problem = format_problem(args.dataset, data[i], i)
        print(formatted_problem)

    # Summary
    if not args.all and end_idx < total_problems:
        remaining = total_problems - end_idx
        print(f"\n... {remaining} more problems available")
        print(f"Use: python3 view_dataset.py {args.dataset} --start {end_idx} --limit {args.limit}")
        print(f"Or:  python3 view_dataset.py {args.dataset} --all | less")

if __name__ == "__main__":
    main()
