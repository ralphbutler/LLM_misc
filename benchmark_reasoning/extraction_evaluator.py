#!/usr/bin/env python3
"""
Extraction Model Evaluator
Tests different local LLMs on their ability to extract structured answers from benchmark responses.
"""

import sys
import os
import json
import time
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from litellm import completion
import litellm

# Enable JSON schema validation
litellm.enable_json_schema_validation = True

class ExtractedAnswer(BaseModel):
    final_answer: str          # "A", "42", "yes", etc.
    confidence: float          # 0.0-1.0 extraction confidence
    reasoning_excerpt: Optional[str] = None  # Key reasoning steps from response
    extraction_source: str     # Exact text that led to answer
    ambiguous: bool           # Multiple valid interpretations?
    answer_format: str        # "multiple_choice", "numerical", "yes_no", "text"

# Test models (hardcoded for now)
TEST_MODELS = [
    ("openai/glm-4.5-air", "http://localhost:1234/v1"),
    ("openai/phi-4-reasoning-plus-mlx", "http://localhost:1234/v1"),
    ("openai/qwen3-32b-mlx", "http://localhost:1234/v1"),
    ("openai/llama-3.2-3b-instruct", "http://localhost:1234/v1"),
    ("openai/deepseek-r1-0528-qwen3-8b-mlx", "http://localhost:1234/v1")
]

def create_extraction_prompt(original_response: str, dataset_type: str, question_text: str = "") -> str:
    """Create prompt for answer extraction"""

    # Determine expected answer format based on dataset
    format_guides = {
        "winogrande": "Expected: Single letter A or B",
        "piqa": "Expected: Single number 0 or 1",
        "gsm8k": "Expected: Pure number like 42 or 3.5",
        "competition_math": "Expected: Number, fraction, or mathematical expression",
        "hellaswag": "Expected: Single number 0, 1, 2, or 3",
        "logiqa": "Expected: Single letter A, B, C, or D"
    }

    format_guide = format_guides.get(dataset_type, "Expected: Extract the final answer")

    return f"""You are an expert at extracting final answers from LLM responses to benchmark questions.

DATASET: {dataset_type}
{format_guide}

ORIGINAL QUESTION: {question_text[:200]}...

LLM RESPONSE TO ANALYZE:
{original_response}

Your task: Extract the final answer and provide structured information about the extraction.

Be very precise. Look for:
- Clear answer statements like "The answer is X"
- Bold or emphasized text
- Numbers at the end of calculations
- Multiple choice selections

Return confidence based on how clear the answer extraction was.
Mark as ambiguous if multiple valid interpretations exist.
"""

def test_model_extraction(model_name: str, api_base: str, test_cases: List[Dict]) -> Dict[str, Any]:
    """Test a single model on extraction tasks"""

    print(f"\nTesting model: {model_name}")
    results = {
        "model": model_name,
        "total_tests": len(test_cases),
        "successful_extractions": 0,
        "pydantic_successes": 0,
        "avg_confidence": 0.0,
        "extraction_times": [],
        "failures": [],
        "extractions": []
    }

    for i, test_case in enumerate(test_cases):
        print(f"  Test {i+1}/{len(test_cases)}: {test_case['dataset']}...")

        try:
            start_time = time.time()

            # Create extraction prompt
            prompt = create_extraction_prompt(
                test_case["llm_response"],
                test_case["dataset"],
                test_case.get("question", "")
            )

            messages = [{"role": "user", "content": prompt}]

            # Try Pydantic approach first
            success = False
            extraction_result = None

            try:
                response = completion(
                    model=model_name,
                    messages=messages,
                    api_base=api_base,
                    max_tokens=1024,
                    temperature=0.0,
                    response_format=ExtractedAnswer
                )

                content = response.choices[0].message.content
                json_content = json.loads(content)
                extraction_result = ExtractedAnswer(**json_content)

                results["pydantic_successes"] += 1
                success = True

            except Exception as e:
                print(f"    Pydantic failed: {str(e)[:100]}...")

                # Try JSON schema fallback
                try:
                    response = completion(
                        model=model_name,
                        messages=messages,
                        api_base=api_base,
                        max_tokens=1024,
                        temperature=0.0,
                        response_format={
                            "type": "json_object",
                            "response_schema": ExtractedAnswer.model_json_schema()
                        }
                    )

                    content = response.choices[0].message.content
                    json_content = json.loads(content)
                    extraction_result = ExtractedAnswer(**json_content)
                    success = True

                except Exception as e2:
                    print(f"    JSON schema also failed: {str(e2)[:100]}...")
                    results["failures"].append({
                        "test_case": i,
                        "error": str(e2)
                    })

            end_time = time.time()
            results["extraction_times"].append(end_time - start_time)

            if success:
                results["successful_extractions"] += 1
                results["extractions"].append({
                    "test_case": i,
                    "dataset": test_case["dataset"],
                    "extracted_answer": extraction_result.final_answer,
                    "confidence": extraction_result.confidence,
                    "ambiguous": extraction_result.ambiguous,
                    "answer_format": extraction_result.answer_format,
                    "extraction_time": end_time - start_time
                })

        except Exception as e:
            print(f"    Complete failure: {str(e)}")
            results["failures"].append({
                "test_case": i,
                "error": str(e)
            })

    # Calculate averages
    if results["extractions"]:
        results["avg_confidence"] = sum(ex["confidence"] for ex in results["extractions"]) / len(results["extractions"])

    results["success_rate"] = results["successful_extractions"] / results["total_tests"]
    results["pydantic_rate"] = results["pydantic_successes"] / results["total_tests"]
    results["avg_extraction_time"] = sum(results["extraction_times"]) / len(results["extraction_times"]) if results["extraction_times"] else 0

    return results

def create_sample_test_cases() -> List[Dict]:
    """Create sample test cases for evaluation"""
    return [
        {
            "dataset": "gsm8k",
            "question": "A store has 42 apples. If they sell 15, how many are left?",
            "llm_response": "I need to subtract 15 from 42.\n\n42 - 15 = 27\n\nTherefore, there are 27 apples left.",
            "expected_answer": "27"
        },
        {
            "dataset": "winogrande",
            "question": "The trophy doesn't fit in the brown suitcase because _ is too large. (A) trophy (B) suitcase",
            "llm_response": "Looking at this sentence, I need to determine what is too large.\n\nThe trophy doesn't fit in the suitcase because the trophy is too large.\n\n**The answer is A.**",
            "expected_answer": "A"
        },
        {
            "dataset": "piqa",
            "question": "How to remove rust from metal? (0) Use lemon juice (1) Use vinegar",
            "llm_response": "Both lemon juice and vinegar contain acid that can help remove rust. However, vinegar is more commonly used and effective for rust removal.\n\nThe answer is 1.",
            "expected_answer": "1"
        },
        {
            "dataset": "hellaswag",
            "question": "A person is cooking pasta. Next they will: (0) drain water (1) add more salt (2) turn off heat (3) serve immediately",
            "llm_response": "When cooking pasta, after it's done cooking, the logical next step would be to drain the water before doing anything else.\n\nChoose 0: drain water",
            "expected_answer": "0"
        },
        {
            "dataset": "competition_math",
            "question": "What is 2/3 + 1/6?",
            "llm_response": "To add these fractions, I need a common denominator.\n\n2/3 = 4/6\n\nSo: 4/6 + 1/6 = 5/6\n\nThe answer is 5/6.",
            "expected_answer": "5/6"
        }
    ]

def main():
    print("Extraction Model Evaluator")
    print("=" * 50)

    # Create test cases
    test_cases = create_sample_test_cases()
    print(f"Created {len(test_cases)} test cases")

    # Test each model
    all_results = []

    for model_name, api_base in TEST_MODELS:
        try:
            model_results = test_model_extraction(model_name, api_base, test_cases)
            all_results.append(model_results)

            print(f"\n{model_name} Results:")
            print(f"  Success Rate: {model_results['success_rate']:.1%}")
            print(f"  Pydantic Rate: {model_results['pydantic_rate']:.1%}")
            print(f"  Avg Confidence: {model_results['avg_confidence']:.2f}")
            print(f"  Avg Time: {model_results['avg_extraction_time']:.2f}s")

        except Exception as e:
            print(f"Failed to test {model_name}: {e}")

    # Summary report
    print("\n" + "=" * 50)
    print("FINAL COMPARISON")
    print("=" * 50)

    for results in sorted(all_results, key=lambda x: x['success_rate'], reverse=True):
        print(f"{results['model']:25} | Success: {results['success_rate']:.1%} | Pydantic: {results['pydantic_rate']:.1%} | Confidence: {results['avg_confidence']:.2f} | Time: {results['avg_extraction_time']:.2f}s")

    # Save detailed results
    with open("extraction_evaluation_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nDetailed results saved to: extraction_evaluation_results.json")

if __name__ == "__main__":
    main()
