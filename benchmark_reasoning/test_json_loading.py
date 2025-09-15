#!/usr/bin/env python3
"""
Test JSON dataset loading - the simple way!
"""

from datasets import load_dataset

def test_json_loading():
    """Test loading waterjug.json directly with load_dataset"""

    print("🧪 Testing direct JSON loading...")

    # Load directly from JSON file
    dataset = load_dataset("json", data_files="rmb_puzzles.json")

    print(f"✅ Loaded dataset: {dataset}")
    print(f"📊 Dataset info: {len(dataset['train'])} problems")

    print("\n🧩 Dataset contents:")
    print("=" * 50)

    # Print the problem
    problem = dataset['train'][0]
    print(f"Question: {problem['question']}")
    print(f"Options: {problem['options']}")
    print(f"Answer: {problem['answer']}")
    print(f"Difficulty: {problem['difficulty']}")

    print("\n🎯 Ready for YAML config!")
    print('Add to your YAML like this:')
    print("""
rmb_puzzles:
  path: "json"
  data_files: "rmb_puzzles.json"
  format: "huggingface"
  split: "train"
  mode: "counts"
  difficulty_source: "real"
  counts:
    easy: 1
""")

if __name__ == "__main__":
    test_json_loading()
