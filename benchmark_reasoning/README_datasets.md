# Dataset Overview

This document describes each dataset used in our benchmark configurations, with examples to help understand what kinds of problems they contain.

## 🧮 Mathematics Datasets

### GSM8K (Grade School Math 8K)
**What it tests**: Elementary mathematical reasoning and word problem solving
**Format**: Word problems requiring multi-step arithmetic
**Difficulty**: Elementary to middle school level
**Answer format**: Numerical (extracted from "#### X" format)

**Example problem**:
```
Question: "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
Answer: "#### 72"
```

**Why useful**: Tests basic mathematical reasoning that any competent model should handle well. Good baseline for mathematical capabilities.

---

### 🏆 AIME 2024/2025 (American Invitational Mathematics Examination)
**What it tests**: Advanced mathematical problem-solving and proof techniques
**Format**: Competition mathematics problems
**Difficulty**: High school mathematical olympiad level
**Answer format**: Integers from 0-999

**Example problem**:
```
"Let $ABCD$ be a rhombus with $AC = 16$ and $BD = 30$. Let $N$ be a point on $\overline{AB}$ with $AN = \frac{5}{13}AB$. Let $P$ be the foot of the perpendicular from $N$ to $\overline{AC}$. Find $NP$."
Answer: 123
```

**Why useful**: Tests upper limits of mathematical reasoning. Even strong models often struggle with these problems. Good for identifying truly exceptional mathematical capability.

---

### 🎓 Numina Math
**What it tests**: University-level mathematics across various domains
**Format**: Advanced mathematical problems and proofs
**Difficulty**: Undergraduate to graduate level mathematics
**Answer format**: Varies (not yet evaluated in our system)

**Why useful**: Fills the gap between elementary GSM8K and competition AIME problems. Tests more advanced mathematical concepts and reasoning.

---

## 🔬 Science & Reasoning Datasets

### ARC Challenge (AI2 Reasoning Challenge)
**What it tests**: Scientific reasoning and knowledge application
**Format**: Multiple choice questions (A, B, C, D)
**Difficulty**: Elementary to middle school science level
**Answer format**: Single letter choice

**Example problem**:
```
Question: "Which of the following is an example of a physical change?"
Choices:
A. Burning wood
B. Rusting iron
C. Melting ice
D. Baking bread
Answer: C
```

**Why useful**: Tests scientific knowledge and reasoning without requiring advanced mathematics. Good for evaluating general reasoning about the physical world.

---

### 🎓 MMLU (Massive Multitask Language Understanding)
**What it tests**: Academic knowledge across 57 subjects from STEM to humanities
**Format**: Multiple choice questions (A, B, C, D)
**Difficulty**: Undergraduate to graduate level across diverse academic domains
**Answer format**: Single letter choice (fully evaluated)

**Example problem**:
```
Subject: Abstract Algebra
Question: "Find the degree for the given field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q."
A. 0
B. 4
C. 2
D. 6
Answer: B
```

**Subject areas include**: Mathematics, Physics, Chemistry, Biology, Computer Science, History, Philosophy, Psychology, Economics, Law, Medicine, and many more.

**Why useful**: Comprehensive test of academic knowledge and reasoning across virtually all scholarly domains. Gold standard for evaluating general intelligence and knowledge breadth. With 14,000+ test questions, provides robust statistical evaluation.

---

## 🧩 Logic & Reasoning Datasets

### ⚔️ Knights & Knaves
**What it tests**: Logical deduction and reasoning about truth/lies
**Format**: Logic puzzles involving truth-tellers (knights) and liars (knaves)
**Difficulty**: Scales with number of people involved (2-8 people)
**Answer format**: Varies (not yet evaluated in our system)

**Example problem**:
```
"You meet two people, A and B. A says: 'I am a knight and B is a knave.' What are A and B?"
Expected reasoning: If A is a knight (truth-teller), then A's statement is true, so B is a knave. If A is a knave (liar), then A's statement is false, but the statement "I am a knight" would be false (which knaves can say), creating consistency issues. Therefore A is a knight and B is a knave.
```

**Why useful**: Tests pure logical reasoning without domain knowledge. Difficulty scales predictably with problem complexity (more people = harder). Good for comparing logical thinking capabilities.

---

### 🧩 Puzzte
**What it tests**: Complex logical reasoning and problem-solving
**Format**: Various types of logic puzzles
**Difficulty**: Varies (measured by "ambiguity score" - higher = more difficult)
**Answer format**: Varies (not yet evaluated in our system)

**Example characteristics**: Logic grid puzzles, constraint satisfaction problems, multi-step reasoning challenges. The ambiguity score provides a proxy for difficulty level.

**Why useful**: Tests complex reasoning patterns. The ambiguity scoring gives us some measure of difficulty progression, though not as clean as Knights & Knaves.

---

### 🤔 Winogrande (Commonsense Coreference)
**What it tests**: Common sense reasoning and coreference resolution
**Format**: Fill-in-the-blank sentences with A/B choices
**Difficulty**: Requires world knowledge and common sense
**Answer format**: A or B choice (fully evaluated)

**Example problem**:
```
"Ian volunteered to eat Dennis's menudo after already having a bowl because _ enjoyed eating intestine."
A. Ian
B. Dennis
Answer: A (Ian enjoyed it, so he volunteered for more)
```

**Why useful**: Tests common sense reasoning about human behavior and world knowledge. More robust than old Winograd format with cleaner A/B evaluation.

---

### 🔧 PIQA (Physical Interaction QA)
**What it tests**: Physical reasoning and real-world problem solving
**Format**: Goal-solution pairs with A/B choices
**Difficulty**: Everyday practical reasoning
**Answer format**: A or B choice (fully evaluated)

**Example problem**:
```
Goal: "How do I ready a guinea pig cage for its new occupants?"
A. "Provide the guinea pig with a cage full of a few inches of bedding made of ripped paper strips, you will also need to supply it with a water bottle and a food dish."
B. "Provide the guinea pig with a cage full of a few inches of bedding made of ripped jeans material, you will also need to supply it with a water bottle and a food dish."
Answer: A (paper is safe bedding; jeans material could be harmful)
```

**Why useful**: Tests practical reasoning about physical interactions and real-world problem solving. Good complement to abstract logical reasoning.

---

### 🎢 HellaSwag (Commonsense Reasoning)
**What it tests**: Commonsense reasoning about likely continuations
**Format**: Scenario descriptions with 4 multiple choice endings
**Difficulty**: Requires understanding of typical human behavior and situations
**Answer format**: A, B, C, or D choice (fully evaluated)

**Example problem**:
```
Context: "A man is sitting on a roof. he"
A. "is using wrap to wrap a pair of skis."
B. "is ripping level tiles off."
C. "is holding a rubik's cube."
D. "starts pulling up roofing on a roof."
Answer: B (most logical continuation for someone on a roof)
```

**Why useful**: Tests ability to predict likely continuations of scenarios based on commonsense knowledge. Large validation set (10K+ examples).

---

### 🧠 LogiQA (Logical Reasoning)
**What it tests**: Formal logical reasoning and critical thinking
**Format**: Context + question with 4 multiple choice answers
**Difficulty**: University-level logical reasoning
**Answer format**: A, B, C, or D choice (fully evaluated)

**Example problem**:
```
Context: "Black Americans are twice as likely to suffer from hypertension as white Americans..."
Question: "Which conclusion, if true, best supports the researchers' hypothesis?"
A-D: Various logical conclusions about the hypothesis
Answer: A (the option that provides the strongest logical support)
```

**Why useful**: Tests formal logical reasoning skills similar to standardized test logic sections. More structured than commonsense reasoning tasks.

---

## 📊 Dataset Comparison & Selection Guide

### For Pure Mathematical Testing:
- **Elementary**: GSM8K
- **Advanced**: Numina Math
- **Expert**: AIME

### For Logical & Commonsense Reasoning:
- **Structured Logic**: Knights & Knaves (clean difficulty progression)
- **Formal Logic**: LogiQA (university-level reasoning)
- **Complex Puzzles**: Puzzte (varied puzzle types)
- **Commonsense**: Winogrande (coreference resolution)
- **Scenario Reasoning**: HellaSwag (situation continuations)
- **Physical Reasoning**: PIQA (real-world problem solving)

### For Scientific & Academic Knowledge:
- **General Science**: ARC Challenge (elementary/middle school level)
- **Comprehensive Academic**: MMLU (57 subjects, university level)

### Gaps & Future Additions:
**Missing capabilities we might want to test**:
- **Language understanding**: Reading comprehension, linguistic reasoning
- **Code reasoning**: Programming logic, algorithm understanding
- **Spatial reasoning**: Geometry, visual-spatial problems
- **Temporal reasoning**: Understanding sequences, cause-and-effect

**Evaluating new puzzle datasets**: Consider whether they fill capability gaps or provide cleaner difficulty progression than existing options. Knights & Knaves is particularly valuable because it has genuine difficulty levels, unlike many datasets where difficulty is assumed or estimated.

---

## 🔧 Custom Datasets

### RMB Puzzles
**What it tests**: Custom puzzle reasoning and problem-solving approach validation
**Format**: Multiple choice questions (1-4) for water jug problems and similar puzzles
**Difficulty**: Easy (expandable)
**Answer format**: Numerical choice (1, 2, 3, or 4)

**Example problem**:
```
Question: "I have a full barrel and empty barrel, plus 12L and 6L jugs. To move exactly 6L from full to empty barrel, what's the minimum number of pour operations?"
Options: ["1", "2", "3", "4"]
Answer: "2"
```

**Why useful**:
- **Custom dataset integration example**: Demonstrates how to add personal puzzles to the benchmark system
- **JSON format validation**: Tests the JSON dataset loading capability added to benchmark_runner.py
- **Multiple choice evaluation**: Validates numerical multiple choice answer extraction
- **Puzzle reasoning baseline**: Provides a foundation for expanding custom puzzle collections

**Technical notes**:
- Uses `format: "json"` in YAML configuration
- Loads from local `rmb_puzzles.json` file
- Custom evaluation logic in `benchmark_runner.py` for `rmb_puzzles` dataset type
- Fully integrated with both regex and LLM extraction systems