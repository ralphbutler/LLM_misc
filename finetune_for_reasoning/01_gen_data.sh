
# python generate_chat_samples.py -n 20 -d "very easy" --1lm phi4 > easy.jsonl

echo DOING VERY EASY
python generate_chat_samples.py -n 100 -d "very easy" --llm "granite3.1-dense" > veryeasy.jsonl

echo DOING EASY
python generate_chat_samples.py -n 100 -d   easy --llm "granite3.1-dense" > easy.jsonl

echo DOING MEDIUM
python generate_chat_samples.py -n 100 -d medium --llm "granite3.1-dense" > medium.jsonl

echo DOING HARD
python generate_chat_samples.py -n 100 -d   hard --llm "granite3.1-dense" > hard.jsonl
