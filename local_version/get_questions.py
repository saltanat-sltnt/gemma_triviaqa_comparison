from datasets import load_dataset

dataset = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")

for i in range(10):
    print(f"{i+1}. {dataset[i]['question']}")
    print(f"Answer: {dataset[i]['answer']['value']}")
    print(f"Aliases: {dataset[i]['answer']['aliases']}")