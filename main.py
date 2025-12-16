"""Main entry point for the index-interview system."""

from pprint import pprint

from codex import add_assistant_message, call_codex, get_history
from dataset import PRBenchDataset


def answer_all_prompts(sample):
    for turn in range(sample['turns']):
        prompt = sample[f'prompt_{turn}']
        history = "\n".join(get_history())

        response = call_codex(prompt)
        print(f"Prompt {turn}: {prompt}")
        print(f"Response {turn}: {response}")

def main():
    """Main system logic."""
    print("\n" + "=" * 60)
    print("Index Interview System")
    print("=" * 60 + "\n")
    
    # Initialize dataset
    prbench = PRBenchDataset()
    prbench.load()
    
    # Get a sample task
    print("\n📋 Getting a sample task from the dataset...")
    sample = prbench.get_sample("finance", 0)

    pprint(sample['turns'])
    pprint(sample['prompt_0'])
    pprint(sample['prompt_1'])
    pprint(sample['prompt_2'])
    pprint(sample['prompt_3'])
    
    # print(f"Topic: {sample['topic']}")
    # print(f"Expert: {sample['expert']}")
    # print(f"Turns: {sample['turns']}")
    
    # # Example: Use codex to analyze a sample
    # print("\n🤖 Testing Codex integration...")
    # prompt = "What is 2+2?"
    # result = call_codex(prompt)
    
    # if result['success']:
    #     print(f"✓ Codex response: {result['output']}")
    # else:
    #     print(f"✗ Codex error: {result['error']}")
    
    # print("\n" + "=" * 60)
    # print("System ready!")
    # print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
