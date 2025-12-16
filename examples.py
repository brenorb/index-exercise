"""Example usage of the system components."""

from codex import call_codex
from dataset import PRBenchDataset


def example_dataset_usage():
    """Examples of using the PRBench dataset."""
    print("\n" + "=" * 60)
    print("DATASET EXAMPLES")
    print("=" * 60 + "\n")
    
    # Initialize and load dataset
    prbench = PRBenchDataset()
    prbench.load()
    
    # Get statistics
    print("📊 Dataset Statistics:")
    stats = prbench.get_stats("finance")
    print(f"  Total samples: {stats['total_samples']}")
    print(f"  Topics: {stats['num_topics']}")
    print(f"  Experts: {stats['experts']}")
    
    # Get a single sample
    print("\n📋 Single Sample:")
    sample = prbench.get_sample("finance", 0)
    print(f"  Task ID: {sample['task'][:30]}...")
    print(f"  Topic: {sample['topic']}")
    print(f"  Field: {sample['field']}")
    
    # Get multiple samples
    print("\n📚 Multiple Samples:")
    samples = prbench.get_samples("finance", 0, 3)
    print(f"  Retrieved {len(samples)} samples")
    
    # Filter by topic
    print("\n🔍 Filtering:")
    topics = prbench.get_topics("finance")
    first_topic = topics[0]
    filtered = prbench.filter_by_field("finance", "topic", first_topic)
    print(f"  Found {len(filtered)} samples for '{first_topic}'")
    
    # Show available splits
    print("\n📂 Available Splits:")
    for split in prbench.available_splits:
        if split in prbench.dataset:
            print(f"  - {split}: {len(prbench.dataset[split])} samples")


def example_codex_usage():
    """Examples of using Codex CLI."""
    print("\n" + "=" * 60)
    print("CODEX EXAMPLES")
    print("=" * 60 + "\n")
    
    # Simple query
    print("🤖 Simple Query:")
    result = call_codex("What is Python?")
    if result['success']:
        print(f"  Response: {result['output'][:100]}...")
    else:
        print(f"  Error: {result['error']}")
    
    # Code generation
    print("\n💻 Code Generation:")
    result = call_codex("Write a Python function to calculate fibonacci")
    if result['success']:
        print(f"  Generated code:\n{result['output'][:200]}...")
    else:
        print(f"  Error: {result['error']}")


def example_combined_usage():
    """Example combining dataset and Codex."""
    print("\n" + "=" * 60)
    print("COMBINED EXAMPLE")
    print("=" * 60 + "\n")
    
    # Load dataset
    prbench = PRBenchDataset()
    prbench.load()
    
    # Get a sample
    sample = prbench.get_sample("finance", 0)
    
    # Use the first prompt from the sample with Codex
    if sample.get('prompt_0'):
        print("📋 Dataset Prompt:")
        print(f"  {sample['prompt_0'][:150]}...\n")
        
        print("🤖 Asking Codex to summarize...")
        prompt = f"Summarize this financial question in one sentence: {sample['prompt_0'][:500]}"
        result = call_codex(prompt)
        
        if result['success']:
            print(f"  Summary: {result['output']}")
        else:
            print(f"  Error: {result['error']}")


if __name__ == "__main__":
    # Run all examples
    example_dataset_usage()
    example_codex_usage()
    example_combined_usage()
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60 + "\n")

