"""PRBench dataset interface module."""

from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import load_dataset, load_from_disk


class PRBenchDataset:
    """
    Interface to access the ScaleAI PRBench dataset from HuggingFace.
    
    PRBench contains expert-authored conversations across Finance and Legal domains
    with expert-curated rubric criteria for evaluating professional reasoning.
    """
    
    def __init__(self, cache_dir: str = "./data/prbench_cache"):
        self.dataset_name = "ScaleAI/PRBench"
        self.dataset = None
        self.available_splits = ["finance", "legal", "finance_hard", "legal_hard"]
        self.cache_dir = Path(cache_dir)
    
    def load(self, split: Optional[str] = None, use_cache: bool = True) -> Any:
        """
        Load the PRBench dataset from cache or download it.
        
        Args:
            split: Optional split name ('finance', 'legal', 'finance_hard', 'legal_hard')
                  If None, loads all splits.
            use_cache: If True, try to load from local cache first
        
        Returns:
            The loaded dataset
        """
        # Check if cache exists
        if use_cache and self.cache_dir.exists():
            print(f"Loading PRBench dataset from cache ({self.cache_dir})...")
            try:
                self.dataset = load_from_disk(str(self.cache_dir))
                print("✓ Dataset loaded from cache")
                return self.dataset
            except Exception as e:
                print(f"⚠ Cache load failed: {e}")
                print("Downloading fresh dataset...")
        else:
            print("Loading PRBench dataset from HuggingFace...")
        
        # Download from HuggingFace
        self.dataset = load_dataset(self.dataset_name)
        
        print("✓ Dataset loaded successfully")
        
        # Auto-save to cache
        if use_cache:
            self.save()
        
        return self.dataset
    
    def save(self, path: Optional[str] = None) -> None:
        """
        Save the dataset to disk for faster loading next time.
        
        Args:
            path: Optional custom path to save to. If None, uses self.cache_dir
        """
        if self.dataset is None:
            raise ValueError("No dataset loaded. Call load() first.")
        
        save_path = Path(path) if path else self.cache_dir
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"Saving dataset to {save_path}...")
        self.dataset.save_to_disk(str(save_path))
        print("✓ Dataset saved successfully")
    
    def clear_cache(self) -> None:
        """
        Delete the cached dataset from disk.
        """
        if self.cache_dir.exists():
            import shutil
            shutil.rmtree(self.cache_dir)
            print(f"✓ Cache cleared: {self.cache_dir}")
        else:
            print("No cache to clear")
    
    def get_sample(self, split: str = "finance", index: int = 0) -> Dict[str, Any]:
        """
        Get a specific sample from the dataset.
        
        Args:
            split: Split name to get sample from
            index: Index of the sample to retrieve
        
        Returns:
            Dictionary containing the sample data
        """
        if self.dataset is None:
            self.load()
        
        return dict(self.dataset[split][index])
    
    def get_samples(self, split: str = "finance", start: int = 0, end: int = 5) -> List[Dict[str, Any]]:
        """
        Get multiple samples from the dataset.
        
        Args:
            split: Split name to get samples from
            start: Starting index
            end: Ending index (exclusive)
        
        Returns:
            List of dictionaries containing sample data
        """
        if self.dataset is None:
            self.load()
        
        return [dict(self.dataset[split][i]) for i in range(start, min(end, len(self.dataset[split])))]
    
    def filter_by_field(self, split: str, field: str, value: Any) -> List[Dict[str, Any]]:
        """
        Filter dataset samples by a specific field value.
        
        Args:
            split: Split name to filter from
            field: Field name to filter on
            value: Value to match
        
        Returns:
            List of matching samples
        """
        if self.dataset is None:
            self.load()
        
        filtered = self.dataset[split].filter(lambda x: x[field] == value)
        return [dict(item) for item in filtered]
    
    def get_topics(self, split: str = "finance") -> List[str]:
        """
        Get unique topics from a split.
        
        Args:
            split: Split name
        
        Returns:
            List of unique topics
        """
        if self.dataset is None:
            self.load()
        
        return list(set(self.dataset[split]["topic"]))
    
    def get_stats(self, split: str = "finance") -> Dict[str, Any]:
        """
        Get statistics about a dataset split.
        
        Args:
            split: Split name
        
        Returns:
            Dictionary with statistics
        """
        if self.dataset is None:
            self.load()
        
        data = self.dataset[split]
        
        return {
            "total_samples": len(data),
            "fields": list(data.features.keys()),
            "topics": list(set(data["topic"])),
            "num_topics": len(set(data["topic"])),
            "experts": list(set(data["expert"])),
        }

