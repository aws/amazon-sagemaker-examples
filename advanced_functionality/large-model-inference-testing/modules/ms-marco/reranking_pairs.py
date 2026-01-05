import datasets
import random
import os

class RerankerInputGenerator:
    """
    Generates query-document pairs for testing reranker models using MS MARCO dataset.
    MS MARCO is a large scale information retrieval corpus with real queries and passages.
    """

    def __init__(self, num_candidates_per_query=5, seed=42) -> None:
        """
        Initialize the reranker input generator.
        
        Args:
            num_candidates_per_query: Number of candidate documents per query
            seed: Random seed for reproducible sampling
        """
        self.num_candidates_per_query = num_candidates_per_query
        self.rng = random.Random(seed)
        
        # Load MS MARCO passage ranking dataset
        # This dataset contains queries with relevant and non-relevant passages
        self.dataset = datasets.load_dataset('ms_marco', 'v1.1', split='validation')
        
        # Create a mapping of queries to their passages for efficient sampling
        self._build_query_passage_mapping()

        self.query_is_array = os.getenv("INFERENCE_SERVER", None) == "triton_inference_server" and \
            os.getenv("INFERENCE_ENGINE", None) == "python"
    
    def _build_query_passage_mapping(self):
        """Build mapping from queries to available passages for efficient sampling"""
        self.query_to_passages = {}
        self.all_passages = set()
        
        for example in self.dataset:
            query = example['query']
            passages = example['passages']['passage_text']
            is_selected = example['passages']['is_selected']
            
            # Store both relevant and non-relevant passages
            if query not in self.query_to_passages:
                self.query_to_passages[query] = {'relevant': [], 'non_relevant': []}
            
            for passage, selected in zip(passages, is_selected):
                self.all_passages.add(passage)
                if selected == 1:
                    self.query_to_passages[query]['relevant'].append(passage)
                else:
                    self.query_to_passages[query]['non_relevant'].append(passage)
        
        self.all_passages = list(self.all_passages)
        print(f"Loaded {len(self.query_to_passages)} unique queries with {len(self.all_passages)} total passages")
    
    def __call__(self) -> list:
        """
        Generate query and candidate documents for reranker testing.
        
        Yields:
            tuple: (query_str, list_of_candidate_documents)
        """
        queries = list(self.query_to_passages.keys())
        
        for query in queries:
            passages_data = self.query_to_passages[query]
            candidates = []
            
            # Always include at least one relevant passage if available
            if passages_data['relevant']:
                candidates.extend(self.rng.sample(
                    passages_data['relevant'], 
                    min(1, len(passages_data['relevant']))
                ))
            
            # Fill remaining slots with mix of non-relevant and random passages
            remaining_slots = self.num_candidates_per_query - len(candidates)
            
            if remaining_slots > 0:
                # Add some non-relevant passages from the same query
                available_non_relevant = passages_data['non_relevant']
                if available_non_relevant:
                    non_relevant_count = min(remaining_slots // 2, len(available_non_relevant))
                    candidates.extend(self.rng.sample(available_non_relevant, non_relevant_count))
                    remaining_slots -= non_relevant_count
                
                # Fill remaining with random passages from the corpus
                if remaining_slots > 0:
                    # Exclude already selected candidates to avoid duplicates
                    available_random = [p for p in self.all_passages if p not in candidates]
                    if available_random:
                        random_count = min(remaining_slots, len(available_random))
                        candidates.extend(self.rng.sample(available_random, random_count))
            
            # Shuffle candidates to randomize order (important for reranking evaluation)
            self.rng.shuffle(candidates)
            
            # Ensure we have the expected number of candidates
            if len(candidates) < self.num_candidates_per_query:
                # Pad with random passages if needed
                while len(candidates) < self.num_candidates_per_query and len(candidates) < len(self.all_passages):
                    random_passage = self.rng.choice(self.all_passages)
                    if random_passage not in candidates:
                        candidates.append(random_passage)
            
            documents = candidates[:self.num_candidates_per_query]
            if self.query_is_array:
                query = [query]
                
            yield [ query, documents, [len(documents)] ]

if __name__ == "__main__":
    # Example usage
    generator = RerankerInputGenerator()
    for sample in generator():
        print(sample)
