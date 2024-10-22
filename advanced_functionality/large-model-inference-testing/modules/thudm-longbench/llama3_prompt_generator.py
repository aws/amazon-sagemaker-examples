import datasets
import random

class PromptGenerator:

    def __init__(self) -> None:
        self.datasets = ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique", \
            "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht", \
            "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]
        self.dataset = datasets.load_dataset('THUDM/LongBench', random.choice(self.datasets), split='test')
    
    def __call__(self) -> str:
        for example in self.dataset:
            input = example["input"]
            context = example["context"]
            context = " ".join(context.split()[:512])
            prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nYou are an assistant for question-answering tasks. Use the following pieces of retrieved context in the section demarcated by "```" to answer the question. The context may contain multiple question answer pairs as an example. Only answer the final question provided in the question section below. If you dont know the answer just say that you dont know.\n\n```{context}```\n\nQuestion: {input}\n\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
            yield prompt
            
            
    