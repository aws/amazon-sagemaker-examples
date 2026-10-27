# Create code embedding training dataset
from datasets import Dataset


def create_code_embedding_dataset():
    """
    Create a comprehensive dataset with code-description pairs for embedding fine-tuning.
    This dataset follows contrastive learning principles for code embeddings.
    """
    code_pairs = [
        {
            "text1": "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
            "text2": "recursive function to calculate fibonacci numbers in Python",
        },
        {
            "text1": "SELECT * FROM users WHERE age > 18 AND status = 'active'",
            "text2": "SQL query to find all active adult users from database",
        },
        {
            "text1": "class Stack: def __init__(self): self.items = []",
            "text2": "stack data structure implementation with initialization method",
        },
        {
            "text1": "for i in range(len(arr)): for j in range(len(arr)-1-i): if arr[j] > arr[j+1]: arr[j], arr[j+1] = arr[j+1], arr[j]",
            "text2": "bubble sort algorithm implementation using nested loops",
        },
        {
            "text1": "import pandas as pd; df = pd.read_csv('data.csv')",
            "text2": "load CSV file into pandas DataFrame for data analysis",
        },
        {
            "text1": "def quicksort(arr): return [] if not arr else quicksort([x for x in arr[1:] if x <= arr[0]]) + [arr[0]] + quicksort([x for x in arr[1:] if x > arr[0]])",
            "text2": "quicksort algorithm implementation using list comprehension",
        },
        {
            "text1": "try: result = func() except Exception as e: print(f'Error: {e}')",
            "text2": "error handling with try-except block and formatted output",
        },
        {"text1": "lambda x: x ** 2", "text2": "lambda function to calculate square of a number"},
        {
            "text1": "def binary_search(arr, target): left, right = 0, len(arr) - 1",
            "text2": "binary search algorithm initialization with left and right pointers",
        },
        {
            "text1": "class LinkedList: def __init__(self): self.head = None",
            "text2": "linked list data structure class definition with head pointer",
        },
    ]

    # Expand dataset with more programming language examples
    additional_pairs = [
        {
            "text1": "function addNumbers(a, b) { return a + b; }",
            "text2": "JavaScript function to add two numbers and return result",
        },
        {
            "text1": "public class Calculator { private int result; }",
            "text2": "Java class definition for calculator with private result field",
        },
        {
            "text1": "def __str__(self): return f'{self.name}: {self.value}'",
            "text2": "Python string representation method for object display",
        },
        {
            "text1": "CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(100))",
            "text2": "SQL table creation statement with primary key and name field",
        },
        {
            "text1": "const result = await fetch('/api/data').then(res => res.json())",
            "text2": "JavaScript async API call with JSON response parsing",
        },
    ]

    # Combine both lists
    code_pairs.extend(additional_pairs)

    # Transform combined list to the format for Dataset.from_dict()
    dataset_dict = {
        "text1": [pair["text1"] for pair in code_pairs],
        "text2": [pair["text2"] for pair in code_pairs],
    }
    return Dataset.from_dict(dataset_dict)
