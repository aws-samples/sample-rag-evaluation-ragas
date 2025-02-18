# Language Model Evaluation Toolkit

This project provides a comprehensive toolkit for evaluating the performance of language models, particularly in the context of question-answering tasks using Amazon Bedrock and other AI services.

The toolkit offers a set of utility functions that enable developers to fetch and process text data, create datasets, interact with Amazon Bedrock Knowledge Bases, and perform various evaluations on language model outputs. It leverages popular libraries such as LangChain, LlamaIndex, and Hugging Face Datasets to provide a robust evaluation framework.

## Repository Structure

- `utils.py`: Core utility functions for text processing, dataset creation, and model evaluation.
- `requirements.txt`: List of Python package dependencies required for the project.
- `CODE_OF_CONDUCT.md`: Guidelines for contributor behavior and community standards.
- `CONTRIBUTING.md`: Instructions for contributing to the project.
- `README.md`: This file, providing an overview and usage instructions for the project.

## Usage Instructions

### Installation

1. Ensure you have Python 3.7 or later installed.
2. Clone the repository to your local machine.
3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Getting Started

1. Set up an Amazon Bedrock Knowledge Base and note its ID.
2. Update the `knowledge_base_id` variable in `utils.py` with your Knowledge Base ID.
3. Configure your AWS credentials to allow access to Bedrock and other AWS services.

### Key Functions

#### Splitting Documents

To split a document from a URL into chunks:

```python
from utils import split_document_from_url

chunks = split_document_from_url("https://example.com", chunk_size=1000, chunk_overlap=100)
```

#### Creating a Bedrock Retriever

To create an Amazon Bedrock Knowledge Base retriever:

```python
from utils import get_bedrock_retriever

retriever = get_bedrock_retriever(text_chunks, region_name="us-west-2")
```

#### Building a Dataset

To create a dataset for evaluation:

```python
from utils import build_dataset

dataset = build_dataset(eval_questions, ground_truth, predictions, text_content)
```

#### Evaluating Model Performance

To evaluate the model using various metrics:

```python
from utils import evaluate_llama_index_metric
from llama_index.core.evaluation import FaithfulnessEvaluator

evaluator = FaithfulnessEvaluator()
results = evaluate_llama_index_metric(evaluator, dataset)
```

### Troubleshooting

#### Common Issues

1. AWS Credentials Not Found
   - Problem: `botocore.exceptions.NoCredentialsError`
   - Solution: Ensure AWS credentials are properly configured in `~/.aws/credentials` or as environment variables.

2. Knowledge Base Not Found
   - Problem: `botocore.exceptions.ClientError: An error occurred (ResourceNotFoundException) when calling the RetrieveOperation`
   - Solution: Verify the `knowledge_base_id` in `utils.py` is correct and the Knowledge Base exists in your AWS account.

#### Debugging

To enable verbose logging:

1. Add the following at the beginning of your script:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

2. Look for log files in your current working directory or check the console output for detailed information.

### Performance Optimization

- Monitor the time taken for document splitting and retrieval operations.
- For large documents, consider increasing the chunk size to reduce the number of API calls.
- Use batch processing when evaluating multiple queries to improve throughput.

## Data Flow

The toolkit processes data through the following steps:

1. Document Retrieval: Fetch documents from web URLs.
2. Text Chunking: Split documents into manageable chunks.
3. Knowledge Base Integration: Store chunks in Amazon Bedrock Knowledge Base.
4. Query Processing: Use the Knowledge Base to retrieve relevant information for queries.
5. Answer Generation: Generate answers using the retrieved information.
6. Evaluation: Assess the quality of generated answers using various metrics.

```
[Web Document] -> [Text Chunker] -> [Bedrock Knowledge Base]
                                           |
                                           v
[User Query] -> [Retriever] -> [Answer Generator] -> [Evaluator]
                                           |
                                           v
                                    [Evaluation Results]
```

Note: The actual answer generation step is not explicitly included in the provided code but is assumed to be part of the workflow when using this evaluation toolkit.