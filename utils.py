import logging
from typing import List

import boto3
import botocore
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.vectorstores import VectorStoreRetriever
from langchain.chains import RetrievalQA
from langchain.retrievers.bedrock import AmazonKnowledgeBasesRetriever
from llama_index.core.evaluation import (
    AnswerRelevancyEvaluator,
    CorrectnessEvaluator,
    EvaluationResult,
    FaithfulnessEvaluator,
)
from llama_index.llms.bedrock import Bedrock
from ragas.integrations.langchain import EvaluatorChain
from datasets import Dataset
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Before you start, setup a Knowledge base manually and put the ID here
knowledge_base_id = 'NW4S3HIGUY'

#from botocore.exceptions import ClientError

def split_document_from_url(
    website_url: str, chunck_size: int, chunk_overlap: int
) -> list:
    """Fetch document from website and split it into chunks of text"""
    web_loader = WebBaseLoader(website_url)
    documents = web_loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunck_size, chunk_overlap=chunk_overlap
    )
    text_chunks = text_splitter.split_documents(documents)
    logging.info(f"Document split into {len(text_chunks)} chunks")
    return text_chunks


def extract_cotext_string(context: List[str]):   
    """Extract context string from the Document object"""
    return list(map(lambda x: x.page_content, context))


def build_dataset(
    eval_questions: List[str], ground_truth: List[str], predictions: List[str], text_content: List[str]
):
    """Build higging face dataset from the evaluation questions, ground truth and predictions"""
    data_samples = {}
    #print(predictions)
    data_samples["question"] = eval_questions
    data_samples["ground_truth"] = ground_truth
    data_samples["answer"] = list(map(lambda x: x["answer"], predictions))
    data_samples["contexts"] = [
        extract_cotext_string(text_content), extract_cotext_string(text_content), extract_cotext_string(text_content), extract_cotext_string(text_content)
    ]

    dataset = Dataset.from_dict(data_samples)
    return dataset


def get_bedrock_retriever(
    text_chunks: List[str],
    region_name: str 
) -> AmazonKnowledgeBasesRetriever:
    """Create retriever using Amazon Bedrock Knowledge Base"""
    
    # Initialize Bedrock clients
    bedrock_agent_runtime = boto3.client('bedrock-agent-runtime', region_name=region_name)
    wisdom = boto3.client('wisdom', region_name=region_name)
    
    logging.info(f"Knowledge Base with ID: {knowledge_base_id}")

    
    logging.info("Content ingested into Knowledge Base")
    retriever = AmazonKnowledgeBasesRetriever(
        knowledge_base_id=knowledge_base_id,
        retrieval_config={"vectorSearchConfiguration": {"numberOfResults": 5}},
    )
    logging.info("Retriever created")
    
    return retriever

def evaluate_llama_index_metric(
    evaluator: EvaluatorChain, dataset: Dataset
) -> pd.DataFrame:
    """Evaluate the dataset using the evaluator chain"""
    eval_results = []

    for i in range(len(dataset)):
        if isinstance(evaluator, FaithfulnessEvaluator | AnswerRelevancyEvaluator):
            eval_result = evaluator.evaluate(
                query=dataset[i]["question"],
                response=dataset[i]["answer"],
                contexts=dataset[i]["contexts"],
            )
        elif isinstance(evaluator, CorrectnessEvaluator):
            eval_result = evaluator.evaluate(
                query=dataset[i]["question"],
                response=dataset[i]["answer"],
                reference=dataset[i]["ground_truth"],
            )
        eval_results.append(eval_result)

    df_results = display_dataframe(eval_results, dataset)
    return df_results


def display_dataframe(
    eval_results: List[EvaluationResult], dataset: Dataset
) -> pd.DataFrame:
    """Display the evaluation results in a pandas dataframe"""
    eval_df = pd.DataFrame(
        {
            "Question": dataset["question"],
            "Answer": dataset["answer"],
            "Score": [result.score for result in eval_results],
            "Feedback": [result.feedback for result in eval_results],
            "Context": dataset["contexts"],
            "Ground Truth": dataset["ground_truth"],
        }
    )
    return eval_df
