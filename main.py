import logging
from llama_index.core.evaluation import FaithfulnessEvaluator
from openai import OpenAI as OpenAIClient
import fitz
from llama_index.core import (
    VectorStoreIndex,
    Document,
    Settings,
    StorageContext,
)
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from environs import Env

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

env = Env()
env.read_env()

client = OpenAIClient(api_key=env.str("OPENAI_API_KEY"))
model_name = env.str("OPENAI_MODEL", "default-model-name")

Settings.llm = OpenAI(model=model_name, temperature=0.7)
Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small", embed_batch_size=100
)


qdrant_client = QdrantClient(path="./qdrant_data")


vector_store = QdrantVectorStore(
    collection_name="daribar_info",
    client=qdrant_client,
    enable_hybrid=True,
    batch_size=20,
)

storage_context = StorageContext.from_defaults(
    vector_store=vector_store
)

Settings.chunk_size = 512


def index_documents(pdf_path):
    """
    Opens PDF file from the path, indexes it using
    a hybrid search enabled vector store.
    """

    doc = fitz.open(pdf_path)
    documents = []

    for page in doc:
        text = page.get_text()
        documents.append(Document(text=text))
    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context
    )

    logger.info(
        "Documents have been indexed with hybrid search enabled."
    )

    return index


def search_with_llama(index, query):
    """
    Search the indexed information using hybrid search mode.
    Returns results (list or None).
    """

    keywords = (
        query.split()
    )

    query_engine = index.as_query_engine(
        similarity_top_k=2,
        sparse_top_k=5,
        vector_store_query_mode="hybrid",
    )
    query_modified = (f"Запрос: '{query}'. Выдели ключевые слово из Запроса, "
                      f"и если по этим словам нет информации, то напиши только "
                      f"'No results'")
    print(query_modified)
    response = query_engine.query(query_modified)
    print(response.response)

    if 'No results' in response.response:
        logger.info("No results'")
        return None

    if not response.response:
        logger.info("No information found in the indexed documents.")
        return None

    found = any(
        keyword.lower() in response.response.lower()
        for keyword in keywords
    )

    if not found:
        return None

    evaluator = FaithfulnessEvaluator()
    eval_result = evaluator.evaluate_response(response=response)
    logger.info(
        f"Search results: {eval_result.response}, Score: {eval_result.score}"
    )

    if not eval_result.passing:
        logger.info(
            "The response is not faithful to the contexts "
            "or information found."
        )
        return None

    return eval_result.response


def generate_response(query, local_results):
    """
    Generates a response to the user's query based on the info
    in the PDF file or general knowledge.
    """

    if local_results:
        prompt = (f"На основании предоставленной информации: "
                  f"'{local_results}' ответьте на вопрос пользователя: "
                  f"'{query}'.")
    else:
        prompt = (f"Ответьте на вопрос пользователя: '{query}', "
                  f"используя общие знания.")

    logger.info(f"Sending prompt to GPT: {prompt}")
    response = client.completions.create(
        model=model_name,
        prompt=prompt,
        max_tokens=150,
        temperature=0.7,
        stop=["."],
    )
    logger.info("Response generated using GPT.")
    return response.choices[0].text.strip()


def main():
    pdf_path = "daribar_info.pdf"
    index = index_documents(pdf_path)
    query = input("Введите ваш вопрос:\n")
    llama_results = search_with_llama(index, query)
    response = generate_response(query, llama_results)
    print("Response:", response)


if __name__ == "__main__":
    main()
