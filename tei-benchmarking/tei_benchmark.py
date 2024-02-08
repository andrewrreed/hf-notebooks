import os
import time
import asyncio
import httpx
from typing import List, Iterable


def generate_batches(text, bs=1, total_chunks=1000):
    """
    Generate batches of text.

    Args:
        text (str): The input text.
        bs (int, optional): The batch size. Defaults to 1.
        total_chunks (int, optional): The total number of chunks to embed.

    Yields:
        list: A batch of text.

    """
    batches = []
    batch = []
    for i in range(total_chunks):
        batch.append(text)
        if len(batch) == bs:
            batches.append(batch)
            batch = []
    if batch:
        batches.append(batch)

    return batches


async def make_request(client: httpx.AsyncClient, batch: List[str]):
    """
    Makes a request to the specified host URL with the given batch of inputs.

    Args:
        client (httpx.AsyncClient): The HTTP client used to make the request.
        batch (list): The batch of inputs to be sent in the request.

    Returns:
        dict: A dictionary containing the response data and request metadata. The dictionary has the following keys:
            - 'embeddings': The response JSON containing the embeddings.
            - 'request_metadata': A dictionary containing metadata about the request, including total time, tokenization time,
              queue time, and inference time.

    """
    host_url = os.getenv("HOST")
    headers = {
        "Accept": "application/json",
        "Authorization": "Bearer " + os.getenv("HF_API_KEY"),
        "Content-Type": "application/json",
    }
    payload = {"inputs": batch, "truncate": True}
    response = await client.post(host_url, headers=headers, json=payload, timeout=10.0)

    if response.status_code == 200:
        request_metadata = {
            "total_time": response.headers["X-Total-Time"],
            "tokenization_time": response.headers["X-Tokenization-Time"],
            "queue_time": response.headers["X-Queue-Time"],
            "inference_time": response.headers["X-Inference-Time"],
        }
        return {
            "embeddings": response.json(),
            "request_metadata": request_metadata,
            "error": False,
        }
    else:
        print(
            f"Request failed with status code {response.status_code}: {response.text}"
        )
        return {
            "embeddings": None,
            "request_metadata": None,
            "error": response.text,
        }


async def embed(
    batches: Iterable[List[str]],
    concurrency_level: int = 1,
    collect_embeddings: bool = True,
):
    """
    Embeds a list of batches of text using an asynchronous HTTP client.

    Args:
        batches (List[List[str]]): A list of batches, where each batch is a list of strings.
        concurrency_level (int, optional): The number of concurrent requests to make. Defaults to 1.
        collect_embeddings (bool, optional): Whether to collect and return the embedding results. Defaults to True.

    Returns:
        dict: A dictionary containing the embedding results and metadata.

    Example:
        >>> batches = [["text1", "text2"], ["text3", "text4"]]
        >>> result = await embed(batches, concurrency_level=2, collect_results=True)
    """
    async with httpx.AsyncClient() as client:
        # Function to process a chunk of batches concurrently
        async def process_chunk(chunk):
            responses = await asyncio.gather(
                *[make_request(client, batch) for batch in chunk],
                return_exceptions=False,
            )
            return responses

        start_time = time.time()

        # Split batches into chunks based on concurrency level
        # A chunk is a list of batches of length `concurrency_level`
        chunks = [
            batches[i : i + concurrency_level]
            for i in range(0, len(batches), concurrency_level)
        ]

        # Process each batch in a chunk concurrently
        results = {"embeddings": [], "request_metadata": [], "errors": []}
        for chunk in chunks:
            result = await process_chunk(chunk)
            results["request_metadata"].extend(
                [batch["request_metadata"] for batch in result]
            )
            results["errors"].extend([batch["error"] for batch in result])

            if collect_embeddings:
                results["embeddings"].extend([batch["embeddings"] for batch in result])

        # gather overall metrics
        end_time = time.time()
        total_time = round(end_time - start_time, 4)
        num_chunks_embedded = sum([len(sublist) for sublist in batches])
        req_per_sec = round(len(chunks) / total_time, 4)
        embed_per_sec = round(num_chunks_embedded / total_time, 4)

        print(
            f"Batch Size: {len(batches[0])}, Concurrency Level: {concurrency_level}, Total Time: {total_time:.2f} seconds, RPS: {req_per_sec:.2f}, Embed per sec: {embed_per_sec:.2f}"
        )

        return {
            "embeddings": results["embeddings"] if collect_embeddings else None,
            "request_metadata": (results["request_metadata"]),
            "metrics": {
                "batch_size": len(batches[0]),
                "concurrency_level": concurrency_level,
                "total_time": total_time,
                "num_chunks_embedded": num_chunks_embedded,
                "req_per_sec": req_per_sec,
                "embed_per_sec": embed_per_sec,
            },
            "errors": results["errors"],
        }
