import os
import time
import asyncio
import aiohttp
import numpy as np
from typing import List, Dict, Any

from utils import save_results


def batch_generator(data: List[Dict[str, Any]], batch_size: int):
    """
    Generates batches of data.

    Args:
        data (List[Dict[str, Any]]): A list of dictionaries representing the data. Each dictionary should have the following keys:
            - "unique_id" (str): A unique identifier for the data.
            - "text" (str): The text data.
        batch_size (int): The size of each batch.

    Yields:
        List[Dict[str, Any]]: A batch of data.

    """
    for i in range(0, len(data), batch_size):
        yield data[i : i + batch_size]


async def fetch_embeddings(
    batch: List[Dict[str, Any]],
    semaphore: asyncio.Semaphore,
    session: aiohttp.ClientSession,
) -> List[Dict[str, Any]]:
    """
    Fetches embeddings for a batch of texts using an API.

    Args:
        batch (List[Dict[str, Any]]): A list of dictionaries representing the batch of texts. Each dictionary should have the following keys:
            - "unique_id" (str): A unique identifier for the text.
            - "text" (str): The text to be embedded.
        semaphore (asyncio.Semaphore): A semaphore to limit the number of concurrent requests.
        session (aiohttp.ClientSession): An HTTP client session for making requests.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries representing the embeddings and timing information for each text in the batch. Each dictionary has the following keys:
            - "unique_id" (str): The unique identifier of the text.
            - "embedding" (str): The embedding of the text. If an error occurred, this will be None.
            - "error" (bool or dict): If an error occurred, this will be True. Otherwise, it will be False. If an error occurred, this will be a dictionary containing the error details.
            - "timing_info" (dict): A dictionary containing timing information for the request. If an error occurred, this will be None.
    """
    async with semaphore:
        try:
            # Prepare request payload based on API requirements
            host_url = os.getenv("HOST")
            headers = {
                "Accept": "application/json",
                "Authorization": "Bearer " + os.getenv("HF_API_KEY"),
                "Content-Type": "application/json",
            }
            payload = {"inputs": [item["text"] for item in batch], "truncate": True}

            async with session.post(
                host_url, headers=headers, json=payload
            ) as response:
                if response.status == 200:
                    embeddings = await response.json()

                    # Extract timing information from headers
                    timing_info = {
                        "total_time": float(response.headers.get("X-Total-Time", 0)),
                        "tokenization_time": float(
                            response.headers.get("X-Tokenization-Time", 0)
                        ),
                        "queue_time": float(response.headers.get("X-Queue-Time", 0)),
                        "inference_time": float(
                            response.headers.get("X-Inference-Time", 0)
                        ),
                    }
                    return [
                        {
                            "unique_id": item["unique_id"],
                            "embedding": emb[:10],
                            "error": False,
                            "timing_info": timing_info,
                        }
                        for item, emb in zip(batch, embeddings)
                    ]
                else:
                    error_details = await response.json()
                    return [
                        {
                            "unique_id": item["unique_id"],
                            "embedding": None,
                            "error": error_details,
                            "timing_info": None,
                        }
                        for item in batch
                    ]
        except Exception as e:
            return [
                {
                    "unique_id": item["unique_id"],
                    "embedding": None,
                    "error": str(e),
                    "timing_info": None,
                }
                for item in batch
            ]


def calculate_statistics(metrics: dict) -> dict:
    """
    Calculate statistics for each metric in the given dictionary.

    Args:
        metrics (dict): A dictionary containing metrics as keys and lists of values as values.

    Returns:
        dict: A dictionary containing statistics for each metric, including min, max, mean, median, p90, and p95.
    """
    statistics = {}
    for key, values in metrics.items():
        if values:  # Ensure the list is not empty
            statistics[key] = {
                "min": min(values),
                "max": max(values),
                "mean": np.mean(values),
                "median": np.median(values),
                "p90": np.percentile(values, 90),
                "p95": np.percentile(values, 95),
            }
    return statistics


# Main coroutine that orchestrates the pipeline
async def run_bulk_embed(
    data: List[Dict[str, Any]],
    batch_size: int,
    concurrency: int,
    filename: str = "embeddings.jsonl",
) -> Dict[str, Any]:
    """
    Run bulk embedding inference pipeline using the given data and parameters.

    Args:
        data (List[Dict[str, Any]]): The input data for generating embeddings.
        batch_size (int): The number of samples in each batch.
        concurrency (int): The maximum number of concurrent requests.
        filename (str, optional): The name of the file to save the results. Defaults to "embeddings.jsonl".

    Returns:
        Dict[str, Any]: A dictionary containing the benchmarking results, including timing statistics,
        total time, embeddings per second, request metrics, and run metadata.
    """

    semaphore = asyncio.Semaphore(concurrency)
    batches = batch_generator(data, batch_size)

    # Initialize metrics collections
    start_time = time.time()
    timing_metrics = {
        "total_time": [],
        "tokenization_time": [],
        "queue_time": [],
        "inference_time": [],
    }
    request_metrics = {"success": 0, "failure": 0, "total": 0}

    async with aiohttp.ClientSession() as session:
        tasks = [fetch_embeddings(batch, semaphore, session) for batch in batches]

        # .as_completed() rather than .gather() to increase throughput
        for batch_future in asyncio.as_completed(tasks):
            results = await batch_future  # returns List[Dict] of length batch_size

            if filename:
                save_results(results, filename)

            # Aggregate timing metrics for successful requests
            for result in results:
                request_metrics["total"] += 1
                if not result.get("error"):
                    request_metrics["success"] += 1
                    for key, value in result["timing_info"].items():
                        timing_metrics[key].append(value)
                else:
                    request_metrics["failure"] += 1

    end_time = time.time()

    # Collect metrics
    total_time = round(end_time - start_time, 4)
    embeddings_per_second = round(request_metrics["success"] / total_time, 4)
    timing_statistics = calculate_statistics(timing_metrics)
    run_metadata = {
        "batch_size": batch_size,
        "concurrency": concurrency,
        "filename": filename,
    }

    print(
        f"Batch Size: {batch_size}, Concurrency Level: {concurrency}, Total Time: {total_time} seconds, Embed per sec: {embeddings_per_second}, Num Success: {request_metrics['success']}, Num Failures: {request_metrics['failure']}"
    )

    return {
        "timing_statistics": timing_statistics,
        "total_time": total_time,
        "embeddings_per_second": embeddings_per_second,
        "request_metrics": request_metrics,
        "run_metadata": run_metadata,
    }


# TO DO:
# - Add logging
# - Add tests
# - Capture RPS
