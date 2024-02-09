import os
import json
import time
import asyncio
import aiohttp
import numpy as np


# Generator for batching data
def batch_generator(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i : i + batch_size]


# Coroutine for fetching embeddings
async def fetch_embeddings(batch, semaphore, session):
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


# Save results in JSON lines format
def save_results(results, filename="embeddings.jsonl"):
    with open(filename, "a") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")


def calculate_statistics(metrics):
    statistics = {}
    for key, values in metrics.items():
        if values:  # Ensure the list is not empty
            statistics[key] = {
                "mean": np.mean(values),
                "min": min(values),
                "median": np.median(values),
                "max": max(values),
                "p90": np.percentile(values, 90),
                "p95": np.percentile(values, 95),
            }
    return statistics


# Main coroutine that orchestrates the pipeline
async def main(data, batch_size, concurrency):
    semaphore = asyncio.Semaphore(concurrency)
    batches = batch_generator(data, batch_size)

    start_time = time.time()

    # Initialize metrics collections
    timing_metrics = {
        "total_time": [],
        "tokenization_time": [],
        "queue_time": [],
        "inference_time": [],
    }

    success_metrics = {"success": 0, "failure": 0, "total": 0}

    async with aiohttp.ClientSession() as session:
        tasks = [fetch_embeddings(batch, semaphore, session) for batch in batches]

        # .as_completed() rather than .gather() to increase throughput
        for batch_future in asyncio.as_completed(tasks):
            results = await batch_future
            save_results(results)

            # Aggregate timing metrics for successful requests
            for result in results:
                success_metrics["total"] += 1
                if not result.get("error"):
                    success_metrics["success"] += 1
                    for key, value in result["timing_info"].items():
                        timing_metrics[key].append(value)
                else:
                    success_metrics["failure"] += 1

    end_time = time.time()
    total_time = round(end_time - start_time, 4)

    # Calculate metrics
    requests_per_second = success_metrics["success"] / total_time

    # Calculate and print statistics
    timing_statistics = calculate_statistics(timing_metrics)
    print("Timing Metrics Statistics: \n")
    for metric, stats in timing_statistics.items():
        print(f"\n{metric}:")
        for stat_name, stat_value in stats.items():
            print(f"  {stat_name}: {round(stat_value*0.001, 4)} seconds")

    print(
        f"Total pipeline execution time (time to embed all data): {total_time} seconds"
    )
    print(f"Total number of chunks embedded: {success_metrics['total']}")
    print(f"Total number of chunks that hit an error: {success_metrics['failure']}")
    print(f"Requests per second (completed requests): {requests_per_second}")
    # print(f"Embeddings per second (completed requests): {embeddings_per_second}")


# TO DO:
# - add metric for embeddings per second
# - Add logging
# - Add error handling
# - Add command-line argument parsing for batch size and concurrency level
# - Add tests


# if __name__ == "__main__":
#     # Example dataset
#     data = [
#         {"unique_id": "1", "text": "Sample text 1"},
#         {"unique_id": "2", "text": "Sample text 2"},
#         # Add more dictionaries as needed
#     ]
#     batch_size = 10
#     concurrency = 5

#     asyncio.run(main(data, batch_size, concurrency))
