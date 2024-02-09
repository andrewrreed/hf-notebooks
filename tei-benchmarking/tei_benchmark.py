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
                            "embedding": emb,
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
def save_results(results, filename):
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
async def main(data, batch_size, concurrency, filename="embeddings.jsonl"):
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
    embeddings_per_second = request_metrics["success"] / total_time
    timing_statistics = calculate_statistics(timing_metrics)
    run_metadata = {
        "batch_size": batch_size,
        "concurrency": concurrency,
        "filename": filename,
    }

    # Print metrics
    print("Timing Metrics Statistics: \n")
    for metric, stats in timing_statistics.items():
        print(f"\n{metric}:")
        for stat_name, stat_value in stats.items():
            print(f"  {stat_name}: {round(stat_value*0.001, 4)} seconds")

    print(
        f"\n\nTotal pipeline execution time (time to embed all data): {total_time} seconds"
    )
    print(f"Total number of chunks to embed: {request_metrics['total']}")
    print(f"Total number of chunks embedded: {request_metrics['success']}")
    print(f"Total number of chunks that hit an error: {request_metrics['failure']}")
    print(f"Embeddings per second (completed requests): {embeddings_per_second}")

    return {
        "timing_statistics": timing_statistics,
        "total_time": total_time,
        "embeddings_per_second": embeddings_per_second,
        "request_metrics": request_metrics,
        "run_metadata": run_metadata,
    }


# TO DO:
# - Add logging
# - Add error handling
# - Add command-line argument parsing for batch size and concurrency level
# - Add tests


# if __name__ == "__main__":
#     data = [
#         {"unique_id": n, "text": f"This is a test sentence - {n}"}
#         for n in range(10_000)
#     ]
#     batch_size = 32
#     concurrency = 10
#     # print(os.getcwd())
#     os.remove("embeddings.jsonl")
#     asyncio.run(main(data, batch_size, concurrency))
