import os
import json
import asyncio
import aiohttp


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
                    return [
                        {
                            "unique_id": item["unique_id"],
                            "embedding": emb[:10],
                            "error": False,
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
                        }
                        for item in batch
                    ]
        except Exception as e:
            return [{"unique_id": item["unique_id"], "error": str(e)} for item in batch]


# Save results in JSON lines format
def save_results(results, filename="embeddings.jsonl"):
    with open(filename, "a") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")


# Main coroutine that orchestrates the pipeline
async def main(data, batch_size, concurrency):
    semaphore = asyncio.Semaphore(concurrency)
    batches = batch_generator(data, batch_size)

    async with aiohttp.ClientSession() as session:
        tasks = [fetch_embeddings(batch, semaphore, session) for batch in batches]

        # .as_completed() rather than .gather() to increase throughput
        for batch_future in asyncio.as_completed(tasks):
            results = await batch_future
            save_results(results)


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
