import os
import re
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List


def save_results(results: List[dict], filename: str) -> None:
    """
    Save the results to a file.

    Args:
        results (List[dict]): The list of results to be saved.
        filename (str): The name of the file to save the results to.

    Returns:
        None
    """
    with open(filename, "a") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")


def load_k6_results(save_dir: str):
    """
    Load k6 benchmark results from the specified directory.

    Args:
        save_dir (str): The directory path where the k6 benchmark result files are saved.

    Returns:
        list: A list of dictionaries containing the extracted metric data from the k6 benchmark result files.
              Each dictionary represents the metric data for a specific combination of virtual users (vu) and batch size (bs).
              The dictionary contains the following keys:
              - "vu": The number of virtual users.
              - "bs": The batch size.
              - "queue_time": The queue time metric.
              - "checks": The checks metric.
              - "embeddings_processed": The embeddings processed metric.
              - "total_time": The total time metric.
              - "iteration_duration": The iteration duration metric.
              - "inference_time": The inference time metric.
              - "iterations": The number of iterations metric.
              - "tokenization_time": The tokenization time metric.
    """

    k6_results = []
    for file in os.listdir(save_dir):
        if file.endswith(".json"):
            file_path = os.path.join(save_dir, file)

            # parse vu and bs settings
            pattern = r"k6-benchmark-result-vu-(\d+)-bs-(\d+).json"
            match = re.match(pattern, file)
            vu, bs = match.groups()

            with open(file_path, "r") as f:
                data = json.load(f)

                # extract relevant data
                metric_data = {}
                metric_data["vu"] = int(vu)
                metric_data["bs"] = int(bs)

                metrics_keys = [
                    "queue_time",
                    "checks",
                    "embeddings_processed",
                    "total_time",
                    "iteration_duration",
                    "inference_time",
                    "iterations",
                    "tokenization_time",
                ]
                for key in metrics_keys:
                    if "error" in data.keys():
                        metric_data[key] = None
                    else:
                        metric_data[key] = data["metrics"][key]

                k6_results.append(metric_data)

    return k6_results


def plot_heatmap(file_path: str):
    """
    Plots heatmaps for throughput and average request latency based on the data in the given JSONL file.

    Args:
        file_path (str): The path to the JSONL file containing the data.

    Returns:
        None
    """

    # Load the JSONL file into a pandas DataFrame
    data = pd.read_json(file_path, lines=True)

    aggregated_data = {
        "vu": [],
        "bs": [],
        "avg_request_latency": [],
        "embeddings_per_second": [],
        "total_embeddings_processed": [],
        "failed_requests": [],
    }

    # Loop through each record to extract and aggregate the data
    for index, row in data.iterrows():
        aggregated_data["vu"].append(row["vu"])
        aggregated_data["bs"].append(row["bs"])
        aggregated_data["avg_request_latency"].append(
            row["iteration_duration"]["values"].get("avg", 0)
            if row["iteration_duration"]
            else None
        )
        aggregated_data["embeddings_per_second"].append(
            row["embeddings_processed"]["values"].get("rate", 0)
            if row["embeddings_processed"]
            else None
        )
        aggregated_data["total_embeddings_processed"].append(
            row["embeddings_processed"]["values"].get("count", 0)
            if row["embeddings_processed"]
            else None
        )
        aggregated_data["failed_requests"].append(
            row["checks"]["values"].get("fails", 0) if row["checks"] else None
        )

    # Convert the aggregated data into a DataFrame
    aggregated_df = pd.DataFrame(aggregated_data)

    # Generate Pivot Tables
    pivot_throughput = aggregated_df.pivot(
        index="vu", columns="bs", values="embeddings_per_second"
    )
    pivot_latency = aggregated_df.pivot(
        index="vu", columns="bs", values="avg_request_latency"
    )

    # Create masks for NaN values
    mask_throughput = pivot_throughput.isnull()
    mask_latency = pivot_latency.isnull()

    # Throughput Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        pivot_throughput, annot=True, fmt=".1f", cmap="coolwarm", mask=mask_throughput
    )
    plt.title("Throughput (embeddings per second) vs. Batch Size & Concurrency Level")
    plt.xlabel("Batch Size (bs)")
    plt.ylabel("Concurrency Level (vu)")
    plt.show()

    # Latency Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        pivot_latency, annot=True, fmt=".2f", cmap="coolwarm", mask=mask_latency
    )
    plt.title("Average Request Latency (ms) vs. Batch Size & Concurrency Level")
    plt.xlabel("Batch Size (bs)")
    plt.ylabel("Concurrency Level (vu)")
    plt.show()
