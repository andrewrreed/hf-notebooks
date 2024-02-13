# TEI Endpoint Inference & Benchmark Toolkit
This repository contains a set of tools designed for benchmarking the performance of an embedding endpoint. The toolkit enables users to measure various metrics such as response times, throughput, and error rates, providing insights into the efficiency and reliability of embedding services.

## Overview
The toolkit comprises several key components designed to facilitate comprehensive benchmarking:

- `tei-benchmarking.ipynb`: A Jupyter notebook that provides a detailed walkthrough of the benchmarking process. It demonstrates how to use the toolkit to conduct benchmarking sessions, analyze results, and visualize performance metrics. This interactive environment is ideal for exploring different benchmarking scenarios and refining testing strategies.

- `bulk_inference_pipeline.py`: The core script for executing bulk inference operations. It orchestrates the process of sending batches of data to the embedding endpoint, fetching embeddings, and collecting performance metrics. This script leverages asynchronous programming to maximize throughput and efficiency.

- `tei-benchmark.js`: A JavaScript file designed to benchmark the embedding endpoint from a web development perspective. This script can be useful for testing the endpoint in environments where JavaScript is the primary language, such as in web applications or serverless functions.

- `utils.py`: A utility module that provides foundational functionalities such as saving results to files. This module supports other scripts by offering common operations needed during the benchmarking process.

## Getting Started
1. Setup Environment: Ensure Python 3.6+ is installed. For JavaScript benchmarks, Node.js is required.

2. Install Dependencies: Install necessary Python packages and Node.js modules as specified in the requirements.txt (for Python) and package.json (for JavaScript).

3. Configure Endpoint: Set the environment variables HOST and HF_API_KEY to point to the embedding endpoint and authenticate requests.

4. Run Benchmarks:

- For Python benchmarks, execute the bulk_inference_pipeline.py script with appropriate parameters for batch size and concurrency.
- For JavaScript benchmarks, run the tei-benchmark.js script.

5. Analyze Results: Use the tei-benchmarking.ipynb notebook to load, analyze, and visualize the benchmarking results. Adjust parameters and repeat tests as necessary.