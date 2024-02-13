import { check } from "k6";
import http from "k6/http";
import { Trend, Counter } from "k6/metrics";
import { textSummary } from "https://jslib.k6.io/k6-summary/0.0.2/index.js";

const host = __ENV.HOST;
const api_key = __ENV.HF_API_KEY;
const vu = __ENV.VU || 1;
const duration = __ENV.DURATION || "10s";
const file_path = __ENV.FILE_PATH || "k6-summary.json";
const batch_size = __ENV.BATCH_SIZE || 10;
const n_tokens = __ENV.N_TOKENS || 512;

const totalTime = new Trend("total_time", true);
const tokenizationTIme = new Trend("tokenization_time", true);
const queueTime = new Trend("queue_time", true);
const inferenceTime = new Trend("inference_time", true);
const embeddingsProcessed = new Counter("embeddings_processed");

let text_chunk = "hello ".repeat(n_tokens);
let inputs = Array.from({ length: batch_size }, () => text_chunk);

export const options = {
  thresholds: {
    http_req_failed: ["rate==0"],
  },
  scenarios: {
    load_test: {
      executor: "constant-vus",
      duration: duration,
      vus: vu,
      gracefulStop: "10s",
    },
  },
};

export default function () {
  const payload = JSON.stringify({
    inputs: inputs,
    truncate: true,
  });

  const headers = {
    Accept: "application/json",
    Authorization: "Bearer " + api_key,
    "Content-Type": "application/json",
  };
  const res = http.post(`${host}/`, payload, {
    headers,
    timeout: "30s",
  });

  check(res, {
    "Post status is 200": (r) => res.status === 200,
  });

  if (res.status === 200) {
    totalTime.add(res.headers["X-Total-Time"]);
    tokenizationTIme.add(res.headers["X-Tokenization-Time"]);
    queueTime.add(res.headers["X-Queue-Time"]);
    inferenceTime.add(res.headers["X-Inference-Time"]);
    embeddingsProcessed.add(batch_size);
  } else {
    console.log(res.error);
  }
}
export function handleSummary(data) {
  return {
    // stdout: textSummary(data, { indent: " ", enableColors: true }),
    [file_path]: JSON.stringify(data), //the default data object
  };
}
