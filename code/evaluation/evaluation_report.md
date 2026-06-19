# Operational Analysis Report

## Architecture & Model Choice
This system uses **Gemini 1.5 Flash** via the `google-genai` SDK with `Pydantic` for strict JSON schema enforcement. Gemini 1.5 Flash balances multi-modal spatial reasoning with high-speed execution, which is crucial for identifying specific object parts on cars, laptops, and packages without triggering rate limits.

## Expected Resource Utilization
* **Model Calls:** 1 call per claim (all images passed in a single context window to minimize latency).
* **Input Tokens:** ~1,100 tokens per call (Text prompt + ~2-3 images). Total for 200 rows: ~220,000 tokens.
* **Output Tokens:** ~150 tokens per call. Total: ~30,000 tokens.

## Estimated Cost & Latency
* **Input/Output Cost:** ~$0.025 total for the entire test dataset on Gemini 1.5 Flash.
* **Latency:** ~1.5 - 2 seconds per claim. 
* **Throttle Strategy:** The `tenacity` library wraps the API call with exponential backoff (`@retry`) to gracefully handle HTTP 429 Too Many Requests errors if TPM/RPM limits are hit. Files are immediately deleted from the API cache after processing.
