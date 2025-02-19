def main():
    import os
    import statistics
    import time

    import pandas as pd
    import requests

    # API Key
    API_KEY = os.environ.get("BASETEN_API_KEY")
    API_URL = "https://model-xxxx.api.baseten.co/environments/production/predict"
    HEADERS = {"Authorization": f"Api-Key {API_KEY}"}

    # Payloads with different lookahead decoding configurations
    payloads = [
        {
            "stream": False,
            "messages": [
                {
                    "role": "user",
                    "content": "Tell me everything you know about optimized inference.",
                }
            ],
            "max_tokens": 512,
            "temperature": 0.0,
            "lookahead_decoding_config": {
                "window_size": 2,
                "ngram_size": 2,
                "verification_set_size": 1,
            },
        },
        {
            "stream": False,
            "messages": [
                {
                    "role": "user",
                    "content": "Tell me everything you know about optimized inference.",
                }
            ],
            "max_tokens": 512,
            "temperature": 0.0,
            "lookahead_decoding_config": {
                "window_size": 3,
                "ngram_size": 3,
                "verification_set_size": 1,
            },
        },
        {
            "stream": False,
            "messages": [
                {
                    "role": "user",
                    "content": "Tell me everything you know about optimized inference.",
                }
            ],
            "max_tokens": 512,
            "temperature": 0.0,
            "lookahead_decoding_config": {
                "window_size": 5,
                "ngram_size": 5,
                "verification_set_size": 3,
            },
        },
    ]

    # Number of iterations per test
    num_trials = 5

    # Store results
    results = []

    for i, payload in enumerate(payloads):
        times = []
        for _ in range(num_trials):
            start_time = time.time()
            resp = requests.post(API_URL, headers=HEADERS, json=payload)
            resp.raise_for_status()
            end_time = time.time()

            elapsed_time = end_time - start_time
            times.append(elapsed_time)
        print(f"Config {i+1}: {times}")
        results.append(
            {
                "Test Case": f"Config {i+1} look-ahead {payload['lookahead_decoding_config']}",
                "Min Time (s)": min(times),
                "Max Time (s)": max(times),
                "Avg Time (s)": statistics.mean(times),
                "Std Dev (s)": statistics.stdev(times),
            }
        )

    df_results = pd.DataFrame(results)
    print(df_results)


if __name__ == "__main__":
    main()
