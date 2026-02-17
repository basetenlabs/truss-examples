# Python sandbox is a project for deploying Python code intepreter on Baseten

Most performant code-sandbox.


## Deployment

Deploy the code sandbox via
```
pip install truss --upgrade
truss push --publish
```

## Usage:

```json
{ "code": "1 + 1", "max_runtime_ms": 10, "max_memory_mb": 5 }
```

```bash
curl -s -X POST http://localhost:3000/execute \
  -H "Content-Type: application/json" \
  -d '{"code": "x = [1, 2, 3]\nlen(x)"}'
```

## Performance:
A single 16-Core GPU can handle around 78000 RPS.

```
==========================================
Heavy Load Test (10k concurrency, 100k requests)
==========================================
Concurrency: 500
Total Requests: 100000
URL: http://localhost:3000/execute


Summary:
  Total:        1.4464 secs
  Slowest:      0.1796 secs
  Fastest:      0.0001 secs
  Average:      0.0068 secs
  Requests/sec: 69137.0176

  Total data:   9065026 bytes
  Size/request: 90 bytes

Response time histogram:
  0.000 [1]     |
  0.018 [84618] |■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
  0.036 [12712] |■■■■■■
  0.054 [1561]  |■
  0.072 [484]   |
  0.090 [287]   |
  0.108 [231]   |
  0.126 [35]    |
  0.144 [37]    |
  0.162 [15]    |
  0.180 [19]    |


Latency distribution:
  10%% in 0.0001 secs
  25%% in 0.0001 secs
  50%% in 0.0004 secs
  75%% in 0.0057 secs
  90%% in 0.0235 secs
  95%% in 0.0307 secs
  99%% in 0.0566 secs

Details (average, fastest, slowest):
  DNS+dialup:   0.0000 secs, 0.0000 secs, 0.0625 secs
  DNS-lookup:   0.0001 secs, 0.0000 secs, 0.1323 secs
  req write:    0.0001 secs, 0.0000 secs, 0.1000 secs
  resp wait:    0.0006 secs, 0.0000 secs, 0.0790 secs
  resp read:    0.0029 secs, 0.0000 secs, 0.1003 secs

Status code distribution:
  [200] 100000 responses
```
