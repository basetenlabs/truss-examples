
# # Function to kill sglang processes and processes using port 3000
# kill_sglang_processes() {
#   echo "Stopping sglang server..."
  
#   # Install required packages if not already installed
#   apt-get update && apt-get install -y psmisc lsof || true

#   # Kill all processes that have sglang::schedul in their command line
#   pkill -9 -f "sglang::schedul" || true
#   # Also try killall in case the process name contains the pattern
#   killall -9 "*sglang*schedul*" 2>/dev/null || true

#   # Kill all processes using port 3000
#   lsof -ti:3000 | xargs -r kill -9 || true
  
#   echo "All sglang processes stopped"
# }

# # load the weights
find /cache/org/DeepSeek-R1 -type f -print0 | xargs -0 -P 16 -I {} dd if="{}" of=/dev/null bs=4M

# launch server
python3 -m sglang.launch_server --served-model-name deepseek --model-path /cache/org/DeepSeek-R1 --tokenizer-path /cache/org/DeepSeek-R1 --trust-remote-code --context-length 34000 --enable-flashinfer-mla --enable-dp-attention --enable-ep-moe  --enable-nccl-nvls --max-prefill-tokens 65536 --tp 8 --port 3000 &

# benchmark
python3 /app/data/bench_one_batch_server.py --model None --base-url http://localhost:3000 --batch-size 1 2 4 8 16 32 64 --input-len 1000 --output-len 1000 --num-shots 50 --num-questions 1024

# # stop the sglang server
# kill_sglang_processes
# # start another server
# python3 -m sglang.launch_server --served-model-name deepseek --model-path /cache/org/DeepSeek-R1 --tokenizer-path /cache/org/DeepSeek-R1 --trust-remote-code --context-length 34000 --enable-flashinfer-mla --enable-ep-moe  --enable-nccl-nvls --max-prefill-tokens 65536 --tp 8 --port 3000 &
# # benchmark
# python3 /app/data/bench_one_batch_server.py --model None --base-url http://localhost:3000 --batch-size 1 2 4 8 16 32 64 --input-len 1000 --output-len 1000 --num-shots 50 --num-questions 1024

# # stop the sglang server after the second benchmark
# kill_sglang_processes