#!/bin/bash
# ===================== launch sample for reference =======================
# bash scripts/infer_run_inference_geneval.sh \
# configs/sana_config/1024ms/Sana_600M_img1024.yaml \
# output/Sana_600M_img1024/checkpoints/xxxxx.pth

# ================= sampler & data =================
np=8    # number of GPU to use
default_step=20   # 14
default_sample_nums=533
default_sampling_algo="flow_dpm-solver"
default_add_label=''

# parser
config_file=$1
model_paths=$2

for arg in "$@"
do
    case $arg in
        --step=*)
        step="${arg#*=}"
        shift
        ;;
        --sampling_algo=*)
        sampling_algo="${arg#*=}"
        shift
        ;;
        --add_label=*)
        add_label="${arg#*=}"
        shift
        ;;
        --model_path=*)
        model_paths="${arg#*=}"
        shift
        ;;
        --cfg_scale=*)
        cfg_scale="${arg#*=}"
        shift
        ;;
        --exist_time_prefix=*)
        exist_time_prefix="${arg#*=}"
        shift
        ;;
        --if_save_dirname=*)
        if_save_dirname="${arg#*=}"
        shift
        ;;
        *)
        ;;
    esac
done

step=${step:-$default_step}
sampling_algo=${sampling_algo:-$default_sampling_algo}
cfg_scale=${cfg_scale:-4.5}
sample_nums=$default_sample_nums
samples_per_gpu=$((sample_nums / np))
add_label=${add_label:-$default_add_label}
ablation_key=${ablation_key:-''}
ablation_selections=${ablation_selections:-''}

echo "Step: $step"
echo "Sample numbers: $sample_nums"
echo "Sampling Algo: $sampling_algo"
echo "CFG scale: $cfg_scale"
echo "Add label: $add_label"
echo "Exist time prefix: $exist_time_prefix"

cmd_template="DPM_TQDM=True python scripts/inference_geneval.py --config={config_file} --model_path={model_path} \
    --sampling_algo $sampling_algo --step $step --cfg_scale $cfg_scale \
    --gpu_id {gpu_id} --start_index {start_index} --end_index {end_index}"
if [ -n "${add_label}" ]; then
    cmd_template="${cmd_template} --add_label ${add_label}"
fi

if [ -n "${ablation_key}" ]; then
    cmd_template="${cmd_template} --ablation_key ${ablation_key} --ablation_selections "${ablation_selections}""
    echo "ablation_key: $ablation_key"
    echo "ablation_selections: $ablation_selections"
fi

if [ -n "${exist_time_prefix}" ]; then
    cmd_template="${cmd_template} --exist_time_prefix ${exist_time_prefix}"
fi

if [ "$if_save_dirname" = true ]; then
    cmd_template="${cmd_template} --if_save_dirname=true"
fi

echo "==================== inferencing ===================="
if [[ "$model_paths" == *.pth ]]; then
  for gpu_id in $(seq 0 $((np - 1))); do
    start_index=$((gpu_id * samples_per_gpu))
    end_index=$((start_index + samples_per_gpu))
    if [ $gpu_id -eq $((np - 1)) ]; then
      end_index=$sample_nums
    fi

    cmd="${cmd_template//\{config_file\}/$config_file}"
    cmd="${cmd//\{model_path\}/$model_paths}"
    cmd="${cmd//\{gpu_id\}/$gpu_id}"
    cmd="${cmd//\{start_index\}/$start_index}"
    cmd="${cmd//\{end_index\}/$end_index}"

    echo "Running on GPU $gpu_id: samples $start_index to $end_index"
    eval CUDA_VISIBLE_DEVICES=$gpu_id $cmd &
  done
  wait

else

  if [ ! -f "$model_paths" ]; then
    echo "Model paths file not found: $model_paths"
    exit 1
  fi
  echo "" >> "$model_paths"   # add a new line to the file avoid skipping last line dir

  while IFS= read -r model_path; do
    if [ -n "$model_path" ] && ! [[ $model_path == \#* ]]; then
      for gpu_id in $(seq 0 $((np - 1))); do
        start_index=$((gpu_id * samples_per_gpu))
        end_index=$((start_index + samples_per_gpu))
        if [ $gpu_id -eq $((np - 1)) ]; then
          end_index=$sample_nums
        fi

        cmd="${cmd_template//\{config_file\}/$config_file}"
        cmd="${cmd//\{model_path\}/$model_path}"
        cmd="${cmd//\{gpu_id\}/$gpu_id}"
        cmd="${cmd//\{start_index\}/$start_index}"
        cmd="${cmd//\{end_index\}/$end_index}"

        echo "Running on GPU $gpu_id: samples $start_index to $end_index"
        eval CUDA_VISIBLE_DEVICES=$gpu_id $cmd &
      done

      wait
    fi
  done < "$model_paths"
fi

echo infer finally done
