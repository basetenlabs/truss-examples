#/bin/bash
set -e

mkdir -p data/data_public
huggingface-cli download  Efficient-Large-Model/sana_data_public --repo-type dataset --local-dir ./data/data_public --local-dir-use-symlinks False

bash train_scripts/train.sh configs/sana_config/512ms/ci_Sana_600M_img512.yaml --data.load_vae_feat=true

bash train_scripts/train.sh configs/sana_config/512ms/ci_Sana_600M_img512.yaml --data.data_dir="[asset/example_data]" --data.type=SanaImgDataset --model.multi_scale=false
