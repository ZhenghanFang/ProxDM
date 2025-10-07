set -e

huggingface-cli download ZhenghanFang/prox-diffusion --local-dir assets/hf
mkdir -p assets/pretrained_models
mv assets/hf/*.pth assets/pretrained_models/
mv assets/hf/fid_stats assets/fid_stats
rm -rf assets/hf