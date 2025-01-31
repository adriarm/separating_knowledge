eval "$(conda shell.bash hook)"

input_folder='../data/shaders_kml/train'
output_folder=$1/shaders_kml_mixup/train
N=2

mkdir -p "$output_folder"
python mixup_dataset.py "$input_folder" "$output_folder" $N