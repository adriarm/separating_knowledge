eval "$(conda shell.bash hook)"

shapes_folder='../data/shaders/train'
textures_folder='../data/shaders/train'
output_folder=$1/shaders_kml/train

mkdir -p "$output_folder"
python fast2leaves.py "$shapes_folder" "$textures_folder" "$output_folder"