from preprocesing import preprocesing

# Classes vectors definition
vector_healthy_grape = [1, 0, 0, 0]
vector_black_rot_grape = [0, 1, 0, 0]
vector_esca_black_measles_grape = [0, 0, 1, 0]
vector_leaf_blight_isariopsis_leaf_spot_grape = [0, 0, 0, 1]

# Input folders definition
input_folder_healthy_grape = "healthy_grape"
input_folder_black_rot_grape = "black-rot_grape"
input_folder_esca_black_measles_grape = "esca-black-measles_grape"
input_folder_leaf_blight_isariopsis_leaf_spot_grape = (
    "leaf-blight-isariopsis-leaf-spot_grape"
)

# Output files definition
output_file_healthy_grape = "dataset_output/healthy_grape.csv"
output_file_black_rot_grape = "dataset_output/black-rot_grape.csv"
output_file_esca_black_measles_grape = "dataset_output/esca-black-measles_grape.csv"
output_file_leaf_blight_isariopsis_leaf_spot_grape = (
    "dataset_output/leaf-blight-isariopsis-leaf-spot_grape.csv"
)

# Resize proportions
x_resize = 0.25
y_resize = 0.25

# Processes
preprocesing(
    vector_healthy_grape,
    input_folder_healthy_grape,
    output_file_healthy_grape,
    x_resize,
    y_resize,
)
preprocesing(
    vector_black_rot_grape,
    input_folder_black_rot_grape,
    output_file_black_rot_grape,
    x_resize,
    y_resize,
)
preprocesing(
    vector_esca_black_measles_grape,
    input_folder_esca_black_measles_grape,
    output_file_esca_black_measles_grape,
    x_resize,
    y_resize,
)
preprocesing(
    vector_leaf_blight_isariopsis_leaf_spot_grape,
    input_folder_leaf_blight_isariopsis_leaf_spot_grape,
    output_file_leaf_blight_isariopsis_leaf_spot_grape,
    x_resize,
    y_resize,
)
