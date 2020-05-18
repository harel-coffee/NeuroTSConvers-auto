# Generate time series from input data
python demo/generate_time_series.py -rg 1 2 3 4 -pmp .  -in demo -ofp ../../OpenFace

# Compute the predictions
python demo/predict.py -rg 1 2 3 4 -pmp . -in demo -t r

# Time series animation using the obtained predictions
python demo/animation.py -in demo

# Visualize the predictions in the brain
python demo/visualization.py -in demo
