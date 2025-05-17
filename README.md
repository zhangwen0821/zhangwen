# Pre
## `conda env create -n flowmse python = 3.11  `

## `pip install -r requirements.txt ` 
# Train
## `python train.py --base_dir data/ --backbone dcunet --ode flowmatching --n_fft 512  `

## `nohup python train.py --base_dir data/ --backbone dcunet --ode flowmatching --n_fft 512 train_log.txt 2>&1  `
