# Speech Enhancement with Flow Matching
## Installation
 ```bash
 conda env create -n flowmse python = 3.11  
```
```bash
 pip install -r requirements.txt 
 ```
## Training
```bash
python train.py --base_dir data/ --backbone dcunet --ode flowmatching --n_fft 512
```
## Evaluation