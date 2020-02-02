# Inf-VAE

This repository contains code for the paper *Inf-VAE: A Variational Autoencoder Framework to Integrate
Homophily and Influence in Diffusion Prediction* ([https://arxiv.org/pdf/2001.00132.pdf](https://arxiv.org/pdf/2001.00132.pdf)).

## Training
### Dependencies
This project is based on `python>=3.6`. The dependent package for this project is listed as follows:

```
tensorflow==1.15.0
numpy==1.18.1
scipy==1.4.1
networkx==2.4
```

### Command
To train and evaluate the model (e.g. on `android`) dataset, please run
```bash
python train.py --dataset android --cuda_device 0
```

Please note that the model is not deterministic. All the experiment results provided in the paper are averages across multiple runs.


## Citation
Please cite the following paper if you are using our code. Thanks!
* Aravind Sankar, Xinyang Zhang, Adit Krishnan and Jiawei Han, "A Deep Generative Approach to Integrate Social Homophily and Temporal Influence in Diffusion Prediction", in Proc. 2020 ACM Int. Conf. on Web Search and Data Mining (WSDM'20), Houston, TX, Feb. 2020
```
@inproceedings{infvae,
  title = {Inf-VAE: A Variational Autoencoder Framework to Integrate
Homophily and Influence in Diffusion Prediction},
  author = {Sankar, Aravind and Zhang, Xinyang and Krishnan, Adit and Han, Jiawei},
  booktitle = {WSDM},
  year = 2020,
}
```
