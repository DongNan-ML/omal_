# Online Meta-Active Learning for Regression

This repository contains code accompanying our project "Online Meta-learned Gradient Norms for Active Learning in Science and Technology". 

## OMAL
<p align="center">
  <img src="myMethod_pipeline.png" width="500" title="myMethod_pipeline" alt="myMethod_pipeline">
</p>

Our method OMAL is described as the above pipeline and can be accessed in: \AL_methods\OMAL 


## Dependencies

The latest tested versions of the dependencies are listed:
- numpy                     1.23.5
- scipy                     1.11.4
- scikit-learn              1.3.0 
- pandas                    2.1.4
- torch                     2.0.0+cu118
- skorch                    0.13.0


## Code Structure

The code is structured as follows:
- Seeds: A set of shared seeds among different datasets for the reproducible results.
- Data_Processing: The processing code for each dataset.
- Datasets: Raw data for each dataset.
- Results: The folder to save the experimental results, and you need to create the subfolders by your way.
- AL_methods:
  - Greedy: The Greedy method proposed by Wu, Lin and Huang (2019). We used the batch-mode Greedy-iGS here.
  - LCMD: The Largest Cluster Maximum Distance Method, proposed by Holzm{\"u}ller, Zaverkin, K{\"a}stner and Steinwart (2023).
  - QBC: The Query-By-Committee method, proposed by RayChaudhuri and Hamey (1995). We used the batch-mode QBC here.
  - Random: Random Sampling.
  - OMAL: The proposed method.

## Datasets

The datasets used in this study are all publicly available. The citations of the datasets are listed under the Datasets folder.


## LCMD

- This repository contains the implementation of the comparison method LCMD method (version 3) from https://github.com/dholzmueller/bmdal_reg with the DOI number: 10.18419/darus-807

- License: (Copied from the original LCMD readme) This source code is licensed under the Apache 2.0 license. However, the implementation of the acs-rf-hyper kernel transformation in `bmdal/features.py` is adapted from the source code at [https://github.com/rpinsler/active-bayesian-coresets](https://github.com/rpinsler/active-bayesian-coresets), which comes with its own (non-commercial) license. Please respect this license when using the acs-rf-hyper transformation directly from `bmdal/features.py` or indirectly through the interface provided at `bmdal/algorithms.py`.


## License
Our source code is licensed under the Apache 2.0 license. 
We also include the MIT License of the modAL package since we used the source code of it.


## Citations

```bibtex
@inproceedings{raychaudhuri1995minimisation,
  title={Minimisation of data collection by active learning},
  author={RayChaudhuri, Tirthankar and Hamey, Leonard GC},
  booktitle={Proceedings of ICNN'95-International Conference on Neural Networks},
  volume={3},
  pages={1338--1341},
  year={1995},
  organization={IEEE}
}

@article{wu2019active,
  title={Active learning for regression using greedy sampling},
  author={Wu, Dongrui and Lin, Chin-Teng and Huang, Jian},
  journal={Information Sciences},
  volume={474},
  pages={90--105},
  year={2019},
  publisher={Elsevier}
}

@article{holzmuller2023framework,
  title={A framework and benchmark for deep batch active learning for regression},
  author={Holzm{\"u}ller, David and Zaverkin, Viktor and K{\"a}stner, Johannes and Steinwart, Ingo},
  journal={Journal of Machine Learning Research},
  volume={24},
  number={164},
  pages={1--81},
  year={2023}
}

@article{modAL2018,
    title={mod{AL}: {A} modular active learning framework for {P}ython},
    author={Tivadar Danka and Peter Horvath},
    url={https://github.com/modAL-python/modAL},
    note={available on arXiv at \url{https://arxiv.org/abs/1805.00979}}
}

```