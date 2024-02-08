# Online Meta-learned Gradient Norms for Active Learning in Science and Technology

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
- Datasets: References list for the data sets used in this research.
- Results: The folder to save the experimental results, and you need to create the subfolders by your way.
- AL_methods:
  - OMAL: The proposed method.
  - Random: The random sampling.
  - Greedy: The Greedy method proposed by Wu, Lin and Huang (2019). We used the random initialisation and batch-mode Greedy-iGS here.
  - QBC: The Query-By-Committee method, proposed by RayChaudhuri and Hamey (1995). We used the batch-mode QBC here.


## LCMD

- The comparison method LCMD method (version 3) can be retrieved from https://github.com/dholzmueller/bmdal_reg with the DOI number: 10.18419/darus-807


## Datasets

The datasets used in this study are all publicly available. The citations of the datasets are listed under the Datasets folder.


## License
Our source code is licensed under the Apache 2.0 license. 
We also include the MIT License of the modAL package since we used the source code of it (multi_argmax function in our method).

## Citations

```bibtex

@article{modAL2018,
    title={mod{AL}: {A} modular active learning framework for {P}ython},
    author={Tivadar Danka and Peter Horvath},
    url={https://github.com/modAL-python/modAL},
    note={available on arXiv at \url{https://arxiv.org/abs/1805.00979}}
}

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


```
