# APAS  <img width="100" alt="portfolio_view" align="right" src="https://upload.wikimedia.org/wikipedia/commons/0/06/Imperial_College_London_new_logo.png"> <img width="100" alt="portfolio_view" align="right" src="https://isla-lab.github.io/images/slider/slider-image.svg"> 
Code for the ECAI 2024 paper: "Rigorous Probabilistic Guarantees for Robust Counterfactual Explanations"

> We study the problem of assessing the robustness of counterfactual explanations for deep learning models. We focus on *plausible model shifts* altering model parameters and propose a novel framework to reason about the robustness property in this setting. To motivate our solution, we begin by showing for the first time that computing the robustness of counterfactuals with respect to plausible model shifts is NP-complete. As this (practically) rules out the existence of scalable algorithms for exactly computing robustness, we propose a novel probabilistic approach which is able to provide tight estimates of robustness with strong guarantees while preserving scalability. Remarkably, and differently from existing solutions targeting plausible model shifts, our approach does not impose requirements on the network to be analyzed, thus enabling robustness analysis on a wider range of architectures. Experiments on four binary classification datasets indicate that our method improves the state of the art in generating robust explanations, outperforming existing methods on a range of metrics.

## Getting Started

**Installation**: Clone the repository and follow the setup instructions provided here below:
   > Part of the code used to collect the results is strongly based on the one of Jiang et al. ("Formalising the Robustness of Counterfactual Explanations for Neural Networks", AAAI 2023) available [here](https://github.com/junqi-jiang/robust-ce-inn).

   
APΔS is tested on Python 3.10. It can be installed easily into a conda environment. If you don't have anaconda, you can install it from here [miniconda](https://docs.conda.io/en/latest/miniconda.html).

```bash
cd APAS
# Remove the old environment, if necessary.
conda deactivate; conda env remove -n apas
# Create a new conda environment
conda env create -f apas-env.yaml
# Activate the environment
conda activate apas
```

*NB: if you don't want to install APΔS using conda, you can check the YAML file and manually install all the dependencies.*

## Repo's organization

- APAS/exp_sec5: reports the scripts to collect the results on the different probabilistic guarantees for existing notions of
model shifts, where we show that the Naturally-Occurring Model Shifts (NOMS) and Plausible Model Shifts (PMS) may capture very different
model changes in general.

- APAS/exp_sec7: reports the script to generate probabilistic robust CFXs using APΔS and also to compute the maximum Δ for a probabilistic Δ-robustness of a given CFX. In detail, inside the folder, you will find the following:
  - /datasets: datasets used for experiments
  - dataset-name.ipynb: experiments for Section 7.2, 7.3
  - correctness_wilks.ipynb: experiments for 7.1
  - /models: pretrained deep neural networks to run the experiments
  - /roar: codes for the baseline method ROAR, Upadhyay et al., "Towards Robust and Reliable Algorithmic Recourse", NeurIPS 2021. Taken from Jiang et al, 2023 and adapted from https://github.com/AI4LIFE-GROUP/ROAR
  - /util_scripts: utility classes for MILP, INN encoding, probabilistic approach, etc.


## Contributors
*  **Luca Marzari** - luca.marzari@univr.it
*  **Francesco Leofante** - f.leofante@imperial.ac.uk

## Reference
If you use this code in your work, please kindly cite our paper:

[Rigorous Probabilistic Guarantees for Robust Counterfactual Explanations]() 
Marzari* L, Leofante* F, Cicalese F and Farinelli A.
```
@incollection{marzari2024Rigorous,
  title={Rigorous Probabilistic Guarantees for Robust Counterfactual Explanations},
  author={Marzari, Luca and Leofante, Francesco and Cicalese, Ferdinando and Farinelli, Alessandro},
  booktitle={ECAI 2024},
  pages={},
  year={2024},
  publisher={IOS Press}
}
```
