# Counterfactual Explanations in Sequential Decision Making Under Uncertainty

This repository contains the code used in the paper [*Counterfactual Explanations in Sequential Decision Making Under Uncertainty*](https://arxiv.org/abs/2107.02776), published at NeurIPS 2021.

## Dependencies

All the experiments were performed using Python 3.9. In order to create a virtual environment and install the project dependencies you can run the following commands:

```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## Code organization

The directory `src` contains the source code for the various experiments. The following table contains a short description for each python file:
| `src/`                                                   | Description                                                  |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [synth_mdp.py](src/synth_mdp.py)                           | Defines a **Synth_MDP** object which handles the generation of synthetic decision making realizations and their counterfactual explanations. |
| [synth_compute_cf_mdps.py](src/synth_compute_cf_mdps.py)    | Generates synthetic decision making realizations, pre-computes the counterfactual transition probabilties for each one of them and saves them as `.pkl` files. |
| [synth_experiment.py](src/synth_experiment.py)                     | Generates optimal counterfactual explanations for each synthetic decision making realization, for various values of the parameter `k`. |
| [therapy_mdp.py](src/therapy_mdp.py)                           | Defines a **Therapy_MDP** object which handles the input of the cognitive behavioral therapy data and the generation of counterfactual explanations for each patient. |
| [therapy_compute_cf_mdps.py](src/therapy_compute_cf_mdps.py)    | Pre-computes the counterfactual transition probabilties for all patients' realizations and saves them as `.pkl` files. |
| [therapy_experiment.py](src/therapy_experiment.py)                     | Generates optimal counterfactual explanations for each patient's realization, for various values of the parameter k. |
| [therapy_evaluation.py](src/therapy_experiment.py)                     | Generates counterfactual explanations for each patient's realization, using our method and various baselines. |


The directory [scripts](scripts/) contains bash scripts that use the aforementioned code and pass parameter values required for the various experiments.

The directory [notebooks](notebooks/) contains jupyter notebooks producing the figures appearing in the paper. Each notebook has script execution prerequisites specified therein.

The directory [figures](figures/) is used for saving the figures produced by the notebooks.

The directory [outputs](outputs/) is used for saving the json outputs produced by the scripts. The sub-directory [cf_mdps](outputs/cf_mdps/) is used for saving intermediate `.pkl` files containing the counterfactual transition probabilities for each single realization.

## Citation

If you use parts of the code in this repository for your own research, please consider citing:

    @article{tsirtsis2021counterfactual,
        title={Counterfactual Explanations in Sequential Decision Making Under Uncertainty},
        author={Tsirtsis, Stratis and De, Abir and Gomez-Rodriguez, Manuel},
        journal={arXiv preprint arXiv:2107.02776},
        year={2021}
    }
