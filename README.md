# Sample-Efficient Reinforcement Learning via Conservative Model-Based Actor-Critic

This is the code of paper 
**Sample-Efficient Reinforcement Learning via Conservative Model-Based Actor-Critic**. 
Zhihai Wang, Jie Wang, Qi Zhou, Bin Li, Houqiang Li. AAAI 2022. [[arXiv](https://arxiv.org/abs/2112.10504)]

## Requirements
- Python 3.6.9
- PyTorch 1.10
- tqdm
- gym 0.21
- mujoco 1.50
```
pip install -r requirements.txt
```

## Reproduce the Results
1. For example, run experiments on Ant 
```
python scripts/run.py configs/ant.json
```

## Citation
If you find this code useful, please consider citing the following paper.
```
@article{wang2021sample,
  title={Sample-Efficient Reinforcement Learning via Conservative Model-Based Actor-Critic},
  author={Wang, Zhihai and Wang, Jie and Zhou, Qi and Li, Bin and Li, Houqiang},
  journal={arXiv preprint arXiv:2112.10504},
  year={2021}
}
```
## Remarks
We will release our data reported in our paper soon.

## Other Repositories
If you are interested in our work, you may find the following papers useful.

**Model-Based Reinforcement Learning via Estimated Uncertainty and Conservative Policy Optimization.**
*Qi zhou, Houqiang Li, Jie Wang.* AAAI 2020. [[paper](https://arxiv.org/abs/1911.12574)] [[code](https://github.com/MIRALab-USTC/RL-POMBU)]