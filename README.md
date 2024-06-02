# DeCoOp: Robust Prompt Tuning with Out-of-Distribution Detection

<p align="center">
🏠 <a href="https://wnjxyk.github.io/DeCoOp" target="_blank">Homepage</a> • 📃 <a href="https://arxiv.org/abs/" target="_blank">Paper</a><br>
</p>

## OPT Problem Setting

![OPT](https://zhouz.dev/DeCoOp/static/images/setting-wide.png)

In this paper, we examine a problem setting known as 
***O**pen-world **P**rompt **T**uning* 
(OPT), which focuses on tuning prompts for base classes and evaluating their performance on a combination of base and new classes. This setting allows for a comprehensive evaluation of the discriminabilities among the base-class, base-to-new, and new-class categories. We demonstrate that the accuracy in OPT does not align consistently with the previous H metric, as shown in the figure in our paper, indicating the need for a new evaluation metric for OPT.

## DeCoOp Approach

![DeCoOp](https://zhouz.dev/DeCoOp/static/images/method.png)

We propose the 
***De**composed **Co**ntext **Op**timization* 
(DeCoOp) approach to solve the OPT problem setting. DeCoOp integrates out-of-distribution (OOD) detection into prompt tuning, introducing new-class detectors to enhance the discriminability between the base and new classes. Additionally, DeCoOp employs sub-classifiers to further enhance the discriminability within the base class, thereby improving the performance of the base-class data. The original prompts are retained to ensure the robust performance of the new-class data. To address the issue of not having knowledge of the new-class data during training, we introduce an ensemble strategy to train the DeCoOp approach. The experimental results demonstrate that our DeCoOp approach surpasses state-of-the-art methods, effectively solving the OPT problem setting.

## Quick Start

### 1. Prepare datasets

Prepare the datasets according to the instructions in [DATASETS](https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md) and put them in the `DATA` directory.

### 2. Prepare Python Environment

Clone DeCoOp repository, create conda environment, and then install the required packages.

```bash
git clone https://github.com/WNJXYK/DeCoOp.git
cd DeCoOp
conda create -n decoop python==3.8
conda activate decoop
pip install -r requirements.txt
```

### 3. Run DeCoOp Approach

Run the DeCoOp approach using the command script `bash eval_decoop.sh {GpuID} {Architecture} {Logdir}`, where `GpuID` is an integer indicating the GPU you want to use, `Architecture` is the backbone model of CLIP with options Vit-B16 or Vit-B32, and `Logdir` is the path to save the experimental results.

For example, to run DeCoOp using GPU 0 and the Vit-B16 backbone model, with the corresponding experimental logs saved to `./results`, use the following command:

```bash
bash eval_decoop.sh 0 Vit-B16 ./results
```

## TODO List

- [x] Launch project homepage
- [x] Release official code
- [ ] Release code based on Dassl.pytorch toolbox

## Citation

Please cite the paper if you refer to our code or paper from DeCoOp.

```plain
@inproceedings{zhou24decoop,
    author       = {Zhi Zhou and Ming Yang and Jiang-Xin Shi and Lan-Zhe Guo and Yu-Feng Li},
    title        = {DeCoOp: Robust Prompt Tuning with Out-of-Distribution Detection},
    booktitle    = {Proceedings of the 41st International Conference on Machine Learning},
    year         = {2024}
}
```