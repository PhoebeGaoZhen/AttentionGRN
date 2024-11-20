This is the repository for the manuscript: *AttentionGRN: A Functional and Directed Graph Transformer for Gene Regulatory network reconstruction from scRNA-seq*


# Requirements

You need to configure the following environment before running the AttentionGRN model. We also provide *environment.yml* for use by our peers.

It should be noted that this project is carried out in the *Windows system*, if you are using Linux system, We hope you can install the corresponding environment version yourself.

* Windows system

* NVIDIA GeForce RTX 3060

* PyCharm 2022

* python = 3.8.19

* dgl = 1.0.1+cu117	

* numpy = 1.24.4

* pandas = 2.0.3

* scikit-learn = 1.3.2

* torch = 1.13.1+cu117	

* torch-cluster = 1.6.1+pt113cu117	

* torch-geometric = 2.5.3	

* torch-scatter = 2.1.1+pt113cu117	

* torch-sparse = 0.6.17+pt113cu117	

* torch-spline-conv = 1.2.2+pt113cu117	

* torchaudio = 0.13.1+cu117	

* torchdata = 0.7.1	

* torchvision = 0.14.1+cu117	

* tornado = 6.3.3	

* networkx = 3.1	

* scipy = 1.10.1

# 1. prepare raw data----BEELINE

URL: https://zenodo.org/records/3701939

Reference: Pratapa, A., Jalihal, A.P., Law, J.N. et al. Benchmarking algorithms for gene regulatory network inference from single-cell transcriptomic data. Nat Methods 17, 147–154 (2020). https://doi.org/10.1038/s41592-019-0690-6.

### content:

**scRNA-seq**  
1. hESC: ExpressionData.csv, GeneOrdering.csv, PseudoTime.csv  
2. hHEP: ExpressionData.csv, GeneOrdering.csv, PseudoTime.csv  
3. mDC: ExpressionData.csv, GeneOrdering.csv, PseudoTime.csv  
4. mESC: ExpressionData.csv, GeneOrdering.csv, PseudoTime.csv  
5. mHSC-E: ExpressionData.csv, GeneOrdering.csv, PseudoTime.csv  
6. mHSC-GM: ExpressionData.csv, GeneOrdering.csv, PseudoTime.csv  
6. mHSC-L: ExpressionData.csv, GeneOrdering.csv, PseudoTime.csv

   
**Networks**
1. human
    * HepG2-ChIP-seq-network.csv  
    * hESC-ChIP-seq-network.csv  
    * Non-specific-ChIP-seq-network.csv  
    * STRING-network.csv  
2. mouse
    * mDC-ChIP-seq-network.csv  
    * mESC-ChIP-seq-network.csv  
    * mESC-lofgof-network.csv  
    * mHSC-ChIP-seq-network.csv  
    * Non-Specific-ChIP-seq-network.csv  
    * STRING-network.csv  
3. human-tfs.csv  
4. mouse-tfs.csv  

# 3. TRN vs GRN

Both GRN and TRN are directed graphs which contain regulatory associations between TF and target genes. The difference between TRN and GRN is whether the TFs can be regulated by other TFs. In GRN, TF can be regulated by other TF, while TFs cannot be regulated by other TF in TRN.

In addition, for GRN inference and TRN inference tasks, we used two different evaluation strategies, namely independent testing and TF-aware three-fold cross-validation. The detailed process of these two evaluation strategies is described in the paper.

GRN inference and TRN inference are evaluated as follows:




|       | GRN inference         |TRN   inference      |
| ----------- | ----------- |----------- |
| evaluation strategy      | independent testing       |TF-aware three-fold cross-validation       |
| datasets   | DATA1-500 and DATA1-1000      |DATA2-500 and DATA2-1000            |


# 4. Evaluate the performance of AttentionGRN in reconstruting GRN

We used independent testing to verify the prediction performance of AttentionGRN in reconstructing GRN.

### train and test
          
  run *./my_test_final/AttentionGRN_GRN/main.py*. 
  
  GRN reconstruction using AttentionGRN model, and the prediction performance of the model is evaluated using independent testing.

# 5. Evaluate the peroformance of AttentionGRN in reconstruting TRN

We used TF-aware three-fold cross-validation to verify the prediction performance of AttentionGRN in reconstructing TRN.

### train and test

  run *./my_test_final/AttentionGRN_TRN/main.py*.

  TRN reconstruction using Model AttentionGRN, and the prediction performance of the model is evaluated using TF-aware three-fold cross validation.

# 6. GRN inference for human mature hepatocytes (hHEP)

**input data**

* gene expression data.
  
|       | gene1   |gene2    |
| ----- | --------|---------|
| cell1 | exp     |exp      |
| cell2 | exp     |exp      |
| cell3 | exp     |exp      |

* prior GRN.

  
| TF   |Target    |
| --------|---------|
| tf1 name      |target gene1 name     |
| tf2 name      |target gene2 name      |
| tf3 name      |target gene3 name        |



* gene list

|       | gene   |index    |
| ----- | --------|---------|
| 0 |gene1 name     |gene1 index      |
| 1 |gene2 name     |gene2 index      |
| 2 |gene3 name     |gene3 index      |


* TF list

|       | TF   |index    |
| ----- | --------|---------|
| 0 |tf1 name     |tf1 index      |
| 1 |tf2 name     |tf2 index      |
| 2 |tf3 name     |tf3 index      |


**procedure**

* run *./my_test_final/AttentionGRN_infer/infer_GRN.py*, you can infer GRN and get the inferred GRN.

**output**

* predict_results/predict_GRN.csv


# GENELink

**requirments**

- The python environment is the same as AttentionGRN
    
**GRN reconstruction via GENELink**

- run *./my_test_final/GENELink/main_GRN_data1.py* to evaluate GENELink in GRN reconstruction on DATA1 using independent testing.


**TRN reconstruction via GENELink**


- run *./my_test_final/GENELink/main_TRN_data2.py* to evaluate GENELink in TRN reconstruction on DATA2 using TF-aware three-fold cross validation.

# GNNLink

**requirments**

- python=3.7
- tensorflow=2.11.0
- keras=2.11.0
- torch=1.12.1
- scikit-learn=1.0.2
- numpy=1.21.6
- pandas=1.3.5

**GRN reconstruction via GNNLink**

- run *./my_test_final/GNNLink/main_GRN_data1.py* to evaluate GNNLink in GRN reconstruction on DATA1 using independent testing.


**TRN reconstruction via GNNLink**

- run *./my_test_final/GNNLink/main_TRN_data2.py* to evaluate GNNLink in TRN reconstruction on DATA2 using TF-aware three-fold cross validation.


# DeepFGRN

**requirments**

- Python=3.6.5
- tensorflow=1.9.0
- keras=2.2.0
- numpy=1.19.5
- pandas=1.1.5
- scikit-learn=0.22.1

    
**GRN reconstruction via DeepFGRN**

- run *./my_test_final/DeepFGRN/main_GRN_data1.py* to evaluate DeepFGRN in GRN reconstruction on DATA1 using independent testing.


**TRN reconstruction via DeepFGRN**


- run *./my_test_final/DeepFGRN/main_TRN_data2.py* to evaluate DeepFGRN in TRN reconstruction on DATA2 using TF-aware three-fold cross validation.


# STGRNS

**requirments**

- The python environment is the same as AttentionGRN
    
**GRN reconstruction via STGRNS**

- run *./my_test_final/STGRNS/main_GRN_data1.py* to evaluate STGRNS in GRN reconstruction on DATA1 using independent testing.


**TRN reconstruction via STGRNS**

- run *./my_test_final/STGRNS/main_TRN_data2.py* to evaluate STGRNS in TRN reconstruction on DATA2 using TF-aware three-fold cross validation.


# CNNC

**requirments**

- python=3.7
- tensorflow=2.6.2
- keras=2.6.0
- scikit-learn=0.24.2
- numpy=1.19.5
- pandas=1.3.5
- matplotlib=3.5.3
- scipy=1.7.3


**GRN reconstruction via CNNC**

- run *./my_test_final/CNNC/main_GRN_data1.py* to evaluate CNNC in GRN reconstruction on DATA1 using independent testing.


**TRN reconstruction via CNNC**


- run *./my_test_final/CNNC/main_TRN_data2.py* to evaluate CNNC in TRN reconstruction on DATA2 using TF-aware three-fold cross validation.

