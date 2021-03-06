# [LOGAN: Local Group Bias Detection by Clustering](https://arxiv.org/abs/2010.02867)
[Jieyu Zhao](https://jyzhao.net), and [Kai-Wei Chang](http://web.cs.ucla.edu/~kwchang/), EMNLP 2020 (short)

----
In this paper, we argue that evaluating bias at the corpus level is not enough for understanding how biases are embedded in a model. In fact, a model with similar aggregated performance between different groups on the entire data may behave differently on instances in a local region. To analyze and detect such local bias, we propose LOGAN, a new bias detection technique based on clustering. Experiments on toxicity classification and object classification tasks show that LOGAN identifies bias in a local region and allows us to better analyze the biases in model predictions.

--- 

### About our code
- Download the scikit-learn package (0.22.2.post1 in our case). And replace the corresponding codes using files under */cluster* folder.
  - Install scikit-learn to /local/Path/scikit-learn (follow [here](https://scikit-learn.org/stable/developers/advanced_installation.html#install-bleeding-edge)).
  - Copy files in */cluster* to /local/Path/scikit-learn/sklearn/cluster/
  - If you change the code under '/cluster/_k_means_fast_logan.pyx', you need to compile the codes with command 
  `gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing -I/usr/include/python3.7 -o _k_means_fast_logan.so _k_means_fast_logan.c`. 
  See [here](https://cython.readthedocs.io/en/latest/src/userguide/source_files_and_compilation.html) for more details.

- Please refer to the [jupyter-notebook](./toxic_clustering_race-2nd2lastlayer.ipynb) for the demo of doing LOGAN on toxicity detection task w.r.t. RACE attribute. Remember to change the path in the script.
- You can download the files needed in the jupyter-notebook from [here](https://drive.google.com/drive/folders/1_bDulU1ksnb5ln9t0F9t55WNDHuj1OfG?usp=sharing). 



