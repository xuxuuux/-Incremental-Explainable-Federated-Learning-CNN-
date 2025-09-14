# -Incremental-Explainable-Federated-Learning-CNN-
An incremental and explainable CNN-based federated learning framework: initially trained on SVHN, then incrementally adapted to MNIST via federated knowledge distillation, with model interpretability analyzed using LIME.

link of MINST dataset: [Yann LeCun's MNIST Database](http://yann.lecun.com/exdb/mnist/)  

link of SVHN dataset:[Stanford SVHN Dataset](http://ufldl.stanford.edu/housenumbers/) 

## federated_learning.py

### Based on the SVHN dataset and average aggregation for federated training

#### IID & Non-IID

My code can generate two types of data distributions: IID or Non-IID. The comparison of accuracy curves under federated learning across different clients is shown below:

<img src="picture\1.png" alt="1" style="zoom:15%;" />

<img src="picture\2.png" alt="1" style="zoom:15%;" />

## transfer_learning_kl.py

This code file uses the MNIST dataset as the incremental dataset and trains the central model, which was initially trained on the SVHN dataset, within a federated incremental learning framework based on knowledge distillation.

<img src="picture\3.png" alt="1" style="zoom:15%;" />

<img src="picture\4.png" alt="1" style="zoom:15%;" />

## final_model_with_LIME_Explaination.py

This code file provides LIME-based visual explanations
for the federated CNN model after incremental training.

In this code, you can freely choose to use digits 0–9 from either
the MNIST or SVHN dataset, view the predictions of the
federated CNN model, and generate LIME-based visual explanations.
All of these interactions are carried out through a human-computer dialogue.

<img src="picture\5.png" alt="1" style="zoom: 50%;" />

<img src="picture\7.png" alt="1" style="zoom: 25%;" />

<img src="picture\8.png" alt="1" style="zoom: 40%;" />

## baseline_cnn.py

This code file is used to train the baseline model and visualize
the performance of the baseline CNN model on the SVHN dataset
using a confusion matrix.
