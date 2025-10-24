# ATTBHFA-Net (Attention BHattacharyya-Hellinger Distance-based Feature Aggregation Network)
This is the code repository for our work ``Enhancing Few-Shot Classification of Benchmark and Disaster Imagery with ATTBHFA-Net'' by Gao Yu Lee, Tanmoy Dam, Md. Meftahul Ferdaus, Daniel Puiu Poenar, and Vu Duong. 

The arXiv preprint of the work is available at: https://arxiv.org/abs/2510.18326 

# Abstract

The increasing frequency of natural and human-induced disasters necessitates advanced visual recognition techniques capable of analyzing critical photographic data. With progress in artificial intelligence and resilient computational systems, rapid and accurate disaster classification has become crucial for efficient rescue operations. However, visual recognition in disaster contexts faces significant challenges due to limited and diverse data from the difficulties in collecting and curating comprehensive, high-quality disaster imagery. Few-Shot Learning (FSL) provides a promising approach to data scarcity, yet current FSL research mainly relies on generic benchmark datasets lacking remote-sensing disaster imagery, limiting its practical effectiveness. Moreover, disaster images exhibit high intra-class variation and inter-class similarity, hindering the performance of conventional metric-based FSL methods. To address these issues, this paper introduces the Attention-based Bhattacharyya-Hellinger Feature Aggregation Network (ATTBHFA-Net), which linearly combines the Bhattacharyya and Hellinger distances to compare and aggregate feature probability distributions for robust prototype formation. The Bhattacharyya distance serves as a contrastive margin that enhances inter-class separability, while the Hellinger distance regularizes same-class alignment. This framework parallels contrastive learning but operates over probability distributions rather than embedded feature points. Furthermore, a Bhattacharyya-Hellinger distance-based contrastive loss is proposed as a distributional counterpart to cosine similarity loss, used jointly with categorical cross-entropy to significantly improve FSL performance. Experiments on four FSL benchmarks and two aerial disaster image datasets demonstrate the superior effectiveness and generalization of ATTBHFA-Net compared to existing approaches.

(The following code repository is for AIDER [1], CDD [2] and miniImageNet [3] evaluation.)

# Code Instructions

1) Run data_prep_AIDER.py (or data_prep_CDD.py) first for uploading the AIDER (CDD) training and testing data from your selected directory. 
2) Run Data_Augmentation.py for the data augmentation code. The type of augmentation is also stated in the main manuscript.
3) Run AIDER_Dataloader.py (or CDD_Dataloader.py), which store the respective training and test tuple (Images, labels) in a Pytorch dataloader format.
4) Run Hyperparameters.py, which contained the hyperparameters for the few-shot learning.
5) Run Attention.py, which contains the codes for the channel-spatial attention mechanism.
6) Run the ResNet-12 backbone, which is a modified ResNet12 with the channel-spatial attention imbued into it.
7) Run Bhattacharyya_Coeff.py, which contains the Bhattacharyya coefficient computation codes.
8) Now run ABHFANet main.py which comprised of the main backbone of the ABHFA-Net.
9) Run Sampler and Loader.py which provides the task sampler for the training and testing dataloader.
10) A series of functions is found inside a file labelled "demo-notebook" (placeholder name). They contained the helper functions to run the Bhattarcharyya Softmax (BHAS) loss function. The sequence in which they should be run is: Common_functions.py -> loss_and_miners_utils.py -> Module_With_Records.py -> Base_Reducers.py -> MeanReducer.py -> MultipleReducers_Do_Nothing_Reducers.py -> BaseDistances.py -> LpDistance.py -> ModulesWithRecordsandReducer.py -> Mixins.py -> BaseMetricLossFunction.py -> GenericPairLoss.py -> BhattLoss.py. The helper functions are mainly adapted from the pytorch metric learning library by Kevin Musgrave: https://github.com/KevinMusgrave/pytorch-metric-learning.
11) Run training_fit.py to begin training.
12) Finally, evaluate the trained model using Eval.py.

# Model Architecture Figure 

This is extracted from Fig.4 from our main preprint manuscript. For more information about the figure please refer to the manuscript.
<img width="1040" height="408" alt="ABHFA-Net" src="https://github.com/user-attachments/assets/1ce3a2f8-a5db-4b31-a8db-b93e58d03700" />
Fig.1: Illustration of the ABHFA-Net algorithmic architecture.

# Datasets

In this work we utilized three datasets designed (and could be designed) for aerial disaster classification: AIDER (Aerial Image Dataset for Emergency Response) by Kykrou and Theocharides, CDD (Comprehensive Disaster Dataset) by Niloy et al., and MEDIC (Multi-Task Learning Dataset for Disaster Image Classification) by Alam et al. 

AIDER comprised of 4 disaster classes along with its aftermath (Fire, Flood, Traffic Accident, Collapsed Buildings), as well as one normal (non-disaster class). This is shown in Fig.2. The non-disaster dataset are of greater quantity than that of the disaster class, hence AIDER is imbalanced by default. However, one can perform undersampling to balance out the class distributions.

<img width="403" height="283" alt="AIDER_samples" src="https://github.com/user-attachments/assets/ce37fc67-c533-4080-a191-f75690add994" /> \
Fig.2. Example images from each AIDER subset disaster classes.


# Preliminary Results 

The few-shot evaluation utilized are the 5-way-1-shot and 5-way-5-shot approach. The table below tabulated the relevant values obtained for the AIDER [1] and CDD [2] dataset when using the 5-way-1-shot and 5-way-5-shot approaches (in %):

| Method | 5-way-1-shot (%) | 5-way-5-shot (%) |
| ------ | ------| ------| 
|ATTBHFA-Net (AIDER)| **67.0 $\pm$ 1.7** | **76.9 $\pm$ 0.3** |
|ATTBHFA-Net (CDD)| **58.8 $\pm$ 1.0** | **62.6 $\pm$ 0.4** |

# Citation

If you find our work useful, you can cite it as follows: 


# Relevant References

[1]  Kyrkou, C., Theocharides, T.: Emergencynet: Efficient aerial image classification for drone-based emergency monitoring using atrous convolutional feature fusion. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing 13, 1687–1699 (2020).

[2] Niloy, F.F., Nayem, A.B.S., Sarker, A., Paul, O., Amin, M.A., Ali, A.A., Zaber, M.I., Rahman, A.M., et al.: A novel disaster image data-set and characteristics analysis using attention model. In: IEEE 2020 25th International Conference on Pattern Recognition (ICPR), pp. 6116–6122 (2021).

[3] Alam, F., Alam, T., Hasan, M.A., Hasnat, A., Imran, M., Ofli, F.: Medic: a multi-task learning dataset for disaster image classification. Neural Computing and Applications 35(3), 2609–2632 (2023).


