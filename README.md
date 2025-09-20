# ABHFA-Net (Attention BHattacharyya Distance-based Feature Aggregation Network)
This is the code repository for our work ``Enhancing Few-Shot Classification of Disaster Imagery with ABHFA-Net'' by Gao Yu Lee, Tanmoy Dam, Md. Meftahul Ferdaus, Daniel Puiu Poenar, and Vu Duong. 

The arXiv preprint of the work will be available soon.

# Abstract

The rising incidence of natural and human-induced calamities necessitates enhanced visual recognition techniques for diverse disaster categories, utilizing critical visual data from photographic sources. Advancements in artificial intelligence and robust computational systems resilient to harsh environments have rendered this field critical to expedite rescue efforts. However, the visual classification of disaster situations faced formidable obstacles due to data constraints, which are primarily due to the limitations of data collection platforms and the complexities inherent in compiling a comprehensive database of high-quality disaster imagery. Few-Shot Learning (FSL) offers a viable solution to these issues but current FSL applications primarily rely on benchmark datasets devoid of disaster-related imagery obtained via remote sensing, thus curtailing its full potential. This paper proposes the Attention BHattacharyya Distance-based Feature Aggregation Network (ABHFA-Net) which performs quantitative comparison and aggregation of similar feature sample probability distributions to formulate class-prototype distributions using the Bhattacharyya distance. Concurrently, an attention mechanism is incorporated into our encoder structure for the query and support image sets. This not only highlights the importance of the attention mechanism and Bhattacharyya distance but also sets a new standard in FSL-based prototype aggregation, especially for disaster image categorization. Additionally, this paper pioneers a Bhattacharyya distance-based contrastive training loss, a more suitable variant of the cosine similarity contrastive loss to compute probability distribution differences. When combined with the categorical cross-entropy loss, it boosts FSL performance to unparalleled levels. Experiments with three separate disaster image classification datasets confirm the effectiveness and superiority of our model over existing FSL methodologies.

(The following code repository is for AIDER [1] evaluation only. The corresponding relevant codes for CDD [2] and MEDIC [3] would be added in shortly.)

# Code Instructions

1) Run data_prep_AIDER.py first for uploading the AIDER training and testing data from your selected directory.
2) Run Data_Augmentation.py for the data augmentation code. The type of augmentation is also stated in the main manuscript.
3) Run AIDER_Dataloader.py, which store the training and test tuple (Images, labels) in a Pytorch dataloader format.
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

# Preliminary Results 

The few-shot evaluation utilized are the 5-way-1-shot and 5-way-5-shot approach. The table below tabulated the relevant values obtained for the AIDER [1], CDD [2] and MEDIC [3] dataset when using the 5-way-1-shot and 5-way-5-shot approaches (in %):

| Method | 5-way-1-shot (%) | 5-way-5-shot (%) |
| ------ | ------| ------| 
|ABHFA-Net (AIDER)| **68.2 $\pm$ 1.19** | **78.3 $\pm$ 0.75** |
|ABHFA-Net (CDD)| **63.0 $\pm$ 0.95** | **74.2 $\pm$ 0.65** |
|ABHFA-Net (MEDIC)| **60.2 $\pm$ 1.22** | **66.5 $\pm$ 0.95** |


# Citation

# Relevant References

[1]  Kyrkou, C., Theocharides, T.: Emergencynet: Efficient aerial image classification for drone-based emergency monitoring using atrous convolutional feature fusion. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing 13, 1687–1699 (2020).

[2] Niloy, F.F., Nayem, A.B.S., Sarker, A., Paul, O., Amin, M.A., Ali, A.A., Zaber, M.I., Rahman, A.M., et al.: A novel disaster image data-set and characteristics analysis using attention model. In: IEEE 2020 25th International Conference on Pattern Recognition (ICPR), pp. 6116–6122 (2021).

[3] Alam, F., Alam, T., Hasan, M.A., Hasnat, A., Imran, M., Ofli, F.: Medic: a multi-task learning dataset for disaster image classification. Neural Computing and Applications 35(3), 2609–2632 (2023).


