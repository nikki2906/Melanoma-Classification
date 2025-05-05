# Melanoma-Classification

## Hypothesis:
We hypothesize that a CNN model trained on melanoma and non-melanoma images can classify melanoma more accurately than logistic regression methods. Modifying a pre-trained CNN model using transfer learning should improve its classification. 
Describe the problem:
Our research question was how machine learning models can help classify skin lesions as melanoma or moles when screening patients for skin cancer
Skin cancer is one of the most common forms of cancer globally, with melanoma being its deadliest variant. Despite being the most dangerous form of skin cancer, melanoma is highly treatable when detected early. Statistics show that
1 in 5 Americans is expected to develop skin cancer by age 70
Melanoma accounts for the majority of skin cancer deaths
Early detection dramatically changes outcomes: the 5-year survival rate is 99% for localized melanoma but drops to below 30% for advanced cases
Our project aims to predict whether a skin lesion is melanoma or benign. We will focus on classifying whether or not a skin lesion is melanoma using machine learning models applied to dermatological images. Our goal is to utilize machine learning algorithms to assist in screening, potentially improving both accuracy and accessibility of skin cancer detection.

## Methodology and Data: 
Dataset link: https://www.kaggle.com/datasets/drscarlat/melanoma. The dataset includes 8,903 melanoma images and 8,902 mon-melanoma images of dermoscopic images in the HAM10k dataset. Data augmentation was performed on the melanoma images to bring the original number of melanoma images (1113) similar to the non-melanoma group. 

<img width="621" alt="Screenshot 2025-05-05 at 10 16 00 PM" src="https://github.com/user-attachments/assets/90e88e4a-8e2a-4c06-a508-4141486af18f" />

### Pre-Processing: 
Loaded and Resized Images to 128x128. Then we converted the images to numpy arrays and flattened them when processing them.

## Logistic Regression Model
As a baseline approach, we implemented the logistic regression model. We used it as a benchmark to compare more complex neural network architectures. For this implementation, we loaded and preprocessed images from both melanoma and nonmelanoma classes and resized all images to 128x128 pixels for consistent dimensionality. We then flattened the RGB pixel values, resulting in 49,152 features per image (128×128×3), and applied standard scaling to normalize the feature values. We trained a logistic regression model with balanced class weights to address potential class imbalance. 

### Logistic Regression Model Result

The logistic regression model results show that the accuracy score is 72.21% on the validation set. The model shows balanced performance across both classes. The ROC curve analysis has an AUC score of 0.782, suggesting the logistic regression method is reasonably reliable for distinguishing melanoma from non-melanoma, though there is still room for improvement. 
In the graph, the model's performance is above the random classifier baseline. While the precision for melanoma detection, which is 0.74, is reasonable, the recall rate, which is 0.68, indicates that the model fails to identify approximately 32% of actual melanoma cases. In a medical context where missing true positives can be dangerous for patients, this highlights a limitation of the baseline approach and motivates us to test other advanced models.

## Baseline CNN Model: 
The baseline CNN model is our foundational model in this project. The model takes in images at 128×128 pixels and processes them through three main convolutional blocks. The first block starts with 64 filters to capture basic features like edges and colors, then the second block uses 32 filters for more complex patterns, and the third uses just 16 filters for higher-level features. After each convolutional layer, there's max pooling to shrink the image size and focus on what's important. The model adds dropout layers (0.25) after the second and third blocks to prevent it from memorizing the training data too closely. After feature extraction, the model uses a Global Average Pooling layer to summarize all the features before making the final yes/no decision about melanoma with a sigmoid activation function.

### Baseline CNN Model Results with default thresholds: 

Looking at our results, the baseline CNN model achieved an accuracy of 80%. The classification metrics show a balanced performance between the two classes, with melanoma detection reaching 0.86 precision and 0.72 recall, while non-melanoma cases show 0.76 precision and 0.88 recall. This balance is important as we want to minimize both false negatives (missed melanomas) and false positives (incorrectly diagnosing melanoma). 

### Baseline CNN Model with optimized thresholds:

Next, we examine how optimizing the decision threshold improves melanoma classification performance. Finding the optimal classification threshold is important in medical image analysis applications like melanoma detection. While machine learning models typically default to a 0.5 threshold for binary classification, this arbitrary value rarely represents the ideal decision boundary. Threshold optimization becomes essential when working with imbalanced datasets, where one class may be significantly underrepresented. More importantly, in medical diagnostics, the consequences of different types of errors vary dramatically, as missing a melanoma diagnosis (false negative) could have dangerous consequences for patients, while incorrectly flagging a benign lesion (false positive) leads to unnecessary worries for patients. By adjusting the threshold, we can fine-tune the precision-recall balance to align with clinical priorities, ultimately maximizing metrics like F1-score, which balance both concerns. 

Analyzing our results, using the default threshold of 0.5, the model achieved 80% overall accuracy with melanoma detection reaching 0.86 precision and 0.72 recall, while non-melanoma cases showed 0.76 precision and 0.88 recall. After optimizing the threshold, the model achieved more balanced detection capabilities between classes, with melanoma recall improving to 0.81 while non-melanoma recall decreased slightly to 0.82%. This adjusted threshold redistributes the error pattern, resulting in more balanced metrics with melanoma precision at 0.82 and non-melanoma precision at 0.81. Despite the small change in overall accuracy to 81%, the optimized threshold better aligns with clinical priorities where a balance between missing melanomas and unnecessary follow-ups is important. The ROC curve shows an AUC of 0.878, demonstrating good discriminative ability across threshold values.

## CNN Model with Transfer Learning
MobileNetV2 is a CNN pre-trained on ImageNet, a dataset containing over 14 million images. While MobileNetV2 is generally used for mobile devices or embedded systems, we picked MobileNetV2 due to its smaller size and efficiency. We also performed transfer learning with ResNet50, a CNN model commonly used for image classification, but only received accuracy scores around 76%. 

Our implementation used MobileNetV2 with input images sized at 128×128 pixels. The base model outputs feature maps of dimension 4×4×1280, which are then processed through a Global Average Pooling layer to reduce dimensionality while preserving spatial information. To stabilize training, we incorporated batch normalization after the pooling layer, followed by a dense layer with 256 neurons that compresses the features extracted by MobileNetV2. Another batch normalization layer was applied before implementing a dropout rate of 0.25 to prevent overfitting. The final layer consists of a single neuron with sigmoid activation, appropriate for our binary classification task of distinguishing melanoma from non-melanoma lesions

### Transfer Learning Model Results:

Our MobileNetV2 transfer learning model achieved an 87% overall accuracy on the validation dataset. The model identified a 100% recall for melanoma detection, with no cancer cases missed, and maintained 80% precision. For non-melanoma cases, the precision achieved 100%, with 75% recall. The classification report indicates F1-scores of 0.86 for non-melanoma and 0.89 for melanoma cases, representing a good balance between precision and recall. The ROC curve analysis shows an AUC of 0.971, which indicates that the model is able to perform well when distinguishing between melanoma and non-melanoma across different classification thresholds. 

### Transfer Learning Model Results with Optimized Thresholds:

Using the default 0.5 threshold, the model achieved 87% accuracy with a bias toward melanoma detection, identifying all melanoma cases (100% recall) but generating false positives (only 80% precision). After optimizing the threshold, the model's accuracy improved to 92% with more balanced class performance. The optimized threshold maintained better melanoma detection sensitivity at 96% while reducing false positives, improving melanoma precision to 89%. Similarly, non-melanoma detection improved from 75% to 88% recall. This balanced approach is reflected in the improved F1-scores for both classes (0.92 for both melanoma and non-melanoma).

## Discussion:
In melanoma cases, missing a melanoma classification (false negative) can be more dangerous than a false positive misclassification. Thus, we tested various thresholds when classifying to prioritize sensitivity (true positive rate) over specificity. Then, re-training based on the optimized threshold produced a higher accuracy with a better balanced class performance on both the baseline CNN and transfer learning models. As an extra check, since the results of our CNN models are above 50%, we can confirm that the results of our model are not due to a random guess. One issue we faced at the beginning of our project was the elongated runtime for our code, which we fixed by changing our hardware accelerator from a CPU to a T4 GPU. This significantly improved our runtime. However, overall, our model was able to accurately predict most of the time if a skin lesion was melanoma or non-melanoma. 

In relation to other work surrounding skin cancer, many CNN models have been created with promising results, with one model obtaining “ an accuracy of 97.78% with a notable precision of 97.9%, recall of 97.9%, and an F2 score of 97.8%..” The FDA has also approved the first AI-Powered Skin Cancer Diagnostic Tool, DermaSensor, which “demonstrates a high rate of sensitivity in the detection of more than 200 types of skin cancers in a clinical study".
Furthermore, machine learning used for detecting cancer has been used for breast cancer datasets as well. For example, one well known K-Nearest Neighbors model is used to predict whether a patient’s tumor is malignant or benign based on the UCI ML Breast Cancer Wisconsin (Diagnostic) dataset. However, this dataset, containing mammography exam results, predicts the result based on features such as radius, texture, perimeter, and area . 





