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
