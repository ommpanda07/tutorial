# 🧠 Alzheimer’s Disease Classification Using Deep Learning and MRI

Alzheimer’s disease is a chronic neurodegenerative condition that primarily affects elderly individuals, leading to progressive deterioration in memory, thinking, and behavioral abilities. As the global population ages, the prevalence of Alzheimer’s is increasing at an alarming rate 📈, posing significant challenges to healthcare systems and patient caregivers 👩‍⚕️👨‍⚕️.

🔍 Accurate and early-stage diagnosis is crucial for managing the disease effectively and improving the quality of life for patients. Traditional diagnostic methods, often relying on clinical assessments and manual analysis of MRI scans 🧬, can be time-consuming, subjective, and error-prone ❗.

In this study, we explore a **deep learning-based approach** to automate the classification of Alzheimer’s disease stages using magnetic resonance imaging (MRI) 🧪🖥️. We employ **transfer learning models**—ResNet50, VGG19, and InceptionV3—due to their strong performance in image recognition tasks 🎯. The models are fine-tuned on a dataset containing MRI images categorized into four classes:

- 🟢 NonDemented  
- 🟡 VeryMildDemented  
- 🟠 MildDemented  
- 🔴 ModerateDemented

🛠️ Data preprocessing and augmentation techniques are applied to enhance the model’s ability to generalize across unseen data. The models are evaluated based on **accuracy**, **loss**, and **AUC** metrics 📊, and the results are visualized using training curves 📈 and confusion matrices 🔎.

Among the models implemented, **ResNet50** demonstrated the highest classification performance with balanced generalization across all categories. 🚀

This research highlights the potential of leveraging state-of-the-art deep learning techniques 🧠💡 to support early detection and diagnosis of Alzheimer’s disease. Such systems can assist medical professionals in making informed decisions and pave the way for **AI-assisted clinical applications** in neurology and radiology 🌐🏥.
