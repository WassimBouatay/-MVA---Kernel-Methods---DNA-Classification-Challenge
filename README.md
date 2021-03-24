# MVA-Kernel-Methods-Challenge

## Team: The WAY
## Team members: Wassim BOUATAY, Amrou CHOUCHENE, Yassine NAJI

## Dependencies
Please install the libraries in the file Requirement.txt

## How to use
**examples:**
```
python start.py --number_of_samples 2000 --classifier SVM --data_type string --Kernel spectrum_kernel
```

**number_of_samples**: We added an option to control the number of samples to consider because some kernels take few minutes and even more than 1 hour for each data set. This will take the first samples without shuffling. We also split the selected samples randomly into a training set (85%) and a validation set (15%)

**Classifier**: The user can choose between 'SVM' and 'RIDGE'

**data_type**: The user can choose whether to use the DNA sequences (string) as input or some extracted numerical features (feature)

**Kernel**: This parameter is required. 
- The user can choose either 'linear' or 'rbf' kernel if he has chosen the extracted features as data_type. 
- The user can choose among the following kernels {spectrum_kernel, SW_kernel, WD_kernel, mismatchKernel} if he has chosen the DNA_sequences as data_type. 

## Remark:
**We added an estimation needed time to compute the kernel matrix per dataset depending on the number of samples chosen by the user and on the chosen kernel. The approximations were done using "i5-8300H CPU @2.30GHz". We also note the testing phase may take an important amount of time depending on those 2 parameters.** 
