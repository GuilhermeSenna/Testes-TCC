import PIL
from tqdm import tqdm
import pickle
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets
from skimage.filters import gabor
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops

trainDset =  datasets.CIFAR10('./cifar10/', train=True, download=True)
testDset =  datasets.CIFAR10('./cifar10/', train=False, download=True)

# Size of train and test datasets
print('No. of samples in train set: '+str(len(trainDset)))
print('No. of samples in test set: '+str(len(testDset)))

# Feature extraction on single image
img = trainDset[0][0] #PIL image
img_gray = img.convert('L') #Converting to grayscale
img_arr = np.array(img_gray) #Converting to array
plt.imshow(img)
# plt.show()

# Finding LBP
feat_lbp = local_binary_pattern(img_arr,8,1,'uniform') #Radius = 1, No. of neighbours = 8
feat_lbp = np.uint8((feat_lbp/feat_lbp.max())*255) #Converting to unit8
lbp_img = PIL.Image.fromarray(feat_lbp) #Conversion from array to PIL image
plt.imshow(lbp_img,cmap='gray') #Displaying LBP
# plt.show()

# Energy and Entropy of LBP feature
lbp_hist,_ = np.histogram(feat_lbp,8)
lbp_hist = np.array(lbp_hist,dtype=float)
lbp_prob = np.divide(lbp_hist,np.sum(lbp_hist))
lbp_energy = np.sum(lbp_prob**2)
lbp_entropy = -np.sum(np.multiply(lbp_prob,np.log2(lbp_prob)))
print('LBP energy = '+str(lbp_energy))
print('LBP entropy = '+str(lbp_entropy))

# Finding GLCM features from co-occurance matrix
gCoMat = graycomatrix(img_arr, [2], [0],256,symmetric=True, normed=True) # Co-occurance matrix
contrast = graycoprops(gCoMat, prop='contrast')
dissimilarity = graycoprops(gCoMat, prop='dissimilarity')
homogeneity = graycoprops(gCoMat, prop='homogeneity')
energy = graycoprops(gCoMat, prop='energy')
correlation = graycoprops(gCoMat, prop='correlation')
print('Contrast = '+str(contrast[0][0]))
print('Dissimilarity = '+str(dissimilarity[0][0]))
print('Homogeneity = '+str(homogeneity[0][0]))
print('Energy = '+str(energy[0][0]))
print('Correlation = '+str(correlation[0][0]))


# Gabor filter
gaborFilt_real,gaborFilt_imag = gabor(img_arr,frequency=0.6)
gaborFilt = (gaborFilt_real**2+gaborFilt_imag**2)//2
# Displaying the filter response
fig, ax = plt.subplots(1,3)
ax[0].imshow(gaborFilt_real,cmap='gray')
ax[1].imshow(gaborFilt_imag,cmap='gray')
ax[2].imshow(gaborFilt,cmap='gray')
# plt.show()


# Energy and Entropy of Gabor filter response
gabor_hist,_ = np.histogram(gaborFilt,8)
gabor_hist = np.array(gabor_hist,dtype=float)
gabor_prob = np.divide(gabor_hist,np.sum(gabor_hist))
gabor_energy = np.sum(gabor_prob**2)
gabor_entropy = -np.sum(np.multiply(gabor_prob,np.log2(gabor_prob)))
print('Gabor energy = '+str(gabor_energy))
print('Gabor entropy = '+str(gabor_entropy))

# Generating training data by extracting features from all images
label = []
featLength = 2+5+2
trainFeats = np.zeros((len(trainDset),featLength)) #Feature vector of each image is of size 1x1030
for tr in tqdm(range(len(trainDset))):

    img = trainDset[tr][0] #One image at a time
    img_gray = img.convert('L') #Converting to grayscale
    img_arr = np.array(img_gray.getdata()).reshape(img.size[1],img.size[0]) #Converting to array
    # LBP
    feat_lbp = local_binary_pattern(img_arr,5,2,'uniform').reshape(img.size[0]*img.size[1])
    lbp_hist,_ = np.histogram(feat_lbp,8)
    lbp_hist = np.array(lbp_hist,dtype=float)
    lbp_prob = np.divide(lbp_hist,np.sum(lbp_hist))
    lbp_energy = np.nansum(lbp_prob**2)
    lbp_entropy = -np.nansum(np.multiply(lbp_prob,np.log2(lbp_prob)))
    # GLCM
    gCoMat = graycomatrix(img_arr, [2], [0],256,symmetric=True, normed=True)
    contrast = graycoprops(gCoMat, prop='contrast')
    dissimilarity = graycoprops(gCoMat, prop='dissimilarity')
    homogeneity = graycoprops(gCoMat, prop='homogeneity')
    energy = graycoprops(gCoMat, prop='energy')
    correlation = graycoprops(gCoMat, prop='correlation')
    feat_glcm = np.array([contrast[0][0],dissimilarity[0][0],homogeneity[0][0],energy[0][0],correlation[0][0]])
    # Gabor filter
    gaborFilt_real,gaborFilt_imag = gabor(img_arr,frequency=0.6)
    gaborFilt = (gaborFilt_real**2+gaborFilt_imag**2)//2
    gabor_hist,_ = np.histogram(gaborFilt,8)
    gabor_hist = np.array(gabor_hist,dtype=float)
    gabor_prob = np.divide(gabor_hist,np.sum(gabor_hist))
    gabor_energy = np.nansum(gabor_prob**2)
    gabor_entropy = -np.nansum(np.multiply(gabor_prob,np.log2(gabor_prob)))
    # Concatenating features(2+5+2)
    concat_feat = np.concatenate(([lbp_energy,lbp_entropy],feat_glcm,[gabor_energy,gabor_entropy]),axis=0)
    trainFeats[tr,:] = concat_feat #Stacking features vectors for each image
    # Class label
    label.append(trainDset[tr][1])
trainLabel = np.array(label) #Conversion from list to array

# Generating testing data by extracting features from all images
label = []
testFeats = np.zeros((len(testDset), featLength))  # Feature vector of each image is of size 1x1030
for ts in tqdm(range(len(testDset))):
    img = testDset[ts][0]  # One image at a time
    img_gray = img.convert('L')  # Converting to grayscale
    img_arr = np.array(img_gray.getdata()).reshape(img.size[1], img.size[0])  # Converting to array
    # LBP
    feat_lbp = local_binary_pattern(img_arr, 5, 2, 'uniform').reshape(img.size[0] * img.size[1])
    lbp_hist, _ = np.histogram(feat_lbp, 8)
    lbp_hist = np.array(lbp_hist, dtype=float)
    lbp_prob = np.divide(lbp_hist, np.sum(lbp_hist))
    lbp_energy = np.nansum(lbp_prob ** 2)
    lbp_entropy = -np.nansum(np.multiply(lbp_prob, np.log2(lbp_prob)))
    # GLCM
    gCoMat = graycomatrix(img_arr, [2], [0], 256, symmetric=True, normed=True)
    contrast = graycoprops(gCoMat, prop='contrast')
    dissimilarity = graycoprops(gCoMat, prop='dissimilarity')
    homogeneity = graycoprops(gCoMat, prop='homogeneity')
    energy = graycoprops(gCoMat, prop='energy')
    correlation = graycoprops(gCoMat, prop='correlation')
    feat_glcm = np.array([contrast[0][0], dissimilarity[0][0], homogeneity[0][0], energy[0][0], correlation[0][0]])
    # Gabor filter
    gaborFilt_real, gaborFilt_imag = gabor(img_arr, frequency=0.6)
    gaborFilt = (gaborFilt_real ** 2 + gaborFilt_imag ** 2) // 2
    gabor_hist, _ = np.histogram(gaborFilt, 8)
    gabor_hist = np.array(gabor_hist, dtype=float)
    gabor_prob = np.divide(gabor_hist, np.sum(gabor_hist))
    gabor_energy = np.nansum(gabor_prob ** 2)
    gabor_entropy = -np.nansum(np.multiply(gabor_prob, np.log2(gabor_prob)))
    # Concatenating features(2+5+2)
    concat_feat = np.concatenate(([lbp_energy, lbp_entropy], feat_glcm, [gabor_energy, gabor_entropy]), axis=0)
    testFeats[ts, :] = concat_feat  # Stacking features vectors for each image
    # Class label
    label.append(testDset[ts][1])
testLabel = np.array(label)

# Normalizing the train features to the range [0,1]
trMaxs = np.amax(trainFeats,axis=0) #Finding maximum along each column
trMins = np.amin(trainFeats,axis=0) #Finding maximum along each column
trMaxs_rep = np.tile(trMaxs,(50000,1)) #Repeating the maximum value along the rows
trMins_rep = np.tile(trMins,(50000,1)) #Repeating the minimum value along the rows
trainFeatsNorm = np.divide(trainFeats-trMins_rep,trMaxs_rep) #Element-wise division
# Normalizing the test features
tsMaxs_rep = np.tile(trMaxs,(10000,1)) #Repeating the maximum value along the rows
tsMins_rep = np.tile(trMins,(10000,1)) #Repeating the maximum value along the rows
testFeatsNorm = np.divide(testFeats-tsMins_rep,tsMaxs_rep) #Element-wise division

# Saving normalized training data and labels
with open("trainFeats.pckl", "wb") as f:
    pickle.dump(trainFeatsNorm, f)
with open("trainLabel.pckl", "wb") as f:
    pickle.dump(trainLabel, f)

# Saving normalized testing data and labels
with open("testFeats.pckl", "wb") as f:
    pickle.dump(testFeatsNorm, f)
with open("testLabel.pckl", "wb") as f:
    pickle.dump(testLabel, f)

print('Files saved to disk!')