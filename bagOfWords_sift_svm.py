import numpy as np
import cv2
import skimage.io as io
from sklearn.cluster import KMeans
import pickle
from sklearn.preprocessing import scale
from sklearn import svm
from sklearn.cross_validation import cross_val_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
#%%
def balanceWhite(img):
    # white balance (gray world assumption)
    B = np.mean(img[:,:,0])
    G = np.mean(img[:,:,1])
    R = np.mean(img[:,:,2])
    KB = (R + G + B) / (3 * B);  
    KG = (R + G + B) / (3 * G);  
    KR = (R + G + B) / (3 * R);  
    img[:,:,0] = KB * img[:,:,0]
    img[:,:,1] = KG * img[:,:,1]
    img[:,:,2] = KR * img[:,:,2]
    return(img)

#%%
## parameters
classes = ['beet_salad', 'caesar_salad', 'caprese_salad', 'greek_salad', 'seaweed_salad']
## 300 for training; 100 for test
numOfTrain = 300; numOfTest = 100;
np.random.seed(3366)
index = np.arange(1000)
np.random.shuffle(index)
img_size = 128;

#%% just for poster

#img = cv2.imread('../data/beet_salad/58733.jpg')
#img = cv2.resize(img,(128,128), interpolation = cv2.INTER_CUBIC)
#img = cv2.medianBlur(img,3)
#img = balanceWhite(img)
#
#
#img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#img = cv2.equalizeHist(img)
#img = cv2.medianBlur(img,3)
#cv2.imwrite('ppt.jpg',img)
## save sift image
#sift = cv2.xfeatures2d.SIFT_create()
#kp = sift.detect(img,None)
#img_ = img
#img__ = cv2.drawKeypoints(img,kp, img_, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#cv2.imwrite('sift_keypoints.jpg',img__)
##%%
## preprocessing
for i in range(len(classes)):
    dir_input = '../data/' + classes[i] + '/*.jpg'
    coll = io.ImageCollection(dir_input)
    for j in range(numOfTrain + numOfTest):
        # loading image    
        img = cv2.imread(coll.files[j]) # 0: B; 1: G; 2: R
        # rescaling to 512*512
        img = cv2.resize(img, (img_size,img_size), interpolation = cv2.INTER_CUBIC)
        # median filter
        img = cv2.medianBlur(img,3)
        # white balance (gray world assumption)
        img_balanceWhite = balanceWhite(img)
        # converting to grayscale image and histogram equalization
        img_gray = cv2.cvtColor(img_balanceWhite, cv2.COLOR_BGR2GRAY)
        img_gray = cv2.equalizeHist(img_gray)
        # median filter
        img_gray = cv2.medianBlur(img_gray,3)
        # training and test dataset
        if (j < numOfTrain):
            dir_save = '../data/preprocessed_' + np.str(img_size) + '/' + 'train_filtered/' \
                        + np.str(i) + '_' + np.str(j) + '.jpg'
        else: 
            dir_save = '../data/preprocessed_' + np.str(img_size) + '/' + 'test_filtered/' \
                        + np.str(i) + '_' + np.str(j) + '.jpg'
        cv2.imwrite(dir_save,img_gray)

#%%
### SIFT + SVM (Bag of Words)
# training feature
sift = cv2.xfeatures2d.SIFT_create()
des = np.empty([1,130])
for i in range(len(classes)):
    for j in range(numOfTrain):      
        ## 1. SIFT feature
        path = '../data/preprocessed_' + np.str(img_size) + '/' + 'train_filtered/' \
                + np.str(i) + '_' + np.str(j) + '.jpg'
        img = cv2.imread(path)
        _, des_1 = sift.detectAndCompute(img,None)
        label_class = np.transpose(np.asmatrix(np.ones(len(des_1)) * i))
        label_num = np.transpose(np.asmatrix(np.ones(len(des_1)) * j))
        des_1 = np.hstack((label_class, label_num, des_1))
        des = np.concatenate((des, des_1))
sift_train = des[1:,:]
del des

des = np.empty([1,130])
for i in range(len(classes)):
    for j in range(numOfTest):      
        ## 1. SIFT feature
        path = '../data/preprocessed_' + np.str(img_size) + '/' + 'test_filtered/' \
                + np.str(i) + '_' + np.str(j+numOfTrain) + '.jpg'
        img = cv2.imread(path)
        _, des_1 = sift.detectAndCompute(img,None)
        label_class = np.transpose(np.asmatrix(np.ones(len(des_1)) * i))
        label_num = np.transpose(np.asmatrix(np.ones(len(des_1)) * j))
        des_1 = np.hstack((label_class, label_num, des_1))
        des = np.concatenate((des, des_1))
sift_test = des[1:,:]
del des

#%% kmeans
#pickle.dump(sift_test, open('test.sav', 'wb'))
#pickle.dump(sift_train, open('train.sav', 'wb'))
#sift_test = pickle.load(open('test.sav', 'rb'))
#sift_train = pickle.load(open('train.sav', 'rb'))
# creating vocabulary based on k-means
k = 500
#import time
#start_time = time.time()
#kmeans = KMeans(n_clusters = k, random_state = 0, n_jobs = -2).fit(sift_train[:,2:]) # 21536 sec for 500 clusters, 300*5
#print("--- %s seconds ---" % (time.time() - start_time)) 
#pickle.dump(kmeans, open('kmeans_filtered_500.sav', 'wb')) # save model

kmeans = pickle.load(open('kmeans_filtered_500.sav', 'rb')) # load model


index_i = np.int32(sift_train[:,0])
index_j = np.int32(sift_train[:,1])
label_kmeans = kmeans.labels_
labelOneHot_kmeans = np.zeros((len(label_kmeans), k))
for i in range(len(label_kmeans)):
    labelOneHot_kmeans[i, label_kmeans[i]] = 1 # onehot encoding

train_data = np.zeros((numOfTrain * len(classes), k+1))
for i in range(len(classes)):
    for j in range(numOfTrain):
        ij = numOfTrain*i + j
        index_sample = np.where((i == index_i) & (j == index_j))
        tmp_start = index_sample[0][0]
        tmp_end = len(index_sample[0]) + tmp_start
        tmp_feature = np.sum(labelOneHot_kmeans[tmp_start:tmp_end,:], axis = 0)
        train_data[ij,1:] = tmp_feature
        train_data[ij,0] = i

# generating test data
label_kmeans_test = kmeans.predict(sift_test[:,2:])

index_i = np.int32(sift_test[:,0])
index_j = np.int32(sift_test[:,1])
label_kmeans = label_kmeans_test
labelOneHot_kmeans = np.zeros((len(label_kmeans), k))
for i in range(len(label_kmeans)):
    labelOneHot_kmeans[i, label_kmeans[i]] = 1 # onehot encoding

test_data = np.zeros((numOfTest * len(classes), k+1))
for i in range(len(classes)):
    for j in range(numOfTest):
        ij = numOfTest*i + j
        index_sample = np.where((i == index_i) & (j == index_j))
        tmp_start = index_sample[0][0]
        tmp_end = len(index_sample[0]) + tmp_start
        tmp_feature = np.sum(labelOneHot_kmeans[tmp_start:tmp_end,:], axis = 0)
        test_data[ij,1:] = tmp_feature
        test_data[ij,0] = i

#%%　svm
##　SVM
#x = [0.001,0.01,0.1,1,10,100,1000,10000, 1e5, 1e6]
x = [0.001,0.01,0.1,1,10,100,1000]
results = []
for i in x:
    X_train = scale(train_data[:,1:], axis = 0)
    y_train = train_data[:,0]
    clf = svm.SVC(C=i, cache_size=100, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
    clf.fit(X_train, y_train)
    X_test = scale(test_data[:,1:], axis = 0)
    y_test = test_data[:,0]
    pred = clf.predict(X_test)
    
    delta = (pred - y_test)
    delta_ = np.where(delta == 0)
    accuracy = np.size(delta_[0])/(numOfTest*len(classes))
    results.append(accuracy)

plt.plot(x, results,dpi=80)
plt.xscale('log')
plt.xlabel('Regularization')
plt.ylabel('Accuracy')
plt.savefig('SVM.png')



