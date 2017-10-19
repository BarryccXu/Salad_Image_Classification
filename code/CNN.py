import numpy as np
import cv2
import skimage.io as io
import tensorflow as tf
import matplotlib.pyplot as plt
import math

#%% data
## parameters
classes = ['beet_salad', 'caesar_salad', 'caprese_salad', 'greek_salad', 'seaweed_salad']
#classes = ['beet_salad','caesar_salad']
## 300 for training; 100 for test
numOfTrain = 300; numOfTest = 100;
np.random.seed(3366)
index = np.arange(1000)
np.random.shuffle(index)
img_size = 128;
## preprocessing
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

X_train = np.zeros((numOfTrain*len(classes),img_size*img_size*3))
y_train = np.zeros(numOfTrain*len(classes), dtype = 'int8')
X_test = np.zeros((numOfTest*len(classes),img_size*img_size*3))
y_test = np.zeros(numOfTest*len(classes), dtype = 'int8')
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
        #white balance (gray world assumption)
        img = balanceWhite(img)
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        # equalize the histogram of the Y channel
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
        # convert the YUV image back to RGB format
        img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        img_flatted = img_output.flatten()
        # training and test dataset
        if (j < numOfTrain):
            X_train[j + numOfTrain*i] = img_flatted
            y_train[j + numOfTrain*i] = i
            #dir_save = '../data/preprocessed_' + np.str(img_size) + '/' + 'trainRGB_filtered/' \
            #           + np.str(i) + '_' + np.str(j) + '.jpg'
        else: 
            X_test[j-numOfTrain + numOfTest*i] = img_flatted
            y_test[j-numOfTrain + numOfTest*i] = i
            #dir_save = '../data/preprocessed_' + np.str(img_size) + '/' + 'testRGB_filtered/' \
            #           + np.str(i) + '_' + np.str(j) + '.jpg'
        #cv2.imwrite(dir_save,img_output)

# convert labels to one_hot
# shuffling labels
index_train = np.arange(numOfTrain*len(classes))
np.random.shuffle(index_train)
index_test = np.arange(numOfTest*len(classes))
np.random.shuffle(index_test)
X_train_shuffled = np.zeros((numOfTrain*len(classes),img_size*img_size*3))
X_test_shuffled = np.zeros((numOfTest*len(classes),img_size*img_size*3))
y_train_oneHot = np.zeros((numOfTrain*len(classes), len(classes)))
y_test_oneHot = np.zeros((numOfTest*len(classes), len(classes)))
#y_train_oneHot_ = np.zeros((numOfTrain*len(classes), len(classes)))


for i in range(numOfTrain*len(classes)):
    ii = index_train[i]
    X_train_shuffled[i,:] = X_train[ii,:]
    y_train_oneHot[i,y_train[ii]] = 1
#    y_train_oneHot_[i,y_train[i]] = 1
for i in range(numOfTest*len(classes)):
    ii = index_test[i]
    X_test_shuffled[i,:] = X_test[ii,:]
    y_test_oneHot[i,y_test[ii]] = 1
    
del X_train, X_test, y_train, y_test

## normalization
#X_train_norm = np.zeros((numOfTrain*len(classes),img_size*img_size*3))
#X_test_norm = np.zeros((numOfTest*len(classes),img_size*img_size*3))
#mean_128 = np.mean(X_test_shuffled, axis = 0)
#for i in range(numOfTrain*len(classes)):
   #X_train_norm[i,:] = (X_train_shuffled[i,:] - 255/2)/2
#X_train_norm = (X_train_shuffled )/255
#X_test_norm = (X_test_shuffled )/255

#%% MODEL
# parameters
n_classes = len(classes)
batch_size = 10

def compute_accuracy(v_xs, v_ys, keep_prob):
    global prediction
    y_pre = sess.run(prediction, feed_dict={Xs: v_xs, keep_prob: keep_prob})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={Xs: v_xs, ys: v_ys, keep_prob: keep_prob})
    return result

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.01, dtype=tf.float32)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape, dtype=tf.float32)
    return tf.Variable(initial)

def conv_3d(x,W):
    # strides [1,x,y,z,1]
    return tf.nn.conv3d(x, W, strides = [1,1,1,1,1], padding = 'SAME')

def conv_2d(x,W):
    return tf.nn.conv2d(x, W, strides = [1,1,1,1], padding = 'SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

def avg_pool_2x2(x):
    return tf.nn.avg_pool(x, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

def max_pool_4x4(x):
    return tf.nn.max_pool(x, ksize = [1,4,4,1], strides = [1,4,4,1], padding = 'SAME')

def norm(x):
     return tf.nn.lrn(x, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
#%%
tf.reset_default_graph()
Xs = tf.placeholder(tf.float32, [batch_size, img_size*img_size*3])
ys = tf.placeholder(tf.float32, [batch_size, n_classes])
# ys = tf.one_hot(ys, depth = n_classes)
X_img = tf.reshape(Xs, [batch_size,128,128,3])
keep_prob = tf.placeholder(tf.float32)
## conv1 layer ##
W_conv1 = weight_variable([1,1,3,16])
b_conv1 = bias_variable([16])
h_conv1 = tf.nn.relu(conv_2d(X_img, W_conv1) + b_conv1) # output size 128x128x16
#h_conv1 = tf.nn.dropout(h_conv1, keep_prob)
h_pool1 = max_pool_2x2(h_conv1)                         # output size 64x64x16
h_norm1 = norm(h_pool1)

## conv2 layer ##
W_conv2 = weight_variable([5,5,16,32])
b_conv2 = bias_variable([32])
h_conv2 = tf.nn.relu(conv_2d(h_norm1, W_conv2) + b_conv2) # output size 64x64x32
#h_conv2 = tf.nn.dropout(h_conv2, keep_prob)
h_pool2 = avg_pool_2x2(h_conv2)                         # output size 32x32x32
h_norm2 = norm(h_pool2)

## conv3 layer ##
W_conv3 = weight_variable([3,3,32,64])
b_conv3 = bias_variable([64])
h_conv3 = tf.nn.relu(conv_2d(h_norm2, W_conv3) + b_conv3) # output size 32x32x64
h_pool3 = avg_pool_2x2(h_conv3)                         # output size 16x16x64
h_norm3 = norm(h_pool3)

## conv4 layer ##
W_conv4 = weight_variable([3,3,64,128])
b_conv4 = bias_variable([128])
h_conv4 = tf.nn.relu(conv_2d(h_norm3, W_conv4) + b_conv4) # output size 16x16x128
h_pool4 = max_pool_2x2(h_conv4)                         # output size 8x8x128
h_norm4 = norm(h_pool4)

## func1 layer ##
W_func1 = weight_variable([8*8*128,2048])
b_func1 = bias_variable([2048])
h_norm4_flat = tf.reshape(h_norm4,[batch_size,8*8*128])
h_func1 = tf.nn.relu(tf.matmul(h_norm4_flat, W_func1) + b_func1)
h_func1_drop = tf.nn.dropout(h_func1, keep_prob)

## func2 layer ##
W_func2 = weight_variable([2048,256])
b_func2 = bias_variable([256])
h_func2 = tf.nn.relu(tf.matmul(h_func1_drop, W_func2) + b_func2)
h_func2_drop = tf.nn.dropout(h_func2, keep_prob)

## func3 layer ##
W_func3 = weight_variable([256, n_classes])
b_func3 = bias_variable([n_classes])
prediction = tf.nn.softmax(tf.matmul(h_func2_drop, W_func3) + b_func3)
correct_prediction = tf.equal(tf.argmax(prediction,1), tf.argmax(ys,1))
Accuracy_tf = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
## loss function ##
#cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices = [1]))
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, ys))
# cross_entropy = tf.reduce_mean(tf.abs(ys-prediction))
#cross_entropy = tf.nn.l2_loss(prediction - ys)
#cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, ys))
#train_step = tf.train.AdamOptimizer(1e-2).minimize(cross_entropy)
optimizer = tf.train.AdamOptimizer(0.0001)
train_step = optimizer.minimize(cross_entropy)

#%%　running
## initial session ##
sess = tf.Session()
sess.run(tf.global_variables_initializer())
with tf.device('/gpu:0'):
    losses_epoc = []; accuracy_epoc_train = []; accuracy_epoc_test = []
    for epoc in range(50):
        losses = []; accuracys_train = []; accuracys_test = []
        for i in range(np.int32(numOfTrain*n_classes/batch_size)):
            i_start = i * batch_size
            i_end = (i+1) * batch_size 
            loss, _ = sess.run([cross_entropy, train_step], 
                              {Xs: X_train_shuffled[i_start:i_end,:], 
                               ys: y_train_oneHot[i_start:i_end,:], keep_prob: 0.7})
            accuracy_train = sess.run(Accuracy_tf, 
                               {Xs: X_train_shuffled[i_start:i_end,:], 
                               ys: y_train_oneHot[i_start:i_end,:], keep_prob: 0.7})
            if i < np.int32(numOfTest*n_classes/batch_size):
                accuracy_test = sess.run(Accuracy_tf, 
                               {Xs: X_test_shuffled[i_start:i_end,:], 
                               ys: y_test_oneHot[i_start:i_end,:], keep_prob: 0.7})

            losses.append(loss)
            accuracys_train.append(accuracy_train)
            accuracys_test.append(accuracy_test)
        print(epoc)
        print('loss:', np.mean(losses))
        losses_epoc.append(np.mean(losses))
        print('accuracy_epoc_train:', np.mean(accuracys_train))
        accuracy_epoc_train.append(np.mean(accuracys_train))
        print('accuracy_epoc_test:', np.mean(accuracys_test))
        accuracy_epoc_test.append(np.mean(accuracys_test))

# running on test data
preds=[]
for i in range(np.int32(numOfTest*n_classes/batch_size)):
    i_start = i * batch_size
    i_end = (i+1) * batch_size 
    pred = sess.run(prediction, {Xs:X_test_shuffled[i_start:i_end,:], ys: y_test_oneHot[i_start:i_end,:], keep_prob: 0.7})
    tmp = np.argmax(pred, axis = 1)
    preds.extend(tmp)
    
delta = preds - np.argmax(y_test_oneHot, axis = 1)
delta_ = np.where(delta == 0)
accuracy = np.size(delta_[0])/(numOfTest*len(classes))

plt.show()
plt.plot(losses_epoc)
plt.xlabel('Epoch')
plt.ylabel('Cross Entropy')

plt.plot(accuracy_epoc_train, 'r')
plt.plot(accuracy_epoc_test, 'b')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Test'], loc = 2)

from sklearn.metrics import confusion_matrix
label = np.argmax(y_test_oneHot, axis = 1)
preds = np.transpose(preds)
confusMat = confusion_matrix(label, preds)
fig = plt.figure()
plt.imshow(confusMat, interpolation= 'None')
plt.colorbar()

#%% plot 
def plot_conv_weights(weights, input_channel=0):
    w = sess.run(weights)
    w_min = np.min(w)
    w_max = np.max(w)
    num_filters = w.shape[3]
    num_grids = math.ceil(math.sqrt(num_filters))
    fig, axes = plt.subplots(num_grids, num_grids)
    for i, ax in enumerate(axes.flat):
        if i<num_filters:
            img = w[:, :, input_channel, i]
            ax.imshow(img, vmin=w_min, vmax=w_max,
                      interpolation='nearest', cmap='seismic')
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()

def plot_conv_layer(layer, image):
    feed_dict = {Xs: image}
    values = sess.run(layer, feed_dict=feed_dict)
    num_filters = values.shape[3]
    num_grids = math.ceil(math.sqrt(num_filters))
    fig, axes = plt.subplots(num_grids, num_grids)
    for i, ax in enumerate(axes.flat):
        if i<num_filters:
            img = values[0, :, :, i]
            ax.imshow(img, interpolation='nearest', cmap='binary')
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()
    
plot_conv_weights(weights=W_conv3)
plot_conv_layer(W_conv1, X_train_shuffled[0:10,:])


plt.show()        
pred = sess.run(prediction, {Xs:X_test_shuffled[10:20], ys: y_test_oneHot[10:20], keep_prob: 0.6})
np.argmax(pred, axis = 1)
np.argmax(y_test_oneHot[0:10], axis = 1)
a = pred[5,:,:,5]

z = np.unique(a)





