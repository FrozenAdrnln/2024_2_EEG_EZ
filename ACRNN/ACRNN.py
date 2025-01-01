
import numpy as np
import pandas as pd
import tensorflow as tf
from cnn import cnn

from channelWiseAttention import  channel_wise_attention
from DiSAN import directional_attention_with_dense, multi_dimensional_attention

import scipy.io
from sklearn.model_selection import train_test_split


def calMinTimeStep(bulkArr):
    keys = list(bulkArr.keys())
    minTimeStep = len(bulkArr[keys[3]][0])
    
    for i in range(4, len(keys)):
        if minTimeStep > len(bulkArr[keys[i]][0]):
            minTimeStep = len(bulkArr[keys[i]][0])
    
    return minTimeStep
        

def preprocessingEEG(eegArr, minTimeStep):
    res = eegArr[0][:minTimeStep].reshape(1, eegArr[0][:minTimeStep].shape[0])
    for i in range(1, len(eegArr)):
        res = np.append(res, eegArr[i][:minTimeStep].reshape(1, eegArr[i][:minTimeStep].shape[0]), axis=0)
    
    return res


window_size = 30000 # 샘플링주파수 x 측정시간 = 128hz x 3초 분량을 추출/ SEED 1000hz x 3초
# the channel of EEG sample, DEAP:32 DREAMER:14, SEED: 62
n_channel = 62 # channel 62개 사용 (SEED Dataset)



###########################################################################
# set model parameters
###########################################################################
# kernel parameter
kernel_height_1st = 62 #DREAMER：14, 채널 수와 관련
kernel_width_1st = 80 # cnn이 한 번에 학습하는 전위데이터 포인트 개수 짧으면 미세한 변화에 민감, 길면 장기적 패턴 감지
kernel_stride = 1 # cnn 탐지가 한 번에 이동하는 거리
conv_channel_num = 40   # 이 개수만큼 필터를 사용하여 입력데이터로부터 이 개수만큼의 특징 맵을 출력
# pooling parameter
pooling_height_1st = 1      # pooling에서 채널 정보는 유지
pooling_width_1st = 75      # 값이 클수록 더 넓고 포괄적인 시간 범위로 데이터를 요약 연산 -> 즉, 연산량이 줄어들지만 시간 축의 해상도도 줄어듦
pooling_stride_1st = 10
# full connected parameter
attention_size = 512             # 최적의 값 찾기
n_hidden_state = 128             # 최적의 값 찾기
###########################################################################
# input channel
input_channel_num = 1            # 1로 설정
# input height
input_height = 62 #DREAMER：14, channel 수
# input width
input_width = 30000
# prediction class
num_labels = 3   # -1: negative, 0: neutral, 1: positive
###########################################################################
# set training parameters
###########################################################################
# step length
num_timestep = 1
# set learning rate
learning_rate = 1e-4
# set maximum traing epochs
training_epochs = 200
# set batch size
#batch_size = -1
# set dropout probability
dropout_prob = 0.5
# instance cnn class
padding = 'VALID'
cnn_2d = cnn(padding=padding)

################################## The ACRNN Model #########################################
# input placeholder

tf.compat.v1.disable_eager_execution()

X = tf.compat.v1.placeholder(tf.float32, shape=[None, input_height, input_width, input_channel_num], name = 'X')
Y = tf.compat.v1.placeholder(tf.float32, shape=[None, num_labels], name = 'Y')
train_phase = tf.compat.v1.placeholder(tf.bool, name = 'train_phase')
keep_prob = tf.compat.v1.placeholder(tf.float32, name='keep_prob')

# channel-wise attention layer
X_1 = tf.transpose(X,[0, 3, 2, 1])

conv = channel_wise_attention(X_1, 1, window_size, n_channel, weight_decay=0.00004, scope='', reuse=None)

conv_1 = tf.transpose(conv,[0, 3, 2, 1])


# CNN layer: 한 층만 사용 (다층으로 사용해볼 수 있지 않을까????)
conv_1 = cnn_2d.apply_conv2d(conv_1, kernel_height_1st, kernel_width_1st, input_channel_num, conv_channel_num, kernel_stride, train_phase)
print("conv 1 shape: ", conv_1.get_shape().as_list())
pool_1 = cnn_2d.apply_max_pooling(conv_1, pooling_height_1st, pooling_width_1st, pooling_stride_1st)
print("pool 1 shape: ", pool_1.get_shape().as_list())
pool_1_shape = pool_1.get_shape().as_list()
pool1_flat = tf.reshape(pool_1, [-1, pool_1_shape[1]*pool_1_shape[2]*pool_1_shape[3]])   #풀링된 특징 맵을 1D 벡터로 변환
fc_drop = tf.nn.dropout(pool1_flat, keep_prob)   # 과적합 방지

# LSTMs layer: cnn layer의 출력을 LSTM layer에 입력해 시계열 데이터의 시간적 패턴을 학습
lstm_in = tf.reshape(fc_drop, [-1, num_timestep, pool_1_shape[1]*pool_1_shape[2]*pool_1_shape[3]])
cells = []
for _ in range(2): # 총 2개의 LSTM cell을 생성
	cell = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(n_hidden_state, forget_bias=1.0, state_is_tuple=True)   # LSTM cell 형성
	cell = tf.compat.v1.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)  # 과적합 방지
	cells.append(cell)
 
	
lstm_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell(cells)   # 다층 LSTM 구조 생성

batch_size = tf.shape(X)[0]
init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32) # 다층 LSTM 구조 초기 상태 형성

# output (run_op) ==> [batch, step, n_hidden_state]
# states: 각 LSTM 셀의 최종 은닉층 상태와 셀 상태를 포함
rnn_op, states = tf.compat.v1.nn.dynamic_rnn(lstm_cell, lstm_in, initial_state=init_state, time_major=False) # LSTM 구조에 CNN input 넣고 출력

#self-attention layer
with tf.name_scope('Attention_layer'):
	attention_op = multi_dimensional_attention(rnn_op, 64, scope=None, keep_prob=1., is_train=None, wd=0., activation='elu', tensor_dict=None, name=None)    # LSTM 결과물에 self-attention 적용

	attention_drop = tf.nn.dropout(attention_op, keep_prob)   # 과적합 방지

	y_ = cnn_2d.apply_readout(attention_drop, rnn_op.shape[2], num_labels)   # 완전연결층에 연결

# softmax layer: probability prediction
y_prob = tf.nn.softmax(y_, name = "y_prob")

# class prediction
y_pred = tf.argmax(y_prob, 1, name = "y_pred") # 확률이 젤 높은 애의 클래스를 반환


# Backpropagation

# cross entropy cost function
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=Y), name = 'loss')
update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
	# set training SGD optimizer
	optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(cost)

# get correctly predicted object
correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(y_), 1), tf.argmax(Y, 1))

# calculate prediction accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name = 'accuracy')

##########################################################################################

############################Experiments on database##############################


mat_file_name = "C:/Users/SAMSUNG/visual/Preprocessed_EEG/1_20131027.mat"
mat_file = scipy.io.loadmat(mat_file_name)

mat_keys = list(mat_file.keys())
minTimeStep = 30000

x = preprocessingEEG(mat_file[mat_keys[3]], minTimeStep)
x = x.reshape(1, x.shape[0], x.shape[1])


for i in range(4, len(mat_keys)):
    newArr = mat_file[mat_keys[i]]
    newArr = preprocessingEEG(newArr, minTimeStep)
    newArr = newArr.reshape(1, newArr.shape[0], newArr.shape[1])
    x = np.append(x, newArr, axis=0)

mat_file_name2 = "C:/Users/SAMSUNG/visual/Preprocessed_EEG/2_20140404.mat"
mat_file2 = scipy.io.loadmat(mat_file_name2)
mat_keys2 = list(mat_file2.keys())

for i in range(3, len(mat_keys2)):
    newArr = mat_file2[mat_keys2[i]]
    newArr = preprocessingEEG(newArr, minTimeStep)
    newArr = newArr.reshape(1, newArr.shape[0], newArr.shape[1])
    x = np.append(x, newArr, axis=0)

mat_file_name3 = "C:/Users/SAMSUNG/visual/Preprocessed_EEG/3_20140603.mat"
mat_file3 = scipy.io.loadmat(mat_file_name3)
mat_keys3 = list(mat_file3.keys())

for i in range(3, len(mat_keys3)):
    newArr = mat_file3[mat_keys3[i]]
    newArr = preprocessingEEG(newArr, minTimeStep)
    newArr = newArr.reshape(1, newArr.shape[0], newArr.shape[1])
    x = np.append(x, newArr, axis=0)

mat_file_name4 = "C:/Users/SAMSUNG/visual/Preprocessed_EEG/1_20131030.mat"
mat_file4 = scipy.io.loadmat(mat_file_name4)
mat_keys4 = list(mat_file4.keys())

for i in range(3, len(mat_keys4)):
    newArr = mat_file4[mat_keys4[i]]
    newArr = preprocessingEEG(newArr, minTimeStep)
    newArr = newArr.reshape(1, newArr.shape[0], newArr.shape[1])
    x = np.append(x, newArr, axis=0)

x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 1)

x = (x - np.mean(x, axis=1, keepdims=True)) / np.std(x, axis=1, keepdims=True)

labels = [0, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0, 0, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0, 0, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0, 0, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0]
num_unique_label = len(np.unique(labels))
identity_matrix = np.eye(num_unique_label)
one_hot_encoded_y = identity_matrix[labels]

X_train, X_test, y_train, y_test = train_test_split(x, one_hot_encoded_y, test_size=0.8)


epochs = 1

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    
    for epoch in range(epochs):
        sess.run(optimizer, feed_dict={X: X_train, Y: y_train, train_phase: True, keep_prob: 0.5})
    
    acc = sess.run(accuracy, feed_dict={X: X_test, Y: y_test, train_phase: False, keep_prob: 1.0})
    
    print(acc)