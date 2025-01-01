# channel-wise attention layer

import tensorflow as tf

# channel_wise_attention: 입력 데이터 각 층에 채널에 따른 각기 다른 가중치를 부여 -> 더 중요한 채널에 집중하도록 함
# attention: 신경망에서 입력 데이터의 중요한 부분에 집중하도록 가중치를 부여하는 메커니즘
# EEG에서 channel은 각 전극에서 얻은 측정값임

tf.compat.v1.disable_eager_execution()

def channel_wise_attention(feature_map, H, W, C, weight_decay=0.00004, scope='', reuse=None):
    with tf.compat.v1.variable_scope(scope, 'ChannelWiseAttention', reuse=reuse):  
        weight = tf.compat.v1.get_variable("weight", [C, C], dtype = tf.float32, initializer = tf.compat.v1.initializers.orthogonal, regularizer = tf.keras.regularizers.L2(weight_decay))
                                                                            
        bias = tf.compat.v1.get_variable("bias", [C], dtype = tf.float32, initializer = tf.compat.v1.initializers.zeros)
         
        transpose_feature_map = tf.transpose(tf.reduce_mean(feature_map, [1, 2], keepdims=True), perm = [0, 3, 1, 2])
        
        
        channel_wise_attention_fm = tf.matmul(tf.reshape(transpose_feature_map, [-1, C]), weight) + bias  
        
        
        channel_wise_attention_fm = tf.nn.sigmoid(channel_wise_attention_fm)


        # channel별 중요도가 반영된 가중치 텐서: channel-wise attention tensor
        attention = tf.reshape(tf.concat([channel_wise_attention_fm] * (H * W), axis=1), [-1, H, W, C])
       
        attended_fm = attention * feature_map
        return attended_fm