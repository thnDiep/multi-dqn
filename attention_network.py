import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation, Layer, LeakyReLU
from keras import backend as K
from enum import Enum

class ModelType(Enum):
    TIME_FRAME_ATN = 'time_frame_atn'  # TimeFrameAttention
    SIMPLE_ATN = 'simple_atn'        # SimpleAttention
    ORIGINAL = 'original'
    
    @classmethod
    def get_values(cls):
        return [e.value for e in cls]

class SimpleAttention(Layer):
    def __init__(self, **kwargs):
        super(SimpleAttention, self).__init__(**kwargs)
        
    def build(self, input_shape):
        # Khởi tạo weights cho attention
        self.W = self.add_weight(name='attention_weight', 
                                 shape=(input_shape[-1], input_shape[-1]),
                                 initializer='glorot_uniform',
                                 trainable=True)
        self.b = self.add_weight(name='attention_bias', 
                                 shape=(input_shape[-1],),
                                 initializer='zeros',
                                 trainable=True)
        super(SimpleAttention, self).build(input_shape)
        
    def call(self, x):
        batch_size = K.shape(x)[0]
        x_flat = K.reshape(x, (batch_size, 68))
        
        # attention scores
        e = K.tanh(K.dot(x_flat, self.W) + self.b)
        
        # softmax -> attention weights
        a = K.softmax(e, axis=-1)
        
        # context vector
        return x_flat * a
        
    def compute_output_shape(self, input_shape):
        return (input_shape[0], 68)
    
        
class TimeFrameAttention(Layer):
    def __init__(self, **kwargs):
        super(TimeFrameAttention, self).__init__(**kwargs)
        
    def build(self, input_shape):
        # Khởi tạo các weights cho từng khung thời gian
        self.W_hour = self.add_weight(name='hour_weight', 
                                     shape=(40, 40),
                                     initializer='glorot_uniform',
                                     trainable=True)
        
        self.W_day = self.add_weight(name='day_weight', 
                                    shape=(20, 20),
                                    initializer='glorot_uniform',
                                    trainable=True)
        
        self.W_week = self.add_weight(name='week_weight', 
                                     shape=(8, 8),
                                     initializer='glorot_uniform',
                                     trainable=True)
        
        # Bias cho từng khung thời gian
        self.b_hour = self.add_weight(name='hour_bias', 
                                     shape=(40,),
                                     initializer='zeros',
                                     trainable=True)
        
        self.b_day = self.add_weight(name='day_bias', 
                                    shape=(20,),
                                    initializer='zeros',
                                    trainable=True)
        
        self.b_week = self.add_weight(name='week_bias', 
                                     shape=(8,),
                                     initializer='zeros',
                                     trainable=True)
                                     
        super(TimeFrameAttention, self).build(input_shape)
        
    def call(self, x):
        # Tách input thành các khung thời gian
        batch_size = K.shape(x)[0]
        x_flat = K.reshape(x, (batch_size, 68))
        
        hour_data = x_flat[:, :40]
        day_data = x_flat[:, 40:60]
        week_data = x_flat[:, 60:]
        
        # Tính attention cho từng khung thời gian
        hour_att = K.softmax(K.tanh(K.dot(hour_data, self.W_hour) + self.b_hour), axis=-1)
        day_att = K.softmax(K.tanh(K.dot(day_data, self.W_day) + self.b_day), axis=-1)
        week_att = K.softmax(K.tanh(K.dot(week_data, self.W_week) + self.b_week), axis=-1)
        
        # Áp dụng attention weights
        hour_weighted = hour_data * hour_att
        day_weighted = day_data * day_att
        week_weighted = week_data * week_att
        
        # Ghép lại
        output = K.concatenate([hour_weighted, day_weighted, week_weighted], axis=-1)
        
        return output
        
    def compute_output_shape(self, input_shape):
        return (input_shape[0], 68)
    

def build_model(model_name, input_shape=(1, 1, 68), nb_actions=3):
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))

    if model_name == ModelType.TIME_FRAME_ATN.value:
        model.add(TimeFrameAttention())
    elif model_name == ModelType.SIMPLE_ATN.value:
        model.add(SimpleAttention())
    
    # Thêm các layer tiếp theo
    model.add(Dense(35, activation='linear'))
    model.add(LeakyReLU(alpha=.001))
    model.add(Dense(nb_actions))
    model.add(Activation('linear'))

    custom_objects = {
        'TimeFrameAttention': TimeFrameAttention,
        'SimpleAttention': SimpleAttention
    }
    
    return model, custom_objects