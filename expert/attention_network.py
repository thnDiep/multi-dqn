from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation, Layer, LeakyReLU
from keras import backend as K
from enum import Enum

class ModelType(Enum):
    ORIGINAL = 'original'
    TIME_FRAME_ATN = 'time_frame_atn'  # TimeFrameAttention
    GLOBAL_FEATURE_ATN = 'global_feature_atn'  # GlobalFeatureAttention
    LOCAL_FEATURE_ATN = 'local_feature_atn'  # LocalFeatureAttention
    
    @classmethod
    def get_values(cls):
        return [e.value for e in cls]

class GlobalFeatureAttention(Layer):
    def __init__(self, **kwargs):
        super(GlobalFeatureAttention, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight', 
                                 shape=(input_shape[-1], input_shape[-1]),
                                 initializer='glorot_uniform',
                                 trainable=True)
        self.b = self.add_weight(name='attention_bias', 
                                 shape=(input_shape[-1],),
                                 initializer='zeros',
                                 trainable=True)
        super(GlobalFeatureAttention, self).build(input_shape)
        
    def call(self, x):
        batch_size = K.shape(x)[0]
        x_flat = K.reshape(x, (batch_size, 68))
        
        # attention scores
        e = K.tanh(K.dot(x_flat, self.W) + self.b)
        
        # softmax -> attention weights
        a = K.softmax(e, axis=-1)
        
        # context vector
        return x_flat * a
        
class LocalFeatureAttention(Layer):
    def __init__(self, **kwargs):
        super(LocalFeatureAttention, self).__init__(**kwargs)
        
    def build(self, input_shape):
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
                                     
        super(LocalFeatureAttention, self).build(input_shape)
        
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
        
        output = K.concatenate([hour_weighted, day_weighted, week_weighted], axis=-1)
        
        return output
        

class TimeFrameAttention(Layer):
    def __init__(self, **kwargs):
        super(TimeFrameAttention, self).__init__(**kwargs)
        self.attention_weights = None

    def build(self, input_shape):
        self.W_hour = self.add_weight(name='hour_weight',
                                    shape=(40, 1),
                                    initializer='glorot_uniform',
                                    trainable=True)
                                    
        self.W_day = self.add_weight(name='day_weight',
                                    shape=(20, 1),
                                    initializer='glorot_uniform',
                                    trainable=True)
                                    
        self.W_week = self.add_weight(name='week_weight',
                                    shape=(8, 1),
                                    initializer='glorot_uniform',
                                    trainable=True)
        
        # Gate weights
        self.gate_weights = self.add_weight(name='gate_weights',
                                          shape=(68, 68),
                                          initializer='glorot_uniform',
                                          trainable=True)
        
        self.b = self.add_weight(name='timeframe_bias',
                                shape=(3,),
                                initializer='zeros',
                                trainable=True)
                                
        super(TimeFrameAttention, self).build(input_shape)
        
    def call(self, x):
        batch_size = K.shape(x)[0]
        x_flat = K.reshape(x, (batch_size, 68))
        
        # Tách input thành các khung thời gian
        hour_data = x_flat[:, :40]
        day_data = x_flat[:, 40:60]
        week_data = x_flat[:, 60:]
        
        # Tính attention cho từng khung thời gian riêng biệt
        hour_score = K.dot(hour_data, self.W_hour)  # [batch_size, 1]
        day_score = K.dot(day_data, self.W_day)    # [batch_size, 1]
        week_score = K.dot(week_data, self.W_week)  # [batch_size, 1]
        
        # Ghép các score lại
        scores = K.concatenate([hour_score, day_score, week_score], axis=1)  # [batch_size, 3]
        
        # Thêm bias và tính softmax
        e = K.tanh(scores + self.b)
        a = K.softmax(e, axis=-1)  # [batch_size, 3]
        self.attention_weights = a
        
        # Áp dụng attention weights cho từng khung thời gian
        hour_weighted = hour_data * a[:, 0:1]
        day_weighted = day_data * a[:, 1:2]
        week_weighted = week_data * a[:, 2:3]
        
        # Ghép lại
        weighted_features = K.concatenate([hour_weighted, day_weighted, week_weighted], axis=-1)
        
        # Áp dụng gate mechanism
        gate = K.sigmoid(K.dot(weighted_features, self.gate_weights))
        output = gate * weighted_features
        return output

def build_model(model_name, input_shape=(1, 1, 68), nb_actions=3):
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))

    if model_name == ModelType.TIME_FRAME_ATN.value:
        model.add(TimeFrameAttention())
    elif model_name == ModelType.GLOBAL_FEATURE_ATN.value:
        model.add(GlobalFeatureAttention())
    elif model_name == ModelType.LOCAL_FEATURE_ATN.value:
        model.add(LocalFeatureAttention())
    
    model.add(Dense(35, activation='linear'))
    model.add(LeakyReLU(alpha=.001))
    model.add(Dense(nb_actions))
    model.add(Activation('linear'))

    custom_objects = {
        'TimeFrameAttention': TimeFrameAttention,
        'GlobalFeatureAttention': GlobalFeatureAttention,
        'LocalFeatureAttention': LocalFeatureAttention
    }
    
    return model, custom_objects