import tensorflow as tf
from tensorflow.keras import layers, Model

class AttentionModule(Model):
    def __init__(self, dim_hour=40, dim_day=20, dim_week=8, proj_dim=64):
        super().__init__()
        self.dim_hour = dim_hour
        self.dim_day = dim_day
        self.dim_week = dim_week

        # Projection để đưa về cùng chiều
        self.hour_proj = layers.Dense(proj_dim)
        self.day_proj = layers.Dense(proj_dim)
        self.week_proj = layers.Dense(proj_dim)

        # Attention weights learned from concatenated input
        self.att_mlp = tf.keras.Sequential([
            layers.Dense(32, activation='relu'),
            layers.Dense(3)  # logits cho 3 trọng số
        ])

        # Output layer để hỗ trợ tính loss
        self.output_fc = layers.Dense(1, activation='sigmoid')  # binary output

    def call(self, X_hour, X_day, X_week):
        # Projection
        Xh = self.hour_proj(X_hour)   # shape: (batch, proj_dim)
        Xd = self.day_proj(X_day)
        Xw = self.week_proj(X_week)

        # Attention weight học từ giá gộp
        X_all = tf.concat([X_hour, X_day, X_week], axis=-1)
        att_logits = self.att_mlp(X_all)
        att_weights = tf.nn.softmax(att_logits, axis=-1)

        # Tách ra từng w
        w_hour = tf.expand_dims(att_weights[:, 0], -1)
        w_day = tf.expand_dims(att_weights[:, 1], -1)
        w_week = tf.expand_dims(att_weights[:, 2], -1)

        # Tổ hợp lại
        X_weighted = w_hour * Xh + w_day * Xd + w_week * Xw

        # Dự đoán trend (để tính loss, không phải mục tiêu chính)
        y_pred = self.output_fc(X_weighted)

        return y_pred, att_weights