# yrqUni trans_layers调用示例（实现Transformer结构）

from trans_layers import Add, LayerNormalization
from trans_layers import MultiHeadAttention, PositionWiseFeedForward
from trans_layers import PositionEncoding

from tensorflow.keras.layers import Input, Embedding, Dense, Dropout, Concatenate, GlobalMaxPooling1D, Flatten
import tensorflow as tf

feed_forward_size_trans = 2048
max_seq_len_trans = 40
model_dim_trans = 128

input_trans = Input(shape=(max_seq_len_trans,), name='input_trans_layer')
input_trans_label = Input(shape=(max_seq_len_trans,), name='input_trans_label_layer')

x = Embedding(input_dim=5307+1,
                output_dim=128,
                weights=[emb1],
                trainable=False,
                input_length=40,
                mask_zero=True)(input_trans)

x_label = Embedding(input_dim=2+1,
                output_dim=128,
                weights=[emb1_label],
                trainable=False,
                input_length=40,
                mask_zero=True)(input_trans_label) # 非标准Transformer部分，可直接注释

encodings = PositionEncoding(model_dim_trans)(x)
encodings = Add()([x, encodings])
encodings = Add()([x_label, encodings]) # 非标准Transformer部分，可直接注释

# encodings = x
masks = tf.equal(input_trans, 0)

# (bs, 100, 128*2)
attention_out = MultiHeadAttention(4, 32)(
    [encodings, encodings, encodings, masks])

# Add & Norm
attention_out += encodings
attention_out = LayerNormalization()(attention_out)
# Feed-Forward
ff = PositionWiseFeedForward(model_dim_trans, feed_forward_size_trans)
ff_out = ff(attention_out)
# Add & Norm
ff_out += attention_out
encodings = LayerNormalization()(ff_out)
encodings = GlobalMaxPooling1D()(encodings)
encodings = Dropout(0.2)(encodings)

output_trans = Dense(5, activation='softmax', name='output_trans_layer')(encodings)