from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from transformers import TFAlbertPreTrainedModel
from transformers.modeling_tf_utils import get_initializer
from model_stuff.AlbertTransformer.modeling_tf_albert import TFALBertMainLayer


class TFAlbertForNaturalQuestionAnswering(TFAlbertPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.albert = TFALBertMainLayer(config)
        self.initializer = get_initializer(config.initializer_range)

        # after we have the bert embeddings we calculate the start token with a fully connected
        self.layer_1 = tf.keras.layers.Dense(512,
                                             kernel_initializer=self.initializer, activation=tf.nn.relu)
        self.layer2 = tf.keras.layers.Dense(256,
                                            kernel_initializer=self.initializer, activation=tf.nn.relu)
        self.start = tf.keras.layers.Dense(1,
                                           kernel_initializer=self.initializer, name="start")

        self.end = tf.keras.layers.Dense(1,
                                         kernel_initializer=self.initializer, name="end")

        self.type = tf.keras.layers.Dense(5, kernel_initializer=self.initializer,
                                          activation=tf.nn.softmax, name="type")

    def call(self, inputs, **kwargs):
        bert_output = self.albert(inputs)
        presoftmax = self.layer2(self.layer_1(bert_output[0]))
        # tf.print(tf.shape(presoftmax)) # [4, 512, 256]

        start_logit = self.start(presoftmax)
        # tf.print(tf.shape(start_logit)) #[4, 512, 1]
        end_logit = self.end(presoftmax)

        start = tf.math.softmax(tf.squeeze(start_logit, axis=-1), axis=1)
        end = tf.math.softmax(tf.squeeze(end_logit, axis=-1), axis=1)
        # tf.print(tf.shape(end)) #[4, 512]

        answer_type = self.type(bert_output[1])

        return start, end, answer_type
