from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from transformers import TFBertMainLayer, TFBertPreTrainedModel
from transformers.modeling_tf_utils import get_initializer


class TFBertForNaturalQuestionAnswering(TFBertPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.bert = TFBertMainLayer(config, name='bert')
        self.initializer = get_initializer(config.initializer_range)

        # after we have the bert embeddings we calculate the start token with a fully connected
        self.layer_1 = tf.keras.layers.Dense(512,
                                             kernel_initializer=self.initializer, activation=tf.nn.relu)
        self.layer2 = tf.keras.layers.Dense(256,
                                            kernel_initializer=self.initializer, activation=tf.nn.relu)
        self.start = tf.keras.layers.Dense(1,
                                           kernel_initializer=self.initializer, name="start", activation=tf.nn.softmax)

        self.end = tf.keras.layers.Dense(1,
                                         kernel_initializer=self.initializer, name="end", activation=tf.nn.softmax)

        self.type = tf.keras.layers.Dense(5, kernel_initializer=self.initializer,
                                          activation=tf.nn.softmax, name="type")

    def call(self, inputs, **kwargs):
        bert_output = self.bert(inputs)
        presoftmax = self.layer2(self.layer_1(bert_output[0]))

        start = self.start(presoftmax)
        end = self.end(presoftmax)
        answer_type = self.type(bert_output[1])

        return start, end, answer_type
