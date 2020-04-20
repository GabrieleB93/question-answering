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
        self.start_short = tf.keras.layers.Dense(1,
                                           kernel_initializer=self.initializer, name="start_short")

        self.end_short = tf.keras.layers.Dense(1,
                                         kernel_initializer=self.initializer, name="end_short")

        self.start_long = tf.keras.layers.Dense(1,
                                        kernel_initializer=self.initializer, name="start_long")

        self.end_long = tf.keras.layers.Dense(1,
                                         kernel_initializer=self.initializer, name="end_long")


    def call(self, inputs, **kwargs):
        bert_output = self.albert(inputs)
        presoftmax = self.layer2(self.layer_1(bert_output[0]))
        # tf.print(tf.shape(presoftmax)) # [4, 512, 256]

        start_logit_short = self.start_short(presoftmax)
        # tf.print(tf.shape(start_logit)) #[4, 512, 1]
        end_logit_short = self.end_short(presoftmax)

        start_logit_long = self.start_long(presoftmax)
        # tf.print(tf.shape(start_logit)) #[4, 512, 1]
        end_logit_long = self.end_long(presoftmax)

        start_short = tf.math.softmax(tf.squeeze(start_logit_short, axis=-1), axis=1)
        end_short = tf.math.softmax(tf.squeeze(end_logit_short, axis=-1), axis=1)

        start_long = tf.math.softmax(tf.squeeze(start_logit_long, axis=-1), axis=1)
        end_long = tf.math.softmax(tf.squeeze(end_logit_long, axis=-1), axis=1)

        # tf.print(tf.shape(end)) #[4, 512]

        #answer_type = tf.math.softmax(self.type(bert_output[1]))

        return start_logit_short, end_logit_short, start_logit_long, end_logit_long
