from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from transformers import TFAlbertPreTrainedModel
# from transformers.modeling_tf_utils import get_initializer
# from model_stuff.AlbertTransformer.modeling_tf_albert import TFALBertMainLayer
from transformers import TFAlbertMainLayer
from transformers.modeling_tf_utils import get_initializer


class TFAlbertForNaturalQuestionAnswering(TFAlbertPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
  
        self.albert = TFAlbertMainLayer(config)

        self.initializer = get_initializer(config.initializer_range)
        self.qa_outputs = tf.keras.layers.Dense(config.num_labels,
                                  kernel_initializer=self.initializer, name='qa_outputs')
        self.long_outputs = tf.keras.layers.Dense(1, kernel_initializer=self.initializer,
                                    name='long_outputs')

    def call(self, inputs, **kwargs):
        outputs = self.albert(inputs, **kwargs)
        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = tf.split(logits, 2, axis=-1)

        start_logits = tf.squeeze(start_logits, -1)
        end_logits = tf.squeeze(end_logits, -1)
        
        long_logits = tf.squeeze(self.long_outputs(sequence_output), -1)
        
        return {"start": start_logits, "end": end_logits, "long":long_logits}
