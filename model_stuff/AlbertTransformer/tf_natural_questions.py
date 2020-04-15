from transformers import BertTokenizer
from transformers import TFBertModel, TFDistilBertModel
from transformers import TFBertMainLayer, TFDistilBertMainLayer, TFBertPreTrainedModel, TFDistilBertPreTrainedModel
from transformers.modeling_tf_utils import get_initializer


class TFNQModel:
    def __init__(self, config, *inputs, **kwargs):
        """
        Subclasses of this class are different in self.backend,
        which should be a model that outputs a tensor of shape
        (batch_size, hidden_dim), and the `backend_call()` method.

        We will use Hugging Face Bert/DistilBert as backend
        in this notebook.
        """

        self.backend = None

        self.seq_output_dropout = tf.keras.layers.Dropout(
            kwargs.get('seq_output_dropout_prob', 0.05))
        self.pooled_output_dropout = tf.keras.layers.Dropout(
            kwargs.get('pooled_output_dropout_prob', 0.05))

        self.pos_classifier = tf.keras.layers.Dense(
            2,
            kernel_initializer=get_initializer(config.initializer_range),
            name='pos_classifier')

        self.answer_type_classifier = tf.keras.layers.Dense(
            NB_ANSWER_TYPES,
            kernel_initializer=get_initializer(config.initializer_range),
            name='answer_type_classifier')

    def backend_call(self, inputs, **kwargs):
        """This method should be implemented by subclasses.

           The implementation should take into account the (somehow) different input formats of Hugging Face's
           models.

           For example, the `TFDistilBert` model, unlike `Bert` model, doesn't have segment_id as input.

           Then it calls `self.backend_call()` to get the outputs from Bert's model, which is used in self.call().
        """

        raise NotImplementedError

    def call(self, inputs, **kwargs):

        # sequence / [CLS] outputs from original bert
        sequence_output, pooled_output = self.backend_call(
            inputs, **kwargs
        )  # shape = (batch_size, seq_len, hidden_dim) / (batch_size, hidden_dim)

        # dropout
        sequence_output = self.seq_output_dropout(sequence_output,
                                                  training=kwargs.get(
                                                      'training', False))
        pooled_output = self.pooled_output_dropout(pooled_output,
                                                   training=kwargs.get(
                                                       'training', False))

        pos_logits = self.pos_classifier(
            sequence_output)  # shape = (batch_size, seq_len, 2)
        start_pos_logits = pos_logits[:, :, 0]  # shape = (batch_size, seq_len)
        end_pos_logits = pos_logits[:, :, 1]  # shape = (batch_size, seq_len)

        answer_type_logits = self.answer_type_classifier(
            pooled_output)  # shape = (batch_size, NB_ANSWER_TYPES)

        outputs = (start_pos_logits, end_pos_logits, answer_type_logits)

        return outputs  # logits


class TFBertForNQ(TFNQModel, TFBertPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):

        TFBertPreTrainedModel.__init__(
            self, config, *inputs, **kwargs)  # explicit calls without super
        TFNQModel.__init__(self, config)

        self.bert = TFBertMainLayer(config, name='bert')

    def backend_call(self, inputs, **kwargs):

        outputs = self.bert(inputs, **kwargs)
        sequence_output, pooled_output = outputs[0], outputs[
            1]  # shape = (batch_size, seq_len, hidden_dim) / (batch_size, hidden_dim)

        return sequence_output, pooled_output


class TFDistilBertForNQ(TFNQModel, TFDistilBertPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):

        TFDistilBertPreTrainedModel.__init__(
            self, config, *inputs, **kwargs)  # explicit calls without super
        TFNQModel.__init__(self, config)

        self.backend = TFDistilBertMainLayer(config, name="distilbert")

    def backend_call(self, inputs, **kwargs):

        if isinstance(inputs, tuple):
            # Distil bert has no segment_id (i.e. `token_type_ids`)
            inputs = inputs[:2]
        else:
            inputs = inputs

        outputs = self.backend(inputs, **kwargs)

        # TFDistilBertModel's output[0] is of shape (batch_size, sequence_length, hidden_size)
        # We take only for the [CLS].

        sequence_output = outputs[
            0]  # shape = (batch_size, seq_len, hidden_dim)
        pooled_output = sequence_output[:,
                                        0, :]  # shape = (batch_size, hidden_dim)

        return sequence_output, pooled_output


def get_pretrained_model(model_name):

    pretrained_path = os.path.join(FLAGS.model_dir, model_name)

    tokenizer = BertTokenizer.from_pretrained(pretrained_path)

    model_type = model_name.split("-")[0]
    if model_type not in model_mapping:
        raise ValueError("Model definition not found.")

    model_class = model_mapping[model_type]
    model = model_class.from_pretrained(pretrained_path)

    return tokenizer, model
