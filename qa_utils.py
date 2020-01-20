def jsonl_iterator(jsonl_files, to_json=False):
    """
    Convert test examples to tf records.
    """
    for file_path in jsonl_files:
        with open(file_path, "r", encoding="UTF-8") as fp:
            for jsonl in fp:
                raw_example = jsonl
                if to_json:
                    raw_example = json.loads(jsonl)
                yield raw_example


#Example
#creator = TFExampleCreator(is_training=False)
#nq_lines = jsonl_iterator([FLAGS.predict_file])
#creator.process_nq_lines(nq_lines=nq_lines, output_tfrecord=FLAGS.test_tf_record, max_examples=0, collect_nq_features=False)


def get_dataset(tf_record_file,
                seq_length,
                batch_size=1,
                shuffle_buffer_size=0,
                is_training=False):
    """
    Given the tfrecord it returns all the features
    """
    if is_training:
        features = {
            "unique_ids": tf.io.FixedLenFeature([], tf.int64),
            "input_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
            "input_mask": tf.io.FixedLenFeature([seq_length], tf.int64),
            "segment_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
            "start_positions": tf.io.FixedLenFeature([], tf.int64),
            "end_positions": tf.io.FixedLenFeature([], tf.int64),
            "answer_types": tf.io.FixedLenFeature([], tf.int64)
        }
    else:
        features = {
            "unique_ids": tf.io.FixedLenFeature([], tf.int64),
            "input_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
            "input_mask": tf.io.FixedLenFeature([seq_length], tf.int64),
            "segment_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
            "token_map": tf.io.FixedLenFeature([seq_length], tf.int64)
        }

    # Taken from the TensorFlow models repository: https://github.com/tensorflow/models/blob/befbe0f9fe02d6bc1efb1c462689d069dae23af1/official/nlp/bert/input_pipeline.py#L24
    def decode_record(record, features):
        """Decodes a record to a TensorFlow example."""
        example = tf.io.parse_single_example(record, features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):

            t = example[name]
            if t.dtype == tf.int64:
                t = tf.cast(t, tf.int32)
            example[name] = t
        return example

    def select_data_from_record(record):

        x = {
            'unique_ids': record['unique_ids'],
            'input_ids': record['input_ids'],
            'input_mask': record['input_mask'],
            'segment_ids': record['segment_ids']
        }

        if not is_training:
            x['token_map'] = record['token_map']

        if is_training:
            y = {
                'start_positions': record['start_positions'],
                'end_positions': record['end_positions'],
                'answer_types': record['answer_types']
            }

            return (x, y)

        return x

    dataset = tf.data.TFRecordDataset(tf_record_file)

    dataset = dataset.map(lambda record: decode_record(record, features))
    dataset = dataset.map(select_data_from_record)

    if shuffle_buffer_size > 0:
        dataset = dataset.shuffle(shuffle_buffer_size)

    dataset = dataset.batch(batch_size)

    return dataset
