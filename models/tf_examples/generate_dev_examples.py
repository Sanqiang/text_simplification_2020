import tensorflow as tf

flags = tf.flags

flags.DEFINE_string(
    'comp_path',
    '/Users/sanqiang/git/ts/text_simplification_data/val2/ncomp/tune.8turkers.tok.norm.ori',
    'The path for comp file.')
flags.DEFINE_string(
    'example_output_path',
    '/Users/Shared/Relocated Items/Security/zfs1/hdaqing/saz31/dataset/tmp_wikilarge/dev/tune.example',
    'The path for ppdb outputs.')

FLAGS = flags.FLAGS


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


writer = tf.python_io.TFRecordWriter(FLAGS.example_output_path)

# max_l = 0
# from language_model.gpt2 import encoder
# import os
#
# models_dir = os.path.expanduser(
#     os.path.expandvars('/Users/sanqiang/git/ts/text_simplification_2020/language_model/gpt2/models'))
# enc = encoder.get_encoder('774M', models_dir)
for line in open(FLAGS.comp_path):

    # l = len(enc.encode(line))
    # max_l = max(l, max_l)

    feature = {
        'src_wds': _bytes_feature([str.encode(line)]),
        'trg_wds': _bytes_feature([str.encode("Nothing")])}
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(example.SerializeToString())

print('Done')