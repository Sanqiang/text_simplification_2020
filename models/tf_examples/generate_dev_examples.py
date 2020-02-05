import tensorflow as tf
from models.utils.control_utils import ControlMethod

flags = tf.flags

flags.DEFINE_string(
    'comp_path',
    '/Users/sanqiang/git/ts/text_simplification_data/val2/ncomp/tune.8turkers.tok.norm.ori',
    'The path for comp file.')
flags.DEFINE_string(
    'example_output_path',
    '/Users/sanqiang/git/ts/ts_2020_data/test.example',
    'The path for ppdb outputs.')
flags.DEFINE_string(
    "ppdb_file", "/Users/sanqiang/git/ts/ts_2020_data/ppdb.txt",
    "The file path of ppdb")

flags.DEFINE_string(
    "ppdb_vocab", "/Users/sanqiang/git/ts/ts_2020_data/vocab",
    "The file path of ppdb vocab generated from train")

flags.DEFINE_string(
    "control_mode", "sent_length|0.5:val:scatter_ppdb:flatten:syntax_gen:", #
    "choice of :")

FLAGS = flags.FLAGS


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


if __name__ == '__main__':
    control_obj = ControlMethod(FLAGS)

    writer = tf.python_io.TFRecordWriter(FLAGS.example_output_path)

    for line in open(FLAGS.comp_path):
        vec, extra_outputs = control_obj.get_control_vec_eval(line)
        feature = {
            'src_wds': _bytes_feature([str.encode(line)]),
            'trg_wds': _bytes_feature([str.encode("Nothing")]),
            'template_comp': _bytes_feature([str.encode(extra_outputs["template_comp"])]),
            'template_simp': _bytes_feature([str.encode("Nothing")]),
            'control_wds': _bytes_feature([str.encode(extra_outputs["external_inputs"])]),
            'control_vec': _float_feature([0.0] * 6)
        }

        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())

        print("======")

    print('Done')