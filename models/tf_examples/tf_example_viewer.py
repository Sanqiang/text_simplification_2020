import tensorflow as tf
from google.protobuf.json_format import MessageToDict
import base64

flags = tf.flags


flags.DEFINE_string(
    'example_path',
    '/Users/sanqiang/git/ts/text_simplification_data/example_v7_val/shard_wikilarge_1976.example',
    'The path for examples.')

FLAGS = flags.FLAGS

if __name__ == '__main__':
    for example in tf.python_io.tf_record_iterator(FLAGS.example_path):
        obj = MessageToDict(tf.train.Example.FromString(example))

        output = ""
        for feature_name in obj["features"]["feature"]:
            for field in obj["features"]["feature"][feature_name]:
                if field == "bytesList":
                    val = base64.b64decode(obj["features"]["feature"][feature_name][field]["value"][0])
                else:
                    val = obj["features"]["feature"][feature_name][field]["value"]
                # if feature_name in ("trg_wds", "src_wds"):
                output += "%s:\t\t\t%s\n" % (feature_name, val)
        print(output)
        print("======")


