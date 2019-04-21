
import re
import os
import sys
import tarfile

import numpy as np
import tensorflow as tf
from six.moves import urllib


class Labeler:

    DATA_URL = (
        "http://download.tensorflow.org/models/image/imagenet/"
        + "inception-2015-12-05.tgz"
    )

    def __init__(self, project_root):
        self.imagenet_location = f"{project_root}/inputs/imagenet"
        self.project_root = project_root
        self._download_and_extract_if_necessary()
        self._create_graph()
        self.node_lookup = NodeLookup(self.imagenet_location)

    def labels(self, fname, top_n_labels=1):
        labels = []
        if not tf.gfile.Exists(fname):
            tf.logging.fatal("File does not exist %s", fname)
        image_data = tf.gfile.GFile(fname, "rb").read()
        with tf.Session() as sess:
            softmax_tensor = sess.graph.get_tensor_by_name("softmax:0")
            predictions = sess.run(
                softmax_tensor, {"DecodeJpeg/contents:0": image_data}
            )
            predictions = np.squeeze(predictions)
            for node_id in predictions.argsort()[-top_n_labels:][::-1]:
                human_string = self.node_lookup.id_to_string(node_id)
                labels.append(human_string)
        return labels

    def _download_and_extract_if_necessary(self):
        if not os.path.exists(self.imagenet_location):
            os.makedirs(self.imagenet_location)
        filename = self.DATA_URL.split("/")[-1]
        filepath = os.path.join(self.imagenet_location, filename)
        if not os.path.exists(filepath):
            filepath, _ = urllib.request.urlretrieve(
                self.DATA_URL, filepath, download_progress
            )
            print()
            print(f"Successfully downloaded {filename}")
        tarfile.open(filepath, "r:gz").extractall(self.imagenet_location)

    def _create_graph(self):
        graph = f"{self.imagenet_location}/classify_image_graph_def.pb"
        with tf.gfile.GFile(graph, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name="")


class NodeLookup(object):
    """Converts integer node ID's to human readable labels."""

    def __init__(self, imagenet_location):
        label_lookup_path = os.path.join(
            imagenet_location, "imagenet_2012_challenge_label_map_proto.pbtxt"
        )
        uid_lookup_path = os.path.join(
            imagenet_location, "imagenet_synset_to_human_label_map.txt"
        )
        self.node_lookup = self.load(label_lookup_path, uid_lookup_path)

    def load(self, label_lookup_path, uid_lookup_path):
        if not tf.gfile.Exists(uid_lookup_path):
            tf.logging.fatal("File does not exist %s", uid_lookup_path)
        if not tf.gfile.Exists(label_lookup_path):
            tf.logging.fatal("File does not exist %s", label_lookup_path)

        proto_as_ascii_lines = tf.gfile.GFile(uid_lookup_path).readlines()
        uid_to_human = {}
        p = re.compile(r"[n\d]*[ \S,]*")
        for line in proto_as_ascii_lines:
            parsed_items = p.findall(line)
            uid = parsed_items[0]
            human_string = parsed_items[2]
            uid_to_human[uid] = human_string

        node_id_to_uid = {}
        proto_as_ascii = tf.gfile.GFile(label_lookup_path).readlines()
        for line in proto_as_ascii:
            if line.startswith("  target_class:"):
                target_class = int(line.split(": ")[1])
            if line.startswith("  target_class_string:"):
                target_class_string = line.split(": ")[1]
                node_id_to_uid[target_class] = target_class_string[1:-2]

        node_id_to_name = {}
        for key, val in node_id_to_uid.items():
            if val not in uid_to_human:
                tf.logging.fatal("Failed to locate: %s", val)
            name = uid_to_human[val]
            node_id_to_name[key] = name

        return node_id_to_name

    def id_to_string(self, node_id):
        if node_id not in self.node_lookup:
            return ""
        return self.node_lookup[node_id]


def download_progress(count, block_size, total_size):
    progress = float(count * block_size) / float(total_size) * 100.0
    sys.stdout.write("\r>> Downloading files: %.1f%%" % (progress))
    sys.stdout.flush()
