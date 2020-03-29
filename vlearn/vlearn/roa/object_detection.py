import os
import pdb
import sys
import cv2
import numpy as np
import tensorflow as tf
from ..vid_reader import VReader
from .utils import label_map_util

ERROR_CODE = -3


class ObjDet(VReader):
    def __init__(self, vpath, label_pbtxt, inf_graph):
        """
        Initializes an instance that uses tensorflow object detection models to 
        determine active regions in a video.
        Args:
            vpath (str): Video file path
            category_index (str): Protobuffer label file path (.pbtxt)
            inf_graph (str): Exported graph path (.pb)
        """
        # Load video
        super(ObjDet, self).__init__(vpath)
        # Load graph
        with tf.gfile.FastGFile(inf_graph, "rb",) as f:
            self.graph_def = tf.GraphDef()
            self.graph_def.ParseFromString(f.read())
        self.category_index = label_map_util.create_category_index_from_labelmap(
            label_pbtxt, use_display_name=True,
        )

    def get_roa(self, pocs, poce, skip=1):
        """
        Returns a numpy array with same resolution as video.
        Each element in this array is normalized score
        that it can be considered as active.

        Args:
            pocs (int): Starting poc
            poce (int): Ending poc. Value of -1 implies we use all frames.
            skip (int): Frames to skip. By default it is 1.
        """
        # poce to ending POC in case -1 is given
        if poce == -1:
            poce = self.nfrms - 1

        # setting video to starting POC (pocs)
        poc = pocs
        self.vro.set(cv2.CAP_PROP_POS_FRAMES, poc)

        with tf.Session() as sess:
            sess.graph.as_default()
            tf.import_graph_def(self.graph_def, name="")

            # Video loop
            roa_vid = np.zeros((self.vht, self.vwd))
            while self.vro.isOpened() and poc <= poce:
                ret, frm = self.vro.read()
                rows = frm.shape[0]
                cols = frm.shape[1]
                inp = frm[:, :, [2, 1, 0]]  # BGR2RGB

                # Run the model
                out = sess.run(
                    [
                        sess.graph.get_tensor_by_name("num_detections:0"),
                        sess.graph.get_tensor_by_name("detection_scores:0"),
                        sess.graph.get_tensor_by_name("detection_boxes:0"),
                        sess.graph.get_tensor_by_name("detection_classes:0"),
                    ],
                    feed_dict={
                        "image_tensor:0": inp.reshape(1, inp.shape[0], inp.shape[1], 3)
                    },
                )

                # create a binary image where detection happened as 1
                bin_img = np.zeros((rows, cols))
                num_detections = int(out[0][0])

                for i in range(num_detections):
                    # classId = int(out[3][0][i])
                    # className = category_index[classId]["name"]
                    score = float(out[1][0][i])
                    bbox = [float(v) for v in out[2][0][i]]
                    if score > 0.1:
                        xtl = int(bbox[1] * cols)
                        ytl = int(bbox[0] * rows)
                        xbr = int(bbox[3] * cols)
                        ybr = int(bbox[2] * rows)
                        bin_img[ytl:ybr, xtl:xbr] = 1

                roa_vid += bin_img

                poc += skip
                if skip > 1:
                    self.vro.set(cv2.CAP_PROP_POS_FRAMES, poc)

        roa_vid = cv2.normalize(roa_vid, None, 0, 1, cv2.NORM_MINMAX)
        return roa_vid
