"""
Testing script to run GQN as a tf.estimator.Estimator.
"""

import sys
import argparse
import numpy as np
import tensorflow as tf

from PIL import Image

from gqn.gqn_graph import gqn_draw
from gqn.gqn_params import create_gqn_config
from data_provider.gqn_tfr_provider import gqn_input_fn

# command line argument parser
ARGPARSER = argparse.ArgumentParser(
    description='Train a GQN as a tf.estimator.Estimator.')
# directory parameters
ARGPARSER.add_argument(
    '--data_dir', type=str, default='/tmp/data/gqn-dataset',
    help='The path to the gqn-dataset directory.')
ARGPARSER.add_argument(
    '--dataset', type=str, default='rooms_ring_camera',
    help='The name of the GQN dataset to use. \
    Available names are: \
    jaco | mazes | rooms_free_camera_no_object_rotations | \
    rooms_free_camera_with_object_rotations | rooms_ring_camera | \
    shepard_metzler_5_parts | shepard_metzler_7_parts')
ARGPARSER.add_argument(
    '--model_dir', type=str, default='/tmp/models/gqn',
    help='The directory where the model will be stored.')
ARGPARSER.add_argument(
    '--debug', default=False, action='store_true',
    help="Enables debugging mode for more verbose logging and tensorboard \
    output.")
# model parameters
ARGPARSER.add_argument(
    '--seq_length', type=int, default=8,
    help='The number of generation steps of the DRAW LSTM.')
ARGPARSER.add_argument(
    '--enc_type', type=str, default='pool',
    help='The encoding architecture type.')
# memory management
ARGPARSER.add_argument(
    '--batch_size', type=int, default=36,  # 36 reported in GQN paper -> multi-GPU?
    help='The number of data points per batch. One data point is a tuple of \
    ((query_camera_pose, [(context_frame, context_camera_pose)]), target_frame).')
ARGPARSER.add_argument(
    '--memcap', type=float, default=1.0,
    help='Maximum fraction of memory to allocate per GPU.')

def main(unparsed_argv):
  """
  Pseudo-main executed via tf.app.run().
  """

  # contains the data
  example = gqn_input_fn(
      dataset=FLAGS.dataset,
      context_size=FLAGS.seq_length,
      batch_size=FLAGS.batch_size,
      root=FLAGS.data_dir,
      mode=tf.estimator.ModeKeys.PREDICT
  )

  custom_params = {
      'ENC_TYPE' : FLAGS.enc_type,
      'CONTEXT_SIZE' : FLAGS.seq_length,
      'SEQ_LENGTH' : FLAGS.seq_length
  }
  gqn_config = create_gqn_config(custom_params)
  model_params = {
      'gqn_params' : gqn_config,
      'debug' : FLAGS.debug,
  }

  # graph definition in test mode
  net, ep_gqn = gqn_draw(
      query_pose=example[0].query_camera,
      target_frame=example[1],
      context_poses=example[0].context.cameras,
      context_frames=example[0].context.frames,
      model_params=gqn_config,
      is_training=False
  )

  saver = tf.train.Saver()
  sess = tf.Session()

  # Don't run initalisers, restore variables instead
  # sess.run(tf.global_variables_initializer())
  latest_checkpoint = tf.train.latest_checkpoint(FLAGS.model_dir)
  saver.restore(sess, latest_checkpoint)

  # Run network forward, shouldn't complain about uninitialised variables
  output, output_gt = sess.run([net, example[1]])
  for j in range(len(output)):
    pixels = np.asarray(np.multiply(output[j], 255.), dtype=np.uint8)
    pixels_gt = np.asarray(np.multiply(output_gt[j], 255.), dtype=np.uint8)
    Image.fromarray(pixels).save('testimgs/outputs{}.jpg'.format(j))
    Image.fromarray(pixels_gt).save('testimgs/output_gts{}.jpg'.format(j))

if __name__ == '__main__':
  print("Testing a GQN.")
  FLAGS, UNPARSED_ARGV = ARGPARSER.parse_known_args()
  print("FLAGS:", FLAGS)
  print("UNPARSED_ARGV:", UNPARSED_ARGV)

  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(argv=[sys.argv[0]] + UNPARSED_ARGV)
