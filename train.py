from __future__ import division, print_function

import operator
import os
import time
from datetime import datetime
from pprint import pprint

import tensorflow as tf
from cascade_models import *
from utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.cuda_device

flags = tf.compat.v1.flags
FLAGS = flags.FLAGS

# Set logs
LOG_DIR = "log/"
OUTPUT_DATA_DIR = "log/output/"
ensure_dir(LOG_DIR)
ensure_dir(OUTPUT_DATA_DIR)
datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
today = datetime.today()
log_file = LOG_DIR + '%s_%s_%s_%s.log' % (FLAGS.dataset.split(
    "/")[0], str(today.year), str(today.month), str(today.day))


def predict(sess, ATT_model, input_feed):
  precisions, recalls, maps, num_samples, topk, decoder_targets =\
      sess.run([ATT_model.precision_score, ATT_model.recall_score, ATT_model.map_score, ATT_model.relevance_scores, ATT_model.topk_filter, ATT_model.decoder_targets],feed_dict = input_feed)
  return precisions, recalls, maps, num_samples.shape[0], topk, decoder_targets


with ExpLogger("DVGAE", log_file=log_file, datadir=OUTPUT_DATA_DIR) as logger:
  # log training parameters
  try:
    logger.log(FLAGS.flag_values_dict())
  except:
    logger.log(FLAGS.__flags.items())

  # Load data
  # the datasets are expected to be pre-processed in a proper format.
  # In general, the assumption that the node appearing in cascades must appear in the graph,
  # while the vice-versa may not be true.
  # So, the node indices will be created based on the graph.

  A = load_graph(FLAGS.dataset)
  if FLAGS.use_feats:
    X = load_feats(FLAGS.dataset)
  else:
    X = np.eye(A.shape[0])

  num_nodes = A.shape[0]
  layers_config = list(map(int, FLAGS.vae_layer_config.split(",")))

  if num_nodes % FLAGS.vae_batch_size == 0:
    num_batches_vae = num_nodes // FLAGS.vae_batch_size
  else:
    num_batches_vae = num_nodes // FLAGS.vae_batch_size + 1

  if FLAGS.graph_AE == 'GCN_AE':
    num_batches_vae = 1

  train_cascades, train_times = load_cascades(FLAGS.dataset, mode='train')
  val_cascades, val_times = load_cascades(FLAGS.dataset, mode='val')
  test_cascades, test_times = load_cascades(FLAGS.dataset, mode='test')

  # load with truncating based on max_seq_length.
  train_examples, train_examples_times = get_data_set(
      FLAGS.dataset,
      train_cascades,
      train_times,
      maxlen=FLAGS.max_seq_length,
      mode='train')
  val_examples, val_examples_times = get_data_set(FLAGS.dataset,
                                                  val_cascades,
                                                  val_times,
                                                  maxlen=FLAGS.max_seq_length,
                                                  mode='val')
  test_examples, test_examples_times = get_data_set(
      FLAGS.dataset,
      test_cascades,
      test_times,
      maxlen=FLAGS.max_seq_length,
      test_min_percent=FLAGS.test_min_percent,
      test_max_percent=FLAGS.test_max_percent,
      mode='test')

  print("# nodes in graph", num_nodes)
  print("# train cascades", len(train_cascades))

  print("Init models")
  CVGAE_model = CVGAE(X.shape[1], A, layers_config, mode='train', feats=X)
  ATT_model = DiffusionAttention(num_nodes + 1,
                                 train_examples,
                                 train_examples_times,
                                 val_examples,
                                 val_examples_times,
                                 test_examples,
                                 test_examples_times,
                                 logging=True,
                                 mode='feed')

  # Initialize session
  config = tf.compat.v1.ConfigProto()
  config.gpu_options.allow_growth = True
  sess = tf.compat.v1.Session(config=config)

  # Init variables
  print("Run global var initializer")
  sess.run(tf.global_variables_initializer())

  print("Starting queue runners")
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(sess=sess, coord=coord)
  z_vae_embeddings = np.zeros([num_nodes + 1, FLAGS.vae_latent_dim])

  # pre-train
  logger.log("======VAE Pre-train=======")
  ## Pre-training using simple VAE.
  for epoch in range(FLAGS.pretrain_epochs):
    losses = []
    for b in range(0, num_batches_vae):
      # Training step
      input_feed = CVGAE_model.construct_feed_dict(
          v_sender_all=z_vae_embeddings,
          v_receiver_all=z_vae_embeddings,
          pre_train=True,
          dropout=FLAGS.vae_dropout_rate)
      _, vae_embeds, indices, train_loss = sess.run([
          CVGAE_model.opt_op, CVGAE_model.z_mean, CVGAE_model.node_indices,
          CVGAE_model.vae_loss
      ], input_feed)
      z_vae_embeddings[indices] = vae_embeds
      losses.append(train_loss)
    epoch_loss = np.mean(losses)
    logger.log("Mean VAE loss at epoch: %04d %.5f" % (epoch + 1, epoch_loss))
  logger.log("Pre-training completed")

  # Initial run to get embeddings.
  logger.log("Initial run to get embeddings")
  for b in range(0, num_batches_vae):
    t = time.time()
    indices, z_val = sess.run([CVGAE_model.node_indices, CVGAE_model.z_mean])
    z_vae_embeddings[indices] = z_val
    s = time.time()


  val_loss_all = []
  sender_embeddings = np.copy(z_vae_embeddings)
  receiver_embeddings = np.copy(z_vae_embeddings)
  for epoch in range(FLAGS.epochs):
    ### TRAIN
    ### Step 1: VAE
    losses = []
    # Construct feed dictionary
    input_feed = CVGAE_model.construct_feed_dict(
        v_sender_all=sender_embeddings,
        v_receiver_all=receiver_embeddings,
        dropout=FLAGS.vae_dropout_rate)
    for b in range(0, num_batches_vae):
      # Training step
      _, vae_embeds, indices, train_loss = sess.run([
          CVGAE_model.opt_op, CVGAE_model.z_mean, CVGAE_model.node_indices,
          CVGAE_model.cvgae_loss
      ], input_feed)
      z_vae_embeddings[indices] = vae_embeds
      losses.append(train_loss)

    epoch_loss = np.mean(losses)
    logger.log("Mean VAE loss at epoch: %04d %.5f" % (epoch + 1, epoch_loss))

    ### Step 2: Cascades
    losses = []
    # Construct feed dictionary
    input_feed = ATT_model.construct_feed_dict(
        z_vae_embeddings=z_vae_embeddings)

    for b in range(0, ATT_model.num_train_batches_attention):
      # Training step
      _, train_loss = sess.run([ATT_model.opt_op, ATT_model.diffusion_loss],
                               input_feed)
      losses.append(train_loss)
    # re-assign based on updated s,r embeddings.
    sender_embeddings = sess.run(ATT_model.structure_proxy_embeddings)
    # currently not used.
    receiver_embeddings = sess.run(ATT_model.receiver_embeddings)
    epoch_loss = np.mean(losses)
    logger.log("Mean Attention loss at epoch: %04d %.5f" %
               (epoch + 1, epoch_loss))

    ### TEST
    if epoch % FLAGS.test_freq == 0:
      input_feed = CVGAE_model.construct_feed_dict(
          v_sender_all=sender_embeddings,
          v_receiver_all=receiver_embeddings,
          dropout=0.)
      for _ in range(0, num_batches_vae):
        vae_embeds, indices = sess.run(
            [CVGAE_model.z_mean, CVGAE_model.node_indices], input_feed)
        z_vae_embeddings[indices] = vae_embeds
      input_feed = ATT_model.construct_feed_dict(
          z_vae_embeddings=z_vae_embeddings, is_test=True)

      total_samples = 0
      num_eval_k = len(ATT_model.k_list)
      avg_map_scores = [0.] * num_eval_k
      avg_precision_scores = [0.] * num_eval_k
      avg_recall_scores = [0.] * num_eval_k

      all_outputs = []
      all_targets = []
      for b in range(0, ATT_model.num_test_batches_attention):
        precisions, recalls, maps, num_samples, decoder_outputs, decoder_targets = predict(
            sess, ATT_model, input_feed)
        all_outputs.append(decoder_outputs)
        all_targets.append(decoder_targets)
        avg_map_scores = list(
            map(operator.add, map(operator.mul, maps,
                                  [num_samples] * num_eval_k), avg_map_scores))
        avg_precision_scores = list(
            map(operator.add,
                map(operator.mul, precisions, [num_samples] * num_eval_k),
                avg_precision_scores))
        avg_recall_scores = list(
            map(operator.add,
                map(operator.mul, recalls, [num_samples] * num_eval_k),
                avg_recall_scores))
        total_samples += num_samples
      all_outputs = np.vstack(all_outputs)
      all_targets = np.vstack(all_targets)
      #print (avg_map_scores)
      avg_map_scores = list(
          map(operator.truediv, avg_map_scores, [total_samples] * num_eval_k))
      avg_precision_scores = list(
          map(operator.truediv, avg_precision_scores,
              [total_samples] * num_eval_k))
      avg_recall_scores = list(
          map(operator.truediv, avg_recall_scores,
              [total_samples] * num_eval_k))

      metrics = dict()
      for k in range(0, num_eval_k):
        K = ATT_model.k_list[k]
        metrics["MAP@%d" % K] = avg_map_scores[k]
        metrics["Precision@%d" % K] = avg_precision_scores[k]
        metrics["Recall@%d" % K] = avg_recall_scores[k]

      logger.update_record(avg_map_scores[0],
                          (all_outputs, all_targets, metrics))

    ### VALIDATION
    if epoch % FLAGS.val_freq == 0:
      input_feed = CVGAE_model.construct_feed_dict(
          v_sender_all=sender_embeddings,
          v_receiver_all=receiver_embeddings,
          dropout=0.)
      for b in range(0, num_batches_vae):
        vae_embeds, indices = sess.run(
            [CVGAE_model.z_mean, CVGAE_model.node_indices], input_feed)
        z_vae_embeddings[indices] = vae_embeds
      losses = []
      num_eval_k = len(ATT_model.k_list)
      input_feed = ATT_model.construct_feed_dict(
          z_vae_embeddings=z_vae_embeddings, is_val=True)
      for b in range(0, ATT_model.num_val_batches_attention):
        val_loss = \
            sess.run([ATT_model.diffusion_loss], input_feed)
        losses.append(val_loss)
      epoch_loss = np.mean(losses)
      val_loss_all.append(epoch_loss)
      logger.log("Validation Attention loss at epoch: %04d %.5f" %
                 (epoch + 1, epoch_loss))

      # early stopping
      if len(
          val_loss_all) >= FLAGS.early_stopping and val_loss_all[-1] > np.mean(
              val_loss_all[-(FLAGS.early_stopping + 1):-1]):
        logger.log("Early stopping at epoch: %04d" % (epoch + 1))
        break

  # print evaluation metrics
  outputs, targets, metrics = logger.best_data
  print("Evaluation metrics on test set:")
  pprint(metrics)
  # logger.save_data(outputs, "outputs")
  # logger.save_data(targets, "targets")

  # stop queue runners
  coord.request_stop()
  coord.join(threads)
