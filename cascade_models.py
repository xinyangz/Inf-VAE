from __future__ import division, print_function

import math

import tensorflow as tf
from eval_metrics import *
from graph_ae import GCN_AE, MLP_AE
from models import Model
from utils import *

flags = tf.compat.v1.flags
FLAGS = flags.FLAGS


class DiffusionAttention(Model):

  def __init__(self,
               num_nodes,
               train_examples,
               train_examples_times,
               val_examples,
               val_examples_times,
               test_examples,
               test_examples_times,
               mode='feed',
               **kwargs):
    super(DiffusionAttention, self).__init__(**kwargs)
    self.k_list = [10, 50, 100]  # for evaluation.
    ### Prepare train, test, and val examples -- use max_seq_length --
    train_examples, train_lengths, train_targets, train_masks, train_examples_times, train_targets_times = prepare_sequences(
        train_examples,
        train_examples_times,
        maxlen=FLAGS.max_seq_length,
        mode='train')
    val_examples, val_lengths, val_targets, val_masks, val_examples_times, val_targets_times = prepare_sequences(
        val_examples,
        val_examples_times,
        maxlen=FLAGS.max_seq_length,
        mode='val')
    test_examples, test_lengths, test_targets, test_masks, test_examples_times, test_targets_times = prepare_sequences(
        test_examples,
        test_examples_times,
        maxlen=FLAGS.max_seq_length,
        mode='test')

    num_train_examples = len(train_examples)
    num_val_examples = len(val_examples)
    num_test_examples = len(test_examples)

    self.num_train_batches_attention = num_train_examples // FLAGS.attention_batch_size
    if num_train_examples % FLAGS.attention_batch_size != 0:
      self.num_train_batches_attention += 1

    self.num_val_batches_attention = num_val_examples // FLAGS.attention_batch_size
    if num_val_examples % FLAGS.attention_batch_size != 0:
      self.num_val_batches_attention += 1

    self.num_test_batches_attention = num_test_examples // FLAGS.attention_batch_size
    if num_test_examples % FLAGS.attention_batch_size != 0:
      self.num_test_batches_attention += 1

    self.lambda_s = FLAGS.lambda_s
    self.lambda_r = FLAGS.lambda_r

    self.lambda_a = FLAGS.lambda_a
    self.embedding_size = FLAGS.vae_latent_dim
    self.mode = mode

    self.decoder_inputs_train = train_examples
    self.decoder_targets_train = train_targets
    self.decoder_inputs_length_train = train_lengths
    self.decoder_masks_train = train_masks
    self.decoder_inputs_train_times = train_examples_times
    self.decoder_targets_train_times = train_targets_times

    self.decoder_inputs_val = val_examples
    self.decoder_targets_val = val_targets
    self.decoder_inputs_length_val = val_lengths
    self.decoder_masks_val = val_masks
    self.decoder_inputs_val_times = val_examples_times
    self.decoder_targets_val_times = val_targets_times

    self.decoder_inputs_test = test_examples
    self.decoder_targets_test = test_targets
    self.decoder_inputs_length_test = test_lengths
    self.decoder_masks_test = test_masks
    self.decoder_inputs_test_times = test_examples_times
    self.decoder_targets_test_times = test_targets_times

    self.num_nodes = num_nodes
    self.init_placeholders()
    self.create_batch_queues()
    self.build()

  def get_init_feec_dict(self, z_vae_embeddings):
    '''
    Feed dict:
            1) Encoder and Decoder inputs : [batch_size, max_time_steps]
            2) Encoder and Decoder inputs length : [batch_size]
            3) Node embeddings for the nodes in the current batch : [num_nodes, embedding_dim] (Passed only once to init)
    '''
    input_feed = {}
    input_feed[self.z_vae_embeddings.name] = z_vae_embeddings
    return input_feed

  def construct_feed_dict(self,
                          z_vae_embeddings=None,
                          is_val=False,
                          is_test=False):
    input_feed = {}
    input_feed[self.is_val.name] = is_val
    input_feed[self.is_test.name] = is_test
    if z_vae_embeddings is not None:
      input_feed[self.z_vae_embeddings.name] = z_vae_embeddings
    return input_feed

  # Create batches of (sequence, target) pairs.
  def create_batch_queues(self):
    num_threads = FLAGS.batch_queue_threads
    ## Feed train/val/test based on is_val/is_test flag.
    [decoder_input_train, decoder_length_train, decoder_target_train, decoder_masks_train, decoder_input_train_times, decoder_target_train_times] = \
    tf.train.slice_input_producer([self.decoder_inputs_train, self.decoder_inputs_length_train, \
    self.decoder_targets_train, self.decoder_masks_train, self.decoder_inputs_train_times, self.decoder_targets_train_times], shuffle=True, capacity=FLAGS.attention_batch_size)

    [decoder_input_val, decoder_length_val, decoder_target_val, decoder_masks_val, decoder_input_val_times, decoder_target_val_times] = \
    tf.train.slice_input_producer([self.decoder_inputs_val, self.decoder_inputs_length_val, \
    self.decoder_targets_val, self.decoder_masks_val, self.decoder_inputs_val_times, self.decoder_targets_val_times], shuffle=False, capacity=FLAGS.attention_batch_size)

    [decoder_input_test, decoder_length_test, decoder_target_test, decoder_masks_test, decoder_input_test_times, decoder_target_test_times] = \
    tf.train.slice_input_producer([self.decoder_inputs_test, self.decoder_inputs_length_test, \
    self.decoder_targets_test, self.decoder_masks_test, self.decoder_inputs_test_times, self.decoder_targets_test_times], shuffle=False, capacity=FLAGS.attention_batch_size)

    min_after_dequeue = FLAGS.attention_batch_size
    capacity = min_after_dequeue + (num_threads +
                                    1) * FLAGS.attention_batch_size

    train_queue = tf.queue.RandomShuffleQueue(capacity=capacity, dtypes=[tf.int32]*6, \
    shapes=[decoder_input_train.get_shape(), decoder_length_train.get_shape(), \
    decoder_target_train.get_shape(), decoder_masks_train.get_shape(), decoder_input_train_times.get_shape(), decoder_target_train_times.get_shape()], min_after_dequeue= min_after_dequeue)

    val_queue = tf.queue.PaddingFIFOQueue(capacity=FLAGS.attention_batch_size, dtypes=[tf.int32]*6, \
    shapes=[decoder_input_val.get_shape(), decoder_length_val.get_shape(), \
    decoder_target_val.get_shape(), decoder_masks_val.get_shape(), decoder_input_val_times.get_shape(), decoder_target_val_times.get_shape()])

    test_queue = tf.queue.PaddingFIFOQueue(capacity=FLAGS.attention_batch_size, dtypes=[tf.int32]*6, \
    shapes=[decoder_input_test.get_shape(), decoder_length_test.get_shape(), \
    decoder_target_test.get_shape(), decoder_masks_test.get_shape(), decoder_input_test_times.get_shape(), decoder_target_test_times.get_shape()])

    train_enqueue_ops = [
        train_queue.enqueue([
            decoder_input_train, decoder_length_train, decoder_target_train,
            decoder_masks_train, decoder_input_train_times,
            decoder_target_train_times
        ])
    ] * num_threads
    qr_train = tf.train.QueueRunner(train_queue, train_enqueue_ops)

    val_enqueue_ops = [
        val_queue.enqueue([
            decoder_input_val, decoder_length_val, decoder_target_val,
            decoder_masks_val, decoder_input_val_times, decoder_target_val_times
        ])
    ] * num_threads
    qr_val = tf.train.QueueRunner(val_queue, val_enqueue_ops)

    test_enqueue_ops = [
        test_queue.enqueue([
            decoder_input_test, decoder_length_test, decoder_target_test,
            decoder_masks_test, decoder_input_test_times,
            decoder_target_test_times
        ])
    ] * num_threads
    qr_test = tf.train.QueueRunner(test_queue, test_enqueue_ops)

    tf.train.add_queue_runner(qr_train)
    tf.train.add_queue_runner(qr_val)
    tf.train.add_queue_runner(qr_test)
    # data_batch = tf.cond(self.is_train, lambda: train_queue.dequeue_many(FLAGS.attention_batch_size), lambda: test_queue.dequeue_many(FLAGS.attention_batch_size))
    data_batch = tf.case(
        {
            self.is_val:
                lambda: val_queue.dequeue_many(FLAGS.attention_batch_size),
            self.is_test:
                lambda: test_queue.dequeue_many(FLAGS.attention_batch_size)
        },
        default=lambda: train_queue.dequeue_many(FLAGS.attention_batch_size),
        exclusive=True)

    self.decoder_inputs, self.decoder_inputs_length, self.decoder_targets, self.decoder_masks, self.decoder_inputs_times, self.decoder_targets_times = data_batch
    self.batch_size = tf.shape(self.decoder_inputs)[0]

  def init_placeholders(self):
    self.global_step = tf.Variable(0, trainable=False, name='global_step')
    self.is_val = tf.compat.v1.placeholder(tf.bool, name='is_val')
    self.is_test = tf.compat.v1.placeholder(tf.bool, name='is_test')
    if self.mode == 'feed':
      # add an embedding input.
      self.z_vae_embeddings = tf.compat.v1.placeholder(
          dtype=tf.float32,
          shape=(self.num_nodes, FLAGS.vae_latent_dim),
          name='z_vae_embeddings')

  def zero_out(self, tensor, mask_value=-1, scope=None):
    """Zero out mask value from the given tensor.
      Args:
        tensor: `Tensor` of size [batch_size, seq_length] of target indices.
        mask_value: Missing label value.
        scope: `string`, scope of the operation
      Returns:
        targets: Targets with zerod-out values.
        mask: Mask values.
      """
    with tf.compat.v1.variable_scope(scope, default_name='zero_out'):
      in_vocab_indicator = tf.not_equal(tensor, mask_value)
      tensor *= tf.cast(in_vocab_indicator, tensor.dtype)
      mask = tf.to_float(in_vocab_indicator)
    return tensor, mask

  def build_decoder(self):
    # Attention decoder.
    with tf.compat.v1.variable_scope('attention_decoder'):
      sqrt3 = math.sqrt(3)  # Uniform(-sqrt(3), sqrt(3)) has variance=1.
      initializer = tf.random_uniform_initializer(-sqrt3, sqrt3)
      self.sender_embeddings = tf.compat.v1.get_variable(
          name='sender_embedding',
          shape=[self.num_nodes, self.embedding_size],
          initializer=initializer)
      self.receiver_embeddings = tf.compat.v1.get_variable(
          name='receiver_embedding',
          shape=[self.num_nodes, self.embedding_size],
          initializer=initializer)
      self.structure_proxy_embeddings = tf.compat.v1.get_variable(
          name='stucture_embedding',
          shape=[self.num_nodes, self.embedding_size],
          initializer=initializer)

      self.attn_w = tf.compat.v1.get_variable(
          name='attention_weights_w',
          shape=[self.embedding_size, self.embedding_size],
          initializer=initializer)

      self.decoder_structure_embedded = tf.nn.embedding_lookup(
          params=self.structure_proxy_embeddings,
          ids=self.decoder_inputs)  # (batch_size, seq_len, embed_size)

      self.decoder_sender_embedded = tf.nn.embedding_lookup(
          params=self.sender_embeddings,
          ids=self.decoder_inputs)  # (batch_size, seq_len, embed_size)

      # mask input sequence
      self.decoder_structure_embedded = self.decoder_structure_embedded *\
          tf.expand_dims(tf.cast(self.decoder_masks, tf.float32), -1)
      self.decoder_sender_embedded = self.decoder_sender_embedded *\
          tf.expand_dims(tf.cast(self.decoder_masks, tf.float32), -1)
      self.position_embeddings = tf.compat.v1.get_variable(
          name='position_embeddings',
          shape=[FLAGS.max_seq_length, self.embedding_size],
          initializer=initializer)
      self.decoder_sender_embedded = self.decoder_sender_embedded + self.position_embeddings

      attn_act = tf.nn.tanh(
          tf.reduce_sum(
              tf.multiply(
                  tf.tensordot(self.decoder_structure_embedded,
                               self.attn_w,
                               axes=[[2], [0]]), self.decoder_sender_embedded),
              2))

      attn_alpha = tf.nn.softmax(attn_act)  # (batch_size, seq_len)
      self.decoder_attended = self.decoder_sender_embedded * tf.expand_dims(
          attn_alpha, -1)
      self.decoder_attended = tf.reduce_sum(self.decoder_attended, 1)

      if FLAGS.sender_only:
        self.decoder_outputs = tf.matmul(self.decoder_attended,
                                         tf.transpose(
                                             self.structure_proxy_embeddings)
                                        )  # Shape -> [batch_size, num_nodes]
      else:
        self.decoder_outputs = tf.matmul(
            self.decoder_attended, tf.transpose(
                self.receiver_embeddings))  # Shape -> [batch_size, num_nodes]

      self.decoder_pred = tf.argmax(self.decoder_outputs,
                                    axis=-1,
                                    name='decoder_pred')

      _, self.topk = tf.nn.top_k(self.decoder_outputs,
                                 k=200)  # [batch_size, 200]
      self.topk_filter = tf.py_func(remove_seeds,
                                    [self.topk, self.decoder_inputs], tf.int32)
      masks = tf.cast(
          tf.reshape(
              tf.py_func(get_masks, [self.topk_filter, self.decoder_inputs],
                         tf.int32), [-1]), tf.bool)
      # For evaluation change, the relevance score computation must include the full target set.
      relevance_scores_all = tf.py_func(
          get_relevance_scores, [self.topk_filter, self.decoder_targets],
          tf.bool)
      #relevance_scores_all = tf.equal(tf.reshape(self.topk_filter, [-1,200]), tf.expand_dims(self.decoder_targets[:, 0],1))
      M = tf.reduce_sum(
          tf.reduce_max(tf.one_hot(self.decoder_targets, self.num_nodes),
                        axis=1), -1)
      self.relevance_scores = tf.cast(
          tf.boolean_mask(tf.cast(relevance_scores_all, tf.float32), masks),
          tf.int32)

      self.mrr_score = tf.py_func(MRR, [self.relevance_scores], tf.float32)

      self.precision_score = [
          tf.py_func(mean_precision_at_k, [self.relevance_scores, k],
                     tf.float32) for k in self.k_list
      ]

      self.recall_score = [
          tf.py_func(mean_recall_at_k, [self.relevance_scores, k, M],
                     tf.float32) for k in self.k_list
      ]

      self.map_score = [
          tf.py_func(MAP, [self.relevance_scores, k, M], tf.float32)
          for k in self.k_list
      ]

      self.ndcg_score = [
          tf.py_func(mean_NDCG_at_k, [self.relevance_scores, k], tf.float32)
          for k in self.k_list
      ]

      self.jaccard_score = [
          tf.py_func(jaccard, [self.relevance_scores, k, M], tf.float32)
          for k in self.k_list
      ]

  def init_optimizer(self):
    # Gradients and SGD update operation for training the model
    trainable_params = tf.compat.v1.trainable_variables()
    if FLAGS.optimizer.lower() == 'adadelta':
      self.optimizer = tf.compat.v1.train.AdadeltaOptimizer(
          learning_rate=FLAGS.attention_learning_rate)
    elif FLAGS.optimizer.lower() == 'adam':
      self.optimizer = tf.compat.v1.train.AdamOptimizer(
          learning_rate=FLAGS.attention_learning_rate)
    elif FLAGS.optimizer.lower() == 'rmsprop':
      self.optimizer = tf.compat.v1.train.RMSPropOptimizer(
          learning_rate=FLAGS.attention_learning_rate)
    else:
      self.optimizer = tf.compat.v1.train.GradientDescentOptimizer(
          learning_rate=FLAGS.attention_learning_rate)
    # Compute gradients of loss w.r.t. all trainable variables
    actual_loss = self.diffusion_loss
    gradients = tf.gradients(actual_loss, trainable_params)
    # Clip gradients by a given maximum_gradient_norm
    clip_gradients, _ = tf.clip_by_global_norm(gradients,
                                               FLAGS.max_gradient_norm)
    # Set the model optimization op.
    self.opt_op = self.optimizer.apply_gradients(zip(clip_gradients,
                                                     trainable_params),
                                                 global_step=self.global_step)

  def _loss(self):
    labels = self.decoder_targets  # [batch_size, max_seq_length]
    labels, mask = self.zero_out(labels, mask_value=-1)

    # tf.one_hot will return zero vector for -1 padding in targets
    labels_k_hot = tf.reduce_max(tf.one_hot(self.decoder_targets,
                                            self.num_nodes),
                                 axis=1)
    if FLAGS.pos_weight < 0:
      self.attention_loss = tf.reduce_mean(
          tf.nn.weighted_cross_entropy_with_logits(
              targets=labels_k_hot,
              logits=self.decoder_outputs,
              pos_weight=(self.num_nodes / FLAGS.max_seq_length
                         ))) + self.lambda_a * tf.nn.l2_loss(self.attn_w)
    else:
      self.attention_loss = tf.reduce_mean(
          tf.nn.weighted_cross_entropy_with_logits(targets=labels_k_hot,
                                                   logits=self.decoder_outputs,
                                                   pos_weight=FLAGS.pos_weight)
      ) + self.lambda_a * tf.nn.l2_loss(self.attn_w)

    # Internally calls 'nn_ops.sparse_softmax_cross_entropy_with_logits' by default
    # For v-loss, the mean is constant while v is the encoder embeddings.
    self.sender_loss = 0.5 * self.lambda_s * tf.reduce_mean(
        tf.reduce_sum(
            tf.square(self.structure_proxy_embeddings - self.z_vae_embeddings),
            1))
    if FLAGS.sender_only:
      self.receiver_loss = 0.
    else:
      self.receiver_loss = 0.5 * self.lambda_r * tf.reduce_mean(
          tf.reduce_sum(
              tf.square(self.receiver_embeddings - self.z_vae_embeddings), 1))
    self.diffusion_loss = self.attention_loss + self.sender_loss + self.receiver_loss

  def _build(self):
    self.build_decoder()
    self._loss()
    self.init_optimizer()


class CVGAE(Model):

  def __init__(self, num_features, adj, vae_layers_config, mode, feats,
               **kwargs):
    super(CVGAE, self).__init__(**kwargs)
    self.mode = mode
    self.num_features = num_features
    self.A = adj.astype(np.float32)
    self.feats = feats.astype(np.float32)
    self.layers_config = vae_layers_config
    self.latent_dim = FLAGS.vae_latent_dim
    self.lambda_s = FLAGS.lambda_s
    self.lambda_r = FLAGS.lambda_r
    self.pos_weight = 100
    self.init_placeholders()
    if FLAGS.graph_AE == 'MLP':
      model_func = MLP_AE
    elif FLAGS.graph_AE == 'GCN_AE':
      model_func = GCN_AE
    self.graph_ae = model_func(layers_config=self.layers_config,
                               num_features=self.num_features,
                               adj=adj,
                               latent_dim=self.latent_dim,
                               placeholders=self.placeholders,
                               pos_weight=self.pos_weight)
    self.build()
    self.node_indices = self.graph_ae.node_indices

  # Feed in v_s, v_r, pre_train flag, dropout -- sender and receiver latent embeddings.
  def construct_feed_dict(self,
                          v_sender_all=None,
                          v_receiver_all=None,
                          pre_train=False,
                          dropout=0.):
    input_feed = {}
    if v_sender_all is not None and v_receiver_all is not None:
      input_feed[self.v_sender_all.name] = v_sender_all
      input_feed[self.v_receiver_all.name] = v_receiver_all
    input_feed[self.pre_train.name] = pre_train
    input_feed[self.dropout.name] = dropout
    return input_feed

  def init_placeholders(self):
    self.pre_train = tf.compat.v1.placeholder(tf.bool, name='pre_train')
    self.dropout = tf.compat.v1.placeholder_with_default(0.,
                                                         shape=(),
                                                         name='dropout')
    self.v_sender_all = tf.compat.v1.placeholder(dtype=tf.float32,
                                                 shape=(None, None),
                                                 name='v_sender_all')
    self.v_receiver_all = tf.compat.v1.placeholder(dtype=tf.float32,
                                                   shape=(None, None),
                                                   name='v_receiver_all')
    self.global_step = tf.Variable(0, trainable=False, name='global_step')
    self.placeholders = {}
    self.placeholders['v_sender_all'] = self.v_sender_all
    self.placeholders['v_receiver_all'] = self.v_receiver_all
    self.placeholders['pre_train'] = self.pre_train
    self.placeholders['dropout'] = self.dropout

  def _loss(self):
    # Define two losses here:
    # a) KL divergence loss
    self.kld_loss = 0.5 * tf.reduce_mean(
        tf.reduce_sum(
            tf.square(self.z_mean) + tf.exp(self.z_log_sigma_sq) -
            self.z_log_sigma_sq - 1, 1))
    # b) Reconstruction or likelihood loss -- assuming bernoulli decoder.
    self.sender_loss = 0.5 * self.lambda_s * tf.reduce_mean(
        tf.reduce_sum(tf.square(self.v_sender - self.z_mean), 1))
    if FLAGS.sender_only:
      self.receiver_loss = 0.
    else:
      self.receiver_loss = 0.5 * self.lambda_r * tf.reduce_mean(
          tf.reduce_sum(tf.square(self.v_receiver - self.z_mean), 1))
    self.likelihood_loss, self.reg_loss = self.graph_ae._loss()
    self.vae_loss = self.likelihood_loss + self.kld_loss + self.reg_loss
    self.cvgae_loss = self.kld_loss + self.likelihood_loss + self.sender_loss + self.receiver_loss + self.reg_loss

  def _build(self):
    self._build_vae()
    self._loss()  # Add additional  loss function -- v_loss
    self.init_optimizer()

  def _build_vae(self):
    self.graph_ae.build_inference_network(self.feats)
    self.z_mean, self.z_log_sigma_sq = self.graph_ae.z_mean, self.graph_ae.z_log_sigma_sq
    # Draw one sample z from Gaussian distribution
    eps = tf.random.normal((self.graph_ae.batch_size, self.latent_dim),
                           0,
                           1,
                           dtype=tf.float32)
    # z = mu + sigma*epsilon
    self.z = tf.add(
        self.z_mean,
        tf.multiply(tf.sqrt(tf.maximum(tf.exp(self.z_log_sigma_sq), 1e-10)),
                    eps))

    self.graph_ae.build_generative_network(self.z)
    self.v_sender = self.graph_ae.v_sender
    self.v_receiver = self.graph_ae.v_receiver

  def init_optimizer(self):
    # Gradients and SGD update operation for training the model
    trainable_params = tf.compat.v1.trainable_variables()
    if FLAGS.optimizer.lower() == 'adadelta':
      self.optimizer = tf.compat.v1.train.AdadeltaOptimizer(
          learning_rate=FLAGS.vae_learning_rate)
    elif FLAGS.optimizer.lower() == 'adam':
      self.optimizer = tf.compat.v1.train.AdamOptimizer(
          learning_rate=FLAGS.vae_learning_rate)
    elif FLAGS.optimizer.lower() == 'rmsprop':
      self.optimizer = tf.compat.v1.train.RMSPropOptimizer(
          learning_rate=FLAGS.vae_learning_rate)
    else:
      self.optimizer = tf.compat.v1.train.GradientDescentOptimizer(
          learning_rate=FLAGS.vae_learning_rate)
    # Compute gradients of loss w.r.t. all trainable variables
    #gradients = tf.cond(self.pre_train, lambda:tf.gradients(self.vae_loss, trainable_params), lambda: tf.gradients(self.cvgae_loss, trainable_params))
    actual_loss = tf.cond(self.pre_train, lambda: self.vae_loss,
                          lambda: self.cvgae_loss)
    gradients = tf.gradients(actual_loss, trainable_params)
    # Clip gradients by a given maximum_gradient_norm
    clip_gradients, _ = tf.clip_by_global_norm(gradients,
                                               FLAGS.max_gradient_norm)
    # Set the model optimization op.
    self.opt_op = self.optimizer.apply_gradients(zip(clip_gradients,
                                                     trainable_params),
                                                 global_step=self.global_step)
