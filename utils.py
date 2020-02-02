from __future__ import print_function

import logging
import os
import pickle
from datetime import datetime

import networkx as nx
import numpy as np
import scipy.sparse as sp

import pandas as pd
import tensorflow as tf

flags = tf.compat.v1.flags
FLAGS = flags.FLAGS


class ExpLogger:

  def __init__(self,
               name,
               cmd_print=True,
               log_file=None,
               spreadsheet=None,
               datadir=None):
    self.datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    self.name = name + "_" + self.datetime_str
    self.cmd_print = cmd_print
    log_level = logging.INFO
    logging.basicConfig(filename=log_file,
                    level=log_level,
                    format='%(asctime)s - %(levelname)s: %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S')
    self.file_logger = logging.getLogger()
    self.spreadsheet = spreadsheet
    self.datadir = datadir
    if self.spreadsheet is not None:
      dirname = os.path.dirname(self.spreadsheet)
      if not os.path.exists(dirname):
        os.makedirs(dirname)
      if os.path.isfile(self.spreadsheet):
        try:
          self.dataframe = pd.read_csv(spreadsheet)
        except:
          self.dataframe = pd.DataFrame()
      else:
        self.dataframe = pd.DataFrame()
    if self.datadir is not None:
      if not os.path.exists(self.datadir):
        os.makedirs(self.datadir)
    self.best_metric = float("-inf")
    self.best_data = None

  def __enter__(self):
    self.log("Logger Started, name: " + self.name)
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    if self.spreadsheet is not None:
      self.dataframe.to_csv(self.spreadsheet, index=False)

  def log(self, content):
    if not isinstance(content, str):
      content = str(content)
    if self.cmd_print:
      print(content)
    if self.file_logger is not None:
      self.file_logger.info(content)

  def debug(self, content):
    if not isinstance(content, str):
      content = str(content)
    if self.cmd_print:
      print("[DEBUG]::: " + str + ":::[DEBUG]")
    if self.file_logger is not None:
      self.file_logger.debug(str)

  def spreadsheet_write(self, val_dict):
    if self.spreadsheet is not None:
      if "name" not in val_dict:
        val_dict["name"] = self.name
      self.dataframe = self.dataframe.append(val_dict, ignore_index=True)

  def save_data(self, data, name):
    name = name + "_" + self.datetime_str
    if isinstance(data, np.ndarray):
      np.savez_compressed(os.path.join(self.datadir, name + ".npz"), data=data)
    else:
      with open(os.path.join(self.datadir, name + ".pkl"), "wb") as f:
        pickle.dump(data, f)

  def update_record(self, metric, data):
    if metric > self.best_metric:
      self.best_metric = metric
      self.best_data = data


def sparse_to_tuple(sparse_mx):
  """Convert sparse matrix to tuple representation."""

  def to_tuple(mx):
    if not sp.isspmatrix_coo(mx):
      mx = mx.tocoo()
    coords = np.vstack((mx.row, mx.col)).transpose()
    values = mx.data
    shape = mx.shape
    return coords, values, shape

  def to_tuple_list(matrices):
    # Input is a list of matrices.
    coords = []
    values = []
    shape = [len(matrices)]
    for i in range(0, len(matrices)):
      mx = matrices[i]
      if not sp.isspmatrix_coo(mx):
        mx = mx.tocoo()
      # Create proper indices - coords is a numpy array of pairs of indices.
      coords_mx = np.vstack((mx.row, mx.col)).transpose()
      z = np.array([np.ones(coords_mx.shape[0]) * i]).T
      z = np.concatenate((z, coords_mx), axis=1)
      z = z.astype(int)
      #coords.extend(z.tolist())
      coords.extend(z)
      values.extend(mx.data)

    shape.extend(matrices[0].shape)
    shape = np.array(shape).astype("int64")
    values = np.array(values).astype("float32")
    coords = np.array(coords)
    # print ("insider", len(coords), len(values), shape)
    return coords, values, shape

  if isinstance(sparse_mx, list) and isinstance(sparse_mx[0], list):
    # Given a list of lists, convert it into a list of tuples.
    for i in range(0, len(sparse_mx)):
      sparse_mx[i] = to_tuple_list(sparse_mx[i])

  elif isinstance(sparse_mx, list):
    for i in range(len(sparse_mx)):
      sparse_mx[i] = to_tuple(sparse_mx[i])
  else:
    sparse_mx = to_tuple(sparse_mx)

  return sparse_mx


def normalize_graph_gcn(adj):
  adj = sp.coo_matrix(adj)
  adj_ = adj + sp.eye(adj.shape[0])
  rowsum = np.array(adj_.sum(1))
  degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
  adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(
      degree_mat_inv_sqrt).tocoo()
  return sparse_to_tuple(adj_normalized)


def to_one_hot(labels, N, multilabel=False):
  """In: list of (nodeId, label) tuples, #nodes N
       Out: N * |label| matrix"""
  ids, labels = zip(*labels)
  lb = MultiLabelBinarizer()
  if not multilabel:
    labels = [[x] for x in labels]
  lbs = lb.fit_transform(labels)
  encoded = np.zeros((N, lbs.shape[1]))
  for i in range(len(ids)):
    encoded[ids[i]] = lbs[i]
  return encoded


def sample_mask(idx, l):
  """Create mask."""
  mask = np.zeros(l)
  mask[idx] = 1
  return np.array(mask, dtype=np.bool)


extra_tokens = ['_GO', 'EOS']


def get_data_set(dataset_str,
                 cascades,
                 timestamps,
                 maxlen=None,
                 test_min_percent=0.1,
                 test_max_percent=0.5,
                 mode='test'):
  dataset = []
  dataset_times = []
  eval_set = []
  eval_set_times = []
  for cascade in cascades:
    if maxlen is None or len(cascade) < maxlen:
      dataset.append(cascade)
    else:
      dataset.append(cascade[0:maxlen])  # truncate

  for ts_list in timestamps:
    if maxlen is None or len(ts_list) < maxlen:
      dataset_times.append(ts_list)
    else:
      dataset_times.append(ts_list[0:maxlen])  # truncate

  for cascade, ts_list in zip(dataset, dataset_times):
    assert len(cascade) == len(ts_list)
    for j in range(1, len(cascade)):
      seedSet = cascade[0:j]
      seedSet_times = ts_list[0:j]
      remain = cascade[j:]
      remain_times = ts_list[j:]
      seed_set_percent = len(seedSet) / (len(seedSet) + len(remain))
      if (mode == 'train' or mode == 'val'):
        eval_set.append((seedSet, remain))
        eval_set_times.append((seedSet_times, remain_times))
      if mode == 'test' and (seed_set_percent > test_min_percent and
                             seed_set_percent < test_max_percent):
        eval_set.append((seedSet, remain))
        eval_set_times.append((seedSet_times, remain_times))
  print("# {} examples {}".format(mode, len(eval_set)))
  return eval_set, eval_set_times


def load_graph(dataset_str):
  """Load graph."""
  print("Loading graph", dataset_str)
  G = nx.Graph()
  with open("data/{}/{}".format(dataset_str, "graph.txt"), 'rb') as f:
    nu = 0
    for line in f:
      nu += 1
      if nu == 1:
        # assuming first line contains number of nodes, edges.
        nNodes, nEdges = [int(x) for x in line.strip().split()]
        for i in range(nNodes):
          G.add_node(i)
        continue
      s, t = [int(x) for x in line.strip().split()]
      G.add_edge(s, t)
  A = nx.adjacency_matrix(G)
  print("# nodes", nNodes, "# edges", nEdges, A.shape)
  global start_token, end_token
  start_token = A.shape[0] + extra_tokens.index('_GO')  # start_token = 0
  end_token = A.shape[0] + extra_tokens.index('EOS')  # end_token = 1
  return A


def load_feats(dataset_str):
  X = np.load("data/{}/{}".format(dataset_str, "feats.npz"))
  return X['arr_0']


def load_cascades(dataset_str, mode='train'):
  """Load data."""
  print("Loading cascade", dataset_str, "mode", mode)
  cascades = []
  global avg_diff
  avg_diff = 0.0
  time_stamps = []
  path = mode + str(".txt")
  with open("data/{}/{}".format(dataset_str, path), 'rb') as f:
    for line in f:
      if len(line) < 1:
        continue
      line = list(map(float, line.split()))
      start = int(line[0])
      rest = line[1:]
      cascade = [start]
      cascade.extend(list(map(int, rest[::2])))
      time_stamp = [0]
      time_stamp.extend(rest[1::2])
      cascades.append(cascade)
      time_stamps.append(time_stamp)
  return cascades, time_stamps


def prepare_batch_sequences(input_sequences, target_sequences, batch_size):
  # Split based on batch_size
  assert (len(input_sequences) == len(target_sequences))
  if len(input_sequences) % batch_size == 0:
    num_batch = len(input_sequences) // batch_size
  else:
    num_batch = len(input_sequences) // batch_size + 1
  batches_x = []
  batches_y = []
  N = len(input_sequences)
  for i in range(0, num_batch):
    start = i * batch_size
    end = min((i + 1) * batch_size, N)
    batches_x.append(input_sequences[start:end])
    batches_y.append(target_sequences[start:end])
  return (batches_x, batches_y)


def prepare_batch_graph(A, batch_size):
  N = A.shape[0]
  num_batch = N // batch_size + 1
  random_ordering = np.random.permutation(N)
  batches = []
  batches_indices = []
  for i in range(0, num_batch):
    start = i * batch_size
    end = min((i + 1) * batch_size, N)
    batch_indices = random_ordering[start:end]
    batch = A[batch_indices, :]
    batches.append(batch.toarray())
    batches_indices.append(batch_indices)
  return batches, batches_indices


def prepare_sequences(examples,
                      examples_times,
                      maxlen=None,
                      attention_batch_size=1,
                      mode='train'):
  seqs_x = list(
      map(lambda seq_t: (seq_t[0][(-1) * maxlen:], seq_t[1]), examples))
  times_x = list(
      map(lambda seq_t: (seq_t[0][(-1) * maxlen:], seq_t[1]), examples_times))
  # add padding.
  lengths_x = [len(s[0]) for s in seqs_x]
  lengths_y = [len(s[1]) for s in seqs_x]

  if len(seqs_x) % attention_batch_size != 0 and (mode == 'test' or
                                                  mode == 'val'):
    # Note: this is required to ensure that each batch is full-sized -- else the
    # data may not be split perfectly while evaluation.
    x_batch_size = (1 +
                    len(seqs_x) // attention_batch_size) * attention_batch_size
    lengths_x.extend([1] * (x_batch_size - len(seqs_x)))
    lengths_y.extend([1] * (x_batch_size - len(seqs_x)))

  x_lengths = np.array(lengths_x).astype('int32')
  maxlen_x = maxlen
  # mask input with start token (n_nodes + 1) to work with embedding_lookup
  x = np.ones((len(lengths_x), maxlen_x)).astype('int32') * start_token
  # mask target with -1 so that tf.one_hot will return a zero vector for padded nodes
  y = np.ones((len(lengths_y), maxlen_x)).astype('int32') * -1  # we u
  x_times = np.ones((len(lengths_x), maxlen_x)).astype('int32') * -1
  y_times = np.ones((len(lengths_y), maxlen_x)).astype('int32') * -1
  mask = np.ones_like(x)
  for idx, (s_x, t) in enumerate(seqs_x):
    end_x = lengths_x[idx]
    end_y = lengths_y[idx]
    x[idx, :end_x] = s_x
    y[idx, :end_y] = t
    mask[idx, end_x:] = 0

  for idx, (s_x, t) in enumerate(times_x):
    end_x = lengths_x[idx]
    end_y = lengths_y[idx]
    x_times[idx, :end_x] = s_x
    y_times[idx, :end_y] = t

  return x, x_lengths, y, mask, x_times, y_times


def ensure_dir(d):
  if not os.path.isdir(d):
    os.makedirs(d)
