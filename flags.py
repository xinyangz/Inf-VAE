import tensorflow as tf

flags = tf.compat.v1.flags
FLAGS = flags.FLAGS

# General params.
flags.DEFINE_string('dataset', 'christianity', 'Dataset string.')
flags.DEFINE_integer('epochs', 100, 'Number of epochs to train.')
flags.DEFINE_integer('pretrain_epochs', 50, 'Number of epochs to train.')
flags.DEFINE_string('optimizer', 'adam',
                    'Optimizer for training: (adadelta, adam, rmsprop)')
flags.DEFINE_float('max_gradient_norm', 1.0, 'Clip gradients to this norm')
flags.DEFINE_string('cuda_device', '0', 'GPU in use')

flags.DEFINE_integer('test_freq', 1, 'Testing frequency')
flags.DEFINE_integer('val_freq', 1, 'Validation frequency')
flags.DEFINE_integer('early_stopping', 10,
                     'Tolerance for early stopping (# of epochs).')

flags.DEFINE_integer('batch_queue_threads', 8,
                     'Threads used for tf batch queue')

flags.DEFINE_string('graph_AE', 'GCN_AE',
                    'Graph AutoEncoder Options (MLP, GCN_AE)')
flags.DEFINE_boolean('use_feats', False, 'Use features in GCN or not')

# Model Hyper-parameters
flags.DEFINE_float('lambda_s', 1.0, 'Lambda_s')
flags.DEFINE_float('lambda_r', 0.01, 'Lambda_r')

flags.DEFINE_float('pos_weight', 1,
                   'Pos weight for cross entropy. -1 decides automatically')
flags.DEFINE_boolean('sender_only', False, 'Use only sender embeddings')

# evaluation parameters
flags.DEFINE_integer('max_seq_length', 100, 'Maximum sequence length')
flags.DEFINE_float('test_min_percent', 0.1,
                   'Minimum seed set percentage for testing.')
flags.DEFINE_float('test_max_percent', 0.5,
                   'Maximum seed set percentage for testing.')

# Attention model params
flags.DEFINE_float('lambda_a', 0.1, 'Lambda_a for attention weights')
flags.DEFINE_float('attention_learning_rate', 0.01,
                   'Initial learning rate for attention model.')
flags.DEFINE_float('attention_dropout_rate', 0.1,
                   'Dropout rate (1 - keep probability).')
flags.DEFINE_integer('attention_batch_size', 64, 'Batch size attention')

flags.DEFINE_boolean('use_dropout', True, 'Use dropout')

# vae paramss
# flags.DEFINE_float('vae_annealing', 1, 'VAE annealing parameter beta in its objective function')
flags.DEFINE_string('vae_layer_config', '256,128,64',
                    'VAE NN layer config: comma separated number of units')
flags.DEFINE_float('vae_dropout_rate', 0.2,
                   'Dropout rate (1 - keep probability).')
flags.DEFINE_string('vae_loss_function', 'cross_entropy',
                    'Loss function for GVAE: (rmse,cross_entropy)')
flags.DEFINE_integer('vae_batch_size', 64, 'Batch size VAE')
flags.DEFINE_integer('vae_latent_dim', 64, 'Latent embedding dimension')
flags.DEFINE_float('vae_learning_rate', 0.01, 'Initial learning rate for vae.')
flags.DEFINE_float('vae_weight_decay', 1e-4,
                   'Weight for L2 loss on embedding matrix.')
