import torch
import torch.nn.functional as F

def regression_loss(logits, labels, num_steps, steps, seq_lens, loss_type,
                    normalize_indices, variance_lambda, huber_delta):
  """Loss function based on regressing to the correct indices.
  In the paper, this is called Cycle-back Regression. There are 3 variants
  of this loss:
  i) regression_mse: MSE of the predicted indices and ground truth indices.
  ii) regression_mse_var: MSE of the predicted indices that takes into account
  the variance of the similarities. This is important when the rate at which
  sequences go through different phases changes a lot. The variance scaling
  allows dynamic weighting of the MSE loss based on the similarities.
  iii) regression_huber: Huber loss between the predicted indices and ground
  truth indices.
  Args:
    logits: Tensor, Pre-softmax similarity scores after cycling back to the
      starting sequence.
    labels: Tensor, One hot labels containing the ground truth. The index where
      the cycle started is 1.
    num_steps: Integer, Number of steps in the sequence embeddings.
    steps: Tensor, step indices/frame indices of the embeddings of the shape
      [N, T] where N is the batch size, T is the number of the timesteps.
    seq_lens: Tensor, Lengths of the sequences from which the sampling was done.
      This can provide additional temporal information to the alignment loss.
    loss_type: String, This specifies the kind of regression loss function.
      Currently supported loss functions: regression_mse, regression_mse_var,
      regression_huber.
    normalize_indices: Boolean, If True, normalizes indices by sequence lengths.
      Useful for ensuring numerical instabilities don't arise as sequence
      indices can be large numbers.
    variance_lambda: Float, Weight of the variance of the similarity
      predictions while cycling back. If this is high then the low variance
      similarities are preferred by the loss while making this term low results
      in high variance of the similarities (more uniform/random matching).
    huber_delta: float, Huber delta described in tf.keras.losses.huber_loss.
  Returns:
     loss: Tensor, A scalar loss calculated using a variant of regression.
  """
  # Just to be safe, we stop gradients from labels as we are generating labels.
  labels = labels.detach()
  steps = steps.detach()

  # print('steps', steps.shape)
  # print('seq_lens', seq_lens)
  # print('labels', labels)
  # always normalize to 0~1
  # print('steps', steps.float())
  # print('seq len', seq_lens.shape)
  # tile_seq_lens = seq_lens.clone().expand(-1, num_steps)

  # NOTICES: have to use repeat than expand, otherwise strange behavior would occur
  tile_seq_lens = seq_lens.repeat(1, num_steps)
  # print('normalized steps, torch div', torch.div(steps.float(), tile_seq_lens))
  steps = steps.float() / tile_seq_lens
  # print('tile', tile_seq_lens.shape, steps.shape, tile_seq_lens)
  # print('normalized steps', steps)
  # print(steps, seq_lens.shape, steps.shape)
  # print('be', logits.shape, labels.shape)

  beta = F.softmax(logits, dim=1)
  # true_time = torch.sum(steps * labels, dim=1)
  true_time = torch.gather(steps, dim=1, index=labels.view(-1, 1)).squeeze(-1)
  pred_time = torch.sum(steps * beta, dim=1)
  # print(true_time, pred_time, true_time.shape, pred_time.shape)
  # print('beta', beta)
  # print('pred shape', true_time.shape, pred_time.shape)
  # print('true, pred time', true_time[:5], pred_time[:5])
  # print('true, pred time', true_time, pred_time)

  if loss_type in ['regression_mse', 'regression_mse_var']:
    if 'var' in loss_type:
      # Variance aware regression.
      # pred_time_tiled = pred_time.view(-1, 1).expand(-1, num_steps)
      pred_time_tiled = pred_time.view(-1, 1).repeat(1, num_steps)
      pred_time_variance = torch.sum((steps - pred_time_tiled) ** 2 * beta, dim=1)

      # Using log of variance as it is numerically stabler.
      pred_time_log_var = torch.log(pred_time_variance)
      squared_error = (true_time - pred_time) ** 2
      # print('squared_error', torch.mean(squared_error))
      # mean_error = torch.exp(-pred_time_log_var) * squared_error
      # print(squared_error[:5], (pred_time_log_var * variance_lambda)[:5])
      return torch.mean(torch.exp(-pred_time_log_var) * squared_error
                        + variance_lambda * pred_time_log_var)

    else:
      # squared_error = (true_time - pred_time) ** 2
      # print('squared_error', torch.mean(squared_error), true_time - pred_time)
      return torch.mean((true_time - pred_time) ** 2)
  else:
    raise ValueError('Unsupported regression loss %s. Supported losses are: '
                     'regression_mse, regresstion_mse_var and regression_huber.'
                     % loss_type)
