import torch
from sklearn import metrics
from tqdm import tqdm
import torch.nn as nn
import numpy as np

#The class that impelements the dataset for arabic tweets
class ArabicDataset(torch.utils.data.Dataset):

  def __init__(self, x, y, pad):
    """
    This is the constructor of the ArabicDataset
    Inputs:
    - x: a list of lists where each list contains the ids of the tokens
    - y: a list of lists where each list contains the label of each token in the sentence
    - pad: the id of the <PAD> token (to be used for padding all sentences and labels to have the same length)
    """
    list_len = [len(i) for i in x]
    MAX_LENGTH = max(list_len) 
    for i in range(len(x)):
      x[i] = np.pad(x[i], (0, MAX_LENGTH-len(x[i])), 'constant', constant_values=(pad))

    self.x = torch.from_numpy(np.array(x)) 
    self.y = torch.from_numpy(np.array(y))

  def __len__(self):
    """
    This function should return the length of the dataset (the number of sentences)
    """
    return self.x.shape[0]

  def __getitem__(self, idx):
    """
    This function returns a subset of the whole dataset
    """
    return (self.x[idx], self.y[idx])

def create_emb_layer(weights_train_matrix, non_trainable=False):
    num_embeddings, embedding_dim = weights_train_matrix.size()
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({'weight': weights_train_matrix})
    if non_trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer, num_embeddings, embedding_dim

class Classifier(nn.Module):
  def __init__(self, weights_train_matrix, embedding_dim=100, hidden_size=100, n_classes=3, n_layer=1):
    """
    The constructor of our NER model
    Inputs:
    - vacab_size: the number of unique words
    - embedding_dim: the embedding dimension
    - n_classes: the number of final classes (tags)
    """
    self.hidden_size = hidden_size
    super(Classifier, self).__init__()
    
    self.embedding, num_embeddings, embedding_dim = create_emb_layer(weights_train_matrix, True)
    self.hidden_size = hidden_size

    self.GRU = nn.GRU(input_size=embedding_dim, hidden_size=hidden_size, batch_first=True, num_layers=n_layer)

    self.linear = nn.Linear(hidden_size, n_classes)

  def forward(self, sentences):
    """
    This function does the forward pass of our model
    Inputs:
    - sentences: tensor of shape (batch_size, max_length)

    Returns:
    - final_output: tensor of shape (batch_size, max_length, n_classes)
    """

    final_output = None
    final_output, _ = self.GRU(self.embedding(sentences))
    final_output = final_output[:, -1, :]
    final_output = self.linear(final_output)
    return final_output

def evaluate(model, test_dataset, batch_size=32):
  """
  This function takes a model and evaluates its performance (accuracy) on a test data
  Inputs:
  - model: the model
  - test_dataset: dataset of type ArabicDataset
  """
  
  # (1) create the test data loader
  test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

  # GPU Configuration
  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")
  if use_cuda:
    model = model.cuda()

  total_acc_test = 0.0
  
  y_test = [] 
  y_predected = [] 
  y_pred = [] 
  # (2) disable gradients
  with torch.no_grad():
    report = None
    for test_input, test_label in tqdm(test_dataloader):
      # (3) move the test input to the device
      test_label = test_label.to(device)

      # (4) move the test label to the device
      test_input = test_input.to(device)

      # (5) do the forward pass
      output = model.forward(sentences=test_input)

      # accuracy calculation (just add the correct predicted items to total_acc_test)
      acc = torch.sum(torch.eq(torch.argmax(output, dim=1), test_label))
      total_acc_test += acc
      
      # f1 score calculation
      y_test +=(list(test_label.view(-1)))
      y_predected +=(list(torch.argmax(output, dim=1).view(-1)))
      y_pred +=(list(np.array(output)))

    # (6) calculate the over all accuracy
    total_acc_test /= len(test_dataset)

  report = metrics.classification_report(y_test, y_predected)
  print("accuracy: {:0.3f}".format(metrics.accuracy_score(y_test, y_predected)))
  print(report)
  
  print(f'\nTest Accuracy: {total_acc_test}')
  return np.array(y_pred)

def train(model, train_dataset, batch_size=32, epochs=10, learning_rate=0.001):
  """
  This function implements the training logic
  Inputs:
  - model: the model ot be trained
  - train_dataset: the training set of type NERDataset
  - batch_size: integer represents the number of examples per step
  - epochs: integer represents the total number of epochs (full training pass)
  - learning_rate: the learning rate to be used by the optimizer
  """

  # (1) create the dataloader of the training set (make the shuffle=True)
  train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

  # (2) make the criterion cross entropy loss
  criterion = nn.CrossEntropyLoss(weight=torch.tensor([.5, .4, .1])) 

  # (3) create the optimizer (Adam)
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

  # GPU configuration
  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")
  if use_cuda:
    model = model.cuda()
    criterion = criterion.cuda()

  for epoch_num in range(epochs):
    if epoch_num % 5 == 0:
      learning_rate /= 10
    total_acc_train = 0
    total_loss_train = 0

    for train_input, train_label in tqdm(train_dataloader):

      # (4) move the train input to the device
      train_label = train_label.to(device)

      # (5) move the train label to the device
      train_input = train_input.to(device)

      # (6) do the forward pass
      output = model.forward(sentences=train_input)
      
      # (7) loss calculation (you need to think in this part how to calculate the loss correctly)
      batch_loss = criterion(output, train_label) 

      # (8) append the batch loss to the total_loss_train
      total_loss_train += batch_loss.item()
      
      # (9) calculate the batch accuracy (just add the number of correct predictions)
      argmax = torch.argmax(output, dim=1)
      acc = torch.sum(torch.eq(argmax, train_label))
      total_acc_train += acc

      # (10) zero your gradients
      optimizer.zero_grad()

      # (11) do the backward pass
      batch_loss.backward()

      # (12) update the weights with your optimizer
      optimizer.step()

    num_of_batches = len(train_dataset) / batch_size
    num_of_batches = int(num_of_batches) + 1
    # epoch loss
    epoch_loss = total_loss_train / num_of_batches
    
    # (13) calculate the accuracy
    epoch_acc = total_acc_train / len(train_dataset)

    print(
        f'Epochs: {epoch_num + 1} | Train Loss: {epoch_loss} \
        | Train Accuracy: {epoch_acc}\n')
