import warnings
from asl_data import SinglesData
from collections import Counter 
from queue import PriorityQueue
import numpy as np

def recognize(models: dict, test_set: SinglesData, verbose=False):
  """ Recognize test word sequences from word models set

  :param models: dict of trained models
     {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
  :param test_set: SinglesData object
  :return: (list, list)  as probabilities, guesses
     both lists are ordered by the test set word_id
     probabilities is a list of dictionaries where each key a word and value is Log Liklihood
         [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
          {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
          ]
     guesses is a list of the best guess words ordered by the test set word_id
         ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
  warnings.filterwarnings("ignore", category=DeprecationWarning)
  probabilities = []
  guesses = []    
  for _, XLengths in test_set.get_all_Xlengths().items():
    prob = {}
    best_guess = None 
    best_score = float("-inf")
    for word, model in models.items():
      if model is not None: 
        try: 
          # for each test record test each trained models 
          logL = model.score(XLengths[0], XLengths[1])
          prob[word] = logL
          # keep track of trained model which matches best
          if logL > best_score: 
            best_score = logL
            best_guess = word
        except ValueError as e: 
          if verbose: 
            print("Error {}, error({}):".format(type(e), e))

    probabilities.append(prob)
    guesses.append(best_guess)
    
  return probabilities, guesses

############ PART 4 ###############
class Node(object):
  """
  Simple node object which represents a word guess, its guess value, 
  its parent and its level (or index) in the sentence
  """
  def __init__(self, value, word, parent, level):
      self.value = value
      self.word = word
      self.parent = parent
      self.level = level

  def __lt__(self, other):
    return self.value < other.value

def get_top_n_models(probabilities, lm, top_n_model, current_depth, memo):
  """
  Returns the n models sorted by model score

  Stores result in a memo table in order to avoid rework
  """
  if current_depth == len(probabilities):
    return []
  if current_depth in memo:
    return memo[current_depth]
  probs = probabilities[current_depth]
  # ref: https://stackoverflow.com/a/11902696
  sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
  common = sorted_probs[:top_n_model]
  memo[current_depth] = common
  return common

def get_node_word_path(node):
  """
  Walks back up the tree using the nodes parents
  to provide a the "sentence" value of the current node
  """
  guesses = []
  current = node 
  while current is not None: 
    if current.parent is not None: 
      guesses.insert(0, current.word)
    current = current.parent
  return guesses

def expand_node(node, probabilities, lm, n_gram, top_n_model, weights, memo):
  """
  Expands current node by obtaining n best models from the model distribution
  at the node's level

  Scores each node using a combination of the model score and the lm score
  """
  word_path = get_node_word_path(node)
  models = get_top_n_models(probabilities, lm, top_n_model, node.level, memo)
  children = []
  for word, logL in models:
    gram_values = word_path[:]
    if node.level == 0:
      gram_values.append("<s>")
    gram_values.append(word)
    try: 
      ngram_arg = ' '.join(gram_values[-n_gram:])
      lm_score = lm.log_p(ngram_arg)
      node_value = np.matmul(weights, [logL, lm_score])
      child = Node(node_value + node.value, word, node, node.level + 1)
      children.append(child)
    except KeyError: 
      continue

  return children

def unifrom_cost_search(probabilities, root, lm, n_gram, top_n_model, weights):
  """
  Explores the directed graph using a PriorityQueue

  Does not keep track of the explored set since this is a directed graph
  """
  frontier = PriorityQueue()
  # ref: https://stackoverflow.com/a/26441285
  frontier.put((-root.value, root))
  memo = dict()
  while frontier:
    node = frontier.get()[1]
    children = expand_node(node, probabilities, lm, n_gram, top_n_model, weights, memo)
    if not children:
      return get_node_word_path(node)
    for child in children:
      frontier.put((-child.value, child))
  return None

def find_guess(probabilities, lm, n_gram, top_n_model, weights):
  """
  Finds best guess using uniform cost search
  """
  root = Node(0, "[DUMMY]", None, 0)
  guess = unifrom_cost_search(probabilities, root, lm, n_gram, top_n_model, weights)
  return guess

def adjust_probs(probabilities):
  """
  "Shifts" probabilities down below zero
  in order to not maintain a valid uniform cost search

  Ref: https://ai-nd.slack.com/files/U5F20UC2H/F6Q8E59NC/Summary_about_part_4__Recognition_project
  """
  adjusted_probabilities = []
  for prob in probabilities:
    adjusted = {}
    max_val = max(prob.values())
    for key, value in prob.items():
      # impl from slack
      adjusted[key] = value - max_val
    adjusted_probabilities.append(adjusted)
  return adjusted_probabilities

def guess_with_lm(probabilities, lm, test_set):
  probabilities = adjust_probs(probabilities)
  # hyper parameters
  hmm_score_w = 50.0
  lm_score_w = 1.3
  weights = [hmm_score_w, lm_score_w]
  top_n_model = 35
  n_gram = 3

  guesses = []
  for video_num in test_set.sentences_index:
    sentence = test_set.sentences_index[video_num]
    start_index = sentence[0]
    end_index = start_index + len(sentence)
    # find best guess for each sentence
    guess = find_guess(probabilities[start_index:end_index], lm, n_gram, top_n_model, weights)
    guesses += guess

  return probabilities, guesses

