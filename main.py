import data
import plot
from pennylane import numpy as np
import training
import fixed_testing as testing
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
def calculate_stats(dicts):
    # Extract keys from the first dictionary assuming all dictionaries have the same keys
    keys = dicts[0].keys()
    
    # Initialize a dictionary to store the results
    results = {}
    
    # Iterate over each key to calculate mean and std deviation
    for key in keys:
        values = [d[key] for d in dicts]  # Extract values for each key from all dictionaries
        mean = np.mean(values)
        std = np.std(values)
        
        # Store the result as a string formatted as "mean ± std"
        results[key] = f"{mean:.4f} ± {std:.4f}"
    
    return results

def run(train_pulsar_global, train_non_pulsar_global,test, initial_weights, train_circuit,test_circuit, title, num_sets,max_epochs,dev):
  global opt_train_pulsar,opt_train_non_pulsar,opt_test_pulsar,opt_test_non_pulsar,results, loss_values,fpr, tpr
  opt_weights,loss_values = training.main(train_pulsar_global,train_non_pulsar_global,initial_weights,train_circuit,title,num_sets,max_epochs)

  opt_train_non_pulsar = np.array([testing.pulsar_probability(train_non_pulsar_global[i,:,:8],opt_weights[i],train_circuit)for i in range(num_sets)])
  opt_train_pulsar = np.array([testing.pulsar_probability(train_pulsar_global[i,:,:8],opt_weights[i],train_circuit) for i in range(num_sets)])
  predictions_results_fpr_tpr = [testing.metrics(test[i], opt_weights[i], test_circuit, dev,i,title) for i in range(num_sets)] #changed this to i if increasing num_sets
  predictions,results,fpr,tpr = zip(*predictions_results_fpr_tpr)
  opt_test_non_pulsar = [testing.pulsar_probability(test_non_pulsar_global[i,:,:8],opt_weights[i],test_circuit)for i in range(num_sets)]
  opt_test_pulsar = [testing.pulsar_probability(test_pulsar_global[i,:,:8],opt_weights[i],test_circuit) for i in range(num_sets)]
  opt_test_non_pulsar = np.array(opt_test_non_pulsar)
  opt_test_pulsar = np.array(opt_test_pulsar)
  
  
  for i in range(num_sets):
      plot.plot_all(opt_train_pulsar[i],opt_train_non_pulsar[i],opt_test_pulsar[i],opt_test_non_pulsar[i],max_epochs,results[i], loss_values[i], title, i,fpr[i], tpr[i])
  averaged_results = calculate_stats(results)

  return opt_weights,results,averaged_results


# Hyperparameters
max_epochs = 300
num_sets = 2
test_size = 400
train_size = 200
# Also the quantum model (found in its file) and its number of shots.

normalized_dataset = data.normalize()
train_pulsar_global,train_non_pulsar_global,test_pulsar_global,test_non_pulsar_global = data.sample_pulsars(normalized_dataset, train_size, test_size, num_sets)
train_data = np.concatenate((train_pulsar_global, train_non_pulsar_global), axis=1)
test = np.concatenate((test_pulsar_global, test_non_pulsar_global), axis=1)

import qcnn_train_encoding_weights
import qcnn_test_encoding_weights #as test_circuit
from qcnn_test_encoding_weights import dev
title="QCNN 8 encoding weights RZ"
np.random.seed(42)  # Set the seed
initial_weights = 2 * np.pi * np.random.random(size=(num_sets,26))

optimized_weights_third,results_third,averaged_result_third = run(train_pulsar_global, train_non_pulsar_global,test, initial_weights, qcnn_train_encoding_weights,qcnn_test_encoding_weights, title, num_sets,max_epochs,dev)
print(averaged_result_third)

