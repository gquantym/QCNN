from collections import Counter
from pennylane import numpy as np
import time
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
import plot
def pulsar_probability(sampled_data, weights, quantum_circuit):
    global expectation_value
    expectation_value = [quantum_circuit.circuit(weights, feature_vector) for feature_vector in sampled_data]
    probability_pulsar = (1 - np.array(expectation_value)) / 2
    return probability_pulsar

def most_common(predictions):
    counter = Counter(predictions)
    most_common_element, _ = counter.most_common(1)[0]
    return most_common_element

def metrics(test_data, optimized_weights, quantum_circuit,dev,set_number,title):
    predictions = []
    labels = []
    
    start_time = time.perf_counter()
    for sample in test_data:
        feature_vector = sample[:8]  # Assuming the first 8 elements are features
        actual_label = sample[8]  # Assuming the 9th element is the actual classification label
        sampled_outputs = [quantum_circuit.circuit(optimized_weights, feature_vector) for _ in range(dev.shots)]
        probability_pulsar = (1 - np.array(sampled_outputs)) / 2
        # Binarize the probabilities to 0 or 1 based on a threshold, here 0.5
        predicted_labels = [1 if prob > 0.5 else 0 for prob in probability_pulsar]
        majority_label = most_common(predicted_labels)
        predictions.append(majority_label)
        labels.append(actual_label)  # Use the actual label from the data
    
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print("Elapsed time {0}".format(elapsed_time))
    #from sklearn.metrics import confusion_matrix
    
    fpr, tpr, _ = roc_curve(labels, predictions) #returning these for ROC curve

    
    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
    
    # Metrics Calculation
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
    npv = tn / (tn + fn) if (tn + fn) != 0 else 0
    balanced_accuracy = (recall + specificity) / 2
    geometric_mean = np.sqrt(recall * specificity)
    informedness = recall + specificity - 1
    
    metric_dict = {
    'accuracy': accuracy,
    'recall': recall,
    'precision': precision,
    'specificity': specificity,
    'npv': npv,
    'balanced_accuracy': balanced_accuracy,
    'geometric_mean': geometric_mean,
    'informedness': informedness}
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"NPV: {npv:.4f}")
    print(f"Balanced Accuracy: {balanced_accuracy:.4f}")
    print(f"Geometric Mean: {geometric_mean:.4f}")
    print(f"Informedness: {informedness:.4f}")
    return predictions, metric_dict,fpr,tpr