from pennylane import numpy as np
import pennylane as qml
import time
import plot
def pulsar_probability(sampled_data, weights, quantum_circuit):
    global expectation_value
    expectation_value = [quantum_circuit.circuit(weights, feature_vector) for feature_vector in sampled_data]
    probability_pulsar = (1 - np.array(expectation_value)) / 2
    return probability_pulsar

def cross_entropy_loss(weights, data, quantum_circuit):
    predictions = np.array([quantum_circuit.circuit(weights, feature_vector) for feature_vector in data[:,:8]])
    predictions = (1-predictions)/2
    targets = np.array([row[-1] for row in data])
    targets = targets.reshape(predictions.shape)
    predictions = np.clip(predictions, 1e-15, 1 - 1e-15)
    #print("Prediction = ",predictions,"targets = ",targets)
    loss = -np.sum(targets * np.log(predictions) + (1 - targets) * np.log(1 - predictions))
    loss = loss / len(targets)
    return loss

def train(initial_weights, train_data, quantum_circuit,set_number,epochs):
    optimizer = qml.AdamOptimizer(stepsize=0.01)
    loss_array = np.array([])
    start_time = time.perf_counter()
    loss_queue = []
    gradient_array = [] 
    grad_fn = qml.grad(cross_entropy_loss, argnum=0)
    
    for epoch in range(epochs):
        current_gradient = grad_fn(initial_weights, train_data, quantum_circuit)
        gradient_array.append(current_gradient)
        optimized_weights, current_loss = optimizer.step_and_cost(lambda w: cross_entropy_loss(w, train_data, quantum_circuit), initial_weights)
        initial_weights = optimized_weights
        loss_array = np.append(loss_array, current_loss)
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch + 1}, Loss: {current_loss:.8f}")
        loss_queue.append(current_loss)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print("Training: Elapsed time (Set Number = {0}): {1}".format(set_number, elapsed_time))
    return optimized_weights,loss_array,gradient_array


def main(train_pulsar,train_non_pulsar,initial_weights,quantum_circuit,title,num_sets,epochs):
    train_data = np.concatenate((train_pulsar, train_non_pulsar), axis=1)
    weights_loss_values_grad = [train(initial_weights[i], train_data[i],quantum_circuit,i,epochs) for i in range(num_sets)]
    opt_weights, loss_values, param_grad = zip(*weights_loss_values_grad)
    
    opt_prob_non_pulsar = [pulsar_probability(train_non_pulsar[i,:,:8],opt_weights[i],quantum_circuit)for i in range(num_sets)]
    opt_prob_pulsar = [pulsar_probability(train_pulsar[i,:,:8],opt_weights[i],quantum_circuit) for i in range(num_sets)]
    opt_prob_non_pulsar = np.array(opt_prob_non_pulsar)
    opt_prob_pulsar = np.array(opt_prob_pulsar)

    return opt_weights,loss_values,param_grad