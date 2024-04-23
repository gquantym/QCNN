import pennylane as qml

# Define the device
dev = qml.device('lightning.qubit', wires=8,shots=1000)

#print("Device:", dev)
#print("Number of shots:", dev.shots)


def encoding(feature_vector):
    for i, x_ in enumerate(feature_vector):
        qml.RY(x_,wires=i)
    
def W(theta):
    for i in range(8):
        qml.RY(theta[i],wires=i)
    qml.CNOT(wires=[0,5])
    qml.CNOT(wires=[1,6])
    qml.CNOT(wires=[2,7])
    qml.RY(theta[8],wires=0)
    qml.RY(theta[9],wires=1)
    qml.RY(theta[10],wires=6)
    qml.RY(theta[11],wires=7)
    qml.CNOT(wires=[3,0])
    qml.CNOT(wires=[4,1])
    qml.CNOT(wires=[7,0])
    qml.CNOT(wires=[6,1])
    qml.RY(theta[12],wires=0)
    qml.RY(theta[13],wires=1)
    qml.CNOT(wires=[0,1])
    qml.RY(theta[14],wires=1)
    qml.RY(theta[15],wires=2)
    qml.CNOT(wires=[1,2])
    qml.RY(theta[16],wires=2)
    qml.CNOT(wires=[2,1])
    qml.RY(theta[17],wires=1)


# Define the quantum function
@qml.qnode(dev, diff_method = "adjoint")
def circuit(weights,feature_vector):
    encoding(feature_vector)
    W(weights)
    return qml.expval(qml.PauliZ(wires=1))
#import numpy as np
#initial_weights = 2 * np.pi * np.random.random(size=(18))
#drawn_circuit = qml.draw(circuit)(initial_weights,np.array([0,1,2,3,4,5,6,7]))

# Display the circuit
#print(drawn_circuit)

