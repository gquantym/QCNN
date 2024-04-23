import pennylane as qml

# Define the device
dev = qml.device('lightning.qubit', wires=8,shots=1000)

#print("Device:", dev)
#print("Number of shots:", dev.shots)


def encoding(feature_vector):
    for i, x_ in enumerate(feature_vector):
        qml.RY(x_,wires=i)
    
def c_1(theta):
    for i in range(8):
        qml.RY(theta[i,0],wires=i)
    qml.CNOT(wires=[0,4])
    qml.CNOT(wires=[1,5])
    qml.CNOT(wires=[2,6])
    qml.CNOT(wires=[3,7])
    for i in range(8):
        qml.RY(theta[i,1],wires=i)
def p_1():
    qml.CNOT(wires=[4,0])
    qml.CNOT(wires=[5,1])
    qml.CNOT(wires=[6,2])
    qml.CNOT(wires=[7,3])

def c_2(theta):
    for i in range(4):
        qml.RY(theta[i,0],wires=i)
    qml.CNOT(wires=[0,2])
    qml.CNOT(wires=[1,3])
    for i in range(4):
        qml.RY(theta[i,1],wires=i)
def p_2():
    qml.CNOT(wires=[2,0])
    qml.CNOT(wires=[3,1])
    
def c_3(theta):
    for i in range(2):
        qml.RY(theta[i,0],wires=i)
    qml.CNOT(wires=[0,1])
    for i in range(2):
        qml.RY(theta[i,1],wires=i)

def p_3():
    qml.CNOT(wires=[1,0])
def c_4(theta):
    qml.RY(theta,wires=0)

# Define the quantum function
@qml.qnode(dev, diff_method = "adjoint")
def circuit(weights,feature_vector):
    encoding(feature_vector)
    theta1 = weights[0:8]
    theta2 = weights[8:12]
    theta3 = weights[12:14]
    theta4 = weights[14][0]
    c_1(theta1)
    p_1()
    c_2(theta2)
    p_2()
    c_3(theta3)
    p_3()
    c_4(theta4)
    
    return qml.expval(qml.PauliZ(wires=0))
#import numpy as np
#initial_weights = 2 * np.pi * np.random.random(size=(15,2))
#drawn_circuit = qml.draw(circuit)(initial_weights,np.array([0,1,2,3,4,5,6,7]))

# Display the circuit
#print(drawn_circuit)


