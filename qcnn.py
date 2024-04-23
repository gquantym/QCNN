import pennylane as qml

# Define the device
dev = qml.device('lightning.qubit', wires=8,shots = 1000)

#print("Device:", dev)
#print("Number of shots:", dev.shots)


def encoding(feature_vector):
    for i, x_ in enumerate(feature_vector):
        qml.RY(x_,wires=i)
    
def convolution_1(theta1):
    qml.RY(theta1[0,0],wires=0)
    qml.RY(theta1[5,0],wires=5)
    qml.RY(theta1[6,0],wires=6)
    qml.RY(theta1[7,0],wires=7)
    qml.CNOT(wires=[0,5])
    qml.RY(theta1[0,1],wires=0)
    qml.RY(theta1[1,0],wires=1)
    qml.CNOT(wires=[1,6])
    qml.RY(theta1[2,0],wires=2)   
    qml.CNOT(wires=[2,7])
    qml.RY(theta1[3,0],wires=3)
    qml.RY(theta1[4,0],wires=4)
    #qml.RY(theta1[5,1],wires=5)
    qml.RY(theta1[6,1],wires=6)
    qml.RY(theta1[7,1],wires=7)
    qml.CNOT(wires=[3,0])
    qml.RY(theta1[1,1],wires=1)
    qml.CNOT(wires=[4,1])
    qml.RY(theta1[2,1],wires=2)
    qml.CNOT(wires=[5,2])
    #qml.RY(theta1[3,1],wires=3)
    qml.CNOT(wires=[6,3])
    #qml.RY(theta1[4,1],wires=4)
    qml.CNOT(wires=[7,4])
    
def pooling_1():
    qml.CNOT(wires=[7,0])
    qml.CNOT(wires=[6,1])
    qml.CNOT(wires=[5,2])
    qml.CNOT(wires=[4,3])
    
def convolution_2(theta2):
    qml.RY(theta2[0,0],wires=0)
    qml.RY(theta2[1,0],wires=1)
    qml.RY(theta2[2,0],wires=2)
    #qml.RY(theta2[3,0],wires=3)
    qml.CNOT(wires=[0,1])
    #qml.RY(theta2[0,1],wires=0)
    qml.RY(theta2[1,1],wires=1)
    qml.CNOT(wires=[1,2])
    qml.RY(theta2[2,1],wires=2)
    qml.CNOT(wires=[2,3])
    #qml.RY(theta2[3,1],wires=3)     
    qml.CNOT(wires=[3,0])
    
def pooling_2():
    qml.CNOT(wires=[3,0])
    qml.CNOT(wires=[2,1])
    
def convolution_3(theta3):
    #qml.RY(theta3[0], wires=0)
    qml.RY(theta3[1], wires=1)
    qml.CNOT(wires=[0,1])

def pooling_3():
    qml.CNOT(wires=[1,0])
    
    
# Define the quantum function
@qml.qnode(dev, diff_method = "adjoint")
def circuit(weights,feature_vector):
    
    theta_1 = weights[0:8]
    theta_2 = weights[8:12]
    theta_3 = weights[12]
    
    encoding(feature_vector)
    convolution_1(theta_1)
    pooling_1()
    convolution_2(theta_2)
    pooling_2()
    convolution_3(theta_3)
    pooling_3()   
    return qml.expval(qml.PauliZ(wires=0))


#drawn_circuit = qml.draw(circuit)()

# Display the circuit
#print(drawn_circuit)


