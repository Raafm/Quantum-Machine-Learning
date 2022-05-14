##!pip install qiskit

import pandas as pd
import numpy as np
from math import ceil
import matplotlib.pyplot as plt


from qiskit import QuantumCircuit, QuantumRegister, AncillaRegister, ClassicalRegister
from qiskit import transpile, assemble
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit import Aer


from sklearn.model_selection import train_test_split



"""# Dataset"""

from sklearn.datasets import load_digits
digits = load_digits()
digits


ones = digits.data[digits.target == 1]
zeros = digits.data[digits.target == 0]



N_1 = ones.shape[0]
N_0 = zeros.shape[0]

print("number if digits 1: ",N_1)
print("number if digits 0: ",N_0)

X = np.vstack([ones,zeros])
y = np.array([1]*N_1+[0]*N_0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

print("N° of images in training_set:      ",X_train.shape[0])
print("N° of labels for the training_set: ",y_train.shape[0])

print("N° of images in testset:      ",X_test.shape[0])
print("N° of labels for the testset: ",y_test.shape[0])

indices = np.random.choice(
                            list(range(X_train.shape[0])),
                            10
                        )

indices

print("training set")
for x_sampled,y_sampled in zip(X_train[indices],y_train[indices]):
    
    
    print()

"""# mean digit"""

ideal_one = np.array( list( ones[:,i].mean() for i in range(64) ) )  
ideal_zero = np.array( list( zeros[:,i].mean() for i in range(64) ) )  

"""# Neurônio do Mangini"""

class QuantumNeuron:
    
    
    def Ui(self,theta):

        #Changing the simulator 
        self.circ.barrier()
        for posicoes,tetha_i in zip( self.posicoes_state_vector[:-1],theta[:-1]):
        
            self.circ.x(posicoes)
            
            # apply multicontrolled Z gate rotation (Gambiarra due to bug in qiskit)
            self.circ.mcx(self.states[:-1],self.ancilla_rotation)
            self.circ.crz(tetha_i,self.ancilla_rotation,self.states[-1])
            self.circ.mcx(self.states[:-1],self.ancilla_rotation)

            # undo all modifications
        
            self.circ.x(posicoes)
            self.circ.barrier()

        tetha_i = theta[-1]
    
        # apply multicontrolled Z gate

        self.circ.mcx(self.states[:-1],self.ancilla_rotation)
        self.circ.crz(tetha_i,self.ancilla_rotation,self.states[-1])
        self.circ.mcx(self.states[:-1],self.ancilla_rotation)
        

        # undo all modifications
        
        self.circ.barrier()

    def todos_binarios(self,N):
        n_qubits = ceil(np.log2(N))
        Nstates = 2**n_qubits

        posicoes_state_vector = []
        for numero in range(Nstates):
            binario = bin(numero)[2:]   # string binaria do estado
            n0_antes = n_qubits - len(binario) # numero de bits 0 que devem ser add a string de bits para completar todos qubits
            posicoes = []
            # apply X gates in bits 0 of the state vector
            for i,c in enumerate(reversed('0'*n0_antes + binario)):
                if c == '0':
                    posicoes.append(i)

            posicoes_state_vector.append(posicoes)

        return posicoes_state_vector

    def __init__(self,size_feat,shots=2**11):

        self.simulator_backend = Aer.get_backend('aer_simulator')
        
        print("__________________________")
        if gpu_flag:
            print("Using gpu")
            self.simulator_backend.set_options(device='GPU')
        else: print("not using gpu")
        print("__________________________")
        
        self.n_qubits =  ceil(np.log2(size_feat))
        self.posicoes_state_vector = self.todos_binarios(2**self.n_qubits)

        self.states = QuantumRegister(self.n_qubits, 'states') 
        self.ancilla = AncillaRegister(1, 'ancilla')
        self.ancilla_rotation = AncillaRegister(1, 'ancilla_rotation')
        self.output = ClassicalRegister(1, 'reg_class')

        self.circ = QuantumCircuit(self.states, self.ancilla,self.output,self.ancilla_rotation ) 
        self.shots = shots

        self.r_param = []
        self.w_param = []

        for i in range(size_feat):
            self.w_param.append(Parameter('w_'+str(i)))
            self.r_param.append(Parameter('r_'+str(i)))

        self.circ.h(self.states)

        self.Ui(self.r_param)     
        self.Ui(self.w_param)     

        self.circ.h(self.states)
        self.circ.x(self.states)

        self.circ.mcx(self.states,self.ancilla)

        self.circ.measure(self.ancilla,self.output)

        self.circ = transpile(self.circ,
                         self.simulator_backend)


    def forward(self, rotations, weights):


        param_dict = {}

        for i in range(len(weights)):
            param_dict[self.w_param[i]] = weights[i]
            param_dict[self.r_param[i]] = rotations[i]



        qobj = assemble(self.circ,
                        shots=self.shots,
                        parameter_binds = [param_dict])


        results = self.simulator_backend.run(qobj).result()
        
        all_counts = results.get_counts()

        if '1' in all_counts:

            counts = np.array(results.get_counts()['1'])
            
            # Compute probabilities for each state
            probabilities = counts / self.shots

            return probabilities

        else:
            return 0

"""# otimizador"""

def distribuicao(L,parameters,N_medicoes = 100):
    medicoes = np.array(
                    [L(parameters) for _ in range(N_medicoes)]
                )
    
    return medicoes

np.random.seed(42)

def grad(L,w,ck):
    p = len(w)
    deltak = np.random.choice([-1, 1], size=p)
    ck_deltak = ck*deltak

    DELTA_L = L(w +  ck_deltak) -  L(w -  ck_deltak)
    return ( DELTA_L ) /(2*ck_deltak)

"""a should be choosen by:


$\frac{a}{(A+1)^{\alpha}} magnitude\_g0 ≈ Δw_0 $
"""

def SPSA(L,parameters,alpha = 0.602,gamma = 0.101, N_iterations = int(1e5), min_loss = 10):

    w = parameters
    p = len(w)
    c = 1e-2
    magnitude_g0 = np.abs(  grad(L,w,c).mean() )

    A = N_iterations*0.1
        
    a = 2*((A+1)**alpha)/magnitude_g0

    for k in range(1,N_iterations):
        
        if k%10 == 0: 
            loss = L(w)
            print(f"it {k} | Loss: {round(loss,4)}")
            
            if loss < min_loss: break
                
        ak = a/((k+A)**(alpha))
        ck = c/(k**(gamma))
    
        gk = grad(L,w,ck)

        w -= ak*gk
    
        
    return w

"""# treinamento"""

############################# gpu settings ##############################
#########################################################################
gpu_flag = input("run on a GPU? y or n:   ")
while gpu_flag != "y" and gpu_flag != "n":
    print(f"{gpu_flag} not an valid awnser")
    gpu_flag = input("run on a GPU? y or n")

gpu_flag =     gpu_flag == 'y'  # if use gpu:  gpu_flag == True

print("_"*50)

def perceptron_weights(weights, input, quantic_neuron = None):

    
    if quantic_neuron is None:
        n_qubits = np.ceil(np.log2(len(weights)))
        activation = np.abs(  1 + np.sum(np.exp(1j*(input-weights)))  )**2/(2**(2*n_qubits)) + (2*np.random.rand(1) - 1)*1e-4

        output = np.float32(activation)
    
        return output[0]
    else:
        output = quantic_neuron.forward(input,weights)

        return output

def loss_function(parameters,x,y,quantic_neuron):

    w = parameters
    outputs = []

    for i in range(len(x)):
    
      outputs.append(  perceptron_weights(w,x[i],quantic_neuron)    )

    outputs = np.array(outputs)

    photo_loss =  ((outputs - y)**2).mean(axis=None )

    return photo_loss

weights_guess = np.random.rand(64)
quantic_neuron = QuantumNeuron(len(weights_guess))

def train(X_train,y_train):

    x,y = X_train,y_train


    print("antes do treino")
    print("prev output:",end = " ")
    for x_i in x:
        print(perceptron_weights(weights_guess,x_i,quantic_neuron),end=" ")
    print()
    print("y = ",y)
    print("\ntreino:\n" + '-'*30)

    loss = lambda parameters: loss_function(parameters,x,y,quantic_neuron)


    w = SPSA( loss, weights_guess,min_loss = 0.2 )


    print("-"*30 )
    print("\ndepois do treino")
    print("output = ",end=" ")
    for x_i in x:
        print(perceptron_weights(w,x_i,quantic_neuron),end=" ")
    print()
    print("era pra ser = ",y)
    print("_"*50)
    print("\n\n\n")

    return w

w=train(X_train,y_train)

raw_output = np.array([perceptron_weights(w,x_i,quantic_neuron) for x_i in X_train])
raw_output

one_indices  = np.where(y_train == 1)
zero_indices = np.where(y_train == 0)
one_indices

one_mean  = raw_output[one_indices].mean()
zero_mean = raw_output[zero_indices].mean()
print(f"mean of output that should be one: {one_mean}")
print(f"mean of output that should be zero: {zero_mean}")

pd.DataFrame(100*raw_output[one_indices],columns = ["ones"])

pd.DataFrame(1e3*raw_output[one_indices],columns = ["ones"]).hist()
plt.show()
print()
pd.DataFrame(1e3*raw_output[zero_indices],columns = ["zeros"]).hist()
plt.show()

