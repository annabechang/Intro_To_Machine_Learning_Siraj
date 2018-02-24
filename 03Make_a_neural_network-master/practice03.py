#video url:https://www.youtube.com/watch?v=p69khggr1Jo&list=PL2-dafEMk2A7YdKv4XfKpfbTH5z6rEEj3&index=3

from numpy import exp, array, random, dot


#define neural netwrok class


class NeuralNetwork():
    def __init__(self):
        #seed the random number generator so it generates the same 
        #num everytime the program run
        random.seed(1)

        
        #we model a single neuron, with 3 input connections and 1 output connection.
        #we assign random weights to a 3 x 1 matrix, with values in range 
        #-1 to 1 and mean = 0
        self.synaptic_weights = 2 * random.random((3, 1)) - 1


        #the activation function, the sigmoid funtion which describes an s shaped curbe
        #we pass the weighted sum of inputs through this function to 
        #convert into probability between 0 and 1 
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

        #the function that culculate the derivative of the sigmoid
        #which give us the gradient / slope
        # this measure how confident we are to the weight and help us up date
        #the prediction

    def __sigmoid_derivative(self, x):
        return x * (1 - x)



    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in xrange(number_of_training_iterations):
            #pass the training set through our neural net 
            output = self.think(training_set_inputs)


            #calculate the error
            error = training_set_outputs - output


            #multiply the error by the input ad afain by the gradient 
            #of the sigmoid curve
            #gradient descent
            adjustment = dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))


            #adj weights. 
            self.synaptic_weights += adjustment

    #use sigmoid in predict funtion which pass inputs 
    #as parameter and pass it through our neural network 

    #def predict(self, inputs):
    #    return self.__sigmoid(dot(inputs, self.synaptic_weights))

    # The neural network thinks.
    def think(self, inputs):
        # Pass inputs through our neural network (our single neuron).
        return self.__sigmoid(dot(inputs, self.synaptic_weights))


#main function
if __name__ == "__main__":

    
    #initialise a single neuron neural network
    neural_network = NeuralNetwork()
    
    #print out starting weights for a reference when we demo
    print ('Random starting synaptic weight :')
    print (neural_network.synaptic_weights)
    
    # in the training set we have 4 exps. each has 3 input
    #1 output 
    #the t function transposes the matrix 
    #from horizontal to vertical
    training_set_inputs = array([[0,0,1],[1,1,1],[1,0,1],[0,1,1]])
    training_set_outputs = array([[0,1,1,0]]).T
    
    #train the neural network with the training set
    #do it 10k times nd make small adjs each time
    neural_network.train(training_set_inputs,training_set_outputs,10000)
    
    print ('New synaptic weights after training:')
    print (neural_network.synaptic_weights)
    
    #test the neural network with a new situation

    
    print ("Considering new situation")
    print neural_network.think(array([1,0,0]))




