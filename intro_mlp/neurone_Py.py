def simple_neuron(input_value, weight, bias):

    return input_value*weight+bias

a = int(input("insert a "))
b = int(input("insert b "))
c = int(input("insert c "))

Y = simple_neuron(a,b,c)

print(Y)