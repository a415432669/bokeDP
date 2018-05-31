import math
import random

random.seed(0) #设置随机数种子


def rand(a, b):
    return (b - a) * random.random() + a #生成从[a,b)的随机数


def make_matrix(m, n, fill=0.0):#创造一个指定大小的矩阵
    mat = []
    for i in range(m):
        mat.append([fill] * n)
    return mat


def sigmoid(x): #设定激活函数为sigmoid哈数
    return 1.0 / (1.0 + math.exp(-x))


def sigmoid_derivative(x):#sigmoid激活函数的导数
    return x * (1 - x)


class BPNeuralNetwork:
    def __init__(self):
        
        self.input_n = 0
        self.hidden_n = 0
        self.output_n = 0
        #三个列表维护输入层，隐含层和输出层神经元，列表中的元素代表对应神经元当前的输出值.
        self.input_cells = []
        self.hidden_cells = []
        self.output_cells = []
        #使用两个二维列表以邻接矩阵的形式维护输入层与隐含层
        self.input_weights = []
        self.output_weights = []
        #隐含层与输出层之间的连接权值， 通过同样的形式保存矫正矩阵.
        self.input_correction = []
        self.output_correction = []

    def setup(self, ni, nh, no):
        self.input_n = ni + 1
        self.hidden_n = nh
        self.output_n = no
        
        # init cells
        self.input_cells = [1.0] * self.input_n #初始化输入层，长度为input_n的列表
        self.hidden_cells = [1.0] * self.hidden_n #初始化隐藏层，长度为hidden_n的列表
        self.output_cells = [1.0] * self.output_n #初始化输出层，长度为output_n的列表
        # init weights
        self.input_weights = make_matrix(self.input_n, self.hidden_n)#输入的权重为，n个输入层给n个隐藏层的2维列表权重
        self.output_weights = make_matrix(self.hidden_n, self.output_n) # 输出的权重为，n个隐藏层给n个输出层的2维列表权重
        # random activate，随机给出初始输入权重
        for i in range(self.input_n):
            for h in range(self.hidden_n):
                self.input_weights[i][h] = rand(-0.2, 0.2)
        for h in range(self.hidden_n):#随机给出初始输出权重
            for o in range(self.output_n):
                self.output_weights[h][o] = rand(-2.0, 2.0)
        # init correction matrix，初始化输入矫正权重和输出矫正权重
        self.input_correction = make_matrix(self.input_n, self.hidden_n)
        self.output_correction = make_matrix(self.hidden_n, self.output_n)
        print(self.input_weights)
        print(self.output_weights)
        print(self.input_correction)
        print(self.output_correction)

    def predict(self, inputs):
        # activate input layer
        for i in range(self.input_n - 1):
            self.input_cells[i] = inputs[i]
        # activate hidden layer
        for j in range(self.hidden_n):
            total = 0.0
            for i in range(self.input_n):
                total += self.input_cells[i] * self.input_weights[i][j]
            self.hidden_cells[j] = sigmoid(total)
        # activate output layer
        for k in range(self.output_n):
            total = 0.0
            for j in range(self.hidden_n):
                total += self.hidden_cells[j] * self.output_weights[j][k]
            self.output_cells[k] = sigmoid(total)
        return self.output_cells[:]

    def back_propagate(self, case, label, learn, correct):
        # feed forward
        self.predict(case)
        # get output layer error
        output_deltas = [0.0] * self.output_n
        for o in range(self.output_n):
            error = label[o] - self.output_cells[o]
            output_deltas[o] = sigmoid_derivative(self.output_cells[o]) * error
        # get hidden layer error
        hidden_deltas = [0.0] * self.hidden_n
        for h in range(self.hidden_n):
            error = 0.0
            for o in range(self.output_n):
                error += output_deltas[o] * self.output_weights[h][o]
            hidden_deltas[h] = sigmoid_derivative(self.hidden_cells[h]) * error
        # update output weights
        for h in range(self.hidden_n):
            for o in range(self.output_n):
                change = output_deltas[o] * self.hidden_cells[h]
                self.output_weights[h][o] += learn * change + correct * self.output_correction[h][o]
                self.output_correction[h][o] = change
        # update input weights
        for i in range(self.input_n):
            for h in range(self.hidden_n):
                change = hidden_deltas[h] * self.input_cells[i]
                self.input_weights[i][h] += learn * change + correct * self.input_correction[i][h]
                self.input_correction[i][h] = change
        # get global error
        error = 0.0
        for o in range(len(label)):
            error += 0.5 * (label[o] - self.output_cells[o]) ** 2
        return error

    def train(self, cases, labels, limit=10000, learn=0.05, correct=0.1):
        for j in range(limit):
            error = 0.0
            for i in range(len(cases)):
                label = labels[i]
                case = cases[i]
                error += self.back_propagate(case, label, learn, correct)

    def test(self):
        cases = [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ]
        labels = [[0], [1], [1], [0]]
        self.setup(2, 5, 1)
        self.train(cases, labels, 10000, 0.05, 0.1)
        for case in cases:
            print(self.predict(case))


if __name__ == '__main__':
    nn = BPNeuralNetwork()
    nn.test()
