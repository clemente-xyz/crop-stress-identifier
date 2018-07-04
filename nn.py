import tensorflow as tf


class Neural_Network:
    def __init__(
        self,
        hl1_neurons,
        hl2_neurons,
        hl3_neurons,
        classes,
        batch_size,  # number of images that enters to the model every iteration
        img_dimension,
    ):
        self.hl1_neurons = hl1_neurons
        self.hl2_neurons = hl2_neurons
        self.hl3_neurons = hl3_neurons
        self.classes = classes
        self.batch_size = batch_size
        self.img_dimension = img_dimension
        self.x = tf.placeholder(
            "float", [None, img_dimension]
        )  # input vars, where we specify the dimension of the img (28*28=784 px)
        self.y = tf.placeholder("float")  # output var

    def model(self, x):
        # define the layers structure
        hidden_layer1 = {
            "weights": tf.Variable(
                tf.random_normal([self.img_dimension, self.hl1_neurons])
            ),
            "biases": tf.Variable(tf.random_normal([self.hl1_neurons])),
        }
        hidden_layer2 = {
            "weights": tf.Variable(
                tf.random_normal([self.hl1_neurons, self.hl2_neurons])
            ),
            "biases": tf.Variable(tf.random_normal([self.hl2_neurons])),
        }
        hidden_layer3 = {
            "weights": tf.Variable(
                tf.random_normal([self.hl2_neurons, self.hl3_neurons])
            ),
            "biases": tf.Variable(tf.random_normal([self.hl3_neurons])),
        }
        output_layer = {
            "weights": tf.Variable(tf.random_normal([self.hl3_neurons, self.classes])),
            "biases": tf.Variable(tf.random_normal([self.classes])),
        }

        # define the model it self: sum((x * weights) + biases) for every hidden layer
        l1 = tf.add(
            tf.matmul(x, hidden_layer1["weights"]), hidden_layer1["biases"]
        )  # the sum block for layer 1
        l1 = tf.nn.relu(l1)  # reactivation function (step function block) for layer 1

        l2 = tf.add(
            tf.matmul(l1, hidden_layer2["weights"]), hidden_layer2["biases"]
        )  # the sum block for layer 2
        l2 = tf.nn.relu(l2)  # reactivation function (step function block) for layer 2

        l3 = tf.add(
            tf.matmul(l2, hidden_layer3["weights"]), hidden_layer3["biases"]
        )  # the sum block for layer 3
        l3 = tf.nn.relu(l3)  # reactivation function (step function block) for layer 3

        output = tf.matmul(l3, output_layer["weights"]) + output_layer["biases"]

        return output

    def train_me(self, data_set):  # takes x: input data
        prediction = self.model(self.x)
        cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=self.y)
        )  # we calculate de difference between the prediction and the optimus value
        optimization = tf.train.AdamOptimizer().minimize(
            cost
        )  # using AdamOptimizer model we minimize the cost (considering learning_rate=0.001)

        no_epochs = 10  # number of cycles feed foward + backprop

        with tf.Session() as session:
            session.run(tf.initialize_all_variables())

            for epoch in range(no_epochs):
                epoch_loss = 0
                for i in range(int(data_set.train.num_examples / self.batch_size)):
                    epoch_x, epoch_y = data_set.train.next_batch(self.batch_size)
                    i, c = session.run(
                        [optimization, cost],
                        feed_dict={self.x: epoch_x, self.y: epoch_y},
                    )
                    epoch_loss = epoch_loss + c

                print(
                    "Epoch: ",
                    epoch,
                    "completed out of: ",
                    no_epochs,
                    "loss: ",
                    epoch_loss,
                )

            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(self.y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, "float"))

            print(
                "Accuracy: ",
                accuracy.eval(
                    {self.x: data_set.test.images, self.y: data_set.test.labels}
                ),
            )

