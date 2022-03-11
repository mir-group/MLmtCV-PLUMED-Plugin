# ANN PLUMED plugin
----

## Features
* use any Tensorflow network's output as collective variables

## Prerequisite
* Tensorflow C API 1.13 + <https://www.tensorflow.org/install/lang_c>
* PLUMED 2.4.3

## Installation

Example bash script see "compile.sh".

1. Copy "ANN.cpp", "tf_utils.hpp", "tf_utils.cpp", to the src/function folder of the PLUMED source code
2. Export the Tensorflow libary and include to environment variables
3. Configure plumed with the Tensorflow api
4. make

## PLUMED script Examples

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
t1: TORSION ATOMS=5,7,9,15 NOPBC
t2: TORSION ATOMS=7,9,15,17 NOPBC
a: ANN ARG=t1,t2 MODELPATH=linear_simple INPUT=x OUTPUT=nn_return
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

a complete example is in the "test" folder.

## Generate Tensorflow model for the plug-in

Use the python Tensorflow to define the network, add names to the jacobian and gradient tensor. Note that the gradient tensor has to be flattened inside the python code.

The jacobian of the network output to the network input can be simply defined as the following

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from tensorflow.python.ops.parallel_for.gradients import jacobian
def grad(x):
    ''' Define the gradients '''
    J = jacobian(Y, X)
    return tf.reshape(J, [-1], name="grad_return")
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Simple_save can be used to save the model.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
sess = tf.keras.backend.get_session()
export_dir = "models/linear_simple"
inputs = {"x": X}
outputs = {"y": Y, "dy_dx": dy_dx}
tf.saved_model.simple_save(sess, export_dir, inputs=inputs,             outputs=outputs)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Complete example python codes to generate a simple ANN and save the model is in the "pysrc" folder.

