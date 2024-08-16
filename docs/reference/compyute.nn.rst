compyute.nn
===========
.. automodule:: compyute.nn
.. currentmodule:: compyute.nn

.. _losses:

Losses
------
.. autoclass:: Loss
   :special-members: __call__
.. autoclass:: BinaryCrossEntropy
   :show-inheritance:
   :special-members: __call__
.. autoclass:: CrossEntropy
   :show-inheritance:
   :special-members: __call__
.. autoclass:: MeanSquaredError
   :show-inheritance:
   :special-members: __call__

.. _metrics:

Metrics
-------
.. autoclass:: Metric
   :special-members: __call__
.. autoclass:: Accuracy
   :show-inheritance:
   :special-members: __call__
.. autoclass:: R2
   :show-inheritance:
   :special-members: __call__

.. _modules:

Modules
-------
.. autoclass:: Module
   :members:
   :special-members: __call__

.. _activations:

Activations
~~~~~~~~~~~
.. autoclass:: ReLU
   :show-inheritance:
.. autoclass:: LeakyReLU
   :show-inheritance:
.. autoclass:: GELU
   :show-inheritance:
.. autoclass:: Sigmoid
   :show-inheritance:
.. autoclass:: SiLU
   :show-inheritance:
.. autoclass:: Softmax
   :show-inheritance:
.. autoclass:: Tanh
   :show-inheritance:

Blocks
~~~~~~
.. autoclass:: Convolution1DBlock
   :show-inheritance:
.. autoclass:: Convolution2DBlock
   :show-inheritance:
.. autoclass:: DenseBlock
   :show-inheritance:

Containers
~~~~~~~~~~
.. autoclass:: Sequential
   :show-inheritance:
.. autoclass:: ParallelConcat
   :show-inheritance:
.. autoclass:: ParallelAdd
   :show-inheritance:
.. autoclass:: ResidualConnection
   :show-inheritance:

Convolution
~~~~~~~~~~~
.. autoclass:: Convolution1D
   :show-inheritance:
.. autoclass:: Convolution2D
   :show-inheritance:
.. autoclass:: MaxPooling2D
   :show-inheritance:
.. autoclass:: AvgPooling2D
   :show-inheritance:

Embedding
~~~~~~~~~
.. autoclass:: Embedding
   :show-inheritance:

Linear
~~~~~~
.. autoclass:: Linear
   :show-inheritance:

Normalizations
~~~~~~~~~~~~~~
.. autoclass:: BatchNorm1D
   :show-inheritance:
.. autoclass:: BatchNorm2D
   :show-inheritance:
.. autoclass:: LayerNorm
   :show-inheritance:
.. autoclass:: RMSNorm
   :show-inheritance:

Recurrent
~~~~~~~~~
.. autoclass:: GRU
   :show-inheritance:
.. autoclass:: LSTM
   :show-inheritance:
.. autoclass:: Recurrent
   :show-inheritance:

Regularization
~~~~~~~~~~~~~~
.. autoclass:: Dropout
   :show-inheritance:

Reshape
~~~~~~~
.. autoclass:: Reshape
   :show-inheritance:
.. autoclass:: Flatten
   :show-inheritance:
.. autoclass:: Moveaxis
   :show-inheritance:

Parameter and Buffer
--------------------
.. autoclass:: Parameter
   :members:
   :show-inheritance:
.. autoclass:: Buffer
   :members:
   :show-inheritance:
