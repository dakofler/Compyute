compyute.nn
===========
.. automodule:: compyute.nn
.. currentmodule:: compyute.nn

Dataloaders
-----------
.. autoclass:: Dataloader
   :members:
   :special-members: __call__

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
.. autoclass:: Tanh
   :show-inheritance:

Blocks
~~~~~~
.. autoclass:: Convolution1dBlock
   :show-inheritance:
.. autoclass:: Convolution2dBlock
   :show-inheritance:
.. autoclass:: DenseBlock
   :show-inheritance:
.. autoclass:: ResidualBlock
   :show-inheritance:

Containers
~~~~~~~~~~
.. autoclass:: Container
   :show-inheritance:
   :members:
.. autoclass:: Sequential
   :show-inheritance:
.. autoclass:: ParallelConcat
   :show-inheritance:
.. autoclass:: ParallelAdd
   :show-inheritance:

Convolution
~~~~~~~~~~~
.. autoclass:: Convolution1d
   :show-inheritance:
.. autoclass:: Convolution2d
   :show-inheritance:
.. autoclass:: MaxPooling2d
   :show-inheritance:
.. autoclass:: AvgPooling2d
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
.. autoclass:: Batchnorm1d
   :show-inheritance:
.. autoclass:: Batchnorm2d
   :show-inheritance:
.. autoclass:: Layernorm
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

Parameter
---------
.. autoclass:: Buffer
   :members:
   :show-inheritance:
.. autoclass:: Parameter
   :members:
   :show-inheritance:
