compyute
========
.. automodule:: compyute
.. currentmodule:: compyute

Tensors
-------
.. autoclass:: Tensor
    :members:
    :show-inheritance:
.. autofunction:: tensor

Creating and Combining Operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: append
.. autofunction:: arange
.. autofunction:: concatenate
.. autofunction:: empty
.. autofunction:: empty_like
.. autofunction:: full
.. autofunction:: full_like
.. autofunction:: identity
.. autofunction:: linspace
.. autofunction:: ones
.. autofunction:: ones_like
.. autofunction:: split
.. autofunction:: stack
.. autofunction:: zeros
.. autofunction:: zeros_like

Reshaping Operations
~~~~~~~~~~~~~~~~~~~~
.. autofunction:: diagonal
.. autofunction:: reshape
.. autofunction:: flatten
.. autofunction:: transpose
.. autofunction:: insert_dim
.. autofunction:: add_dims
.. autofunction:: resize
.. autofunction:: repeat
.. autofunction:: tile
.. autofunction:: pad
.. autofunction:: pad_to_shape
.. autofunction:: moveaxis
.. autofunction:: squeeze
.. autofunction:: flip
.. autofunction:: broadcast_to

Selecting Operations
~~~~~~~~~~~~~~~~~~~~
.. autofunction:: argmax
.. autofunction:: get_diagonal
.. autofunction:: tril
.. autofunction:: triu
.. autofunction:: unique

Transforming and Computing Operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: abs
.. autofunction:: clip
.. autofunction:: cos
.. autofunction:: cosh
.. autofunction:: dot
.. autofunction:: einsum
.. autofunction:: exp
.. autofunction:: fft1d
.. autofunction:: fft2d
.. autofunction:: histogram
.. autofunction:: inner
.. autofunction:: ifft1d
.. autofunction:: ifft2d
.. autofunction:: log
.. autofunction:: log2
.. autofunction:: log10
.. autofunction:: max
.. autofunction:: maximum
.. autofunction:: mean
.. autofunction:: min
.. autofunction:: minimum
.. autofunction:: outer
.. autofunction:: prod
.. autofunction:: real
.. autofunction:: round
.. autofunction:: sech
.. autofunction:: sin
.. autofunction:: sinh
.. autofunction:: sqrt
.. autofunction:: std
.. autofunction:: tan
.. autofunction:: tanh
.. autofunction:: tensorprod
.. autofunction:: tensorsum
.. autofunction:: sum
.. autofunction:: var

Data Types
----------
.. autoclass:: int8
.. autoclass:: int16
.. autoclass:: int32
.. autoclass:: int64
.. autoclass:: float16
.. autoclass:: float32
.. autoclass:: float64
.. autoclass:: complex64
.. autoclass:: complex128

Devices
-------
.. autoclass:: cpu
.. autoclass:: cuda