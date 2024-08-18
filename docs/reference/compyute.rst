compyute
===========
.. currentmodule:: compyute


Tensors
-------
.. autosummary::
    :toctree: ../_generated/compyute
    
    tensor
    Tensor


Tensor Operations
-----------------

Creating and Combining Operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
    :toctree: ../_generated/compyute
    
    append
    arange
    concat
    empty
    empty_like
    full
    full_like
    identity
    linspace
    ones
    ones_like
    split
    stack
    zeros
    zeros_like


Reshaping Operations
~~~~~~~~~~~~~~~~~~~~
.. autosummary::
    :toctree: ../_generated/compyute
    
    diagonal
    reshape
    flatten
    transpose
    insert_dim
    add_dims
    resize
    repeat
    tile
    pad
    pad_to_shape
    moveaxis
    squeeze
    flip
    broadcast_to


Selecting Operations
~~~~~~~~~~~~~~~~~~~~
.. autosummary::
    :toctree: ../_generated/compyute

    argmax
    get_diagonal
    tril
    triu
    unique


Transforming and Computing Operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
    :toctree: ../_generated/compyute

    abs
    all
    allclose
    clip
    cos
    cosh
    dot
    einsum
    exp
    fft1d
    fft2d
    histogram
    inner
    ifft1d
    ifft2d
    log
    log2
    log10
    max
    maximum
    mean
    min
    minimum
    norm
    outer
    prod
    real
    round
    sech
    sin
    sinh
    sqrt
    std
    tan
    tanh
    tensorprod
    tensorsum
    sum
    var


Data Types
----------
.. autoclass:: bool
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


Utility Functions
-----------------
.. autosummary::
    :toctree: ../_generated/compyute

    save
    load