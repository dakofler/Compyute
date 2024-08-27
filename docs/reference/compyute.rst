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
    convolve1d_fft
    convolve2d_fft
    cos
    cosh
    dot
    einsum
    exp
    fft1d
    fft2d
    histogram
    ifft1d
    ifft2d
    inner
    invert
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

Available data types are:

.. autosummary::
    :toctree: ../_generated/compyute
    
    bool_
    int8
    int16
    int32
    int64
    float16
    float32
    float64
    complex64
    complex128
    use_dtype


Devices
-------
.. autosummary::
    :toctree: ../_generated/compyute
    
    Device
    cpu
    cuda
    use_device


Utility Functions
-----------------
.. autosummary::
    :toctree: ../_generated/compyute

    save
    load