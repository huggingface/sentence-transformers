similarity
==========

``cos_sim`` returns a PyTorch tensor by default. Set ``convert_to_numpy=True`` to
receive the cosine similarity matrix as a NumPy array:

.. code-block:: python

    import numpy as np
    from sentence_transformers.util import cos_sim

    a = np.array([[1.0, 0.0]])
    b = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    scores = cos_sim(a, b, convert_to_numpy=True)
    print(scores)  # [[ 1.  0. -1.]]

NumPy output is detached from autograd and moved to CPU. Score values and dtypes
follow the default calculation. The flag defaults to ``False``, preserving the
tensor's device and gradients. ``pytorch_cos_sim`` always returns a tensor.

.. automodule:: sentence_transformers.util.similarity
    :members:
