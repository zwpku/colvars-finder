Quick Start
===========

Example 1: Define an ANN that includes alignment layer. 
This gives a neural network function that is invariant under both rotation and translation.

.. code-block:: python

    import MDAnalysis as mda
    from molann.ann import AlignmentLayer, FeatureLayer, PreprocessingANN, MolANN, create_sequential_nn
    from molann.feature import Feature

