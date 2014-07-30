Module :py:mod:`pyqt_fit.kde_methods`
=====================================

.. automodule:: pyqt_fit.kde_methods

Univariate KDE estimation methods
---------------------------------

The exact definition of such a method is found in :py:attr:`pyqt_fit.kde.KDE1D.method`

.. autofunction:: generate_grid

.. autoclass:: KDE1DMethod
   :members: unbounded, __call__, grid, __str__


Estimation methods
``````````````````

Here are the methods implemented in pyqt_fit. To access these methods, the simplest is to use the instances provided:

.. py:data:: renormalization

    Instance of the :py:class:`RenormalizationMethod` class.

.. py:data:: reflection

    Instance of the :py:class:`ReflectionMethod` class.

.. py:data:: linear_combination

    Instance of the :py:class:`LinearCombinationMethod` class.

.. py:data:: cyclic

    Instance of the :py:class:`CyclicMethod` class.

Classes implementing the estimation methods
```````````````````````````````````````````

.. autoclass:: RenormalizationMethod

.. autoclass:: ReflectionMethod

.. autoclass:: LinearCombinationMethod

.. autoclass:: CyclicMethod

