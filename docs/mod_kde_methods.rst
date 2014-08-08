Module :py:mod:`pyqt_fit.kde_methods`
=====================================

.. automodule:: pyqt_fit.kde_methods

.. py:currentmodule:: pyqt_fit.kde_methods

Univariate KDE estimation methods
---------------------------------

The exact definition of such a method is found in :py:attr:`pyqt_fit.kde.KDE1D.method`

.. autofunction:: generate_grid

.. autofunction:: compute_bandwidth

.. autoclass:: KDE1DMethod

   The following methods are interface methods that should be overriden with ones specific to the implemented method. 

   .. automethod:: fit

   .. automethod:: pdf

   .. automethod:: __call__

   .. automethod:: grid

   .. automethod:: cdf

   .. automethod:: cdf_grid

   .. automethod:: icdf

   .. automethod:: icdf_grid

   .. automethod:: sf

   .. automethod:: sf_grid

   .. automethod:: isf

   .. automethod:: isf_grid

   .. automethod:: hazard

   .. automethod:: hazard_grid

   .. automethod:: cumhazard

   .. automethod:: cumhazard_grid

   .. attribute:: name

      :type: str

      Specify a human-readable name for the method, for presentation purposes.

   But the class also provide a number of utility methods to help implementing you own:

   .. automethod:: numeric_cdf

   .. automethod:: numeric_cdf_grid

Estimation methods
``````````````````
Here are the methods implemented in pyqt_fit. To access these methods, the simplest is to use the instances provided.

.. py:data:: unbounded

    Instance of the :py:class:`KDE1DMethod` class.

.. py:data:: renormalization

    Instance of the :py:class:`RenormalizationMethod` class.

.. py:data:: reflection

    Instance of the :py:class:`ReflectionMethod` class.

.. py:data:: linear_combination

    Instance of the :py:class:`LinearCombinationMethod` class.

.. py:data:: cyclic

    Instance of the :py:class:`CyclicMethod` class.

.. autofunction:: transformKDE1D

.. py:data:: default_method

    Method used by :py:class:`pyqt_fit.kde.KDE1D` by default.
    :Default: :py:data:`reflection`

Classes implementing the estimation methods
```````````````````````````````````````````

.. autoclass:: RenormalizationMethod

.. autoclass:: ReflectionMethod

.. autoclass:: LinearCombinationMethod

.. autoclass:: CyclicMethod

.. autofunction:: create_transform

.. autoclass:: TransformKDE1DMethod

