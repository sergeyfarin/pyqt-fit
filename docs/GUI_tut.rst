.. Tutorial on the use of the GUI

Regression using the GUI - tutorial
===================================

Using the interface
-------------------

The script is starting from the command line with:

.. code-block:: shell

  $ pyqt_fit1d.py

Once starting the script, the interface will look like this:

.. image:: PyQt-GUI.png

The interface is organised in 4 sections:

1. the top-left of the window to define the data to load and process;
2. the bottom-left to define the function to be fitted and its parameters;
3. the top-right to define the options to compute confidence intervals;
4. the bottom-right to define the output options.

Loading the Data
^^^^^^^^^^^^^^^^
The application can load CSV files. The first line of the file must be the name
of the available datasets. In case of missing data, only what is available on
the two selected datasets are kept.

Once loaded, the available data sets will appear as option in the combo-boxes.
You need to select for the X axis the explaining variable and the explained
variable on the Y axis.

Defining the regression function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
First, you will want to choose the function. The available functions are listed
in the combo box. When selecting a function, the list of parameters appear in
the list below. The value presented are estimated are a quick estimation from
the data. You can edit them by double-clicking. It is also where you can
specify if the parameter is of known value, and should therefore be fixed.

If needed, you can also change the computation of the residuals. By default there are two kind of residuals:

  Standard
    residuals are simply the difference between the estimated and observed value.

  Difference of the logs
    residual are the difference of the log of the values.


Plotting and output
^^^^^^^^^^^^^^^^^^^
By default, the output consists in the data points, and the fitted function,
interpolated on the whole range of the input data. If is, however, possible to
both change the range of data, or even evaluate the function on the existing
data points rather than interpolated ones.

The result of the fitting can also be output. What is written correspond
exactly to what is displayed. The output is also a CSV file, and is meant to be
readable by a human.

Confidence interval
^^^^^^^^^^^^^^^^^^^
Confidence interval can be computed using bootstrapping. There are two kinds of
boostrapping implemented:

  regular bootstrapping
    The data are resampled, the pairs :math:`(x,y)` are kept. There is no
    assumption made. But is is often troublesome in regression, tending to
    flatten the results.

  residual resampling
    After the first evaluation, for each pair :math:`(x,y)`, we find the
    estimated value :math:`\hat{y}`. Then, the residuals are re-sampled, and
    new pairs :math:`(x,\hat{y}+r')` are recreated, :math:`r'` being the
    resampled residual.

Defining your own function
--------------------------
First, you need to define the environment variable ``PYQTFIT_PATH`` and add a
list of colon-separated folders. In each folder, you can add python modules in
a ``functions`` sub-folder. For example, if the path ``~/.pyqtfit`` is in
``PYQTFIT_PATH``, then you need to create a folder ``~/.pyqtfit/functions``, in
which you can add your own python modules.

Which module will be loaded, and the functions defined in it will be added in
the interface. A function is a class or an object with the following
properties:

  name
    Name of the function

  description
    Equation of the function

  args
    List of arguments

  ``init_args(x,y)``
    Function guessing some initial arguments from the data. It must return a
    list or tuple of values, one per argument to the function.

  ``__call__(args, x)``
    Compute the function. The ``args`` argument is a tuple or list with as many
    elements are in the ``args`` attribute of the function.

  ``Dfun(args, x)``
    Compute the jacobian of the residuals. If the function is not provided, the
    attribute should be set to None, and the jacobian will be estimated
    numerically.

Defining your own residual
--------------------------

