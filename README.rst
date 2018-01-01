=========
rr.approx
=========

Approximate floating point arithmetic library. This simple module can be used to compare numbers using a relative and absolute tolerance, to mitigate (not eliminate!) floating point rounding errors.

.. code-block:: python

    from rr.approx import Approx

    x = Approx(0.1) * 3
    x == 0.3  # True

The ``Approx`` class is very simple to use as a replacement for "regular" floats -- you can use ``Approx`` objects instead of floats in most (if not all) contexts: arithmetic and comparisons.

The ``Approx.context()`` context manager allows temporary modification of relative and absolute tolerance parameters.

.. code-block:: python

    print(Approx.rtol, Approx.atol)  # display current parameters
    print(Approx(0.1) * 3 == 0.3)  # make a test comparison
    with Approx.context(rtol=0, atol=0):  # temporary modification
        print(Approx.rtol, Approx.atol)  # show modified tolerances
        print(Approx(0.1) * 3 == 0.3)  # rerun the comparison
    print(Approx.rtol, Approx.atol)  # back to original parameters


Disclaimer
==========

Please note that this module **does not solve** any problems with floating point numbers. What it does is provide a little (or big, depending on how you configure your tolerance parameters) buffer to compensate for rounding errors, but as these errors accumulate you can always get unexpected results.

**Float rounding is inherently inaccurate** in all computers and programming languages due to representing a possibly recurring decimal in a finite number of digits. If you really care about exact results and don't mind paying a performance penalty, you should check out the `decimal <https://docs.python.org/3/library/decimal.html>`_ module from the standard library or some other alternative.


Compatibility
=============

Developed and tested in Python 3.6+. The code may or may not work under earlier versions of Python 3 (perhaps back to 3.3).


Installation
============

From the github repo:

.. code-block:: bash

    pip install git+https://github.com/2xR/rr.approx.git


License
=======

This library is released as open source under the MIT License.

Copyright (c) 2017 Rui Rei
