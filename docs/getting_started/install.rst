.. _install:

.. highlight:: shell

Installation
============

**Pyreal** can be installed in two ways:

* From PyPI
* From source

Stable Release
--------------

To install **Pyreal**, run the following command in your terminal `pip`:

.. code-block:: console

    pip install pyreal

This is the preffered method to install **Pyreal**, as it will always install the most recent
and stable release.

If you don't have `pip`_ installed, this `Python installation guide`_ can guide
you through the process.

From source
-----------

The source code of **Pyreal** can be downloaded from the `Github repository`_

You can clone the repository and install with the following command in your terminal:

You can clone the repository and install it from source by running ``make install`` on the
``stable`` branch:

.. code-block:: console

    git clone git://github.com/dai-lab/pyreal
    cd pyreal
    git checkout stable
    make install

.. note:: The ``master`` branch of the SDV repository contains the latest development version.
          If you want to install the latest stable version, make sure not to omit the
          ``git checkout stable`` indicated above.

If you are installing **Pyreal** in order to modify its code, the installation must be done
from its sources, in the editable mode, and also including some additional dependencies in
order to be able to run the tests and build the documentation.

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/
.. _Github repository: https://github.com/dai-lab/pyreal
