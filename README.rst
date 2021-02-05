OutPyRX: A Novel Bayesian Outlier Score Based on the
  Negative Binomial Distribution for
  Detecting Aberrantly Expressed Genes in
  RNA-Seq
  Gene Expression Count Data
=========================================================================================================

OutPyRX has been tested on Linux (Ubuntu)
and Windows 10.
Note that OutPyRX is still in
alpha stage,
so encountering bugs while
running it is expected.

Installation
------------
The recommended way is to install
OutPyRX using ``pip`` inside of a
virtualenv virtual environment,
after downloading this
whole repository as a zip file::

  pip install outpyrx.zip

It might take some time to install as it
needs to install some large packages
it depends on, such as TensorFlow.

Usage
-----
We recommend that you create a
separate workspace
directory for every data file that you
process as ``outpyrx`` will create
some files and folders inside the
directory where you invoke it.
The install process will create
an ``outpyrx`` executable
inside your environment.
Then you can run ``outpyrx``
on your data as follows::

 outpyrx [-h] [-s SETTINGS]
              [-p {normalize_sf}] [-n]
              data_file

The only parameter that you have to supply
to ``outpyrx`` is the ``data_file``,
which should be a tab-separated
pandas-compatible CSV file containing
gene expression counts.
Its index (first column) should
contain names of genes,
while its columns header (first row)
should contain the names of samples
that the counts were sequenced from.
Other cells should contain
integer count data.
The other (optional) options
of ``outpyrx`` are:

-h       prints help
-s, --settings=file        Python ``file`` to
 tweak additional ``outpyrx`` settings
-p, --preprocessing  what kind of preprocessing
 to use;
 for now only ``normalize_sf`` is supported,
 which will normalize the counts using
 DESeq size factors
-n, --nosubset  Don't filter/subset the Kremer dataset

A usage primer
--------------
We will run OutPyRX using the publicly
available dataset from
https://media.nature.com/original/nature-assets/ncomms/2017/170612/ncomms15824/extref/ncomms15824-s1.txt,
which was published as part of the
supplemental material
of the paper [Kremer2017]_.

The terminal commands below are for a Linux OS,
but they should be straightforward to
"translate" to Windows.

::

 mkdir workspace
 cd workspace
 outpyrx -p normalize_sf https://media.nature.com/original/nature-assets/ncomms/2017/170612/ncomms15824/extref/ncomms15824-s1.txt

If you supply an URL as its ``data_file``
parameter it will save the file from
the URL.
In the example above, it will save
to the working directory the file
``ncomms15824-s1.txt``.

After OutPyRX finishes processing the data,
it will create a CSV files containing
the p-values.
This file will end with
"-opx-pv.csv".

It will also create a directory that
holds the full trace of the inferred
parameters of the underlying negative
binomial (NB) process.
For the example above it will
create the directory
``[ALL]-sp-normalized_sf-from-ncomms15824-s1``..

Note that Python itself might create
a ``__pycache__`` directory.

Browsing the trace
------------------
While still being in your workspace
directory and inside the OutPyRX-enabled
virtual environment,
you can explore the trace directory from
Python.
E.g. after opening a Python shell,
run the following Python commands::

 import outpyr.helpers as h
 import outpyr.helpers_tensorflow as htf

 j = 0
 ti = htf.TraceInspector(r'[ALL]-sp-normalized_sf-from-ncomms15824-s1')
 ti.plot_r_j_trace(j)

which will plot the trace (after warm-up)
for the dispersion parameter of the first
gene (``j = 0``) in the dataset.
If you want to see the full trace,
including the warm-up run::

 h.plot(ti._get_param_j_trace('r', j))

And you can also plot a histogram of the
trace::

 h.histogram(ti._get_param_j_trace('r', j))

Note that OutPyRX uses
standard numpy arrays for trace(s).

To get the p-value for a particular count
in the dataset, let's say for gene ``j = 0`` and
sample ``i = 1`` use::

 j = 0
 i = 1
 ti.get_p_value_matrix()[j, i]

You can also get a trace of the p-value
with::

 h.plot(ti.get_p_value_ji_trace(j, i))

And there is also a full-trace version::

 h.plot(ti._get_p_value_ji_trace(j, i))

Finally, there is a function that will
sort all p-values in the p-value matrix
in ascending order,
and show their indices.
E.g. the following will show the first
10 counts with lowest p-values::

 h.sort_p_values(ti.get_p_value_matrix())[:10]

.. [Kremer2017] Kremer, L.,
 Bader, D., Mertes, C. et al.
 Genetic diagnosis of Mendelian disorders
 via RNA sequencing. Nature Communications 8,
 15824 (2017) doi:10.1038/ncomms15824
