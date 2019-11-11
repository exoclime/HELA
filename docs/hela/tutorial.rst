Tutorial
========

Fitting a line
--------------

First, we must generate some example data, which we can do using a built-in
function called `~hela.generate_example_data`, which returns the path to the
example file directory, the training dataset path, and the path to the samples
which we'd like to predict on:

.. code-block:: python

    from hela import generate_example_data
    # Generate an example dataset directory
    example_dir, training_dataset, samples_path = generate_example_data()

This handy command created an example directory called ``linear_data``,
which contains a training dataset described by the metadata file located at path
``training_dataset``. This training dataset contains a JSON file describing the
free parameters, which looks like this:

.. code-block:: python

    {"metadata":
        {"names": ["slope", "intercept"],
         "ranges": [[0, 1], [0, 1]],
         "colors": ["#F14532", "#4a98c9"],
         "num_features": 1000},
     "training_data": "training.npy",
     "testing_data": "testing.npy"}

This file tells the model what the two fitting parameters are and their ranges,
where to grab the training and testing datasets (in the npy pickle files), the
number of features (1000), the colors to use for each parameter in the plots.

We also generated a bunch of samples with a known slope and intercept, called
``samples_path``, on which we'll apply our trained random forest to estimate
the slope and intercept.

Once we have these three data structures written and their paths saved, we can
run ``hela`` on the data. First, we'll initialize a `~hela.Retrieval` object
with the paths to the three files/directories that it needs to know about:

.. code-block:: python

    from hela import Retrieval
    import matplotlib.pyplot as plt

    # Initialize a retrieval model object:
    r = Retrieval(training_dataset, example_dir, samples_path)

We now have a Retrieval object ``r`` which is ready for training. We can
train the random forest with 1000 trees and on a single processor:

.. code-block:: python

    # Train the random forest:
    r2scores = r.train(num_trees=1000, num_jobs=1)

    # Plot the results:
    r.plot_correlations()
    plt.show()

.. plot::

    from hela import generate_example_data
    # Generate an example dataset directory
    example_dir, training_dataset, samples_path = generate_example_data()

    from hela import Retrieval
    import matplotlib.pyplot as plt

    # Initialize a random forest object:
    r = Retrieval(training_dataset, example_dir, samples_path)

    # Train the random forest:
    r2scores = r.train(num_trees=1000, num_jobs=1)
    r.plot_correlations()
    plt.show()

The `~hela.Retrieval.train` method returns a dictionary called ``r2scores``
which contains the :math:`R^2` scores of the slope and intercept, which should
both be close to unity for this example.

Finally, let's estimate the posterior distributions for the slope and intercept
using the trained random forest on the sample data in ``samples_path``, where
the true values of the slope and intercept are :math:`m=0.2` and :math:`b=0.5`
using the `~hela.Retrieval.predict` method:

.. code-block:: python

    # Predict posterior distributions from random forest
    posterior = r.predict()
    posterior_slopes, posterior_intercepts = posterior.samples.T

    # Plot the posteriors
    r.plot_posterior()
    plt.show()

.. plot::

    from hela import generate_example_data
    # Generate an example dataset directory
    example_dir, training_dataset, samples_path = generate_example_data()

    from hela import Retrieval
    import matplotlib.pyplot as plt

    # Initialize a random forest object:
    r = Retrieval(training_dataset, example_dir, samples_path)

    # Predict posterior distributions from random forest
    posterior = r.predict()
    posterior_slopes, posterior_intercepts = posterior.samples.T
    r.plot_posterior()
    plt.tight_layout()
    plt.show()

