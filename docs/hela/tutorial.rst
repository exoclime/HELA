Tutorial
========

First, we must generate some example data, which we can do using a built-in
function called `~hela.generate_example_data`, which returns the path to the
example file directory, the training dataset path, and the path to the samples
which we'd like to predict on:

.. code-block:: python

    from hela import generate_example_data
    # Generate an example dataset directory
    example_dir, training_dataset, samples_path = generate_example_data()

What did that just do? We created an example directory called ``linear_data``,
which contains a training dataset described by the metadata file
``training_dataset``. This training dataset has...

We also generated a bunch of samples with a known slope and intercept, called
``samples_path``, on which we'll apply our trained random forest to estimate
the slope and intercept.

Once we have these three data structures written and their paths saved, we can
run ``hela`` on the data. First, we'll initialize a `~hela.RandomForest` object
with the paths to the three files/directories that it needs to know about:

.. code-block:: python

    from hela import RandomForest
    import matplotlib.pyplot as plt

    # Initialize a random forest object:
    rf = RandomForest(training_dataset, example_dir, samples_path)

We now have a random forest object ``rf`` which is ready for training. We can
train the random forest with 1000 trees and on a single processor:

.. code-block:: python

    # Train the random forest:
    r2scores = rf.train(num_trees=1000, num_jobs=1)
    plt.show()

.. plot::

    from hela import generate_example_data
    # Generate an example dataset directory
    example_dir, training_dataset, samples_path = generate_example_data()

    from hela import RandomForest
    import matplotlib.pyplot as plt

    # Initialize a random forest object:
    rf = RandomForest(training_dataset, example_dir, samples_path)

    # Train the random forest:
    r2scores = rf.train(num_trees=1000, num_jobs=1)
    plt.show()

The `~hela.RandomForest.train` method returns a dictionary called `r2scores`
which contains the :math:`R^2` scores of the slope and intercept.

Finally, let's estimate the posterior distributions for the slope and intercept
using the trained random forest on the sample data in ``samples_path``:

.. code-block:: python

    # Predict posterior distirbutions from random forest
    posterior_slopes, posterior_intercepts = rf.predict(plot_posterior=True)
    plt.show()

.. plot::

    from hela import generate_example_data
    # Generate an example dataset directory
    example_dir, training_dataset, samples_path = generate_example_data()

    from hela import RandomForest
    import matplotlib.pyplot as plt

    # Initialize a random forest object:
    rf = RandomForest(training_dataset, example_dir, samples_path)

    # Train the random forest:
    r2scores = rf.train(num_trees=1000, num_jobs=1)
    plt.close()

    # Predict posterior distirbutions from random forest
    posterior_slopes, posterior_intercepts = rf.predict(plot_posterior=True)
    plt.show()

