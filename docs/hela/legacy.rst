Legacy API
==========

We start with training our forest, on the example dataset provided. To check how
to run the training stage, you can run::

    python rfretrieval.py train -h

This will show you the usage of ``train``, in case you need a reminder. So, we
run training as follows::

    python rfretrieval.py train example_dataset/example_dataset.json example_model/


The ``training_dataset`` refers to the ``.json`` file in the dataset folder.
The ``training.npy`` and ``testing.npy`` files must also be in this folder.
The ``model_path`` is just some new output path you need to choose a name for.
It will be created.

You can also edit the number of trees used, and the number of jobs, and find the
feature importances, by running with the extra optional arguments::

    python rfretrieval.py train example_dataset/example_dataset.json example_model/ --num-trees 100 --num-jobs 3 --feature-importance

The default number of trees is 1000. The default number of jobs is 5. The
default does not run the feature importance. This is because it requires
training a new forest for each parameter, so makes the process much slower, and
you may not need the feature importance every time you use HELA.

Once running, HELA will update you at several stages of training, telling you
how long each stage has taken. E.g.::

    [Parallel(n_jobs=5)]: Done  40 tasks      | elapsed:    5.0s

The ``40 tasks`` refers to the first 40 trees having been trained.

After training is complete, HELA will run testing. It will print an :math:`R^2`
score for each parameter, and plot the results. The forest itself, ``model.pkl``
and the predicted vs real graph can now be found in the ``example_model/``
folder.

You can now use your forest to predict on data. In the example dataset we have
included the WASP-12b data, for which this particular training set was tailored
for. You can check how the prediction stage runs by running::

    python rfretreival.py predict -h

For this stage, you must provide the model's path, the data file, and an output
folder. Whether the posteriors are plotted or not is optional. So, to include
the posteriors, we run::

    python rfretrieval.py predict example_model/ example_dataset/WASP12b.npy example_plots/ --plot-posterior

This will give you a prediction for each parameter on this data file. The
numbers given are the median, and in brackets the 16th and 84th percentiles, of
the posteriors. The posterior matrix can now be found in the ``example_plots/``
folder.
