

def test_linear_end_to_end():
    from ..model import generate_example_data
    example_dir, training_dataset, samples_path = generate_example_data()

    # Import RandomForest object from HELA
    from ..model import Model

    # Initialize a model:
    m = Model(training_dataset, example_dir, samples_path)

    # Train the random forest:
    r2scores = m.train(num_trees=1000, num_jobs=1)

    # Do a rough check that the R^2 values are near unity
    assert abs(r2scores['slope'] - 1) < 0.01
    assert abs(r2scores['intercept'] - 1) < 0.01

    # Predict posterior distributions from random forest
    posterior = m.predict()
    posterior_slopes, posterior_intercepts = posterior.samples.T

    # Do a very generous check that the posterior distributions match
    # the expected values
    assert abs(posterior_slopes.mean() - 0.3) < 3 * posterior_slopes.std()
    assert (abs(posterior_intercepts.mean() - 0.5) <
            3 * posterior_intercepts.std())
