

def test_linear_end_to_end():
    from ..retrieval import Retrieval, generate_example_data
    from ..dataset import Dataset

    # Create an example dataset
    training_dataset, example_dir, example_data = generate_example_data()

    # Load the dataset
    dataset = Dataset.load_json("linear_dataset/example_dataset.json")

    # Train the model
    r = Retrieval()
    r2scores = r.train(dataset, num_trees=1000, num_jobs=5)

    # Predict posterior distribution for slope and intercept of example data
    posterior = r.predict(example_data)

    # Do a rough check that the R^2 values are near unity
    assert abs(r2scores['slope'] - 1) < 0.3
    assert abs(r2scores['intercept'] - 1) < 0.3

    posterior_slopes, posterior_intercepts = posterior.samples.T

    # Do a very generous check that the posterior distributions match
    # the expected values
    assert abs(posterior_slopes.mean() - 0.7) < 3 * posterior_slopes.std()
    assert (abs(posterior_intercepts.mean() - 0.5) <
            3 * posterior_intercepts.std())
