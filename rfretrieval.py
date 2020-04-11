
import argparse
import os
import logging

import numpy as np
from sklearn import metrics, multioutput
import joblib

import hela

LOGGER = logging.getLogger(__name__)


def train_model(dataset, num_trees, num_jobs, verbose=1):
    pipeline = hela.Model(
        num_trees=num_trees,
        num_jobs=num_jobs,
        verbose=verbose
    )
    pipeline.fit(dataset.training_x, dataset.training_y)
    return pipeline


def test_model(model, dataset, output_path):

    if dataset.testing_x is None:
        return

    LOGGER.info("Testing model...")
    pred = model.predict(dataset.testing_x)
    # pred = model.predict_median(dataset.testing_x)
    r2scores = {
        name_i: metrics.r2_score(real_i, pred_i)
        for name_i, real_i, pred_i in zip(dataset.names,
                                          dataset.testing_y.T,
                                          pred.T)
    }
    print("Testing scores:")
    for name, values in r2scores.items():
        print("\tR^2 score for {}: {:.3f}".format(name, values))

    LOGGER.info("Plotting testing results...")
    fig = hela.predicted_vs_real(
        dataset.testing_y,
        pred,
        names=dataset.names,
        ranges=dataset.ranges
    )
    fig.savefig(os.path.join(output_path, "predicted_vs_real.pdf"),
                bbox_inches='tight')


def _plot_feature_importances(model, dataset, output_path):

    LOGGER.info("Computing feature importances for individual parameters...")
    regr = multioutput.MultiOutputRegressor(model, n_jobs=1)
    regr.fit(dataset.training_x, dataset.training_y)

    fig = hela.plot.feature_importances(
        forests=(
            [i.random_forest for i in regr.estimators_]
            + [model.random_forest]
        ),
        names=dataset.names + ["joint prediction"],
        colors=dataset.colors + ["C0"]
    )

    fig.savefig(os.path.join(output_path, "feature_importances.pdf"),
                bbox_inches='tight')


def _plot_feature_importances_breakdown(model, dataset, output_path):

    LOGGER.info("Computing feature importances per output...")
    importances = hela.importances_per_output(
        model.random_forest,
        dataset.training_x,
        model.scaler_transform(dataset.training_y)
    )

    fig = hela.plot.stacked_feature_importances(
        importances,
        dataset.names,
        dataset.colors
    )

    fig.tight_layout()
    fig.savefig(
        os.path.join(output_path, "feature_importances_breakdown.pdf"),
        bbox_inches='tight'
    )


def _compute_median_fit(model, training_x, query, out_filename):

    posterior_x = model.posterior(query, prior_samples=training_x)
    median_fit = hela.posterior_percentile(posterior_x, 50)
    np.save(out_filename, median_fit)


def data_ranges(posterior, percentiles=(50, 16, 84)):

    values = hela.posterior_percentile(posterior, percentiles)
    ranges = np.array([values[0], values[2]-values[0], values[0]-values[1]])
    return ranges.T


def main_train(
        training_dataset,
        model_path,
        num_trees,
        num_jobs,
        feature_importances,
        feature_importances_breakdown,
        quiet,
        **_):

    LOGGER.info("Loading dataset '%s'...", training_dataset)
    dataset = hela.load_dataset(training_dataset)

    LOGGER.info("Training model...")
    model = train_model(dataset, num_trees, num_jobs, not quiet)

    os.makedirs(model_path, exist_ok=True)
    model_file = os.path.join(model_path, "model.pkl")
    LOGGER.info("Saving model to '%s'...", model_file)
    joblib.dump(model, model_file)

    LOGGER.info("Printing model information...")
    print("OOB score: {:.4f}".format(model.random_forest.oob_score_))

    test_model(model, dataset, model_path)

    if feature_importances:
        model.enable_posterior = False
        _plot_feature_importances(model, dataset, model_path)

    if feature_importances_breakdown:
        _plot_feature_importances_breakdown(model, dataset, model_path)


def main_predict(
        training_dataset,
        model_path,
        data_file,
        output_path,
        plot_posterior,
        save_median_fit,
        **_):

    LOGGER.info("Loading dataset '%s'...", training_dataset)
    dataset = hela.load_dataset(training_dataset, load_testing_data=False)

    model_file = os.path.join(model_path, "model.pkl")
    LOGGER.info("Loading random forest from '%s'...", model_file)
    model: hela.Model = joblib.load(model_file)

    LOGGER.info("Loading data from '%s'...", data_file)
    data, _ = hela.load_data_file(data_file, model.random_forest.n_features_)

    os.makedirs(output_path, exist_ok=True)

    posterior = model.posterior(data[0])

    posterior_ranges = data_ranges(posterior)
    for name_i, pred_range_i in zip(dataset.names, posterior_ranges):
        format_str = "Prediction for {}: {:.3g} [+{:.3g} -{:.3g}]"
        print(format_str.format(name_i, *pred_range_i))

    if plot_posterior:
        LOGGER.info("Plotting the posterior matrix...")

        fig = hela.plot.posterior_matrix(
            posterior,
            names=dataset.names,
            ranges=dataset.ranges,
            colors=dataset.colors
        )
        os.makedirs(output_path, exist_ok=True)
        LOGGER.info("Saving the figure....")
        fig.savefig(os.path.join(output_path, "posterior_matrix.pdf"),
                    bbox_inches='tight')
        LOGGER.info("Done.")

    if save_median_fit:
        LOGGER.info("Computing and saving median fit...")
        _compute_median_fit(
            model,
            dataset.training_x,
            data[0],
            os.path.join(output_path, "median_fit.npy")
        )
        LOGGER.info("Done.")


def show_usage(parser, **_):
    parser.print_help()


def main():

    parser = argparse.ArgumentParser(
        description="rfretrieval: Atmospheric retrieval with random forests."
    )
    parser.set_defaults(func=show_usage, parser=parser)
    parser.add_argument("--quiet", action='store_true')
    subparsers = parser.add_subparsers()

    parser_train = subparsers.add_parser('train', help="train a model")
    parser_train.set_defaults(func=main_train)
    parser_train.add_argument(
        "training_dataset", type=str,
        help="JSON file with the training dataset description"
    )
    parser_train.add_argument(
        "model_path", type=str,
        help="path where the trained model will be saved"
    )
    parser_train.add_argument(
        "--num-trees", type=int, default=1000,
        help="number of trees in the forest"
    )
    parser_train.add_argument(
        "--num-jobs", type=int, default=5,
        help="number of parallel jobs for fitting the random forest"
    )
    parser_train.add_argument(
        "--feature-importances", action='store_true',
        help="plot feature importances after training"
    )
    parser_train.add_argument(
        "--feature-importances-breakdown", action='store_true',
        help="plot feature importances per output after training"
    )

    parser_test = subparsers.add_parser(
        'predict',
        help="use a trained model to perform a prediction"
    )
    parser_test.set_defaults(func=main_predict)
    parser_test.add_argument(
        "training_dataset", type=str,
        help="JSON file with the training dataset description"
    )
    parser_test.add_argument(
        "model_path", type=str,
        help="path to the trained model"
    )
    parser_test.add_argument(
        "data_file", type=str,
        help="NPY file with the data for the prediction"
    )
    parser_test.add_argument(
        "output_path", type=str,
        help="path to write the results of the prediction"
    )
    parser_test.add_argument(
        "--plot-posterior", action='store_true',
        help="plot and save the scatter matrix of the posterior distribution"
    )
    parser_test.add_argument(
        "--save-median-fit", action='store_true',
        help="save the median fit"
    )

    args = parser.parse_args()
    hela.config_logger(level=logging.WARNING if args.quiet else logging.INFO)
    args.func(**vars(args))


if __name__ == '__main__':
    main()
