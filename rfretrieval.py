
import argparse
import os
import logging

import numpy as np
from sklearn import metrics, multioutput
import joblib

from dataset import load_dataset, load_data_file
from models import Model
from utils import config_logger
from wpercentile import wpercentile
import plot

logger = logging.getLogger(__name__)


def train_model(dataset, num_trees, num_jobs, verbose=1):
    pipeline = Model(num_trees, num_jobs,
                     names=dataset.names,
                     ranges=dataset.ranges,
                     colors=dataset.colors,
                     verbose=verbose)
    pipeline.fit(dataset.training_x, dataset.training_y)
    return pipeline


def test_model(model, dataset, output_path):
    
    if dataset.testing_x is None:
        return
    
    logger.info("Testing model...")
    pred = model.predict(dataset.testing_x)
    r2scores = {name_i: metrics.r2_score(real_i, pred_i)
                for name_i, real_i, pred_i in zip(dataset.names, dataset.testing_y.T, pred.T)}
    print("Testing scores:")
    for name, values in r2scores.items():
        print("\tR^2 score for {}: {:.3f}".format(name, values))
    
    logger.info("Plotting testing results...")
    fig = plot.predicted_vs_real(dataset.testing_y, pred, dataset.names, dataset.ranges)
    fig.savefig(os.path.join(output_path, "predicted_vs_real.pdf"),
                bbox_inches='tight')


def compute_feature_importance(model, dataset, output_path):
    
    logger.info("Computing feature importance for individual parameters...")
    regr = multioutput.MultiOutputRegressor(model, n_jobs=1)
    regr.fit(dataset.training_x, dataset.training_y)
    
    fig = plot.feature_importances(forests=[i.rf for i in regr.estimators_] + [model.rf],
                                   names=dataset.names + ["joint prediction"],
                                   colors=dataset.colors + ["C0"])
    
    fig.savefig(os.path.join(output_path, "feature_importances.pdf"),
                bbox_inches='tight')


def data_ranges(posterior, percentiles=(50, 16, 84)):
    
    samples, weights = posterior
    values = wpercentile(samples, weights, percentiles, axis=0)
    ranges = np.array([values[0], values[2]-values[0], values[0]-values[1]])
    return ranges.T


def main_train(training_dataset, model_path,
               num_trees, num_jobs,
               feature_importance, quiet,
               **_):
    
    logger.info("Loading dataset '{}'...".format(training_dataset))
    dataset = load_dataset(training_dataset)
    
    logger.info("Training model...")
    model = train_model(dataset, num_trees, num_jobs, not quiet)
    
    os.makedirs(model_path, exist_ok=True)
    model_file = os.path.join(model_path, "model.pkl")
    logger.info("Saving model to '{}'...".format(model_file))
    joblib.dump(model, model_file)
    
    logger.info("Printing model information...")
    print("OOB score: {:.4f}".format(model.rf.oob_score_))
    
    test_model(model, dataset, model_path)
    
    if feature_importance:
        model.enable_posterior = False
        compute_feature_importance(model, dataset, model_path)


def main_predict(model_path, data_file, output_path, plot_posterior, **_):
    
    model_file = os.path.join(model_path, "model.pkl")
    logger.info("Loading random forest from '{}'...".format(model_file))
    model = joblib.load(model_file)
    
    logger.info("Loading data from '{}'...".format(data_file))
    data, _ = load_data_file(data_file, model.rf.n_features_)
    
    posterior = model.posterior(data[0])
    
    posterior_ranges = data_ranges(posterior)
    for name_i, pred_range_i in zip(model.names, posterior_ranges):
        print("Prediction for {}: {:.3g} [+{:.3g} -{:.3g}]".format(name_i, *pred_range_i))
    
    if plot_posterior:
        logger.info("Plotting the posterior matrix...")
        
        fig = plot.posterior_matrix(
            posterior,
            names=model.names,
            ranges=model.ranges,
            colors=model.colors
        )
        os.makedirs(output_path, exist_ok=True)
        logger.info("Saving the figure....")
        fig.savefig(os.path.join(output_path, "posterior_matrix.pdf"),
                    bbox_inches='tight')
        logger.info("Done.")


def show_usage(parser, **_):
    parser.print_help()


def main():
    
    parser = argparse.ArgumentParser(description="rfretrieval: Atmospheric retrieval with random forests.")
    parser.set_defaults(func=show_usage, parser=parser)
    parser.add_argument("--quiet", action='store_true')
    subparsers = parser.add_subparsers()
    
    parser_train = subparsers.add_parser('train', help="train a model")
    parser_train.add_argument("training_dataset", type=str,
                              help="JSON file with the training dataset description")
    parser_train.add_argument("model_path", type=str,
                              help="path where the trained model will be saved")
    parser_train.add_argument("--num-trees", type=int, default=1000,
                              help="number of trees in the forest")
    parser_train.add_argument("--num-jobs", type=int, default=5,
                              help="number of parallel jobs for fitting the random forest")
    parser_train.add_argument("--feature-importance", action='store_true',
                              help="compute feature importances after training")
    parser_train.set_defaults(func=main_train)
    
    parser_test = subparsers.add_parser('predict', help="use a trained model to perform a prediction")
    parser_test.set_defaults(func=main_predict)
    parser_test.add_argument("model_path", type=str,
                             help="path to the trained model")
    parser_test.add_argument("data_file", type=str,
                             help="NPY file with the data for the prediction")
    parser_test.add_argument("output_path", type=str,
                             help="path to write the results of the prediction")
    parser_test.add_argument("--plot-posterior", action='store_true',
                             help="plot and save the scatter matrix of the posterior distribution")
    
    args = parser.parse_args()
    config_logger(level=logging.WARNING if args.quiet else logging.INFO)
    args.func(**vars(args))


if __name__ == '__main__':
    main()
