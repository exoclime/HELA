# HELA  ![](img/Hela_logo1.png)

A Random Forest retrieval algorithm, here used to perform atmospheric retrieval on exoplanet atmospheres.

The theory paper can be found here: https://arxiv.org/abs/1806.03944

Please cite this when using HELA. 

The set-up here is simply a Random Forest algorithm, for use on a training set provided by the user. We have uploaded an example dataset, the one used in the above paper, to demonstrate how it is used. 

## Requirements

HELA is developed for use with Python 3 and requires the following packages:
- numpy
- sklearn
- matplotlib


## Running HELA

Here we explain how to run HELA with the example dataset provided. 

Once HELA is downloaded, go to your HELA directory. Everything is run in this directory. There are two stages to HELA - the training stage and the predicting stage. Inside the training stage is also the testing stage. 

We start with training our forest, on the example dataset provided. To check how to run the training stage, you can run:

```
python rfretrieval.py train -h
```

This will show you the usage of 'train', in case you need a reminder. So, we run training as follows:

```
python rfretrieval.py train example_dataset/example_dataset.json example_model/
```

The ```training_dataset``` refers to the ```.json``` file in the dataset folder. The ```training.npy``` and ```testing.npy``` files must also be in this folder. The ```model_path``` is just some new output path you need to choose a name for. It will be created. 

You can also edit the number of trees used, and the number of jobs, and find the feature importances, by running with the extra optional arguments:

```
python rfretrieval.py train example_dataset/example_dataset.json example_model/ --num-trees 100 --num-jobs 3 --feature-importance
```

The default number of trees is 1000. The default number of jobs is 5. The default does not run the feature importance. This is because it requires training a new forest for each parameter, so makes the process much slower, and you may not need the feature importance every time you use HELA. 

Once running, HELA will update you at several stages of training, telling you how long each stage has taken. E.g.

```
[Parallel(n_jobs=5)]: Done  40 tasks      | elapsed:    5.0s
```

The ```40 tasks``` refers to the first 40 trees having been trained. 

After training is complete, HELA will run testing. It will print an R^2 score for each parameter, and plot the results. The forest itself, ```model.pkl``` and the predicted vs real graph can now be found in the ```example_model/``` folder. 

You can now use your forest to predict on data. In the example dataset we have included the WASP-12b data, for which this particular training set was tailored for. You can check how the prediction stage runs by running:

```
python rfretreival.py predict -h 
```

For this stage, you must provide the model's path, the data file, and an output folder. Whether the posteriors are plotted or not is optional. So, to include the posteriors, we run:

```
python rfretrieval.py predict example_model/ example_dataset/WASP12b.npy example_plots/ --plot-posterior
```

This will give you a prediction for each parameter on this data file. The numbers given are the median, and in brackets the 16th and 84th percentiles, of the posteriors. The posterior matrix can now be found in the ```example_plots/``` folder. 

## Your Training Data

### Creating the Training Set

To use HELA with your own data, you must create a training set that matches the format of the data you aim to analyse. So, it must have the same number of points. You cannot give HELA the x-axis of your data, it only takes the points in the order given, so this must match the data. You also cannot pass in error bars, so the way to do this is to make sure you sample the noise thoroughly, so HELA can see several noisy examples of a particular spectrum. 

We add noise to a noise-free model by running:

```
y_noisy = np.random.normal(y_noisefree, errorbar_width)
```

For each parameter you have, you must create sample spectrum for the whole range. HELA cannot extrapolate, so you must contain the edges of your parameter space in the training set. Of course, this depends on how you are generating your data. If you are generating it on the fly, and you are generating a large number of models, then it is sufficient to use random choices for your parameters. For example, for the example dataset we created 100,000 models by randomly selecting the parameter values as follows:

```
parameter_value = np.random.uniform(parameter_min, parameter_max)
```

We then use these parameter values to create the spectrum, add noise, and then add it to the training set. If you are starting from a model grid, as opposed to generating your own models, then you simply need to draw from the model grid and add noise. 

When adding a spectrum to the training set, you must add the parameter values on the end. This is how HELA knows what values correspond to which spectrum. So one line in your training set should look like, for a spectrum with 3 data points (e.g. 3 wavelength points) and 2 parameters (e.g. temperature and metallicity) for example, 

```
[spectrum_point_1, spectrum_point_2, spectrum_point_3, parameter_value_1, parameter_value_2]
```

These are then stacked into a numpy array, with each row being a new spectrum.

The training set then needs to be split into training and testing. These MUST NOT contain the same parameter values. For example, if there is a spectrum in the training file with the same set of a parameters as a spectrum in the testing file, then despite the addition of some variable noise, it is likely that the forest will predict very well on this spectrum, thus giving unrealistic testing scores. You don't want to trick yourself into thinking you have a perfect forest!

So, if you start from a grid it is probably easiest to first split the grid into some models for training and some for testing, and then draw these separately and add noise, to guarantee no cross over. If you are drawing randomly, as above, then it is highly unlikely you will have ever drawn exactly the same parameters twice, so it is sufficient to just split the entire set at some point. 

We suggest roughly 80% for training and 20% for testing, but this is not a rule. It is just better to have more in the training set as this should give you a better performing forest in the end. 

The training and testing files then need to be saved as ```.npy``` files, in your dataset folder. 

### The dataset.json File

Inside your dataset folder you must have the json file like ```example_dataset.json``` file in the example. You need to edit this to match your data. The ```"names"``` correspond to the labels that will go on the graphs. The ```"ranges"``` correspond to the ranges that will be set on the graphs (so should match your whole range for each parameter - you don't want to cut anything out of the graphs!). The ```"colors"``` are just codes for the colors used in plotting. 

Lastly, the ```"num_features"``` is important. This is the number of points in your spectra. It tells HELA where the division is between data points and parameter values in your training and testing sets (e.g. in the example shown above it would be ```"num_features": 3``` . If you have the wrong value set here, it will only show an error if the number is too large. If the number is too small, it will split the spectra too early, and assign the left over data points to the parameter values, giving you some weird results. So if something looks odd in your predicted vs real graph, check you have set the right number of features!

### Your Data File

The data that you wish to predict on needs to also be a ```.npy``` file, with a length matching the ```"num_features"``` you have set. 


## Remarks

If you have any questions or suggestions please contact either [Pablo Márquez-Neila](mailto:pablo.marquez@artorg.unibe.ch) or [Chloe Fisher](mailto:chloe.fisher@csh.unibe.ch).

Enjoy using HELA!

We acknowledge partial financial support from the Center for Space and Habitability, the University of Bern International 2021 Ph.D Fellowship, the PlanetS National Center of Competence in Research, the Swiss National Science Foundation, the European Research Council and the Swiss-based MERAC Foundation.
