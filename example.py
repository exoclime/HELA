import matplotlib.pyplot as plt
from hela import Retrieval, Dataset, generate_example_data

# Create an example dataset
training_dataset, example_dir, example_data = generate_example_data()

# Load the dataset
dataset = Dataset.load_json("linear_dataset/example_dataset.json")

# Train the model
r = Retrieval()
r2scores = r.train(dataset, num_trees=1000, num_jobs=5)

# Print OOB score (optional)
print(r.oob)

# Plot predicted vs real:
fig, ax = r.plot_predicted_vs_real(dataset)

# Save the model (optional)
# from hela import save_model
# save_model("linear_dataset/model.pkl", r.model)

# Predict posterior distribution for slope and intercept of example data
posterior = r.predict(example_data)

# Print posterior ranges (optional)
posterior.print_percentiles(dataset.names)

fig2, ax2 = posterior.plot_posterior_matrix(dataset)

plt.show()