import numpy as np
import shap
import matplotlib.pyplot as plt

# Compute SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

if isinstance(shap_values, list):
    shap_values = shap_values[1]

# Get predicted probabilities
probs = model.predict_proba(X)[:, 1]

# Select some representative samples
landslide_indices = np.where(y == 1)[0][:2]        # 2 landslide cases
nonlandslide_indices = np.where(y == 0)[0][:2]     # 2 non-landslide cases

selected_indices = np.concatenate([landslide_indices, nonlandslide_indices])

print("Displaying SHAP Force Plots...\n")

for idx in selected_indices:

    sample_shap = shap_values[idx]
    sample_features = X[idx]

    # Keep top 6 important features
    top_n = 6
    top_idx = np.argsort(np.abs(sample_shap))[::-1][:top_n]

    top_shap = np.round(sample_shap[top_idx], 2)
    top_features = np.round(sample_features[top_idx], 2)
    top_names = [PARAMS[i] for i in top_idx]

    formatted_labels = [
        f"{name} = {value:.2f}"
        for name, value in zip(top_names, top_features)
    ]

    # Base value
    if isinstance(explainer.expected_value, np.ndarray):
        base_value = explainer.expected_value[0]
    else:
        base_value = explainer.expected_value

    # Print probability
    prob = probs[idx]
    print(f"Sample Index: {idx}")
    print(f"Predicted Landslide Probability: {prob:.4f}\n")

    # Plot
    plt.figure(figsize=(18,4))

    shap.force_plot(
        base_value,
        top_shap,
        formatted_labels,
        matplotlib=True,
        show=False
    )

    plt.subplots_adjust(top=0.85)
    plt.show()
