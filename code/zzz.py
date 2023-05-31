import re
import matplotlib.pyplot as plt

loss_values = []

with open('../out/nohup.out', 'r') as file:
    for line in file:
        # Extract the loss values using regular expressions
        match = re.search(r'Classification loss: (\d+\.\d+) \| Regression loss: (\d+\.\d+)', line)
        if match:
            classification_loss = float(match.group(1))
            regression_loss = float(match.group(2))
            loss_values.append(classification_loss + regression_loss)

# Plot the loss curve
plt.plot(loss_values)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.show()
