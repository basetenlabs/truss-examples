# Dynamic Batching in Truss

This repository contains an implementation designed to enable dynamic batching for machine learning models within the Truss framework. The core of this implementation lies in the `model/model.py` file, which introduces a `MlBatcher` class extending `AsyncBatcher`. This class is responsible for collecting individual prediction requests and processing them in batches, thereby improving throughput and efficiency.

## Key Features

- **Dynamic Batching:** The `MlBatcher` class dynamically batches incoming prediction requests, allowing for more efficient use of resources and faster response times.
- **Asynchronous Processing:** Utilizes asynchronous programming to handle concurrent prediction requests without blocking, ensuring high throughput.
- **Easy Integration:** Designed to be deployed as a normal Truss, making integration into existing projects straightforward.

## Deployment

To deploy this as a normal Truss, ensure you have the Truss CLI installed and configured. Then, follow these steps:

1. Clone this repository to your local machine.
2. Navigate to the repository directory and build the Truss using the command `truss up`.
3. Once the build completes, deploy the Truss to your desired environment.

## Configuration

The `config.yaml` file contains configuration options for the model, including the Python version, required packages, and runtime settings such as `predict_concurrency`. Adjust these settings as needed to optimize performance for your specific use case.

## Testing

The `test.py` file provides an example of how to send concurrent requests to the deployed model for testing purposes. Modify the URL and data as needed to match your deployment.

## Conclusion

This implementation showcases how dynamic batching can be seamlessly integrated into the Truss framework, providing a scalable and efficient solution for handling machine learning inference at scale.
