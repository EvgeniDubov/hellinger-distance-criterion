# Build inside a Docker container
To build this project inside a Docker container just run:
``` docker build . -t hellinger && docker run hellinger ```

The advantage of using Docker in this case is the isolation and the dependency management.
To use this docker image with your code, just replace "random_forest_train.py" with your python script in the Dockerfile.
