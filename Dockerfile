# Use a PyTorch official image with CUDA and cuDNN pre-installed
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Set the working directory in the container to /workspace
WORKDIR /workspace

# Copy the current directory contents into the container at /workspace
COPY . /workspace

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 47907 available to the world outside this container
EXPOSE 47907

# Define environment variable
ENV NAME World

# Run app.py when the container launches
CMD ["python", "app.py"]
