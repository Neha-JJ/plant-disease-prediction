# Use the official Python image from the Docker Hub
FROM python:3.10-slim

# Copy the current directory contents into the container at /app
COPY . /app

# Set the working directory in the container
WORKDIR /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Create .streamlit directory in the home directory
RUN mkdir -p ~/.streamlit

# Copy Streamlit configuration file
COPY config.toml ~/.streamlit/config.toml

# Copy Streamlit credentials file if it exists
COPY credentials.toml ~/.streamlit/credentials.toml

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Run app.py when the container launches
CMD ["streamlit", "run", "main.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
