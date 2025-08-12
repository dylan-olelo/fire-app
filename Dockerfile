# Use a standard base image provided by Hugging Face
FROM hf-base-v1.1.0

# Copy all your project files into the /app directory in the container
COPY . /app

# Set the working directory inside the container
WORKDIR /app

# Install all the Python libraries from your requirements.txt file
RUN pip install -r requirements.txt

# Tell the environment that your app will be listening on port 5000
EXPOSE 5000

# The final command to run when your container starts
CMD ["python", "app.py"]