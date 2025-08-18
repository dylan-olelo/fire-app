# Start from a clean, official Python image
FROM python:3.11-slim

# Set environment variables to control the cache location
ENV SENTENCE_TRANSFORMERS_HOME=/app/cache

# Copy all your project files into the /app directory
COPY . /app

# Set the working directory
WORKDIR /app

# Create the cache directory as root
RUN mkdir -p /app/cache

# Install all the Python libraries from your requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

RUN python download_models.py

# --- NEW: PERMISSION FIX ---
# Create a new non-root user named "user"
RUN useradd -m -u 1000 user

# Give the new user ownership of the app and cache directories
RUN chown -R user:user /app

# Switch to the non-root user
USER user
# --- END OF FIX ---

# Tell the environment that your app will be listening on port 7860
EXPOSE 7860

# The command to run your Flask app when the container starts
# This will now be run as the non-root 'user'
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "app:app"]