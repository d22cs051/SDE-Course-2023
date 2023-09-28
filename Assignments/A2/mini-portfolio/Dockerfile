# Use an official Nginx runtime as a parent image
FROM nginx:latest

# Copy the HTML files from your local machine to the container's default Nginx web root directory
COPY . /usr/share/nginx/html

# Expose port 80 to the outside world
EXPOSE 80

# Start Nginx in the foreground
CMD ["nginx", "-g", "daemon off;"]
