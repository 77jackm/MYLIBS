### SNIPPETS of Useful Code

# Conditional Block ("with") and Exception Handling
import requests 
import filepath as PATH

try:
      file_response = requests.get(file_url, stream=True)
      file_response.raise_for_status()

      with open(file_path, 'wb') as f:
        for chunk in file_response.iter_content(chunk_size=8192):
          f.write(chunk)

      print(f"Downloaded: {file_name}")

    except requests.exceptions.RequestException as e:
      print(f"Error downloading {file_name}: {e}")