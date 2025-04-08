# Run This Command
##  Replace ``` path = "file/path/to/output/folder" ``` with desired path

```
import urllib.request; path = "file/path/to/output/folder"; script_url = "https://raw.githubusercontent.com/IamNishanKhan/RadioTherapy/main/version5.1.py"; script_content = urllib.request.urlopen(script_url).read().decode('utf-8'); modified_script = script_content.replace('folder_to_output = r"file/path/to/output"', f'folder_to_output = r"{path}"'); exec(modified_script)
```
