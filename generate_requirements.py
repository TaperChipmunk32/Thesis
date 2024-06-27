import os
from nbconvert import PythonExporter

# Define the root directory to start the search
root_dir = os.getcwd()

py_files = []
# Traverse through directories and subdirectories
for root, dirs, files in os.walk(root_dir):
    for file in files:
        if file.endswith('.ipynb'):
            # Get the full path of the .ipynb file
            ipynb_file = os.path.join(root, file)
            
            # Convert .ipynb to .py
            py_exporter = PythonExporter()
            (py_code, _) = py_exporter.from_filename(ipynb_file)
            
            # Create the .py file name
            py_file = os.path.splitext(ipynb_file)[0] + '.py'
            
            # Write the converted code to the .py file
            with open(py_file, 'w') as f:
                f.write(py_code)
            py_files.append(py_file)
            print(f"Converted {ipynb_file} to {py_file}")
#rename existing requirements.txt file
os.rename('requirements.txt', 'requirements_old.txt')

#run pipreqs command
os.system('pipreqs .')

#append Werkzeug to requirements.txt file
with open('requirements.txt', 'a') as f:
    f.write('Werkzeug==2.2.2')

#comare the two files
with open('requirements_old.txt', 'r') as file1:
    with open('requirements.txt', 'r') as file2:
        diff = set(file1).difference(file2)
        if diff:
            print('Requirements have changed')
            print(diff)
        else:
            print('Requirements have not changed')

if not diff:
    os.remove('requirements_old.txt')

#delete py files
for file in py_files:
    os.remove(file)