import nbformat
import sys
import re

def convert_py_to_ipynb(py_path, ipynb_path):
    with open(py_path, 'r', encoding='utf-8') as f:
        content = f.read()

    cells = content.split('# %%')
    
    nb = nbformat.v4.new_notebook()
    
    for cell in cells:
        cell_content = cell.strip()
        if not cell_content:
            continue
            
        if cell_content.startswith('[markdown]'):
            # Strip '[markdown]' and leading/trailing whitespace
            text = cell_content[len('[markdown]'):].strip()
            # Remove `# ` prefixes mimicking generic markdown 
            text = '\n'.join([line[2:] if line.startswith('# ') else line for line in text.split('\n')])
            nb.cells.append(nbformat.v4.new_markdown_cell(text))
        elif cell_content.startswith('[code]'):
            code = cell_content[len('[code]'):].strip()
            nb.cells.append(nbformat.v4.new_code_cell(code))
        else:
            # default to code cell
            nb.cells.append(nbformat.v4.new_code_cell(cell_content))
            
    with open(ipynb_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

if __name__ == '__main__':
    convert_py_to_ipynb(sys.argv[1], sys.argv[2])
