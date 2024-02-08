import sys
import re

def process_md_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    # Replace "your" with "our"
    content = re.sub(r'\byour\b', 'our', content, flags=re.IGNORECASE)

    # Replace "you" with "we"
    content = re.sub(r'\byou\b', 'we', content, flags=re.IGNORECASE)

    # Replace "I'm" with "we're"
    content = re.sub(r'\bI\'m\b', "we're", content, flags=re.IGNORECASE)

    # Replace words with underscore or f(x) substring with $...$
    content = re.sub(r'\b(\w*_[\w\d_]+|\w*f\(x\)\w*)\b', r'$\1$', content)

    # Replace "w1" with "$w_{1}$", "B12" with "$b_{12}$", etc
    content = re.sub(r'\b([wbWB])(\d+)\b', lambda match: f"${match.group(1).lower()}_{match.group(2)}$", content)
    # Replace "w_1" with "$w_{1}$", "B_12" with "$b_{12}$", etc
    content = re.sub(r'\b([wbWB])_(\d+)\b', lambda match: f"${match.group(1).lower()}_{match.group(2)}$", content)
    # Replace "w^1" with "$w^{1}$", "B^12" with "$b^{12}$", etc
    content = re.sub(r'\b([wbWB])\^(\d+)\b', lambda match: f"${match.group(1).lower()}^{match.group(2)}$", content)

    # Use regex to replace standalone "w"
    content = re.sub(r'\s+(w)\s+', r' $\1$ ', content)
    # Use regex to replace standalone "b"
    content = re.sub(r'\s+(b)\s+', r' $\1$ ', content)
    # Use regex to replace standalone "x"
    content = re.sub(r'\s+(x)\s+', r' $\1$ ', content)
    # Use regex to replace standalone "y"
    content = re.sub(r'\s+(y)\s+', r' $\1$ ', content)


    # Replace tensor flow' with 'Tensorflow'
    content = re.sub(r'tensor flow', 'Tensorflow', content)

    # Replace video with 'section'
    content = re.sub(r'video', 'section', content)

    # Repalce error expressions (train, cv, test) with subscripts
    content = re.sub(r'\bJ train\b', r'$J_{train}$', content)
    content = re.sub(r'\bJ cv\b', r'$J_{cv}$', content)
    content = re.sub(r'\bJ test\b', r'$J_{test}$', content)

    # Split the content into sentences
    sentences = re.split(r'(?<=[.!?])\s+', content)

    # Create a new paragraph every three sentences
    for i in range(2, len(sentences), 3):
        sentences[i] = '\n\n' + sentences[i]

    # Join the sentences back into the content
    content = ' '.join(sentences)

    # Write the modified content back to the file
    with open(file_path, 'w') as file:
        file.write(content)

if __name__ == "__main__":
    # Check if a file path is provided as a command-line argument
    if len(sys.argv) != 2:
        print("Usage: python script_name.py <markdown_file_path>")
        sys.exit(1)

    # Get the file path from the command-line argument
    md_file_path = sys.argv[1]

    # Process the markdown file
    process_md_file(md_file_path)

    print(f"Processing complete for {md_file_path}.")