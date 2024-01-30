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

    # Use regex to replace standalone "w"
    content = re.sub(r'\b(w)\b', r'$\1$', content)
    # Use regex to replace standalone "b"
    content = re.sub(r'\b(b)\b', r'$\1$', content)

    # Replace tensor flow' with 'Tensorflow'
    content = re.sub(r'tensor flow', 'Tensorflow', content)

    # Replace video with 'section'
    content = re.sub(r'video', 'section', content)

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