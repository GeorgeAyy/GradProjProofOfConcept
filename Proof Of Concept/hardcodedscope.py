import pytesseract
from PIL import Image
import cv2
import google.generativeai as genai
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2

# Configure the Gemini API
try:
    api_key = "AIzaSyDOExmBe0spo7h7PXGRbFqiPRPzfn5FdxE"  # Replace with your actual API key
    genai.configure(api_key=api_key)
    print("Gemini API configured successfully")
except Exception as e:
    print(f"Error configuring Gemini API: {e}")

# Step 1: Preprocess the image and extract text
def preprocess_image(image_path):
    print(f"Preprocessing image: {image_path}")
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    print(f"Image loaded, shape: {img.shape if img is not None else 'None'}")
    img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)[1]  # Use binary threshold
    print("Image preprocessing completed")
    return img

def extract_text_from_image(image_path):
    print(f"Extracting text from image: {image_path}")
    preprocessed_image = preprocess_image(image_path)
    text = pytesseract.image_to_string(preprocessed_image)
    print(f"Extracted text: {text[:500]}...")  # Print first 500 characters for debugging
    return text

# Step 2: Send extracted text to Gemini for structuring
def generate_structured_output(extracted_text):
    model = genai.GenerativeModel("gemini-1.5-flash")
    print("Sending extracted text to Gemini for structuring")

    # Define the prompt to ask for structured output
    prompt = f"Based on the following text from a class diagram, extract and structure the information into classes with their attributes and methods:\n\n{extracted_text}\n\nPlease return the structured classes in the following format:\n\nClass: [Class Name]\nAttributes: [List of Attributes]\nMethods: [List of Methods]"
    
    print(f"Prompt sent to Gemini: {prompt[:500]}...")  # Print the first 500 characters of the prompt for debugging
    
    try:
        response = model.generate_content(prompt)
        print("Received structured response from Gemini")
        return response.text
    except Exception as e:
        print(f"Error during API call: {e}")
        return None

# Step 3: Generate the system scope based on structured classes
def generate_system_scope(structured_output):
    model = genai.GenerativeModel("gemini-1.5-flash")
    print("Generating system scope from structured output")

    # Define the prompt to ask Gemini to create the system scope
    prompt = f"Create a system scope based on the following structured class diagram. Assume the functionality of each class and write a clear system scope:\n\n{structured_output}\n\nReturn the system scope in a well-structured paragraph."
    
    print(f"Prompt sent to Gemini for system scope: {prompt[:500]}...")  # Debugging
    
    try:
        response = model.generate_content(prompt)
        print("Received system scope from Gemini")
        return response.text
    except Exception as e:
        print(f"Error during API call: {e}")
        return None

# Step 4: Hardcode the system scope for comparison
def hardcoded_system_scope():
    system_scope = """
    The system is designed to manage student records, track attendance, and facilitate communication between students, parents, and teachers. 
    The system allows teachers to assign grades, monitor attendance, and send notifications to parents. Parents can view their children's grades and attendance records and communicate with teachers. 
    Students can access their grades, view their attendance history, and receive notifications for upcoming assignments and events. The system integrates with external messaging platforms for real-time notifications.
    """
    print("Hardcoded system scope:")
    print(system_scope[:500])  # Print first 500 characters for debugging
    return system_scope

# Step 5: Compare the Gemini output with the hardcoded system scope
def compare_scopes(gemini_scope, hardcoded_scope):
    print("Comparing Gemini-generated scope with hardcoded system scope")
    
    documents = [gemini_scope, hardcoded_scope]
    
    # Convert the documents to a TF-IDF matrix
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()
    
    # Compute the cosine similarity
    cosine_sim = cosine_similarity(vectors)
    
    # The matrix contains similarity scores. We want the score between the two scopes.
    similarity_score = cosine_sim[0][1]
    print(f"Similarity score: {similarity_score:.2f}")
    
    return similarity_score

# Example usage:
image_path = 'ClassDiagram.jpg'

print(f"Starting process for image: {image_path}")
extracted_text = extract_text_from_image(image_path)

# Step 1: Send the extracted text to Gemini for structuring into classes, attributes, and methods
structured_output = generate_structured_output(extracted_text)

# Step 2: Generate the system scope from the structured output
if structured_output:
    print("Structured Output from Gemini:")
    print(structured_output)

    # Step 3: Generate the system scope
    system_scope = generate_system_scope(structured_output)
    
    if system_scope:
        print("System Scope from Gemini:")
        print(system_scope)

        # Step 4: Use the hardcoded system scope for comparison
        hardcoded_scope = hardcoded_system_scope()

        # Step 5: Compare the Gemini system scope with the hardcoded system scope
        similarity = compare_scopes(system_scope, hardcoded_scope)
        
        print(f"The similarity between the hardcoded scope and the Gemini-generated scope is: {similarity:.2f}")
    else:
        print("Failed to generate system scope from Gemini")
else:
    print("Failed to generate structured output from Gemini")
