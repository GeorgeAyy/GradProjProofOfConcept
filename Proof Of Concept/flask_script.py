from flask import Flask, request, jsonify
import pytesseract
from PIL import Image
import cv2
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configure Gemini API
api_key = "AIzaSyDOExmBe0spo7h7PXGRbFqiPRPzfn5FdxE"
genai.configure(api_key=api_key)

app = Flask(__name__)

# Function to preprocess the image
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)[1]
    return img

# Function to extract text from image
def extract_text_from_image(image_path):
    preprocessed_image = preprocess_image(image_path)
    text = pytesseract.image_to_string(preprocessed_image)
    return text

# Function to generate structured output
def generate_structured_output(extracted_text):
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f"Based on the following text from a class diagram, extract and structure the information into classes with their attributes and methods:\n\n{extracted_text}\n\nPlease return the structured classes in the following format:\n\nClass: [Class Name]\nAttributes: [List of Attributes]\nMethods: [List of Methods]"
    response = model.generate_content(prompt)
    return response.text

# Function to generate system scope
def generate_system_scope(structured_output):
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f"Create a system scope based on the following structured class diagram. Assume the functionality of each class and write a clear system scope:\n\n{structured_output}\n\nReturn the system scope in a well-structured paragraph."
    response = model.generate_content(prompt)
    return response.text

# Hardcoded system scope for comparison
def hardcoded_system_scope():
    return """
    The system is designed to manage student records, track attendance, and facilitate communication between students, parents, and teachers. 
    The system allows teachers to assign grades, monitor attendance, and send notifications to parents. Parents can view their children's grades and attendance records and communicate with teachers. 
    Students can access their grades, view their attendance history, and receive notifications for upcoming assignments and events. The system integrates with external messaging platforms for real-time notifications.
    """

# Function to compare scopes using cosine similarity
def compare_scopes(gemini_scope, hardcoded_scope):
    documents = [gemini_scope, hardcoded_scope]
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()
    cosine_sim = cosine_similarity(vectors)
    similarity_score = cosine_sim[0][1]
    return similarity_score

# Flask route to process the image and paragraph
@app.route('/process', methods=['POST'])
def process_image():
    data = request.get_json()
    image_path = data['imagePath']
    paragraph = data['paragraph']

    try:
        # Step 1: Extract text from the image
        extracted_text = extract_text_from_image(image_path)

        # Step 2: Generate structured output from extracted text
        structured_output = generate_structured_output(extracted_text)

        # Step 3: Generate system scope from structured output
        system_scope = generate_system_scope(structured_output)

        # Step 4: Compare the generated system scope with the hardcoded system scope
        hardcoded_scope = hardcoded_system_scope()
        similarity = compare_scopes(system_scope, hardcoded_scope)

        # Return the results
        return jsonify({
            'structuredOutput': structured_output,
            'systemScope': system_scope,
            'similarity': similarity
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
