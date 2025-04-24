import nltk
import random
import string
import re
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import wordnet, stopwords
from nltk.tag import pos_tag
from nltk import Tree
from nltk.chunk import ne_chunk
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('wordnet')

app = Flask(__name__)
CORS(app)

class MCQGenerator:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        
    def preprocess_text(self, text):
        """Clean and preprocess the input text."""
        # Remove extra spaces and special chars
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Remove unnecessary punctuation but keep sentence structure
        text = re.sub(r'[^\w\s\.\,\?\!\:\;]', '', text)
        
        # Split into sentences
        sentences = sent_tokenize(text)
        return sentences
    
    def get_key_phrases(self, sentence):
        """Extract potential key phrases from a sentence."""
        words = word_tokenize(sentence)
        tagged = pos_tag(words)
        
        # Extract noun phrases and named entities
        key_phrases = []
        
        # Get named entities
        named_entities = []
        ne_tree = ne_chunk(tagged)
        for subtree in ne_tree:
            if type(subtree) == Tree:
                entity = " ".join([word for word, tag in subtree.leaves()])
                named_entities.append(entity)
        
        # Identify noun phrases
        current_phrase = []
        for word, tag in tagged:
            if tag.startswith('NN'):
                current_phrase.append(word)
            elif current_phrase:
                key_phrases.append(" ".join(current_phrase))
                current_phrase = []
                
        if current_phrase:  # Add the last phrase if any
            key_phrases.append(" ".join(current_phrase))
            
        # Add named entities to key phrases
        key_phrases.extend(named_entities)
        
        # Remove stop words and short phrases
        key_phrases = [phrase for phrase in key_phrases 
                      if len(phrase) > 2 and 
                      not all(word.lower() in self.stop_words for word in word_tokenize(phrase))]
        
        return list(set(key_phrases))
    
    def generate_distractors(self, answer, count=3):
        """Generate distractor options for the answer."""
        distractors = []
        
        # Try to get synonyms from WordNet
        synonyms = []
        for syn in wordnet.synsets(answer):
            for lemma in syn.lemmas():
                if lemma.name().lower() != answer.lower() and '_' not in lemma.name():
                    synonyms.append(lemma.name())
        
        # If we have synonyms, use them
        if synonyms:
            distractors = list(set(synonyms))[:count]
        
        # If not enough distractors from synonyms, add random words
        if len(distractors) < count:
            # Get words with similar part of speech
            answer_pos = pos_tag([answer])[0][1]
            all_words = [word for word, pos in pos_tag(word_tokenize(' '.join(list(wordnet.words())[:1000]))) 
                        if pos == answer_pos and word.lower() != answer.lower()]
            
            if all_words:
                random_words = random.sample(all_words, min(count - len(distractors), len(all_words)))
                distractors.extend(random_words)
        
        # If still not enough, just create variations
        while len(distractors) < count:
            letters = string.ascii_lowercase
            distractor = ''.join(random.choice(letters) for i in range(len(answer)))
            distractors.append(distractor)
        
        return distractors[:count]
    
    def create_fill_in_blank(self, sentence, key_phrase):
        """Create a fill-in-the-blank question from a sentence."""
        # Replace the key phrase with a blank
        blank_sentence = sentence.replace(key_phrase, "________")
        
        return {
            "question_type": "fill-in-blank",
            "question": blank_sentence,
            "answer": key_phrase,
            "options": [key_phrase] + self.generate_distractors(key_phrase)
        }
    
    def create_definition_question(self, key_phrase):
        """Create a 'What is' definition question."""
        question = f"What is meant by {key_phrase}?"
        
        # Try to get a definition from WordNet
        definition = ""
        synsets = wordnet.synsets(key_phrase.replace(" ", "_"))
        if synsets:
            definition = synsets[0].definition()
        
        if not definition:
            # If no definition found, use a generic approach
            definition = f"The correct meaning of {key_phrase}"
        
        return {
            "question_type": "definition",
            "question": question,
            "answer": definition,
            "options": [definition] + self.generate_distractors(definition)
        }
    
    def create_true_false(self, sentence, key_phrase):
        """Create a true/false question based on the sentence."""
        # Create negation of the sentence
        words = word_tokenize(sentence)
        tagged = pos_tag(words)
        
        # Find a verb to negate
        verb_index = -1
        for i, (word, tag) in enumerate(tagged):
            if tag.startswith('VB'):
                verb_index = i
                break
        
        false_statement = sentence
        if verb_index >= 0:
            # Negate the verb
            if words[verb_index].lower() in ['is', 'are', 'was', 'were']:
                words[verb_index] += " not"
            elif words[verb_index].lower() == 'have':
                words[verb_index] = "don't have"
            else:
                words[verb_index] = "don't " + words[verb_index]
            
            false_statement = ' '.join(words)
        
        # Randomly choose if true or false question
        is_true = random.choice([True, False])
        
        return {
            "question_type": "true_false",
            "question": sentence if is_true else false_statement,
            "answer": "True" if is_true else "False",
            "options": ["True", "False"]
        }
    
    def generate_mcqs(self, text, num_questions=5):
        """Generate multiple-choice questions from the given text."""
        sentences = self.preprocess_text(text)
        mcqs = []
        
        # Track used sentences to avoid duplicates
        used_sentences = set()
        
        # Process each sentence
        for sentence in sentences:
            if len(mcqs) >= num_questions:
                break
                
            if sentence in used_sentences:
                continue
                
            key_phrases = self.get_key_phrases(sentence)
            if not key_phrases:
                continue
            
            # Choose a random key phrase
            key_phrase = random.choice(key_phrases)
            
            # Generate different question types
            question_types = [
                lambda s, kp: self.create_fill_in_blank(s, kp),
                lambda s, kp: self.create_definition_question(kp),
                lambda s, kp: self.create_true_false(s, kp)
            ]
            
            question_func = random.choice(question_types)
            question = question_func(sentence, key_phrase)
            
            # Shuffle options
            all_options = question["options"]
            random.shuffle(all_options)
            
            # Track the correct answer index
            correct_index = all_options.index(question["answer"])
            
            question["options"] = all_options
            question["correct_index"] = correct_index
            question["source_sentence"] = sentence
            
            mcqs.append(question)
            used_sentences.add(sentence)
        
        return mcqs

    def analyze_mcqs(self, mcqs):
        """Analyze the generated MCQs for quality and coverage."""
        analysis = {
            "total_questions": len(mcqs),
            "question_types": {},
            "difficulty_estimate": {},
            "coverage_score": 0
        }
        
        # Count question types
        for mcq in mcqs:
            q_type = mcq["question_type"]
            analysis["question_types"][q_type] = analysis["question_types"].get(q_type, 0) + 1
        
        # Estimate difficulty (simplified)
        difficulty_scores = []
        for mcq in mcqs:
            # Factors affecting difficulty:
            # 1. Length of the question
            # 2. Complexity of options
            q_length = len(mcq["question"].split())
            options_similarity = self._calculate_options_similarity(mcq["options"])
            
            # Combine factors for a difficulty score (0-1)
            difficulty = min(1.0, (q_length / 30) * 0.5 + options_similarity * 0.5)
            difficulty_scores.append(difficulty)
            
        # Calculate overall difficulty metrics
        if difficulty_scores:
            analysis["difficulty_estimate"] = {
                "average": np.mean(difficulty_scores),
                "min": min(difficulty_scores),
                "max": max(difficulty_scores)
            }
        
        # Calculate coverage (how much of the text is covered by questions)
        coverage = min(1.0, len(mcqs) / 10)  # Simple approximation
        analysis["coverage_score"] = coverage
        
        return analysis
    
    def _calculate_options_similarity(self, options):
        """Calculate how similar the options are (higher = more difficult)."""
        if len(options) <= 1:
            return 0
            
        # Simple method: average length of common prefix
        total_similarity = 0
        comparisons = 0
        
        for i in range(len(options)):
            for j in range(i+1, len(options)):
                opt1 = options[i]
                opt2 = options[j]
                
                # Calculate similarity based on length and common characters
                total_chars = len(opt1) + len(opt2)
                if total_chars == 0:
                    continue
                
                common_prefix = len(os.path.commonprefix([opt1, opt2]))
                total_similarity += common_prefix / total_chars
                comparisons += 1
                
        if comparisons == 0:
            return 0
        
        return total_similarity / comparisons


@app.route('/generate_mcqs', methods=['POST'])
def generate_mcqs():
    """API endpoint for generating MCQs from input text."""
    data = request.json
    text = data.get('text', '')
    num_questions = data.get('num_questions', 5)
    
    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    generator = MCQGenerator()
    mcqs = generator.generate_mcqs(text, num_questions)
    
    # Analyze the questions generated
    analysis = generator.analyze_mcqs(mcqs)
    
    return jsonify({"mcqs": mcqs, "analysis": analysis})



@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
